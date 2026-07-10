"""Stable, process-safe resource locking for Capstan lifecycle work.

This module deliberately has no runner, service, or CLI integration.  It
turns filesystem paths, endpoint identities, and explicit port reservations
into stable resource keys, then serializes those keys with POSIX ``flock``.

The filesystem defenses protect cooperating same-UID Capstan processes from
normal recursive cleanup and creation races.  They do not attempt to defend
against an intentionally malicious same-UID actor that can chmod directories
or SIGKILL a holder; that actor is outside this module's threat model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import errno
import fcntl
import hashlib
import ipaddress
import os
import pwd
import stat
import threading


_LOCK_DIRECTORY_MODE = 0o700
_LOCK_RESOURCE_DIRECTORY_MODE = 0o500
_LOCK_FILE_MODE = 0o600
_LOCK_FILE_NAME = "lock"
_LOCK_FILE_CREATE_RETRIES = 8
_LOCK_ROOT_COMPONENTS = (".capstan", "locks")
_FILESYSTEM_RESOURCE_PREFIXES = ("path:", "file:", "pid:", "manifest:")


@dataclass(frozen=True, slots=True)
class EndpointIdentity:
    """The lock-relevant endpoint identity: normalized host and effective port."""

    host: str
    port: int


@dataclass(frozen=True, slots=True)
class LockFailure:
    """A structured, safe-to-report reason a lock could not be acquired."""

    code: str
    message: str
    resource: str | None = None
    path: str | None = None
    errno: int | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.resource is not None:
            result["resource"] = self.resource
        if self.path is not None:
            result["path"] = self.path
        if self.errno is not None:
            result["errno"] = self.errno
        return result


class LifecycleLockError(RuntimeError):
    """Raised by the context-manager API when acquisition fails.

    The corresponding structured failure is available as ``.failure``.
    Code that needs to avoid exceptions can use :func:`acquire_locks`.
    """

    def __init__(self, failure: LockFailure):
        self.failure = failure
        super().__init__(f"{failure.code}: {failure.message}")


@dataclass(frozen=True, slots=True)
class LockInfo:
    operation: str
    resources: tuple[str, ...]
    paths: tuple[str, ...]
    lock_root: str
    resource_paths: tuple[str, ...]


@dataclass(slots=True)
class LockAcquireResult:
    """The explicit result surface for :func:`acquire_locks`."""

    lock: ResourceLock | None
    error: LockFailure | None

    @property
    def ok(self) -> bool:
        return self.lock is not None and self.error is None


class ResourceLock:
    """An acquired resource lock. ``release`` is idempotent."""

    def __init__(self, info: LockInfo, release: Callable[[], None]) -> None:
        self.info = info
        self._release = release
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._release()

    def __enter__(self) -> LockInfo:
        return self.info

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.release()


@dataclass(slots=True)
class _HeldScope:
    root: str
    resources: tuple[str, ...]
    paths: dict[str, str]
    resource_paths: dict[str, str]
    held_resources: list[_HeldResource]
    depth: int = 1


@dataclass(slots=True)
class _ThreadLockEntry:
    key: tuple[str, str]
    lock: threading.RLock
    references: int = 0


@dataclass(slots=True)
class _ThreadLockLease:
    entry: _ThreadLockEntry
    acquired: bool = False


@dataclass(slots=True)
class _HeldResource:
    resource: str
    directory_name: str
    directory_path: Path
    file_path: Path
    directory_fd: int
    file_fd: int
    directory_identity: tuple[int, int]
    file_identity: tuple[int, int]
    thread_lock: _ThreadLockLease


class _FailureSignal(Exception):
    def __init__(self, failure: LockFailure):
        self.failure = failure
        super().__init__(failure.message)


_thread_lock_guard = threading.Lock()
_thread_locks: dict[tuple[str, str], _ThreadLockEntry] = {}
_held = threading.local()


def default_lock_root() -> Path:
    """Return the fixed per-user lock namespace without consulting XDG state."""

    uid = os.geteuid()
    try:
        home = pwd.getpwuid(uid).pw_dir
    except KeyError as exc:  # pragma: no cover - unusual POSIX account setup
        raise LifecycleLockError(
            LockFailure("user_home_unavailable", f"no passwd entry for effective uid {uid}")
        ) from exc
    return Path(home, *_LOCK_ROOT_COMPONENTS)


def canonical_filesystem_path(path: str | Path) -> Path:
    """Return an absolute path with every currently resolvable symlink removed."""

    return Path(path).expanduser().resolve(strict=False)


def resolve_manifest_path(manifest: Any, value: str | Path) -> Path:
    """Resolve a manifest-relative filesystem value, including symlink aliases."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(manifest.path).parent / path
    return canonical_filesystem_path(path)


def endpoint_identity(endpoint: str) -> EndpointIdentity:
    """Normalize an HTTP(S) endpoint to its host and effective port.

    IPv6 literals are represented without URL brackets in ``EndpointIdentity``;
    :func:`endpoint_resource` adds brackets where needed to make an unambiguous
    key.
    """

    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid endpoint {endpoint!r}: {exc}") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.netloc or not host:
        raise ValueError(f"endpoint must be an absolute http(s) URL: {endpoint!r}")
    effective_port = port if port is not None else (443 if scheme == "https" else 80)
    if not 1 <= effective_port <= 65535:
        raise ValueError(f"endpoint port must be between 1 and 65535: {endpoint!r}")
    return _endpoint_identity(host, effective_port, endpoint=endpoint)


def _endpoint_identity(host: str, port: int, *, endpoint: str) -> EndpointIdentity:
    normalized_host = host.rstrip(".").lower()
    if not normalized_host:
        raise ValueError(f"endpoint has no host: {endpoint!r}")
    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        # DNS names are case-insensitive but must not be resolved here: a lock
        # key must be deterministic and never make a network call.
        if normalized_host == "localhost":
            normalized_host = "local"
    else:
        mapped = getattr(address, "ipv4_mapped", None)
        if address.is_loopback or address.is_unspecified or (mapped is not None and mapped.is_loopback):
            normalized_host = "local"
        else:
            normalized_host = address.compressed
    return EndpointIdentity(host=normalized_host, port=port)


def endpoint_resource(endpoint: str) -> str:
    identity = endpoint_identity(endpoint)
    return _endpoint_identity_resource(identity)


def _endpoint_identity_resource(identity: EndpointIdentity) -> str:
    host = identity.host
    rendered_host = f"[{host}]" if ":" in host else host
    return f"endpoint:{rendered_host}:{identity.port}"


def filesystem_resource(path: str | Path, *, kind: str = "path") -> str:
    """Make a filesystem resource key whose aliases resolve to one identity."""

    if kind not in {"path", "file", "pid", "manifest"}:
        raise ValueError(f"unsupported filesystem resource kind: {kind!r}")
    return f"{kind}:{canonical_filesystem_path(path)}"


def port_resource(port: int) -> str:
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError(f"port must be an integer between 1 and 65535: {port!r}")
    return f"tcp-port:{port}"


def _state_home() -> Path:
    """Compatibility-only PID/log location; never used for the lock namespace."""

    configured = os.environ.get("XDG_STATE_HOME")
    if configured:
        path = Path(configured).expanduser()
        if path.is_absolute():
            return path
    return Path.home() / ".local" / "state"


def pid_state_path(manifest: Any) -> Path:
    start = getattr(manifest, "start", None)
    configured = getattr(start, "pid_path", None) if start is not None else None
    if configured:
        return resolve_manifest_path(manifest, configured)
    return canonical_filesystem_path(_state_home() / "modelctl" / f"{manifest.id}.pid.json")


def log_path(manifest: Any) -> Path:
    start = getattr(manifest, "start", None)
    configured = getattr(start, "log_path", None) if start is not None else None
    if configured:
        return resolve_manifest_path(manifest, configured)
    return canonical_filesystem_path(_state_home() / "modelctl" / f"{manifest.id}.log")


def lifecycle_resources(*manifests: Any) -> tuple[str, ...]:
    """Return deterministic resource keys for one or more manifest lifecycles.

    Service labels are intentionally not part of this Slice 2 core.
    """

    resources: set[str] = set()
    for manifest in manifests:
        resources.add(filesystem_resource(manifest.path, kind="manifest"))
        resources.add(filesystem_resource(pid_state_path(manifest), kind="pid"))
        resources.add(endpoint_resource(manifest.endpoint))
        preflight = getattr(manifest, "preflight", None)
        for port in getattr(preflight, "exclusive_ports", ()):
            resources.add(port_resource(port))
    return tuple(sorted(resources))


def lifecycle_lock_paths(*manifests: Any, lock_root: str | Path | None = None) -> tuple[str, ...]:
    root = _normalized_lock_root(lock_root)
    return tuple(str(_lock_path(root, resource)) for resource in lifecycle_resources(*manifests))


def _normalized_lock_root(lock_root: str | Path | None) -> Path:
    try:
        raw = default_lock_root() if lock_root is None else Path(lock_root).expanduser()
    except LifecycleLockError as exc:
        raise _FailureSignal(exc.failure) from exc
    if not raw.is_absolute():
        raise _FailureSignal(
            LockFailure("invalid_lock_root", "lock root must be absolute", path=str(raw))
        )
    return Path(os.path.abspath(raw))


def _lock_path(root: Path, resource: str) -> Path:
    return _resource_directory_path(root, resource) / _LOCK_FILE_NAME


def _resource_directory_path(root: Path, resource: str) -> Path:
    digest = hashlib.sha256(resource.encode("utf-8")).hexdigest()
    return root / digest


def _normalize_resource(resource: str) -> str:
    if not isinstance(resource, str) or not resource or "\x00" in resource:
        raise _FailureSignal(
            LockFailure("invalid_resource", "resource keys must be non-empty strings without NUL")
        )
    for prefix in _FILESYSTEM_RESOURCE_PREFIXES:
        if resource.startswith(prefix):
            value = resource.removeprefix(prefix)
            if not value:
                raise _FailureSignal(
                    LockFailure("invalid_resource", f"filesystem resource {prefix!r} has no path", resource=resource)
                )
            try:
                return f"{prefix}{canonical_filesystem_path(value)}"
            except (OSError, ValueError) as exc:
                raise _FailureSignal(
                    LockFailure("invalid_resource", f"cannot canonicalize filesystem resource: {exc}", resource=resource)
                ) from exc
    if resource.startswith("endpoint:"):
        endpoint = resource.removeprefix("endpoint:")
        try:
            if "://" in endpoint:
                return endpoint_resource(endpoint)
            normalized_endpoint = _normalize_endpoint_resource(endpoint, resource)
            return resource if normalized_endpoint is None else normalized_endpoint
        except ValueError as exc:
            raise _FailureSignal(LockFailure("invalid_resource", str(exc), resource=resource)) from exc
    return resource


def _normalize_endpoint_resource(value: str, resource: str) -> str | None:
    if value.startswith("["):
        closing = value.find("]")
        if closing < 0 or not value[closing + 1 :].startswith(":"):
            return None
        host = value[1:closing]
        raw_port = value[closing + 2 :]
    else:
        host, separator, raw_port = value.rpartition(":")
        if not separator:
            return None
    try:
        port = int(raw_port)
    except ValueError as exc:
        return None
    if not 1 <= port <= 65535:
        return None
    return _endpoint_identity_resource(_endpoint_identity(host, port, endpoint=resource))


def _normalized_resources(resources: Iterable[str]) -> tuple[str, ...]:
    try:
        normalized = {_normalize_resource(resource) for resource in resources}
    except TypeError as exc:
        raise _FailureSignal(
            LockFailure("invalid_resource", "resources must be an iterable of strings")
        ) from exc
    if not normalized:
        raise _FailureSignal(LockFailure("invalid_resource", "at least one resource is required"))
    return tuple(sorted(normalized))


def _failure_from_oserror(code: str, message: str, *, path: Path | None = None, resource: str | None = None, exc: OSError) -> _FailureSignal:
    return _FailureSignal(
        LockFailure(code, f"{message}: {exc.strerror or exc}", resource=resource, path=None if path is None else str(path), errno=exc.errno)
    )


def _directory_open_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise _FailureSignal(
            LockFailure("platform_unsupported", "safe lifecycle locks require O_NOFOLLOW and O_DIRECTORY")
        )
    return os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0)


def _validate_user_owned_ancestor(st: os.stat_result, path: Path, *, seen_user_owned: bool) -> bool:
    """Validate the user-controlled suffix while allowing trusted system parents.

    A non-user-owned writable directory can only be a root-owned sticky
    temporary ancestor.  That permits normal POSIX temp roots such as ``/tmp``
    without trusting arbitrary shared writable namespaces.
    """

    uid = os.geteuid()
    if st.st_uid == uid:
        if stat.S_IMODE(st.st_mode) & 0o022:
            raise _FailureSignal(
                LockFailure("unsafe_lock_ancestor_mode", "lock namespace ancestor is group- or world-writable", path=str(path))
            )
        return True
    if seen_user_owned:
        raise _FailureSignal(
            LockFailure("unsafe_lock_ancestor_owner", "lock namespace ancestor is not owned by the effective user", path=str(path))
        )
    mode = stat.S_IMODE(st.st_mode)
    if mode & 0o022 and not (st.st_uid == 0 and mode & stat.S_ISVTX):
        raise _FailureSignal(
            LockFailure(
                "unsafe_lock_ancestor_mode",
                "non-user-owned writable lock namespace ancestor is not a trusted root-owned sticky directory",
                path=str(path),
            )
        )
    return False


def _open_lock_root(root: Path) -> int:
    """Create/open a trusted 0700 root, refusing every symlink in its ancestry."""

    flags = _directory_open_flags()
    try:
        fd = os.open("/", flags)
    except OSError as exc:
        raise _failure_from_oserror("namespace_open_failed", "cannot open filesystem root", path=Path("/"), exc=exc) from exc

    current = Path("/")
    seen_user_owned = False
    try:
        for component in root.parts[1:]:
            current = current / component
            created = False
            try:
                before = os.stat(component, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                try:
                    os.mkdir(component, _LOCK_DIRECTORY_MODE, dir_fd=fd)
                    created = True
                    before = os.stat(component, dir_fd=fd, follow_symlinks=False)
                except FileExistsError:
                    before = os.stat(component, dir_fd=fd, follow_symlinks=False)
                except OSError as exc:
                    raise _failure_from_oserror("namespace_create_failed", "cannot create lock namespace", path=current, exc=exc) from exc
            except OSError as exc:
                raise _failure_from_oserror("namespace_inspection_failed", "cannot inspect lock namespace", path=current, exc=exc) from exc

            if stat.S_ISLNK(before.st_mode):
                raise _FailureSignal(
                    LockFailure("symlink_ancestor", "lock namespace contains a symlink ancestor", path=str(current))
                )
            if not stat.S_ISDIR(before.st_mode):
                raise _FailureSignal(
                    LockFailure("unsafe_lock_ancestor_type", "lock namespace ancestor is not a directory", path=str(current))
                )
            seen_user_owned = _validate_user_owned_ancestor(before, current, seen_user_owned=seen_user_owned)
            try:
                next_fd = os.open(component, flags, dir_fd=fd)
            except OSError as exc:
                raise _failure_from_oserror("namespace_open_failed", "cannot open lock namespace", path=current, exc=exc) from exc
            opened = os.fstat(next_fd)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                os.close(next_fd)
                raise _FailureSignal(
                    LockFailure("namespace_changed", "lock namespace changed while being opened", path=str(current))
                )
            if created:
                try:
                    os.fchmod(next_fd, _LOCK_DIRECTORY_MODE)
                except OSError as exc:
                    os.close(next_fd)
                    raise _failure_from_oserror("namespace_mode_failed", "cannot secure new lock namespace", path=current, exc=exc) from exc
                opened = os.fstat(next_fd)
            os.close(fd)
            fd = next_fd

        root_stat = os.fstat(fd)
        if root_stat.st_uid != os.geteuid():
            raise _FailureSignal(
                LockFailure("unsafe_lock_root_owner", "lock root is not owned by the effective user", path=str(root))
            )
        if stat.S_IMODE(root_stat.st_mode) != _LOCK_DIRECTORY_MODE:
            raise _FailureSignal(
                LockFailure("unsafe_lock_root_mode", "lock root mode must be 0700", path=str(root))
            )
        return fd
    except Exception:
        os.close(fd)
        raise


def _validate_lock_file(st: os.stat_result, path: Path) -> None:
    if stat.S_ISLNK(st.st_mode):
        raise _FailureSignal(
            LockFailure("symlink_lock_file", "lock file must not be a symlink", path=str(path))
        )
    if not stat.S_ISREG(st.st_mode):
        raise _FailureSignal(
            LockFailure("unsafe_lock_file_type", "lock file must be a regular file", path=str(path))
        )
    if st.st_uid != os.geteuid():
        raise _FailureSignal(
            LockFailure("unsafe_lock_file_owner", "lock file is not owned by the effective user", path=str(path))
        )
    if stat.S_IMODE(st.st_mode) != _LOCK_FILE_MODE:
        raise _FailureSignal(
            LockFailure("unsafe_lock_file_mode", "lock file mode must be 0600", path=str(path))
        )
    if st.st_nlink != 1:
        raise _FailureSignal(
            LockFailure("unsafe_lock_file_links", "lock file must have exactly one hard link", path=str(path))
        )


def _identity(st: os.stat_result) -> tuple[int, int]:
    return st.st_dev, st.st_ino


def _validate_resource_directory(
    st: os.stat_result, path: Path, *, allow_initialization: bool
) -> None:
    if stat.S_ISLNK(st.st_mode):
        raise _FailureSignal(
            LockFailure("symlink_resource_directory", "resource directory must not be a symlink", path=str(path))
        )
    if not stat.S_ISDIR(st.st_mode):
        raise _FailureSignal(
            LockFailure("unsafe_resource_directory_type", "resource path must be a directory", path=str(path))
        )
    if st.st_uid != os.geteuid():
        raise _FailureSignal(
            LockFailure("unsafe_resource_directory_owner", "resource directory is not owned by the effective user", path=str(path))
        )
    allowed_modes = {_LOCK_RESOURCE_DIRECTORY_MODE}
    if allow_initialization:
        allowed_modes.add(_LOCK_DIRECTORY_MODE)
    if stat.S_IMODE(st.st_mode) not in allowed_modes:
        raise _FailureSignal(
            LockFailure(
                "unsafe_resource_directory_mode",
                "resource directory mode must be 0500 outside lock-file initialization",
                path=str(path),
            )
        )


def _open_resource_directory(root_fd: int, root: Path, resource: str) -> tuple[int, str, Path]:
    path = _resource_directory_path(root, resource)
    directory_name = path.name
    flags = _directory_open_flags()
    try:
        try:
            before = os.stat(directory_name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            try:
                # A freshly created resource directory is already immutable.
                # It is made writable only for the brief, fd-backed creation
                # of a missing lock file below, where all failures restore 0500.
                os.mkdir(directory_name, _LOCK_RESOURCE_DIRECTORY_MODE, dir_fd=root_fd)
                before = os.stat(directory_name, dir_fd=root_fd, follow_symlinks=False)
            except FileExistsError:
                before = os.stat(directory_name, dir_fd=root_fd, follow_symlinks=False)
            except OSError as exc:
                raise _failure_from_oserror(
                    "resource_directory_create_failed",
                    "cannot create resource directory",
                    path=path,
                    resource=resource,
                    exc=exc,
                ) from exc
        except OSError as exc:
            raise _failure_from_oserror(
                "resource_directory_inspection_failed",
                "cannot inspect resource directory",
                path=path,
                resource=resource,
                exc=exc,
            ) from exc

        _validate_resource_directory(before, path, allow_initialization=True)
        try:
            directory_fd = os.open(directory_name, flags, dir_fd=root_fd)
        except OSError as exc:
            raise _failure_from_oserror(
                "resource_directory_open_failed",
                "cannot open resource directory",
                path=path,
                resource=resource,
                exc=exc,
            ) from exc
        try:
            opened = os.fstat(directory_fd)
            if _identity(opened) != _identity(before):
                raise _FailureSignal(
                    LockFailure(
                        "resource_directory_changed",
                        "resource directory changed while being opened",
                        resource=resource,
                        path=str(path),
                    )
                )
            _validate_resource_directory(opened, path, allow_initialization=True)
            return directory_fd, directory_name, path
        except Exception:
            os.close(directory_fd)
            raise
    except _FailureSignal:
        raise


def _secure_resource_directory(directory_fd: int, path: Path, resource: str) -> tuple[int, int]:
    try:
        os.fchmod(directory_fd, _LOCK_RESOURCE_DIRECTORY_MODE)
        secured = os.fstat(directory_fd)
    except OSError as exc:
        raise _failure_from_oserror(
            "resource_directory_mode_failed",
            "cannot secure resource directory",
            path=path,
            resource=resource,
            exc=exc,
        ) from exc
    _validate_resource_directory(secured, path, allow_initialization=False)
    return _identity(secured)


def _open_lock_file(
    root_fd: int,
    root: Path,
    resource: str,
    thread_lock: _ThreadLockLease,
) -> _HeldResource:
    directory_fd, directory_name, directory_path = _open_resource_directory(root_fd, root, resource)
    path = directory_path / _LOCK_FILE_NAME
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        os.close(directory_fd)
        raise _FailureSignal(LockFailure("platform_unsupported", "safe lifecycle locks require O_NOFOLLOW"))
    flags |= nofollow
    file_fd = -1
    try:
        for _attempt in range(_LOCK_FILE_CREATE_RETRIES):
            created = False
            try:
                before = os.stat(_LOCK_FILE_NAME, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                try:
                    os.fchmod(directory_fd, _LOCK_DIRECTORY_MODE)
                    file_fd = os.open(
                        _LOCK_FILE_NAME,
                        flags | os.O_CREAT | os.O_EXCL,
                        _LOCK_FILE_MODE,
                        dir_fd=directory_fd,
                    )
                    created = True
                except FileExistsError:
                    continue
                except OSError as exc:
                    raise _failure_from_oserror(
                        "lock_file_open_failed",
                        "cannot create lock file",
                        path=path,
                        resource=resource,
                        exc=exc,
                    ) from exc
            except OSError as exc:
                raise _failure_from_oserror(
                    "lock_file_inspection_failed",
                    "cannot inspect lock file",
                    path=path,
                    resource=resource,
                    exc=exc,
                ) from exc
            else:
                _validate_lock_file(before, path)
                try:
                    file_fd = os.open(_LOCK_FILE_NAME, flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise _failure_from_oserror(
                        "lock_file_open_failed",
                        "cannot open lock file",
                        path=path,
                        resource=resource,
                        exc=exc,
                    ) from exc

            try:
                if created:
                    os.fchmod(file_fd, _LOCK_FILE_MODE)
                opened = os.fstat(file_fd)
                _validate_lock_file(opened, path)
                if not created and _identity(opened) != _identity(before):
                    raise _FailureSignal(
                        LockFailure(
                            "lock_file_changed",
                            "lock file changed while being opened",
                            resource=resource,
                            path=str(path),
                        )
                    )
                directory_identity = _secure_resource_directory(directory_fd, directory_path, resource)
                file_identity = _identity(opened)
                return _HeldResource(
                    resource=resource,
                    directory_name=directory_name,
                    directory_path=directory_path,
                    file_path=path,
                    directory_fd=directory_fd,
                    file_fd=file_fd,
                    directory_identity=directory_identity,
                    file_identity=file_identity,
                    thread_lock=thread_lock,
                )
            except Exception:
                os.close(file_fd)
                file_fd = -1
                raise
        raise _FailureSignal(
            LockFailure(
                "lock_file_creation_race",
                f"lock file kept changing during {_LOCK_FILE_CREATE_RETRIES} creation attempts",
                resource=resource,
                path=str(path),
            )
        )
    except Exception:
        if file_fd >= 0:
            os.close(file_fd)
        try:
            _secure_resource_directory(directory_fd, directory_path, resource)
        finally:
            os.close(directory_fd)
        raise


def _validate_held_resource(root_fd: int, held_resource: _HeldResource) -> None:
    try:
        named_directory = os.stat(
            held_resource.directory_name, dir_fd=root_fd, follow_symlinks=False
        )
        opened_directory = os.fstat(held_resource.directory_fd)
        _validate_resource_directory(
            named_directory, held_resource.directory_path, allow_initialization=False
        )
        _validate_resource_directory(
            opened_directory, held_resource.directory_path, allow_initialization=False
        )
        if (
            _identity(named_directory) != held_resource.directory_identity
            or _identity(opened_directory) != held_resource.directory_identity
        ):
            raise _FailureSignal(
                LockFailure(
                    "resource_directory_changed",
                    "resource directory changed during lock acquisition",
                    resource=held_resource.resource,
                    path=str(held_resource.directory_path),
                )
            )
        named_file = os.stat(_LOCK_FILE_NAME, dir_fd=held_resource.directory_fd, follow_symlinks=False)
        opened_file = os.fstat(held_resource.file_fd)
        _validate_lock_file(named_file, held_resource.file_path)
        _validate_lock_file(opened_file, held_resource.file_path)
        if (
            _identity(named_file) != held_resource.file_identity
            or _identity(opened_file) != held_resource.file_identity
        ):
            raise _FailureSignal(
                LockFailure(
                    "lock_file_changed",
                    "lock file changed during lock acquisition",
                    resource=held_resource.resource,
                    path=str(held_resource.file_path),
                )
            )
    except _FailureSignal:
        raise
    except OSError as exc:
        raise _failure_from_oserror(
            "lock_resource_validation_failed",
            "cannot validate acquired lock resource",
            path=held_resource.file_path,
            resource=held_resource.resource,
            exc=exc,
        ) from exc


def _reserve_thread_lock(root: Path, resource: str) -> _ThreadLockLease:
    key = (str(root), resource)
    with _thread_lock_guard:
        entry = _thread_locks.get(key)
        if entry is None:
            entry = _ThreadLockEntry(key=key, lock=threading.RLock())
            _thread_locks[key] = entry
        entry.references += 1
    return _ThreadLockLease(entry=entry)


def _acquire_thread_lock(root: Path, resource: str, *, blocking: bool) -> _ThreadLockLease | None:
    lease = _reserve_thread_lock(root, resource)
    try:
        lease.acquired = lease.entry.lock.acquire(blocking=blocking)
    except Exception:
        _release_thread_lock(lease)
        raise
    if lease.acquired:
        return lease
    _release_thread_lock(lease)
    return None


def _release_thread_lock(lease: _ThreadLockLease) -> None:
    if lease.acquired:
        try:
            lease.entry.lock.release()
        finally:
            lease.acquired = False
    with _thread_lock_guard:
        lease.entry.references -= 1
        if lease.entry.references == 0 and _thread_locks.get(lease.entry.key) is lease.entry:
            del _thread_locks[lease.entry.key]


def _release_scope(scope: _HeldScope) -> None:
    for held_resource in reversed(scope.held_resources):
        try:
            fcntl.flock(held_resource.file_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            try:
                os.close(held_resource.file_fd)
            except OSError:
                pass
    for held_resource in reversed(scope.held_resources):
        try:
            os.close(held_resource.directory_fd)
        except OSError:
            pass
    for held_resource in reversed(scope.held_resources):
        try:
            _release_thread_lock(held_resource.thread_lock)
        except RuntimeError:
            pass


def _release_held_scope(scope: _HeldScope) -> None:
    current = getattr(_held, "scope", None)
    if current is not scope:
        return
    scope.depth -= 1
    if scope.depth > 0:
        return
    try:
        _release_scope(scope)
    finally:
        del _held.scope


def _failure_result(failure: LockFailure) -> LockAcquireResult:
    return LockAcquireResult(lock=None, error=failure)


def acquire_locks(
    resources: Iterable[str],
    *,
    operation: str,
    lock_root: str | Path | None = None,
    blocking: bool = True,
) -> LockAcquireResult:
    """Acquire sorted resource locks and return either a lease or ``LockFailure``.

    Nested acquisitions by the same thread may request a subset of the outer
    resources. Widening a nested scope is rejected instead of risking a lock
    order inversion and deadlock.
    """

    try:
        if not isinstance(operation, str) or not operation:
            raise _FailureSignal(LockFailure("invalid_operation", "operation must be a non-empty string"))
        normalized_resources = _normalized_resources(resources)
        root = _normalized_lock_root(lock_root)
        held = getattr(_held, "scope", None)
        if held is not None:
            if held.root != str(root):
                raise _FailureSignal(
                    LockFailure("nested_lock_root_mismatch", "nested acquisition must use the outer lock root", path=str(root))
                )
            if not set(normalized_resources).issubset(held.resources):
                raise _FailureSignal(
                    LockFailure("nested_scope_widening", "nested acquisition cannot add resources", resource=",".join(normalized_resources))
                )
            held.depth += 1
            info = LockInfo(
                operation,
                normalized_resources,
                tuple(held.paths[resource] for resource in normalized_resources),
                held.root,
                tuple(held.resource_paths[resource] for resource in normalized_resources),
            )
            return LockAcquireResult(lock=ResourceLock(info, lambda: _release_held_scope(held)), error=None)

        root_fd = _open_lock_root(root)
        held_resources: list[_HeldResource] = []
        paths: dict[str, str] = {}
        resource_paths: dict[str, str] = {}
        try:
            for resource in normalized_resources:
                thread_lock = _acquire_thread_lock(root, resource, blocking=blocking)
                if thread_lock is None:
                    raise _FailureSignal(
                        LockFailure("contended", "resource is held by another thread", resource=resource)
                    )
                try:
                    held_resource = _open_lock_file(root_fd, root, resource, thread_lock)
                except Exception:
                    _release_thread_lock(thread_lock)
                    raise
                held_resources.append(held_resource)
                paths[resource] = str(held_resource.file_path)
                resource_paths[resource] = str(held_resource.directory_path)
                try:
                    flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
                    fcntl.flock(held_resource.file_fd, flags)
                except OSError as exc:
                    code = "contended" if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK} else "flock_failed"
                    raise _failure_from_oserror(
                        code,
                        "cannot acquire resource lock",
                        path=held_resource.file_path,
                        resource=resource,
                        exc=exc,
                    ) from exc
                _validate_held_resource(root_fd, held_resource)
            # Keep both directory and file inode identities valid through the
            # entire multi-resource acquisition, not merely each individual
            # flock call.
            for held_resource in held_resources:
                _validate_held_resource(root_fd, held_resource)
        except Exception:
            _release_scope(
                _HeldScope(str(root), normalized_resources, paths, resource_paths, held_resources)
            )
            raise
        finally:
            os.close(root_fd)

        scope = _HeldScope(str(root), normalized_resources, paths, resource_paths, held_resources)
        _held.scope = scope
        info = LockInfo(
            operation,
            normalized_resources,
            tuple(paths[resource] for resource in normalized_resources),
            str(root),
            tuple(resource_paths[resource] for resource in normalized_resources),
        )
        return LockAcquireResult(lock=ResourceLock(info, lambda: _release_held_scope(scope)), error=None)
    except _FailureSignal as signal:
        return _failure_result(signal.failure)
    except OSError as exc:
        return _failure_result(
            LockFailure("lock_system_error", f"lifecycle lock system error: {exc.strerror or exc}", errno=exc.errno)
        )
    except Exception as exc:
        return _failure_result(LockFailure("lock_system_error", f"lifecycle lock system error: {type(exc).__name__}: {exc}"))


@contextmanager
def resource_lock(
    resources: Iterable[str],
    *,
    operation: str,
    lock_root: str | Path | None = None,
    blocking: bool = True,
) -> Iterator[LockInfo]:
    """Exception-oriented context manager over :func:`acquire_locks`."""

    result = acquire_locks(resources, operation=operation, lock_root=lock_root, blocking=blocking)
    if not result.ok:
        assert result.error is not None
        raise LifecycleLockError(result.error)
    assert result.lock is not None
    try:
        yield result.lock.info
    finally:
        result.lock.release()


@contextmanager
def lifecycle_lock(
    operation: str,
    *manifests: Any,
    lock_root: str | Path | None = None,
    blocking: bool = True,
) -> Iterator[LockInfo]:
    """Lock all lifecycle resources for the supplied manifests."""

    with resource_lock(
        lifecycle_resources(*manifests),
        operation=operation,
        lock_root=lock_root,
        blocking=blocking,
    ) as info:
        yield info


__all__ = [
    "EndpointIdentity",
    "LifecycleLockError",
    "LockAcquireResult",
    "LockFailure",
    "LockInfo",
    "ResourceLock",
    "acquire_locks",
    "canonical_filesystem_path",
    "default_lock_root",
    "endpoint_identity",
    "endpoint_resource",
    "filesystem_resource",
    "lifecycle_lock",
    "lifecycle_lock_paths",
    "lifecycle_resources",
    "log_path",
    "pid_state_path",
    "port_resource",
    "resolve_manifest_path",
    "resource_lock",
]
