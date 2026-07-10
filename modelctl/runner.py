from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import hashlib
import json
import os
import secrets
import stat
import subprocess
import time

from .http import http_json
from .lifecycle import LifecycleLockError, endpoint_identity, lifecycle_lock
from .manifest import ModelManifest
from .system import (
    ProcessIdentity,
    capture_process_identity,
    live_process_group_members,
    prove_endpoint_owned_by_identity,
    reap_popen,
    retain_popen,
    terminate_process_identity,
)

_MAX_STATE_BYTES = 128 * 1024


class PIDStateError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _Snapshot:
    raw: Any
    payload: bytes
    device: int
    inode: int


def default_pid_path(manifest: ModelManifest) -> Path:
    if manifest.start and manifest.start.pid_path:
        return Path(manifest.start.pid_path)
    state_dir = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "modelctl"
    return state_dir / f"{manifest.id}.pid.json"


def _state_path(manifest: ModelManifest) -> Path:
    configured = default_pid_path(manifest).expanduser()
    if not configured.is_absolute():
        configured = Path(manifest.path).parent / configured
    return configured.parent.resolve(strict=False) / configured.name


def default_log_path(manifest: ModelManifest) -> Path:
    if manifest.start and manifest.start.log_path:
        return Path(manifest.start.log_path)
    state_dir = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "modelctl"
    return state_dir / f"{manifest.id}.log"


def _manifest_path(manifest: ModelManifest) -> str:
    return str(Path(manifest.path).expanduser().resolve(strict=False))


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _read_snapshot(path: Path) -> _Snapshot | None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PIDStateError(f"untrusted pid state: {type(exc).__name__}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise PIDStateError("untrusted pid state: not a regular file")
        if before.st_uid != os.geteuid() or stat.S_IMODE(before.st_mode) != 0o600 or before.st_nlink != 1:
            raise PIDStateError("untrusted pid state: owner, mode, or link count")
        if before.st_size < 0 or before.st_size > _MAX_STATE_BYTES:
            raise PIDStateError("untrusted pid state: size out of bounds")
        payload = b""
        while len(payload) < before.st_size:
            chunk = os.read(fd, before.st_size - len(payload))
            if not chunk:
                raise PIDStateError("untrusted pid state: short read")
            payload += chunk
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino, after.st_size, after.st_nlink, stat.S_IMODE(after.st_mode), after.st_uid) != (
            before.st_dev, before.st_ino, before.st_size, 1, 0o600, os.geteuid()
        ):
            raise PIDStateError("untrusted pid state: changed while reading")
        try:
            raw = json.loads(payload.decode("utf-8", "strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PIDStateError(f"invalid pid state: {type(exc).__name__}: {exc}") from exc
        return _Snapshot(raw=raw, payload=payload, device=before.st_dev, inode=before.st_ino)
    finally:
        os.close(fd)


def _atomic_write(path: Path, raw: Any, *, exclusive: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(raw, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if len(payload) > _MAX_STATE_BYTES:
        raise PIDStateError("pid state exceeds maximum size")
    tmp = path.with_name(f".{path.name}.tmp.{secrets.token_hex(16)}")
    fd: int | None = None
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
        os.fchmod(fd, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("short pid-state write")
            offset += written
        os.fsync(fd)
        os.close(fd)
        fd = None
        if exclusive:
            os.link(tmp, path, follow_symlinks=False)
            os.unlink(tmp)
        else:
            os.replace(tmp, path)
        return path
    finally:
        if fd is not None:
            os.close(fd)
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def read_pid_state(manifest: ModelManifest) -> dict[str, Any] | None:
    try:
        snapshot = _read_snapshot(_state_path(manifest))
    except PIDStateError as exc:
        # Never expose untrusted bytes as state. Return a synthetic blocker so
        # legacy doctor/reporting callers preserve the path instead of deleting it.
        return {
            "schema_version": 2,
            "kind": "blocked_untrusted",
            "manifest": _manifest_path(manifest),
            "error": str(exc),
        }
    return snapshot.raw if snapshot is not None and isinstance(snapshot.raw, dict) else None


def write_pid_state(manifest: ModelManifest, state: dict[str, Any]) -> Path:
    """Compatibility writer; lifecycle start uses exclusive pending publication."""
    return _atomic_write(_state_path(manifest), state, exclusive=False)


def _canonical_command(command: list[str]) -> list[str]:
    result = list(command)
    if result and (os.path.isabs(result[0]) or os.sep in result[0]):
        result[0] = str(Path(result[0]).expanduser().resolve(strict=False))
    return result


def _launch_fingerprint(manifest: ModelManifest) -> dict[str, Any]:
    if manifest.start is None:
        raise ValueError("manifest has no [start] section")
    command = _canonical_command(manifest.start.command)
    cwd = Path(manifest.start.cwd or manifest.path.parent).expanduser()
    if not cwd.is_absolute():
        cwd = manifest.path.parent / cwd
    cwd_text = str(cwd.resolve(strict=False))
    digest = hashlib.sha256(json.dumps({"command": command, "cwd": cwd_text}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"command": command, "cwd": cwd_text, "sha256": digest}


def _pending_state(manifest: ModelManifest, transaction_id: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "kind": "launch_pending",
        "transaction_id": transaction_id,
        "manifest": _manifest_path(manifest),
        "created_at": _timestamp(),
    }


def _active_state(manifest: ModelManifest, identity: ProcessIdentity, log_path: Path) -> dict[str, Any]:
    endpoint = endpoint_identity(manifest.endpoint)
    return {
        "schema_version": 2,
        "kind": "active",
        "pid": identity.leader_pid,
        "pgid": identity.pgid,
        "birth_token": identity.birth_token,
        "manifest": _manifest_path(manifest),
        "endpoint": {"host": endpoint.host, "port": endpoint.port},
        "fingerprint": _launch_fingerprint(manifest),
        "log_path": str(log_path.expanduser().resolve(strict=False)),
        "started_at": _timestamp(),
    }


def _validate_active(manifest: ModelManifest, raw: Any) -> ProcessIdentity | None:
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw.get("schema_version") != 2 or raw.get("kind") != "active":
        return None
    if raw.get("manifest") != _manifest_path(manifest) or raw.get("fingerprint") != _launch_fingerprint(manifest):
        return None
    expected_endpoint = endpoint_identity(manifest.endpoint)
    if raw.get("endpoint") != {"host": expected_endpoint.host, "port": expected_endpoint.port}:
        return None
    expected_log = default_log_path(manifest).expanduser()
    if not expected_log.is_absolute():
        expected_log = manifest.path.parent / expected_log
    if raw.get("log_path") != str(expected_log.resolve(strict=False)):
        return None
    started_at = raw.get("started_at")
    if not isinstance(started_at, str) or not started_at.endswith("Z"):
        return None
    try:
        datetime.fromisoformat(started_at[:-1] + "+00:00")
    except ValueError:
        return None
    pid, pgid, token = raw.get("pid"), raw.get("pgid"), raw.get("birth_token")
    if type(pid) is not int or type(pgid) is not int or pid <= 0 or pgid <= 0 or not isinstance(token, str) or not token:
        return None
    try:
        return ProcessIdentity(pid, pgid, token)
    except ValueError:
        return None


def _is_pending(manifest: ModelManifest, raw: Any, transaction_id: str | None = None) -> bool:
    return bool(
        isinstance(raw, dict)
        and raw.get("schema_version") == 2
        and raw.get("kind") == "launch_pending"
        and raw.get("manifest") == _manifest_path(manifest)
        and isinstance(raw.get("transaction_id"), str)
        and raw.get("transaction_id")
        and (transaction_id is None or raw.get("transaction_id") == transaction_id)
    )


def _inspect(manifest: ModelManifest) -> dict[str, Any]:
    path = _state_path(manifest)
    try:
        snapshot = _read_snapshot(path)
    except PIDStateError as exc:
        return {"status": "blocked", "error": str(exc), "path": path}
    if snapshot is None:
        return {"status": "absent", "path": path}
    raw = snapshot.raw
    if _is_pending(manifest, raw):
        return {"status": "pending", "raw": raw, "snapshot": snapshot, "path": path}
    identity = _validate_active(manifest, raw)
    if identity is None:
        status = "owner_mismatch" if isinstance(raw, dict) and raw.get("manifest") not in {None, _manifest_path(manifest)} else "blocked"
        return {"status": status, "raw": raw, "snapshot": snapshot, "path": path}
    try:
        current = capture_process_identity(identity.leader_pid)
    except Exception as exc:
        return {"status": "unproven", "raw": raw, "identity": identity, "error": f"{type(exc).__name__}: {exc}", "path": path}
    if current is None:
        try:
            members = live_process_group_members(identity.pgid)
        except Exception as exc:
            return {"status": "unproven", "raw": raw, "identity": identity, "error": f"{type(exc).__name__}: {exc}", "path": path}
        if members:
            return {"status": "group_live", "raw": raw, "identity": identity, "members": members, "path": path}
        return {"status": "dead", "raw": raw, "identity": identity, "snapshot": snapshot, "path": path}
    if current != identity:
        return {"status": "identity_mismatch", "raw": raw, "identity": identity, "observed": current, "path": path}
    return {"status": "live", "raw": raw, "identity": identity, "snapshot": snapshot, "path": path}


def pid_state_owner_mismatch(manifest: ModelManifest, state: dict[str, Any] | None = None) -> bool:
    path = _state_path(manifest)
    if state is None:
        state = read_pid_state(manifest)
        if state is None:
            return path.exists() or path.is_symlink()
    if not isinstance(state, dict):
        return True
    if state.get("manifest") not in {None, _manifest_path(manifest), str(manifest.path)}:
        return True
    if state.get("kind") == "launch_pending":
        return True
    if state.get("schema_version") == 2 or state.get("kind") == "active":
        return _validate_active(manifest, state) is None
    return False


def active_pid(manifest: ModelManifest) -> int | None:
    inspection = _inspect(manifest)
    identity = inspection.get("identity")
    return identity.leader_pid if inspection.get("status") == "live" and isinstance(identity, ProcessIdentity) else None


def readiness_check(manifest: ModelManifest, timeout: float = 10.0) -> dict[str, Any]:
    url = manifest.start.readiness_url if manifest.start and manifest.start.readiness_url else manifest.models_url
    contains = manifest.start.readiness_contains if manifest.start else manifest.model_id
    status, body, text = http_json("GET", url, timeout=timeout)
    ready = 200 <= status < 300 and (not contains or contains in text)
    return {"ready": ready, "status": status, "url": url, "contains": contains, "body": body if isinstance(body, dict) else text[:500]}


def wait_ready(manifest: ModelManifest, timeout_sec: float | None = None) -> dict[str, Any]:
    if timeout_sec is None:
        timeout_sec = manifest.start.startup_timeout_sec if manifest.start else 120
    deadline = time.time() + timeout_sec
    last: dict[str, Any] | None = None
    while time.time() < deadline:
        if manifest.start and active_pid(manifest) is None:
            return {"ready": False, "error": "process exited before readiness", "last": last}
        try:
            last = readiness_check(manifest, timeout=max(0.001, min(2.0, deadline - time.time())))
            if last.get("ready"):
                return last
        except Exception as exc:
            last = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
        time.sleep(min(0.2, max(0.01, deadline - time.time())))
    return {"ready": False, "error": "timeout", "last": last}


def _remove_if(manifest: ModelManifest, predicate: Callable[[Any], bool]) -> bool:
    path = _state_path(manifest)
    try:
        snapshot = _read_snapshot(path)
    except PIDStateError:
        return False
    if snapshot is None or not predicate(snapshot.raw):
        return False
    path.unlink()
    return True


def _guarded_remove(manifest: ModelManifest, predicate: Callable[[Any], bool]) -> tuple[bool, str | None]:
    try:
        return _remove_if(manifest, predicate), None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _replace_pending(manifest: ModelManifest, transaction_id: str, active: dict[str, Any]) -> Path:
    snapshot = _read_snapshot(_state_path(manifest))
    if snapshot is None or not _is_pending(manifest, snapshot.raw, transaction_id):
        raise PIDStateError("launch-pending transaction changed before activation")
    return _atomic_write(_state_path(manifest), active, exclusive=False)


def _capture_new_identity(proc: subprocess.Popen[Any]) -> ProcessIdentity | None:
    for _ in range(20):
        identity = capture_process_identity(proc.pid)
        if identity is not None:
            return identity
        poll = getattr(proc, "poll", None)
        if callable(poll) and poll() is not None:
            return None
        time.sleep(0.01)
    return None


def _retain_status(proc: subprocess.Popen[Any]) -> dict[str, Any]:
    try:
        retain_popen(proc)
        return {"retained": True}
    except Exception as exc:
        return {"retained": False, "error": f"{type(exc).__name__}: {exc}"}


def _cleanup_direct(proc: subprocess.Popen[Any]) -> dict[str, Any]:
    try:
        poll = getattr(proc, "poll", None)
        returncode = poll() if callable(poll) else None
        if returncode is None:
            terminate = getattr(proc, "terminate", None)
            wait = getattr(proc, "wait", None)
            kill = getattr(proc, "kill", None)
            if not callable(terminate) or not callable(wait):
                return {"leader_reaped": False, "group_death_certified": False, "error": "incomplete Popen handle", "retention": _retain_status(proc)}
            terminate()
            try:
                wait(timeout=1)
            except subprocess.TimeoutExpired:
                if not callable(kill):
                    raise
                kill()
                wait(timeout=1)
        reaped = bool(callable(poll) and poll() is not None)
        result: dict[str, Any] = {"leader_reaped": reaped, "group_death_certified": False}
        if not reaped:
            result["retention"] = _retain_status(proc)
        return result
    except Exception as exc:
        return {"leader_reaped": False, "group_death_certified": False, "error": f"{type(exc).__name__}: {exc}", "retention": _retain_status(proc)}


def _cleanup_exact(
    manifest: ModelManifest,
    identity: ProcessIdentity,
    proc: subprocess.Popen[Any],
    *,
    transaction_id: str,
) -> dict[str, Any]:
    try:
        terminated = terminate_process_identity(identity, timeout_sec=5)
    except Exception as exc:
        try:
            retain_popen(proc)
        except Exception:
            pass
        return {"group_death_certified": False, "error": f"{type(exc).__name__}: {exc}"}
    if not terminated:
        try:
            retain_popen(proc)
        except Exception:
            pass
        return {"group_death_certified": False}
    try:
        reap = reap_popen(proc, timeout_sec=1)
        retention = None if reap is not None else _retain_status(proc)
    except Exception as exc:
        reap = {"reaped": False, "error": f"{type(exc).__name__}: {exc}"}
        retention = _retain_status(proc)
    try:
        removed = _remove_if(
            manifest,
            lambda raw: _is_pending(manifest, raw, transaction_id) or _validate_active(manifest, raw) == identity,
        )
        state_error = None
    except Exception as exc:
        removed = False
        state_error = f"{type(exc).__name__}: {exc}"
    return {"group_death_certified": True, "state_removed": removed, "state_error": state_error, "reap": reap, "retention": retention}


def _durability_fields(manifest: ModelManifest, transaction_id: str, identity: ProcessIdentity | None = None) -> dict[str, Any]:
    try:
        snapshot = _read_snapshot(_state_path(manifest))
    except PIDStateError:
        return {"durable_blocker": True, "durable_pending": False, "durable_state_kind": "untrusted"}
    if snapshot is None:
        return {"durable_blocker": False, "durable_pending": False, "durable_state_kind": None}
    if _is_pending(manifest, snapshot.raw, transaction_id):
        return {"durable_blocker": True, "durable_pending": True, "durable_state_kind": "launch_pending"}
    if identity is not None and _validate_active(manifest, snapshot.raw) == identity:
        return {"durable_blocker": True, "durable_pending": False, "durable_state_kind": "active"}
    return {"durable_blocker": True, "durable_pending": False, "durable_state_kind": "other"}


def _endpoint_owned(manifest: ModelManifest, identity: ProcessIdentity) -> dict[str, Any]:
    try:
        proof = prove_endpoint_owned_by_identity(manifest.endpoint, identity)
    except Exception as exc:
        return {"owned": False, "error": f"{type(exc).__name__}: {exc}"}
    if proof is None:
        return {"owned": False}
    return {"owned": True, "owner_pids": sorted(proof.owner_pids), "identity": {"pid": identity.leader_pid, "pgid": identity.pgid, "birth_token": identity.birth_token}}


def _start_locked(manifest: ModelManifest, wait: bool) -> dict[str, Any]:
    if manifest.start is None:
        raise RuntimeError("manifest has no [start] section")
    inspection = _inspect(manifest)
    status = inspection["status"]
    if status == "owner_mismatch":
        raise RuntimeError(f"pid state at {default_pid_path(manifest)} is owned by another manifest")
    if status == "live":
        identity = inspection["identity"]
        result: dict[str, Any] = {"ok": True, "started": False, "already_running": True, "pid": identity.leader_pid, "pid_path": str(default_pid_path(manifest))}
        if wait:
            try:
                readiness = wait_ready(manifest)
                ownership = _endpoint_owned(manifest, identity) if readiness.get("ready") else {"owned": False}
            except Exception as exc:
                result.update({"ok": False, "status": "readiness_exception", "error": f"{type(exc).__name__}: {exc}"})
                return result
            result.update({"ok": bool(readiness.get("ready") and ownership.get("owned")), "readiness": readiness, "endpoint_ownership": ownership})
        return result
    if status == "dead":
        identity = inspection["identity"]
        removed, remove_error = _guarded_remove(manifest, lambda raw: _validate_active(manifest, raw) == identity)
        if not removed:
            return {"ok": False, "started": False, "status": "dead_state_remove_failed", "safe_to_start": False, "state_error": remove_error}
    elif status != "absent":
        return {"ok": False, "started": False, "status": "pid_state_blocked", "pid_state_status": status, "safe_to_start": False, "pid_path": str(default_pid_path(manifest))}

    transaction_id = secrets.token_hex(16)
    pending = _pending_state(manifest, transaction_id)
    try:
        _atomic_write(_state_path(manifest), pending, exclusive=True)
    except FileExistsError:
        return {"ok": False, "started": False, "status": "pid_state_appeared", "safe_to_start": False}
    except Exception as exc:
        return {"ok": False, "started": False, "status": "pending_write_failed", "error": f"{type(exc).__name__}: {exc}", "safe_to_start": False}

    log_path = default_log_path(manifest)
    if not log_path.is_absolute():
        log_path = manifest.path.parent / log_path
    log_path = log_path.expanduser().resolve(strict=False)
    proc: subprocess.Popen[Any] | None = None
    log = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("ab", buffering=0)
        env = os.environ.copy()
        env.update(manifest.start.env)
        cwd = Path(manifest.start.cwd or manifest.path.parent).expanduser()
        if not cwd.is_absolute():
            cwd = manifest.path.parent / cwd
        proc = subprocess.Popen(
            manifest.start.command,
            cwd=str(cwd.resolve(strict=False)),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        if log is not None:
            try:
                log.close()
            except Exception:
                pass
        if proc is None:
            removed, remove_error = _guarded_remove(manifest, lambda raw: _is_pending(manifest, raw, transaction_id))
            return {
                "ok": False,
                "started": False,
                "status": "popen_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "pending_removed": removed,
                "state_error": remove_error,
                **_durability_fields(manifest, transaction_id),
            }
        cleanup = _cleanup_direct(proc)
        return {"ok": False, "started": True, "status": "post_spawn_exception", "pid": proc.pid, "cleanup": cleanup, **_durability_fields(manifest, transaction_id), "error": f"{type(exc).__name__}: {exc}"}

    capture_error: str | None = None
    try:
        identity = _capture_new_identity(proc)
    except Exception as exc:
        identity = None
        capture_error = f"{type(exc).__name__}: {exc}"
    try:
        log.close()
    except Exception as exc:
        cleanup = _cleanup_exact(manifest, identity, proc, transaction_id=transaction_id) if identity else _cleanup_direct(proc)
        return {"ok": False, "started": True, "status": "log_close_failed", "pid": proc.pid, "cleanup": cleanup, **_durability_fields(manifest, transaction_id, identity), "error": f"{type(exc).__name__}: {exc}"}
    if identity is None:
        cleanup = _cleanup_direct(proc)
        result = {"ok": False, "started": True, "status": "identity_capture_failed", "pid": proc.pid, "cleanup": cleanup, **_durability_fields(manifest, transaction_id)}
        if capture_error is not None:
            result["error"] = capture_error
        return result

    try:
        _replace_pending(manifest, transaction_id, _active_state(manifest, identity, log_path))
        retain_popen(proc)
    except Exception as exc:
        cleanup = _cleanup_exact(manifest, identity, proc, transaction_id=transaction_id)
        return {"ok": False, "started": True, "status": "activation_failed", "pid": proc.pid, "cleanup": cleanup, **_durability_fields(manifest, transaction_id, identity), "error": f"{type(exc).__name__}: {exc}"}

    result: dict[str, Any] = {"ok": True, "started": True, "pid": identity.leader_pid, "pid_path": str(default_pid_path(manifest)), "log_path": str(default_log_path(manifest))}
    if wait:
        try:
            readiness = wait_ready(manifest)
            ownership = _endpoint_owned(manifest, identity) if readiness.get("ready") else {"owned": False}
        except Exception as exc:
            cleanup = _cleanup_exact(manifest, identity, proc, transaction_id=transaction_id)
            result.update({
                "ok": False,
                "status": "readiness_exception",
                "error": f"{type(exc).__name__}: {exc}",
                "cleanup": cleanup,
                **_durability_fields(manifest, transaction_id, identity),
            })
            return result
        result.update({"readiness": readiness, "endpoint_ownership": ownership})
        if not readiness.get("ready") or not ownership.get("owned"):
            cleanup = _cleanup_exact(manifest, identity, proc, transaction_id=transaction_id)
            result.update({
                "ok": False,
                "status": "readiness_failed" if not readiness.get("ready") else "endpoint_ownership_failed",
                "cleanup": cleanup,
                **_durability_fields(manifest, transaction_id, identity),
            })
    return result


def start(manifest: ModelManifest, wait: bool = False) -> dict[str, Any]:
    try:
        with lifecycle_lock("start", manifest):
            return _start_locked(manifest, wait)
    except LifecycleLockError as exc:
        return {"ok": False, "started": False, "status": "lock_failed", "lock": exc.failure.as_dict()}


def _stop_locked(manifest: ModelManifest, timeout_sec: int) -> dict[str, Any]:
    inspection = _inspect(manifest)
    status = inspection["status"]
    pid_path = default_pid_path(manifest)
    if status == "absent":
        return {"ok": True, "stopped": False, "already_stopped": True, "safe_to_start": True, "pid_path_removed": str(pid_path)}
    if status == "owner_mismatch":
        return {"ok": False, "stopped": False, "already_stopped": False, "owner_mismatch": True, "safe_to_start": False, "pid_path": str(pid_path), "pid_state": inspection.get("raw")}
    if status == "dead":
        identity = inspection["identity"]
        removed, remove_error = _guarded_remove(manifest, lambda raw: _validate_active(manifest, raw) == identity)
        return {"ok": removed, "stopped": False, "already_stopped": removed, "safe_to_start": removed, "pid_path_removed": str(pid_path) if removed else None, "state_error": remove_error}
    if status != "live":
        return {"ok": False, "stopped": False, "already_stopped": False, "safe_to_start": False, "pid_state_status": status, "pid_path": str(pid_path)}
    identity = inspection["identity"]
    try:
        terminated = terminate_process_identity(identity, timeout_sec=timeout_sec)
    except Exception as exc:
        return {"ok": False, "stopped": False, "known_pid_stopped": False, "safe_to_start": False, "unexpected_active_pid": identity.leader_pid, "error": f"{type(exc).__name__}: {exc}"}
    if not terminated:
        return {"ok": False, "stopped": False, "known_pid_stopped": False, "safe_to_start": False, "unexpected_active_pid": identity.leader_pid}
    removed, remove_error = _guarded_remove(manifest, lambda raw: _validate_active(manifest, raw) == identity)
    return {"ok": removed, "stopped": True, "pid": identity.leader_pid, "known_pid_stopped": True, "safe_to_start": removed, "unexpected_active_pid": None, "pid_path": str(pid_path), "state_error": remove_error}


def stop(manifest: ModelManifest, timeout_sec: int = 10) -> dict[str, Any]:
    try:
        with lifecycle_lock("stop", manifest):
            return _stop_locked(manifest, timeout_sec)
    except LifecycleLockError as exc:
        return {"ok": False, "stopped": False, "already_stopped": False, "safe_to_start": False, "status": "lock_failed", "lock": exc.failure.as_dict()}


def _manifest_ref(manifest: ModelManifest) -> dict[str, Any]:
    return {"id": manifest.id, "model_id": manifest.model_id, "endpoint": manifest.endpoint, "path": str(manifest.path)}


def rotate(
    current: ModelManifest,
    target: ModelManifest,
    *,
    readiness_timeout_sec: float | None = None,
    stop_timeout_sec: int = 10,
    rollback: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rotate from current manifest process to target with readiness-gated PID ownership handoff."""
    base: dict[str, Any] = {
        "ok": False,
        "action": "rotate",
        "from": _manifest_ref(current),
        "to": _manifest_ref(target),
        "readiness_timeout_sec": readiness_timeout_sec,
        "stop_timeout_sec": stop_timeout_sec,
        "rollback_enabled": rollback,
        "dry_run": dry_run,
    }
    if current.start is None:
        return {**base, "status": "invalid_request", "error": "current manifest has no [start] section", "issues": ["current_missing_start"]}
    if target.start is None:
        return {**base, "status": "invalid_request", "error": "target manifest has no [start] section", "issues": ["target_missing_start"]}
    if current.endpoint != target.endpoint or current.model_id != target.model_id:
        return {
            **base,
            "status": "invalid_request",
            "error": "target must preserve current endpoint and model_id for stable-lane rotation",
            "issues": ["target_identity_mismatch"],
        }
    timeout = readiness_timeout_sec if readiness_timeout_sec is not None else target.start.startup_timeout_sec
    base["readiness_timeout_sec"] = timeout
    current_pid_path = default_pid_path(current)
    target_pid_path = default_pid_path(target)
    plan = {
        "steps": ["stop_current", "start_target", "verify_target_readiness", "atomically_handoff_pid_state"],
        "current_pid_path": str(current_pid_path),
        "target_pid_path": str(target_pid_path),
    }
    if dry_run:
        return {**base, "ok": True, "status": "planned", "plan": plan}

    old_pid = active_pid(current)
    stop_current = stop(current, timeout_sec=stop_timeout_sec)
    if stop_current.get("owner_mismatch") or (old_pid is not None and not stop_current.get("stopped")):
        return {
            **base,
            "status": "current_stop_failed",
            "old_pid": old_pid,
            "stop_current": stop_current,
            "issues": ["current_stop_failed"],
        }

    def rollback_current() -> dict[str, Any]:
        if not rollback:
            return {"attempted": False}
        try:
            rollback_start = start(current, wait=False)
            rollback_timeout = current.start.startup_timeout_sec if current.start else timeout
            rollback_readiness = wait_ready(current, timeout_sec=rollback_timeout)
            return {"attempted": True, "start": rollback_start, "readiness": rollback_readiness}
        except Exception as exc:
            return {"attempted": True, "error": f"{type(exc).__name__}: {exc}"}

    try:
        target_start = start(target, wait=False)
    except Exception as exc:
        rollback_result = rollback_current()
        return {
            **base,
            "status": "target_start_failed",
            "old_pid": old_pid,
            "stop_current": stop_current,
            "error": f"{type(exc).__name__}: {exc}",
            "rollback": rollback_result,
            "issues": ["target_start_failed"],
        }
    readiness = wait_ready(target, timeout_sec=timeout)
    if not readiness.get("ready"):
        target_stop = stop(target, timeout_sec=stop_timeout_sec)
        rollback_result = rollback_current()
        return {
            **base,
            "status": "target_not_ready",
            "old_pid": old_pid,
            "stop_current": stop_current,
            "target_start": target_start,
            "readiness": readiness,
            "target_stop": target_stop,
            "rollback": rollback_result,
            "issues": ["target_not_ready"],
        }

    target_state = read_pid_state(target)
    expected_pid = target_start.get("pid") if isinstance(target_start, dict) else None
    target_state_valid = bool(target_state) and isinstance(target_state.get("pid") if target_state else None, int)
    if target_state_valid and isinstance(expected_pid, int) and target_state and target_state.get("pid") != expected_pid:
        target_state_valid = False
    if target_state_valid and target_state and target_state.get("manifest") != str(target.path):
        target_state_valid = False
    if not target_state_valid or target_state is None:
        target_stop = stop(target, timeout_sec=stop_timeout_sec)
        rollback_result = rollback_current()
        return {
            **base,
            "status": "target_pid_state_missing",
            "old_pid": old_pid,
            "stop_current": stop_current,
            "target_start": target_start,
            "readiness": readiness,
            "target_stop": target_stop,
            "rollback": rollback_result,
            "issues": ["target_pid_state_missing"],
        }

    handoff_state = dict(target_state)
    handoff_state["manifest"] = _manifest_path(current)
    if handoff_state.get("schema_version") == 2 and handoff_state.get("kind") == "active":
        handoff_state["fingerprint"] = _launch_fingerprint(current)
        current_log = default_log_path(current).expanduser()
        if not current_log.is_absolute():
            current_log = current.path.parent / current_log
        handoff_state["log_path"] = str(current_log.resolve(strict=False))
    handoff_state["source_manifest"] = str(target.path)
    handoff_state["rotated_from"] = _manifest_ref(current)
    handoff_state["rotated_to"] = _manifest_ref(target)
    handoff_state["rotated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    handoff_state["source_pid_path"] = str(target_pid_path)
    handoff_state["owner_pid_path"] = str(current_pid_path)
    try:
        if target_pid_path == current_pid_path:
            write_pid_state(current, handoff_state)
            removed = False
        else:
            _atomic_write(_state_path(current), handoff_state, exclusive=False)
            _state_path(target).unlink()
            removed = True
    except Exception as exc:
        target_stop = stop(target, timeout_sec=stop_timeout_sec)
        rollback_result = rollback_current()
        return {
            **base,
            "status": "handoff_failed",
            "old_pid": old_pid,
            "stop_current": stop_current,
            "target_start": target_start,
            "readiness": readiness,
            "target_stop": target_stop,
            "rollback": rollback_result,
            "error": f"{type(exc).__name__}: {exc}",
            "issues": ["handoff_failed"],
        }

    return {
        **base,
        "ok": True,
        "status": "rotated",
        "old_pid": old_pid,
        "new_pid": handoff_state.get("pid"),
        "stop_current": stop_current,
        "target_start": target_start,
        "readiness": readiness,
        "handoff": {
            "current_pid_path": str(current_pid_path),
            "target_pid_path": str(target_pid_path),
            "target_pid_state_removed": removed,
            "atomic": True,
        },
    }