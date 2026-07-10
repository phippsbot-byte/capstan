from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import atexit
import ctypes
import errno
import hashlib
import ipaddress
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

GIB = 1024 ** 3


class ProcessIdentityError(RuntimeError):
    """The operating system could not provide trustworthy process identity data."""


class EndpointOwnershipError(ProcessIdentityError):
    """The operating system could not verify ownership of a local TCP listener."""


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """The non-reusable identity of a process-group leader.

    ``leader_pid`` identifies the process to inspect, while ``pgid`` and the
    kernel-issued ``birth_token`` prevent a stale state record from addressing
    a later process or process group that reused the same numeric IDs.
    """

    leader_pid: int
    pgid: int
    birth_token: str

    def __post_init__(self) -> None:
        _positive_kernel_id(self.leader_pid, "leader_pid")
        _positive_kernel_id(self.pgid, "pgid")
        if not isinstance(self.birth_token, str) or not self.birth_token:
            raise ValueError("birth_token must be a non-empty string")


@dataclass(frozen=True, slots=True)
class EndpointOwnershipProof:
    """A complete, identity-checked inventory of one local TCP listener."""

    host: str
    port: int
    identity: ProcessIdentity
    owner_pids: frozenset[int]


def _positive_kernel_id(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, not bool")
    return value


def validate_pid(pid: object) -> int:
    """Return a validated PID, rejecting bools and every non-positive value."""
    return _positive_kernel_id(pid, "pid")


def validate_pgid(pgid: object) -> int:
    """Return a validated process-group ID, rejecting bools and non-positive values."""
    return _positive_kernel_id(pgid, "pgid")


def run(cmd: list[str], timeout: int = 10) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return proc.returncode, proc.stdout.strip()
    except Exception as exc:  # pragma: no cover
        return 999, f"{type(exc).__name__}: {exc}"


def _pid_stat(pid: int) -> str | None:
    try:
        proc = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def pid_alive(pid: int) -> bool:
    try:
        checked_pid = validate_pid(pid)
    except ValueError:
        return False
    try:
        os.kill(checked_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if sys.platform == "darwin":
        info = _darwin_proc_bsdinfo(checked_pid)
        # libproc stops returning task BSD info once a process has exited but
        # remains as a zombie.  Treat that state as dead, matching the Linux
        # stat-state behavior below.
        if info is None or info.pbi_status == 5:  # SZOMB
            return False
        return True
    stat = _pid_stat(checked_pid)
    if stat and stat.lstrip().startswith("Z"):
        return False
    return True


def _linux_proc_stat(pid: int) -> tuple[str, int, int] | None:
    """Return ``(state, pgrp, starttime)`` from /proc, or None if it vanished."""
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except PermissionError as exc:
        raise ProcessIdentityError(f"cannot inspect /proc/{pid}/stat") from exc
    except OSError as exc:
        raise ProcessIdentityError(f"cannot inspect /proc/{pid}/stat: {exc}") from exc
    closing_paren = text.rfind(")")
    if closing_paren < 1:
        raise ProcessIdentityError(f"unparseable /proc/{pid}/stat")
    fields = text[closing_paren + 1 :].split()
    # Fields after ``comm`` start with state (3); pgrp is 5 and starttime is 22.
    if len(fields) <= 19:
        raise ProcessIdentityError(f"incomplete /proc/{pid}/stat")
    try:
        return fields[0], int(fields[2]), int(fields[19])
    except ValueError as exc:
        raise ProcessIdentityError(f"unparseable /proc/{pid}/stat") from exc


class _ProcBsdInfo(ctypes.Structure):
    """Darwin ``PROC_PIDTBSDINFO`` layout from <libproc.h>."""

    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


def _darwin_proc_bsdinfo(pid: int) -> _ProcBsdInfo | None:
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidinfo = libproc.proc_pidinfo
        proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        proc_pidinfo.restype = ctypes.c_int
        info = _ProcBsdInfo()
        # PROC_PIDTBSDINFO is the documented libproc flavor for proc_bsdinfo.
        received = proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
    except (AttributeError, OSError):
        return None
    if received < ctypes.sizeof(info):
        return None
    return info


def _darwin_bsdinfo_unavailable_errno(pid: int) -> int | None:
    """Return libproc's failure errno when task BSD info is unavailable."""
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidinfo = libproc.proc_pidinfo
        proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        proc_pidinfo.restype = ctypes.c_int
        info = _ProcBsdInfo()
        ctypes.set_errno(0)
        received = proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
    except (AttributeError, OSError):
        return None
    return ctypes.get_errno() if received < ctypes.sizeof(info) else None


def _darwin_birth_token(pid: int) -> str | None:
    info = _darwin_proc_bsdinfo(pid)
    if (
        info is None
        or info.pbi_status == 5  # SZOMB
        or info.pbi_start_tvsec <= 0
        or not 0 <= info.pbi_start_tvusec < 1_000_000
    ):
        return None
    return f"darwin:{info.pbi_start_tvsec}:{info.pbi_start_tvusec}"


def process_birth_token(pid: object) -> str | None:
    """Return an exact kernel birth token for a live PID, otherwise ``None``.

    The token is deliberately absent when the platform cannot prove an exact
    start identity: callers must not fall back to PID-only state.
    """
    try:
        checked_pid = validate_pid(pid)
    except ValueError:
        return None
    if sys.platform == "darwin":
        return _darwin_birth_token(checked_pid)
    if sys.platform.startswith("linux"):
        try:
            stat = _linux_proc_stat(checked_pid)
        except ProcessIdentityError:
            return None
        if stat is None:
            return None
        state, _pgrp, starttime = stat
        if state == "Z" or starttime < 0:
            return None
        return f"linux:{starttime}"
    return None


def capture_process_identity(leader_pid: object) -> ProcessIdentity | None:
    """Capture a leader's exact PID, process-group, and birth identity.

    A ``None`` result means the operating system cannot prove all three values
    for a live, non-zombie process.  Callers that may signal a process group
    must fail closed rather than reconstructing this value from a PID alone.
    """
    try:
        checked_pid = validate_pid(leader_pid)
    except ValueError:
        return None
    if sys.platform == "darwin":
        info = _darwin_proc_bsdinfo(checked_pid)
        if (
            info is None
            or info.pbi_pid != checked_pid
            or info.pbi_status == 5  # SZOMB
            or info.pbi_pgid <= 0
            or info.pbi_start_tvsec <= 0
            or not 0 <= info.pbi_start_tvusec < 1_000_000
        ):
            return None
        return ProcessIdentity(
            leader_pid=checked_pid,
            pgid=int(info.pbi_pgid),
            birth_token=f"darwin:{info.pbi_start_tvsec}:{info.pbi_start_tvusec}",
        )
    if sys.platform.startswith("linux"):
        try:
            stat = _linux_proc_stat(checked_pid)
        except ProcessIdentityError:
            return None
        if stat is None:
            return None
        state, pgid, starttime = stat
        if state == "Z" or pgid <= 0 or starttime < 0:
            return None
        return ProcessIdentity(leader_pid=checked_pid, pgid=pgid, birth_token=f"linux:{starttime}")
    return None


def _linux_live_process_group_members(pgid: int) -> list[int]:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        raise ProcessIdentityError("/proc is unavailable for process-group inspection")
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise ProcessIdentityError("cannot enumerate /proc") from exc
    members: list[int] = []
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            stat = _linux_proc_stat(pid)
        except ProcessIdentityError:
            raise
        if stat is None:
            continue
        state, process_group, _starttime = stat
        if process_group == pgid and state != "Z":
            members.append(pid)
    return sorted(members)


def _ps_live_process_group_members(pgid: int) -> list[int]:
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid=,pgid=,stat="],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
    except Exception as exc:
        raise ProcessIdentityError("cannot enumerate process groups with ps") from exc
    if proc.returncode != 0:
        raise ProcessIdentityError(f"ps process-group enumeration failed: {proc.stderr.strip()}")
    members: list[int] = []
    for row in proc.stdout.splitlines():
        fields = row.split()
        if not fields:
            continue
        if len(fields) != 3:
            raise ProcessIdentityError(f"unparseable ps process row: {row!r}")
        try:
            member_pid, member_pgid = int(fields[0]), int(fields[1])
        except ValueError as exc:
            raise ProcessIdentityError(f"unparseable ps process row: {row!r}") from exc
        if member_pgid == pgid and not fields[2].startswith("Z"):
            members.append(member_pid)
    return sorted(members)


def _darwin_list_pids(list_type: int, typeinfo: int) -> list[int]:
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_listpids = libproc.proc_listpids
        proc_listpids.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int]
        proc_listpids.restype = ctypes.c_int
    except (AttributeError, OSError) as exc:
        raise ProcessIdentityError("libproc PID enumeration is unavailable") from exc
    int_size = ctypes.sizeof(ctypes.c_int)
    capacity = 0
    # A list can grow between libproc's sizing and fill calls.  ``proc_listpids``
    # reports an upper-bound sizing query for group lists, so a short fill is a
    # complete snapshot; a fill that exactly consumes its buffer is ambiguous
    # and must be retried with more space.
    for _attempt in range(4):
        ctypes.set_errno(0)
        required = proc_listpids(list_type, typeinfo, None, 0)
        errno = ctypes.get_errno()
        if required < 0 or (required == 0 and errno != 0) or required % int_size:
            raise ProcessIdentityError("libproc PID enumeration returned an invalid size")
        if required == 0:
            return []
        capacity = max(capacity, required)
        buffer = (ctypes.c_int * (capacity // int_size))()
        ctypes.set_errno(0)
        received = proc_listpids(list_type, typeinfo, ctypes.byref(buffer), capacity)
        errno = ctypes.get_errno()
        if (
            received < 0
            or received > capacity
            or received % int_size
            or (received == 0 and errno != 0)
        ):
            raise ProcessIdentityError("libproc PID enumeration returned an incomplete result")
        if received < capacity:
            return [pid for pid in buffer[: received // int_size] if pid > 0]
        capacity = max(capacity * 2, received + int_size)
    raise ProcessIdentityError("libproc PID inventory did not stabilize")


def _darwin_live_process_group_members(pgid: int) -> list[int]:
    # PROC_PGRP_ONLY asks the kernel for the group directly, avoiding a lossy
    # `ps` snapshot and keeping a departed group leader from hiding children.
    members: list[int] = []
    for pid in _darwin_list_pids(2, pgid):  # PROC_PGRP_ONLY
        info = _darwin_proc_bsdinfo(pid)
        if info is None:
            # libproc reports ESRCH for its intentionally unavailable task BSD
            # info on an unreaped zombie (and for a PID that vanished).  Any
            # other unavailable result for a live PID is inventory uncertainty.
            if _darwin_bsdinfo_unavailable_errno(pid) == errno.ESRCH:
                continue
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            except PermissionError as exc:
                raise ProcessIdentityError(f"cannot inspect group member {pid}") from exc
            raise ProcessIdentityError(f"cannot inspect live group member {pid}")
        if info.pbi_pgid != pgid:
            raise ProcessIdentityError(f"libproc group membership changed for pid {pid}")
        if info.pbi_status != 5:  # SZOMB
            members.append(pid)
    return sorted(members)


def live_process_group_members(pgid: object) -> list[int]:
    """Enumerate all non-zombie members of a process group.

    Failure to obtain a complete enough kernel inventory is an error rather
    than an empty group.  Empty is therefore safe to use as a shutdown proof.
    """
    checked_pgid = validate_pgid(pgid)
    if sys.platform == "darwin":
        return _darwin_live_process_group_members(checked_pgid)
    if sys.platform.startswith("linux"):
        return _linux_live_process_group_members(checked_pgid)
    return _ps_live_process_group_members(checked_pgid)


def _valid_timeout(timeout_sec: object) -> float:
    if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float)):
        raise ValueError("timeout_sec must be a non-negative number")
    timeout = float(timeout_sec)
    if timeout < 0 or timeout == float("inf") or timeout != timeout:
        raise ValueError("timeout_sec must be a finite non-negative number")
    return timeout


def _wait_for_empty_process_group(pgid: int, deadline: float) -> bool:
    while True:
        try:
            if not live_process_group_members(pgid):
                return True
        except ProcessIdentityError:
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(0.05, max(0.001, deadline - time.monotonic())))


def _terminate_process_group_legacy(pgid: object, timeout_sec: int | float = 10) -> bool:
    """Legacy raw-PGID shutdown; new lifecycle code must use identity safety.

    This intentionally cannot defend against PID/PGID reuse because callers
    supply only a numeric group.  It remains for existing compatibility until
    the Slice 2 lifecycle integration has moved to
    :func:`terminate_process_identity`.
    """
    try:
        checked_pgid = validate_pgid(pgid)
        timeout = _valid_timeout(timeout_sec)
        if not live_process_group_members(checked_pgid):
            return True
    except (ProcessIdentityError, ValueError):
        return False
    try:
        os.killpg(checked_pgid, signal.SIGTERM)
    except ProcessLookupError:
        try:
            return not live_process_group_members(checked_pgid)
        except ProcessIdentityError:
            return False
    except (PermissionError, OSError):
        return False
    if _wait_for_empty_process_group(checked_pgid, time.monotonic() + timeout):
        return True
    try:
        os.killpg(checked_pgid, signal.SIGKILL)
    except ProcessLookupError:
        try:
            return not live_process_group_members(checked_pgid)
        except ProcessIdentityError:
            return False
    except (PermissionError, OSError):
        return False
    # KILL should be immediate, but allow the kernel scheduler a bounded grace
    # period before declaring that a live member could not be certified gone.
    return _wait_for_empty_process_group(checked_pgid, time.monotonic() + max(0.1, min(1.0, timeout)))


def terminate_process_group(pgid: object, timeout_sec: int | float = 10) -> bool:
    """Legacy compatibility wrapper for a raw process-group shutdown.

    New lifecycle callers must retain a :class:`ProcessIdentity` and call
    :func:`terminate_process_identity`; this wrapper has no birth-token proof.
    """
    return _terminate_process_group_legacy(pgid, timeout_sec)


def _initial_group_anchors(identity: ProcessIdentity) -> dict[int, str]:
    """Capture original group members that can prevent later PGID reuse."""
    anchors = {identity.leader_pid: identity.birth_token}
    try:
        members = live_process_group_members(identity.pgid)
    except ProcessIdentityError:
        return anchors
    for pid in members:
        member = capture_process_identity(pid)
        if member is not None and member.pgid == identity.pgid:
            anchors[pid] = member.birth_token
    return anchors


def _authenticated_group_state(pgid: int, anchors: dict[int, str]) -> str:
    """Return ``empty``, ``anchored``, or ``unsafe`` for a post-TERM group."""
    try:
        members = live_process_group_members(pgid)
    except ProcessIdentityError:
        return "unsafe"
    if not members:
        return "empty"
    for pid in members:
        expected_token = anchors.get(pid)
        if expected_token is None:
            continue
        member = capture_process_identity(pid)
        if member is not None and member.pgid == pgid and member.birth_token == expected_token:
            # A surviving member from the authenticated pre-TERM group proves
            # that this numeric PGID has not been emptied and reused.
            return "anchored"
    return "unsafe"


def _wait_for_empty_authenticated_group(pgid: int, anchors: dict[int, str], deadline: float) -> bool:
    while True:
        state = _authenticated_group_state(pgid, anchors)
        if state == "empty":
            return True
        if state != "anchored" or time.monotonic() >= deadline:
            return False
        time.sleep(min(0.05, max(0.001, deadline - time.monotonic())))


def terminate_process_identity(identity: object, timeout_sec: int | float = 10) -> bool:
    """Safely TERM then KILL the exact group authenticated by ``identity``.

    The leader PID, PGID, and birth token are read together and compared
    immediately before the first group signal.  After TERM, a surviving member
    from the initial authenticated group must remain as an anchor before KILL;
    a reused or uncertain PGID is never signaled again.
    """
    if not isinstance(identity, ProcessIdentity):
        return False
    try:
        timeout = _valid_timeout(timeout_sec)
    except ValueError:
        return False
    # Capture possible child anchors first, then make the leader identity read
    # the last operation before SIGTERM.  Do not insert a membership snapshot
    # between this comparison and killpg.
    anchors = _initial_group_anchors(identity)
    if capture_process_identity(identity.leader_pid) != identity:
        return False
    try:
        os.killpg(identity.pgid, signal.SIGTERM)
    except ProcessLookupError:
        return _authenticated_group_state(identity.pgid, anchors) == "empty"
    except (PermissionError, OSError):
        return False
    if _wait_for_empty_authenticated_group(identity.pgid, anchors, time.monotonic() + timeout):
        return True
    # This check is intentionally adjacent to SIGKILL.  If the original group
    # vanished after TERM, no remaining anchor means the numeric PGID could now
    # identify a foreign group and must not be signaled.
    if _authenticated_group_state(identity.pgid, anchors) != "anchored":
        return False
    try:
        os.killpg(identity.pgid, signal.SIGKILL)
    except ProcessLookupError:
        return _authenticated_group_state(identity.pgid, anchors) == "empty"
    except (PermissionError, OSError):
        return False
    return _wait_for_empty_authenticated_group(
        identity.pgid,
        anchors,
        time.monotonic() + max(0.1, min(1.0, timeout)),
    )


def _valid_port(port: object) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65_535:
        raise ValueError("port must be an integer from 1 through 65535")
    return port


def _normalize_host(host: object) -> str:
    if not isinstance(host, str) or not host or host != host.strip():
        raise ValueError("endpoint host must be a non-empty string")
    bare_host = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    if not bare_host or "%" in bare_host:
        raise ValueError("endpoint host must not be empty or contain an IPv6 zone")
    try:
        return str(ipaddress.ip_address(bare_host))
    except ValueError:
        if any(char in bare_host for char in "/:@[]"):
            raise ValueError("endpoint host is malformed")
        return bare_host.rstrip(".").lower()


def normalize_endpoint_host_port(endpoint_or_host: object, port: object | None = None) -> tuple[str, int]:
    """Normalize an endpoint URL or host/port pair to ``(host, effective_port)``.

    HTTP and HTTPS URLs may omit their port, in which case their protocol
    defaults are used.  A host/port pair always requires an explicit port.
    """
    if not isinstance(endpoint_or_host, str):
        raise ValueError("endpoint must be a string")
    if port is not None:
        return _normalize_host(endpoint_or_host), _valid_port(port)
    try:
        parsed = urlsplit(endpoint_or_host)
        parsed_port = parsed.port
    except ValueError as exc:
        raise ValueError(f"malformed endpoint: {exc}") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.netloc or parsed.hostname is None:
        raise ValueError("endpoint must be an absolute http or https URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint must not contain user credentials")
    effective_port = parsed_port if parsed_port is not None else (443 if scheme == "https" else 80)
    return _normalize_host(parsed.hostname), _valid_port(effective_port)


def _local_host_candidates(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    if host == "localhost":
        return [ipaddress.IPv4Address("127.0.0.1"), ipaddress.IPv6Address("::1")]
    try:
        return [ipaddress.ip_address(host)]
    except ValueError as exc:
        # Resolving arbitrary names can silently turn an ownership check into a
        # network-dependent operation.  Refuse it instead of guessing localness.
        raise EndpointOwnershipError("listener ownership requires an IP literal or localhost") from exc


def _listener_address_matches(host: str, family: int, listener_address: str) -> bool:
    try:
        listener = ipaddress.ip_address(listener_address)
    except ValueError as exc:
        raise EndpointOwnershipError(f"unparseable listener address: {listener_address!r}") from exc
    for requested in _local_host_candidates(host):
        if requested.version != listener.version:
            continue
        if requested == listener:
            return True
        if listener.version == 4 and listener == ipaddress.IPv4Address("0.0.0.0"):
            return True
        if listener.version == 6 and listener == ipaddress.IPv6Address("::"):
            return True
    return False


def _linux_listener_inodes(host: str, port: int) -> set[str]:
    tables = ((Path("/proc/net/tcp"), socket.AF_INET), (Path("/proc/net/tcp6"), socket.AF_INET6))
    matched: set[str] = set()
    for table, family in tables:
        try:
            rows = table.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise EndpointOwnershipError(f"cannot inspect {table}") from exc
        if not rows:
            raise EndpointOwnershipError(f"empty socket table: {table}")
        for row in rows[1:]:
            fields = row.split()
            if len(fields) < 10:
                raise EndpointOwnershipError(f"unparseable socket table row in {table}")
            if fields[3] != "0A":  # TCP_LISTEN
                continue
            try:
                packed_address, text_port = fields[1].split(":", 1)
                row_port = int(text_port, 16)
                if family == socket.AF_INET:
                    raw = bytes.fromhex(packed_address)
                    if len(raw) != 4:
                        raise ValueError("invalid IPv4 length")
                    address = socket.inet_ntop(socket.AF_INET, raw[::-1])
                else:
                    raw = bytes.fromhex(packed_address)
                    if len(raw) != 16:
                        raise ValueError("invalid IPv6 length")
                    # Linux exposes each 32-bit IPv6 word in host byte order.
                    address = socket.inet_ntop(socket.AF_INET6, b"".join(raw[index : index + 4][::-1] for index in range(0, 16, 4)))
            except (OSError, ValueError) as exc:
                raise EndpointOwnershipError(f"unparseable socket table row in {table}") from exc
            if row_port == port and _listener_address_matches(host, family, address):
                matched.add(fields[9])
    return matched


_SOCKET_LINK = re.compile(r"^socket:\[(\d+)\]$")


def _linux_pid_socket_inodes(pid: int) -> set[str]:
    fd_dir = Path("/proc") / str(pid) / "fd"
    try:
        descriptors = list(fd_dir.iterdir())
    except FileNotFoundError:
        return set()
    except PermissionError as exc:
        raise EndpointOwnershipError(f"cannot inspect file descriptors for pid {pid}") from exc
    except OSError as exc:
        raise EndpointOwnershipError(f"cannot inspect file descriptors for pid {pid}: {exc}") from exc
    inodes: set[str] = set()
    for descriptor in descriptors:
        try:
            target = os.readlink(descriptor)
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise EndpointOwnershipError(f"cannot inspect file descriptor for pid {pid}") from exc
        except OSError as exc:
            raise EndpointOwnershipError(f"cannot inspect file descriptor for pid {pid}: {exc}") from exc
        match = _SOCKET_LINK.match(target)
        if match:
            inodes.add(match.group(1))
    return inodes


def _linux_all_pids() -> list[int]:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        raise EndpointOwnershipError("/proc is unavailable for listener-owner inspection")
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise EndpointOwnershipError("cannot enumerate /proc for listener owners") from exc
    return sorted(int(entry.name) for entry in entries if entry.name.isdecimal())


def _linux_listener_owner_birth_tokens(host: str, port: int) -> dict[int, str]:
    """Return every exact live owner of the matching Linux listener inodes."""
    initial_inodes = _linux_listener_inodes(host, port)
    if not initial_inodes:
        return {}
    owners: dict[int, str] = {}
    for pid in _linux_all_pids():
        before = process_birth_token(pid)
        try:
            pid_inodes = _linux_pid_socket_inodes(pid)
        except EndpointOwnershipError:
            # An inaccessible PID might own a matching socket, so a complete
            # owner inventory is impossible.
            raise
        matches = initial_inodes & pid_inodes
        if not matches:
            continue
        if before is None:
            raise EndpointOwnershipError(f"cannot prove identity of listener owner {pid}")
        after = process_birth_token(pid)
        if after is None or after != before:
            raise EndpointOwnershipError(f"listener owner {pid} changed identity during inspection")
        owners[pid] = before
    final_inodes = _linux_listener_inodes(host, port)
    if final_inodes != initial_inodes:
        raise EndpointOwnershipError("listener inode inventory changed during inspection")
    if not owners:
        raise EndpointOwnershipError("matching listener has no provable owning PID")
    return owners


def _lsof_listener_owner_pids(host: str, port: int) -> set[int]:
    """Inventory all matching macOS listener owners from lsof field records."""
    executable = shutil.which("lsof")
    if executable is None:
        raise EndpointOwnershipError("lsof is unavailable for listener ownership inspection")
    try:
        proc = subprocess.run(
            [executable, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpfnt"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
    except Exception as exc:
        raise EndpointOwnershipError("lsof listener ownership inspection failed") from exc
    if proc.returncode not in {0, 1}:
        raise EndpointOwnershipError(f"lsof listener ownership inspection failed: {proc.stderr.strip()}")
    owners: set[int] = set()
    current_pid: int | None = None
    current_file = False
    file_type: str | None = None
    for line in proc.stdout.splitlines():
        if not line:
            continue
        field, value = line[0], line[1:]
        if field == "p":
            try:
                current_pid = validate_pid(int(value))
            except ValueError as exc:
                raise EndpointOwnershipError(f"unparseable lsof PID record: {line!r}") from exc
            current_file = False
            file_type = None
            continue
        if field == "f":
            if current_pid is None:
                raise EndpointOwnershipError("lsof file record appeared before its process")
            current_file = True
            file_type = None
            continue
        if field == "t":
            if current_pid is None or not current_file:
                raise EndpointOwnershipError("lsof type record appeared outside a file record")
            file_type = value
            continue
        if field != "n":
            continue
        if current_pid is None or not current_file:
            raise EndpointOwnershipError("lsof name record appeared outside a file record")
        name = value.removesuffix(" (LISTEN)")
        if name.startswith("TCP "):
            name = name[4:]
        if name.startswith("["):
            closing = name.find("]")
            listener_host = name[1:closing] if closing >= 0 and name[closing + 1 :].startswith(":") else ""
            text_port = name[closing + 2 :] if listener_host else ""
        else:
            listener_host, separator, text_port = name.rpartition(":")
            if not separator:
                raise EndpointOwnershipError(f"unparseable lsof listener name: {value!r}")
        try:
            listener_port = _valid_port(int(text_port))
        except ValueError as exc:
            raise EndpointOwnershipError(f"unparseable lsof listener port: {value!r}") from exc
        if listener_port != port:
            continue
        if file_type == "IPv4":
            family, wildcard = socket.AF_INET, "0.0.0.0"
        elif file_type == "IPv6":
            family, wildcard = socket.AF_INET6, "::"
        else:
            raise EndpointOwnershipError(f"unparseable lsof listener type: {file_type!r}")
        address = wildcard if listener_host == "*" else listener_host
        if _listener_address_matches(host, family, address):
            owners.add(current_pid)
    return owners


def _darwin_listener_owner_birth_tokens(host: str, port: int) -> dict[int, str]:
    """Return a stable, complete lsof owner snapshot for one local listener."""
    before_pids = _lsof_listener_owner_pids(host, port)
    if not before_pids:
        return {}
    before_tokens: dict[int, str] = {}
    for pid in before_pids:
        token = process_birth_token(pid)
        if token is None:
            raise EndpointOwnershipError(f"cannot prove identity of listener owner {pid}")
        before_tokens[pid] = token
    after_pids = _lsof_listener_owner_pids(host, port)
    if after_pids != before_pids:
        raise EndpointOwnershipError("lsof listener-owner inventory changed during inspection")
    for pid, before in before_tokens.items():
        after = process_birth_token(pid)
        if after is None or after != before:
            raise EndpointOwnershipError(f"listener owner {pid} changed identity during inspection")
    return before_tokens


def _listener_owner_birth_tokens(host: str, port: int) -> dict[int, str]:
    if sys.platform.startswith("linux"):
        return _linux_listener_owner_birth_tokens(host, port)
    return _darwin_listener_owner_birth_tokens(host, port)


def _lsof_listener_owned_by_pid(host: str, port: int, pid: int) -> bool:
    """Legacy PID-only matcher; lifecycle proof callers must not use this."""
    return pid in _darwin_listener_owner_birth_tokens(host, port)


def _endpoint_owned_by_pid(host: str, port: int, pid: int) -> bool:
    if sys.platform.startswith("linux"):
        listener_inodes = _linux_listener_inodes(host, port)
        return bool(listener_inodes & _linux_pid_socket_inodes(pid))
    return _lsof_listener_owned_by_pid(host, port, pid)


def _endpoint_owner_arguments(
    endpoint_or_host: object,
    port_or_owner: object,
    owner: object | None,
    owner_name: str,
) -> tuple[str, int, int]:
    if owner is None:
        host, port = normalize_endpoint_host_port(endpoint_or_host)
        owner_id = _positive_kernel_id(port_or_owner, owner_name)
    else:
        host, port = normalize_endpoint_host_port(endpoint_or_host, port_or_owner)
        owner_id = _positive_kernel_id(owner, owner_name)
    return host, port, owner_id


def endpoint_owned_by_pid(endpoint_or_host: object, port_or_pid: object, pid: object | None = None) -> bool:
    """Legacy PID-only listener matcher, without a process identity proof.

    Use either ``(host, port, pid)`` or ``(endpoint_url, pid)``.  Inventory
    failures raise :class:`EndpointOwnershipError`; they never become a match.
    """
    host, port, checked_pid = _endpoint_owner_arguments(endpoint_or_host, port_or_pid, pid, "pid")
    return _endpoint_owned_by_pid(host, port, checked_pid)


def endpoint_owned_by_pgid(endpoint_or_host: object, port_or_pgid: object, pgid: object | None = None) -> bool:
    """Legacy PGID-only listener matcher, without a complete owner inventory.

    Use either ``(host, port, pgid)`` or ``(endpoint_url, pgid)``.  Detached
    ``setsid`` children do not match their former group's ID.
    """
    host, port, checked_pgid = _endpoint_owner_arguments(endpoint_or_host, port_or_pgid, pgid, "pgid")
    for member_pid in live_process_group_members(checked_pgid):
        if _endpoint_owned_by_pid(host, port, member_pid):
            return True
    return False


def prove_endpoint_owned_by_identity(
    endpoint_or_host: object,
    port_or_identity: object,
    identity: object | None = None,
) -> EndpointOwnershipProof | None:
    """Return a complete endpoint ownership proof for an authenticated group.

    Use either ``(host, port, identity)`` or ``(endpoint_url, identity)``.
    The proof is absent whenever the listener has no owner, an owner changed
    identity, inventory is incomplete, or any owner lies outside the exact
    tracked process group.  This is the API intended for Slice 2 lifecycle
    integration; the older boolean helpers above are compatibility-only.
    """
    try:
        if identity is None:
            host, port = normalize_endpoint_host_port(endpoint_or_host)
            checked_identity = port_or_identity
        else:
            host, port = normalize_endpoint_host_port(endpoint_or_host, port_or_identity)
            checked_identity = identity
        if not isinstance(checked_identity, ProcessIdentity):
            return None
        if capture_process_identity(checked_identity.leader_pid) != checked_identity:
            return None
        before_members = live_process_group_members(checked_identity.pgid)
        before_tokens: dict[int, str] = {}
        for pid in before_members:
            member = capture_process_identity(pid)
            if member is None or member.pgid != checked_identity.pgid:
                return None
            before_tokens[pid] = member.birth_token
        if before_tokens.get(checked_identity.leader_pid) != checked_identity.birth_token:
            return None
        owner_tokens = _listener_owner_birth_tokens(host, port)
        if not owner_tokens or not set(owner_tokens).issubset(before_tokens):
            return None
        for pid, token in owner_tokens.items():
            if before_tokens.get(pid) != token:
                return None
        after_members = live_process_group_members(checked_identity.pgid)
        after_tokens: dict[int, str] = {}
        for pid in after_members:
            member = capture_process_identity(pid)
            if member is None or member.pgid != checked_identity.pgid:
                return None
            after_tokens[pid] = member.birth_token
        if capture_process_identity(checked_identity.leader_pid) != checked_identity:
            return None
        if any(after_tokens.get(pid) != token for pid, token in owner_tokens.items()):
            return None
        return EndpointOwnershipProof(host, port, checked_identity, frozenset(owner_tokens))
    except (EndpointOwnershipError, ProcessIdentityError, ValueError):
        return None


# The listener_* spelling reads naturally at call sites and keeps the endpoint
# spelling available for state formats and external callers.
listener_owned_by_pid = endpoint_owned_by_pid
listener_owned_by_pgid = endpoint_owned_by_pgid
prove_listener_owned_by_identity = prove_endpoint_owned_by_identity
prove_endpoint_owned_by_process_identity = prove_endpoint_owned_by_identity


def effective_launch_environment(
    overrides: dict[str, str] | None = None,
    *,
    cwd: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Return the exact environment inherited by a managed child process."""
    env = dict(os.environ)
    for volatile_key in ("_", "SHLVL", "OLDPWD"):
        env.pop(volatile_key, None)
    if cwd is not None:
        env["PWD"] = os.path.realpath(os.path.abspath(os.fspath(cwd)))
    if overrides:
        env.update(overrides)
    return env


def command_cwd_fingerprint(command: Sequence[str], cwd: str | os.PathLike[str]) -> str:
    """Return a stable SHA-256 fingerprint for command/cwd PID-state metadata."""
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence) or not command:
        raise ValueError("command must be a non-empty sequence of strings")
    if any(not isinstance(part, str) for part in command):
        raise ValueError("command must contain only strings")
    try:
        cwd_text = os.fspath(cwd)
    except TypeError as exc:
        raise ValueError("cwd must be a path string") from exc
    if not isinstance(cwd_text, str) or not cwd_text:
        raise ValueError("cwd must be a non-empty path string")
    payload = json.dumps(
        {"command": list(command), "cwd": os.path.realpath(os.path.abspath(cwd_text))},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


pid_state_fingerprint = command_cwd_fingerprint


_retained_popen_lock = threading.RLock()
_retained_popens: set[subprocess.Popen[Any]] = set()


def retain_popen(proc: subprocess.Popen[Any]) -> subprocess.Popen[Any]:
    """Keep a Popen alive so completed children can be reaped explicitly.

    Retained model-server children are intentionally allowed to outlive a CLI
    process.  The atexit hook only reaps already-completed children; it never
    terminates or waits for a running retained child.
    """
    if not isinstance(proc, subprocess.Popen):
        raise TypeError("proc must be subprocess.Popen")
    with _retained_popen_lock:
        _retained_popens.add(proc)
    return proc


def reap_popen(proc: subprocess.Popen[Any], timeout_sec: int | float | None = 0) -> int | None:
    """Reap one retained Popen, returning its exit code or ``None`` if running."""
    if not isinstance(proc, subprocess.Popen):
        raise TypeError("proc must be subprocess.Popen")
    if timeout_sec is None:
        returncode = proc.wait()
    else:
        timeout = _valid_timeout(timeout_sec)
        try:
            returncode = proc.poll() if timeout == 0 else proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
    if returncode is not None:
        with _retained_popen_lock:
            _retained_popens.discard(proc)
    return returncode


def reap_retained_popens() -> dict[int, int]:
    """Non-blockingly reap completed retained children, keyed by PID."""
    with _retained_popen_lock:
        retained = tuple(_retained_popens)
    reaped: dict[int, int] = {}
    for proc in retained:
        returncode = reap_popen(proc, timeout_sec=0)
        if returncode is not None:
            reaped[proc.pid] = returncode
    return reaped


@atexit.register
def _reap_completed_popens_at_exit() -> None:
    """Reap completed retained children without changing running-child lifetime."""
    reap_retained_popens()


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True


def disk_free_gib(path: str) -> float:
    return shutil.disk_usage(path).free / GIB


def swap_used_gib() -> float | None:
    if sys.platform == "darwin":
        rc, out = run(["sysctl", "vm.swapusage"], timeout=5)
        if rc != 0:
            return None
        match = re.search(r"used\s*=\s*([0-9.]+)([MGT])", out)
        if not match:
            return None
        value = float(match.group(1))
        unit = match.group(2)
        return value / 1024 if unit == "M" else value if unit == "G" else value * 1024
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        data: dict[str, int] = {}
        for line in meminfo.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].rstrip(":") in {"SwapTotal", "SwapFree"}:
                data[parts[0].rstrip(":")] = int(parts[1])
        if "SwapTotal" in data and "SwapFree" in data:
            return (data["SwapTotal"] - data["SwapFree"]) * 1024 / GIB
    return None


def path_size_bytes(path: str) -> int:
    p = Path(path)
    if not p.exists() and not p.is_symlink():
        return 0
    if p.is_file() or p.is_symlink():
        try:
            return p.lstat().st_size
        except OSError:
            return 0
    total = 0
    for root, dirs, files in os.walk(p, onerror=lambda _e: None):
        for name in files:
            try:
                total += (Path(root) / name).lstat().st_size
            except OSError:
                pass
        for name in dirs:
            fp = Path(root) / name
            if fp.is_symlink():
                try:
                    total += fp.lstat().st_size
                except OSError:
                    pass
    return total


def human_bytes(num: int | float) -> str:
    value = float(num)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TiB"
