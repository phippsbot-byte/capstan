from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import gc
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import warnings
from unittest.mock import patch

import modelctl.system as system

from modelctl.system import (
    ProcessIdentity,
    capture_process_identity,
    command_cwd_fingerprint,
    endpoint_owned_by_pgid,
    endpoint_owned_by_pid,
    live_process_group_members,
    normalize_endpoint_host_port,
    pid_alive,
    process_birth_token,
    prove_endpoint_owned_by_identity,
    reap_popen,
    reap_retained_popens,
    retain_popen,
    terminate_process_identity,
    terminate_process_group,
    validate_pgid,
    validate_pid,
)


LISTENER_SCRIPT = r'''
import os
from pathlib import Path
import signal
import socket
import sys
import time

output = Path(sys.argv[1])
family = socket.AF_INET6 if sys.argv[2] == "6" else socket.AF_INET
host = "::1" if family == socket.AF_INET6 else "127.0.0.1"
sock = socket.socket(family, socket.SOCK_STREAM)
try:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, 0))
    sock.listen()
except OSError as exc:
    output.write_text("error:" + repr(exc), encoding="utf-8")
    raise SystemExit(0)
output.write_text(f"{os.getpid()}:{sock.getsockname()[1]}", encoding="utf-8")
signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
while True:
    time.sleep(1)
'''


LEADER_EXITS_WITH_TERM_IGNORING_CHILD = r'''
from pathlib import Path
import signal
import subprocess
import sys
import time

child_ready = Path(str(sys.argv[1]) + ".ready")
child = subprocess.Popen([
    sys.executable,
    "-c",
    "from pathlib import Path; import signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); Path(sys.argv[1]).write_text('ready', encoding='utf-8'); time.sleep(60)",
    str(child_ready),
])
while not child_ready.exists():
    time.sleep(0.01)
Path(sys.argv[1]).write_text(str(child.pid), encoding="utf-8")
signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
while True:
    time.sleep(1)
'''


DETACHED_LISTENER_PARENT = r'''
from pathlib import Path
import subprocess
import sys

child = subprocess.Popen([sys.executable, "-c", sys.argv[2], sys.argv[1], "4"])
Path(sys.argv[3]).write_text(str(child.pid), encoding="utf-8")
'''


DETACH_AND_LISTEN = r'''
import os
os.setsid()
exec(compile(%r, "listener.py", "exec"))
''' % LISTENER_SCRIPT


INHERITED_FOREIGN_LISTENER_OWNER = r'''
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

output = Path(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("127.0.0.1", 0))
sock.listen()
child = subprocess.Popen(
    [sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"],
    pass_fds=(sock.fileno(),),
    start_new_session=True,
)
output.write_text(f"{os.getpid()}:{child.pid}:{sock.getsockname()[1]}", encoding="utf-8")
signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
while True:
    time.sleep(1)
'''


def wait_until(predicate, timeout: float = 8.0, message: str = "condition was not met") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.03)
    raise AssertionError(message)


class ProcessIdentityTests(unittest.TestCase):
    def terminate(self, proc: subprocess.Popen[object]) -> None:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def spawn_listener(self, root: Path, family: int = 4) -> tuple[subprocess.Popen[object], int]:
        ready = root / f"listener-{family}.txt"
        proc = subprocess.Popen(
            [sys.executable, "-c", LISTENER_SCRIPT, str(ready), str(family)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            wait_until(ready.exists, message="listener did not report readiness")
            text = ready.read_text(encoding="utf-8")
            if text.startswith("error:"):
                self.terminate(proc)
                self.skipTest(text)
            listener_pid, port = text.split(":", 1)
            self.assertEqual(int(listener_pid), proc.pid)
            return proc, int(port)
        except BaseException:
            self.terminate(proc)
            raise

    def test_pid_and_pgid_validation_rejects_non_positive_and_bool_values(self):
        self.assertEqual(validate_pid(123), 123)
        self.assertEqual(validate_pgid(456), 456)
        for value in (True, False, 0, -1, 1.0, "1", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_pid(value)
                with self.assertRaises(ValueError):
                    validate_pgid(value)

    def test_invalid_or_bool_group_ids_are_never_signaled(self):
        from unittest.mock import patch

        with patch("modelctl.system.os.killpg") as killpg, patch("modelctl.system.os.kill") as kill:
            for value in (True, False, 0, -1, 1.0, "1", None):
                with self.subTest(value=value):
                    self.assertFalse(terminate_process_group(value, timeout_sec=0))
        killpg.assert_not_called()
        kill.assert_not_called()

    def test_process_birth_token_is_stable_then_absent_after_reap(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        try:
            first = process_birth_token(proc.pid)
            self.assertIsInstance(first, str)
            self.assertTrue(first)
            self.assertEqual(first, process_birth_token(proc.pid))
        finally:
            self.terminate(proc)
        self.assertIsNone(process_birth_token(proc.pid))

    def test_process_birth_token_rejects_zombies_on_linux_and_darwin(self):
        with patch.object(system.sys, "platform", "linux"), patch.object(
            system, "_linux_proc_stat", return_value=("Z", 12, 34)
        ):
            self.assertIsNone(process_birth_token(12))

        zombie = system._ProcBsdInfo()
        zombie.pbi_status = 5  # SZOMB
        zombie.pbi_start_tvsec = 1
        with patch.object(system.sys, "platform", "darwin"), patch.object(
            system, "_darwin_proc_bsdinfo", return_value=zombie
        ):
            self.assertIsNone(process_birth_token(12))

    def test_stale_process_identity_never_signals_a_reused_group(self):
        identity = ProcessIdentity(leader_pid=314, pgid=271, birth_token="linux:old")
        replacements = (
            ProcessIdentity(leader_pid=314, pgid=271, birth_token="linux:new"),
            ProcessIdentity(leader_pid=314, pgid=272, birth_token="linux:old"),
            ProcessIdentity(leader_pid=315, pgid=271, birth_token="linux:old"),
        )
        for replacement in replacements:
            with self.subTest(replacement=replacement), patch.object(
                system, "capture_process_identity", return_value=replacement
            ), patch.object(system.os, "killpg") as killpg:
                self.assertFalse(terminate_process_identity(identity, timeout_sec=0))
            killpg.assert_not_called()

    def test_group_shutdown_kills_child_and_never_false_positive_certifies(self):
        with tempfile.TemporaryDirectory() as td:
            child_file = Path(td) / "child.pid"
            leader = subprocess.Popen(
                [sys.executable, "-c", LEADER_EXITS_WITH_TERM_IGNORING_CHILD, str(child_file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                wait_until(child_file.exists, message="leader did not create child")
                child_pid = int(child_file.read_text(encoding="utf-8"))
                identity = capture_process_identity(leader.pid)
                self.assertIsNotNone(identity)
                self.assertIn(child_pid, live_process_group_members(leader.pid))
                certified = terminate_process_identity(identity, timeout_sec=10.0)
                wait_until(lambda: not pid_alive(child_pid), message="TERM-ignoring child survived group KILL")
                if certified:
                    self.assertEqual(live_process_group_members(leader.pid), [], "termination must never certify a non-empty group")
            finally:
                try:
                    os.killpg(leader.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                self.terminate(leader)
                self.assertIsNotNone(leader.returncode)

    def test_normalized_endpoint_host_and_effective_port(self):
        self.assertEqual(normalize_endpoint_host_port("https://EXAMPLE.test/v1"), ("example.test", 443))
        self.assertEqual(normalize_endpoint_host_port("http://[::1]:8123/v1"), ("::1", 8123))
        self.assertEqual(normalize_endpoint_host_port("127.0.0.1", 8124), ("127.0.0.1", 8124))
        for endpoint in ("", "http:///v1", "http://127.0.0.1:0", "ftp://127.0.0.1"):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(ValueError):
                    normalize_endpoint_host_port(endpoint)

    def test_listener_owner_rejects_unrelated_pid_and_accepts_tracked_group(self):
        with tempfile.TemporaryDirectory() as td:
            proc, port = self.spawn_listener(Path(td))
            try:
                identity = capture_process_identity(proc.pid)
                self.assertIsNotNone(identity)
                proof = prove_endpoint_owned_by_identity("127.0.0.1", port, identity)
                self.assertIsNotNone(proof)
                self.assertEqual(proof.owner_pids, frozenset({proc.pid}))
                self.assertTrue(endpoint_owned_by_pid("127.0.0.1", port, proc.pid))
                self.assertTrue(endpoint_owned_by_pgid("127.0.0.1", port, proc.pid))
                self.assertFalse(endpoint_owned_by_pid("127.0.0.1", port, os.getpid()))
            finally:
                self.terminate(proc)

    def test_endpoint_proof_rejects_foreign_co_listener_owner(self):
        tracked = ProcessIdentity(leader_pid=100, pgid=100, birth_token="linux:tracked")
        foreign = ProcessIdentity(leader_pid=200, pgid=200, birth_token="linux:foreign")

        def current(pid: object):
            return tracked if pid == 100 else foreign if pid == 200 else None

        with patch.object(system, "capture_process_identity", side_effect=current), patch.object(
            system, "live_process_group_members", return_value=[100]
        ), patch.object(
            system, "_listener_owner_birth_tokens", return_value={100: "linux:tracked", 200: "linux:foreign"}
        ):
            self.assertIsNone(prove_endpoint_owned_by_identity("127.0.0.1", 8123, tracked))

    def test_endpoint_proof_returns_only_a_complete_authenticated_owner_set(self):
        tracked = ProcessIdentity(leader_pid=100, pgid=100, birth_token="linux:tracked")
        with patch.object(system, "capture_process_identity", return_value=tracked), patch.object(
            system, "live_process_group_members", return_value=[100]
        ), patch.object(system, "_listener_owner_birth_tokens", return_value={100: "linux:tracked"}):
            proof = prove_endpoint_owned_by_identity("127.0.0.1", 8123, tracked)
        self.assertIsNotNone(proof)
        self.assertEqual(proof.owner_pids, frozenset({100}))

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux /proc FD inheritance")
    def test_endpoint_proof_rejects_detached_inherited_listener_owner(self):
        with tempfile.TemporaryDirectory() as td:
            ready = Path(td) / "owners.txt"
            parent = subprocess.Popen(
                [sys.executable, "-c", INHERITED_FOREIGN_LISTENER_OWNER, str(ready)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            child_pid = None
            try:
                wait_until(ready.exists, message="inherited listener did not report readiness")
                parent_pid, child_text, port_text = ready.read_text(encoding="utf-8").split(":", 2)
                self.assertEqual(int(parent_pid), parent.pid)
                child_pid = int(child_text)
                identity = capture_process_identity(parent.pid)
                self.assertIsNotNone(identity)
                self.assertIsNone(prove_endpoint_owned_by_identity("127.0.0.1", int(port_text), identity))
            finally:
                self.terminate(parent)
                if child_pid is not None:
                    try:
                        os.killpg(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    wait_until(lambda: not pid_alive(child_pid), message="detached inherited listener leaked")

    def test_endpoint_owner_pid_reuse_between_inspection_checks_fails_closed(self):
        with patch.object(system, "_linux_listener_inodes", side_effect=[{"91"}, {"91"}]), patch.object(
            system, "_linux_all_pids", return_value=[77]
        ), patch.object(system, "_linux_pid_socket_inodes", return_value={"91"}), patch.object(
            system, "process_birth_token", side_effect=["linux:before", "linux:after"]
        ):
            with self.assertRaises(system.EndpointOwnershipError):
                system._linux_listener_owner_birth_tokens("127.0.0.1", 8123)

    def test_darwin_inventory_failures_are_never_treated_as_empty_or_complete(self):
        class ShortListPids:
            def __call__(self, *_args):
                return 3

        with patch.object(system.ctypes, "CDLL", return_value=SimpleNamespace(proc_listpids=ShortListPids())):
            with self.assertRaises(system.ProcessIdentityError):
                system._darwin_list_pids(2, 99)

        values = iter([4, 4, 8, 8, 16, 16, 32, 32])

        class GrowingListPids:
            def __call__(self, *_args):
                return next(values)

        with patch.object(system.ctypes, "CDLL", return_value=SimpleNamespace(proc_listpids=GrowingListPids())):
            with self.assertRaises(system.ProcessIdentityError):
                system._darwin_list_pids(2, 99)

        with patch.object(system, "_darwin_list_pids", return_value=[44]), patch.object(
            system, "_darwin_proc_bsdinfo", return_value=None
        ), patch.object(system, "_darwin_bsdinfo_unavailable_errno", return_value=system.errno.EACCES), patch.object(
            system.os, "kill", return_value=None
        ):
            with self.assertRaises(system.ProcessIdentityError):
                system._darwin_live_process_group_members(99)

    def test_darwin_lsof_wildcards_remain_family_specific(self):
        result = SimpleNamespace(returncode=0, stderr="", stdout="p88\nf4\ntIPv6\nn*:8123\n")
        with patch.object(system.shutil, "which", return_value="/usr/sbin/lsof"), patch.object(
            system.subprocess, "run", return_value=result
        ), patch.object(system, "process_birth_token", return_value="darwin:1:2"):
            self.assertEqual(system._darwin_listener_owner_birth_tokens("127.0.0.1", 8123), {})
            self.assertEqual(system._darwin_listener_owner_birth_tokens("::1", 8123), {88: "darwin:1:2"})

    def test_listener_detached_with_setsid_is_rejected_for_original_group(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ready = root / "detached-listener.txt"
            child_file = root / "detached-child.pid"
            parent = subprocess.Popen(
                [sys.executable, "-c", DETACHED_LISTENER_PARENT, str(ready), DETACH_AND_LISTEN, str(child_file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            child_pid = None
            try:
                wait_until(ready.exists, message="detached listener did not report readiness")
                child_pid = int(child_file.read_text(encoding="utf-8"))
                listener_state = ready.read_text(encoding="utf-8")
                if listener_state.startswith("error:"):
                    self.skipTest(listener_state)
                _reported_pid, port_text = listener_state.split(":", 1)
                port = int(port_text)
                parent.wait(timeout=5)
                self.assertNotEqual(os.getpgid(child_pid), parent.pid)
                self.assertTrue(endpoint_owned_by_pid("127.0.0.1", port, child_pid))
                self.assertFalse(endpoint_owned_by_pgid("127.0.0.1", port, parent.pid))
            finally:
                self.terminate(parent)
                if child_pid is not None:
                    terminate_process_group(child_pid, timeout_sec=0.2)
                    wait_until(lambda: not pid_alive(child_pid), message="detached listener leaked")

    def test_ipv6_listener_ownership_when_available(self):
        if not socket.has_ipv6:
            self.skipTest("IPv6 is unavailable")
        with tempfile.TemporaryDirectory() as td:
            proc, port = self.spawn_listener(Path(td), family=6)
            try:
                self.assertTrue(endpoint_owned_by_pid("[::1]", port, proc.pid))
                self.assertTrue(endpoint_owned_by_pgid(f"http://[::1]:{port}/v1", proc.pid))
            finally:
                self.terminate(proc)

    def test_command_cwd_fingerprint_is_deterministic_and_sensitive_to_identity(self):
        first = command_cwd_fingerprint(["python", "serve.py", "--port", "8123"], "/tmp/work")
        self.assertEqual(first, command_cwd_fingerprint(["python", "serve.py", "--port", "8123"], "/tmp/work"))
        self.assertNotEqual(first, command_cwd_fingerprint(["python", "serve.py", "--port", "8124"], "/tmp/work"))
        self.assertNotEqual(first, command_cwd_fingerprint(["python", "serve.py", "--port", "8123"], "/tmp/other"))

    def test_retained_popen_can_be_reaped_without_leaking_a_child(self):
        proc = retain_popen(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.25)"]))
        self.assertIsNone(reap_popen(proc, timeout_sec=0))
        self.assertEqual(reap_popen(proc, timeout_sec=5), 0)
        self.assertNotIn(proc.pid, reap_retained_popens())

        retained = retain_popen(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.1)"]))
        retained_pid = retained.pid
        del retained
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            gc.collect()
        reaped: dict[int, int] = {}

        def reaped_child() -> bool:
            nonlocal reaped
            reaped = reap_retained_popens()
            return retained_pid in reaped

        wait_until(reaped_child, message="retained Popen was not reaped")
        self.assertEqual(reaped[retained_pid], 0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            gc.collect()
