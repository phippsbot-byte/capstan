from __future__ import annotations

import json
from pathlib import Path
import errno
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from modelctl.manifest import ModelManifest, PreflightConfig, StartConfig


ROOT = Path(__file__).resolve().parents[1]


class LifecycleLockingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        # macOS commonly exposes /var as a symlink to /private/var. The core
        # intentionally rejects symlink ancestors, so test roots use the
        # physical temporary directory rather than that compatibility alias.
        self.root = Path(self.temp.name).resolve()
        self.lock_root = self.root / "locks"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def manifest(self, name: str, endpoint: str, pid_path: Path, ports: list[int] | None = None) -> ModelManifest:
        manifest_path = self.root / f"{name}.toml"
        return ModelManifest(
            path=manifest_path,
            id=name,
            model_id="test-model",
            endpoint=endpoint,
            start=StartConfig(command=[sys.executable, "-c", "pass"], pid_path=str(pid_path)),
            preflight=PreflightConfig(exclusive_ports=ports or []),
        )

    def test_default_namespace_ignores_xdg_state_home(self) -> None:
        from modelctl.lifecycle import default_lock_root

        original = os.environ.get("XDG_STATE_HOME")
        try:
            os.environ["XDG_STATE_HOME"] = str(self.root / "xdg-one")
            first = default_lock_root()
            os.environ["XDG_STATE_HOME"] = str(self.root / "xdg-two")
            second = default_lock_root()
        finally:
            if original is None:
                os.environ.pop("XDG_STATE_HOME", None)
            else:
                os.environ["XDG_STATE_HOME"] = original
        self.assertEqual(first, second)
        self.assertNotIn("xdg-", str(first))

    def test_resource_keys_canonicalize_symlink_paths_and_normalize_ipv6_endpoints(self) -> None:
        from modelctl.lifecycle import EndpointIdentity, endpoint_identity, lifecycle_resources, resolve_manifest_path

        real = self.root / "real-state"
        real.mkdir()
        alias = self.root / "alias-state"
        alias.symlink_to(real, target_is_directory=True)
        one = self.manifest("one", "HTTP://[2001:0db8:0:0::1]/v1", real / "lane.pid", [8123])
        two = self.manifest("two", "http://[2001:db8::1]:80/v1", alias / "lane.pid", [8123])

        self.assertEqual(resolve_manifest_path(two, two.start.pid_path), (real / "lane.pid").resolve())
        self.assertEqual(
            endpoint_identity(one.endpoint),
            EndpointIdentity(host="2001:db8::1", port=80),
        )
        resources = lifecycle_resources(one, two)
        self.assertEqual(resources, tuple(sorted(resources)))
        self.assertEqual(sum(resource.startswith("pid:") for resource in resources), 1)
        self.assertIn("endpoint:[2001:db8::1]:80", resources)
        self.assertIn("tcp-port:8123", resources)
        self.assertFalse(any(resource.startswith("service:") for resource in resources))

    def test_local_endpoint_aliases_and_manifest_path_serialize_without_explicit_pid_path(self) -> None:
        from modelctl.lifecycle import EndpointIdentity, acquire_locks, endpoint_identity, lifecycle_resources

        manifest_path = self.root / "shared.toml"
        aliases = [
            "http://localhost:8123/v1",
            "http://127.0.0.1:8123/v1",
            "http://0.0.0.0:8123/v1",
            "http://[::1]:8123/v1",
            "http://[::]:8123/v1",
        ]
        manifests = [
            ModelManifest(
                path=manifest_path,
                id=f"alias-{index}",
                model_id="test-model",
                endpoint=endpoint,
                start=StartConfig(command=[sys.executable, "-c", "pass"]),
            )
            for index, endpoint in enumerate(aliases)
        ]
        expected_endpoint = EndpointIdentity(host="local", port=8123)
        self.assertTrue(all(endpoint_identity(endpoint) == expected_endpoint for endpoint in aliases))

        original = os.environ.get("XDG_STATE_HOME")
        try:
            os.environ["XDG_STATE_HOME"] = str(self.root / "xdg-one")
            first_resources = lifecycle_resources(manifests[0])
            os.environ["XDG_STATE_HOME"] = str(self.root / "xdg-two")
            second_resources = lifecycle_resources(manifests[-1])
        finally:
            if original is None:
                os.environ.pop("XDG_STATE_HOME", None)
            else:
                os.environ["XDG_STATE_HOME"] = original

        manifest_resource = f"manifest:{manifest_path.resolve()}"
        self.assertIn("endpoint:local:8123", first_resources)
        self.assertIn("endpoint:local:8123", second_resources)
        self.assertIn(manifest_resource, first_resources)
        self.assertIn(manifest_resource, second_resources)
        self.assertNotEqual(
            next(resource for resource in first_resources if resource.startswith("pid:")),
            next(resource for resource in second_resources if resource.startswith("pid:")),
        )

        ready = threading.Event()
        release = threading.Event()
        errors: list[BaseException] = []

        def hold() -> None:
            try:
                result = acquire_locks(first_resources, operation="local-alias-holder", lock_root=self.lock_root)
                self.assertTrue(result.ok, result.error)
                assert result.lock is not None
                try:
                    ready.set()
                    release.wait(5)
                finally:
                    result.lock.release()
            except BaseException as exc:
                errors.append(exc)

        holder = threading.Thread(target=hold)
        holder.start()
        try:
            self.assertTrue(ready.wait(3), "holder did not complete lock acquisition")
            contender = acquire_locks(
                second_resources,
                operation="local-alias-contender",
                lock_root=self.lock_root,
                blocking=False,
            )
            self.assertFalse(contender.ok)
            self.assertIsNotNone(contender.error)
            assert contender.error is not None
            self.assertEqual(contender.error.code, "contended")
        finally:
            release.set()
            holder.join(5)
        self.assertFalse(holder.is_alive())
        self.assertFalse(errors, errors)

    def test_acquires_files_with_owner_only_modes_and_releases(self) -> None:
        from modelctl.lifecycle import acquire_locks

        result = acquire_locks(["path:/tmp/capstan-test-resource"], operation="mode-test", lock_root=self.lock_root)
        self.assertTrue(result.ok, result.error)
        assert result.lock is not None
        try:
            info = result.lock.info
            self.assertEqual(info.resources, ("path:" + str(Path("/tmp/capstan-test-resource").resolve()),))
            self.assertEqual(stat.S_IMODE(self.lock_root.stat().st_mode), 0o700)
            self.assertEqual(self.lock_root.stat().st_uid, os.geteuid())
            self.assertEqual(len(info.paths), 1)
            lock_path = Path(info.paths[0])
            self.assertTrue(lock_path.is_file())
            self.assertEqual(stat.S_IMODE(lock_path.stat().st_mode), 0o600)
            self.assertEqual(lock_path.stat().st_uid, os.geteuid())
        finally:
            result.lock.release()

    def test_fails_closed_for_unsafe_namespace_and_final_lock_file(self) -> None:
        from modelctl.lifecycle import acquire_locks

        unsafe_root = self.root / "unsafe"
        unsafe_root.mkdir(mode=0o700)
        unsafe_root.chmod(0o755)
        unsafe = acquire_locks(["alpha"], operation="unsafe-root", lock_root=unsafe_root)
        self.assertFalse(unsafe.ok)
        self.assertIsNotNone(unsafe.error)
        assert unsafe.error is not None
        self.assertEqual(unsafe.error.code, "unsafe_lock_root_mode")

        initial = acquire_locks(["alpha"], operation="create", lock_root=self.lock_root)
        self.assertTrue(initial.ok, initial.error)
        assert initial.lock is not None
        lock_path = Path(initial.lock.info.paths[0])
        initial.lock.release()
        lock_path.parent.chmod(0o700)
        lock_path.chmod(0o644)
        bad_file_mode = acquire_locks(["alpha"], operation="bad-file-mode", lock_root=self.lock_root)
        self.assertFalse(bad_file_mode.ok)
        self.assertIsNotNone(bad_file_mode.error)
        assert bad_file_mode.error is not None
        self.assertEqual(bad_file_mode.error.code, "unsafe_lock_file_mode")
        lock_path.chmod(0o600)
        self.assertEqual(stat.S_IMODE(lock_path.parent.stat().st_mode), 0o500)
        lock_path.parent.chmod(0o700)
        lock_path.unlink()
        target = self.root / "target"
        target.write_text("not a lock", encoding="utf-8")
        lock_path.symlink_to(target)
        poisoned = acquire_locks(["alpha"], operation="symlink-final", lock_root=self.lock_root)
        self.assertFalse(poisoned.ok)
        self.assertIsNotNone(poisoned.error)
        assert poisoned.error is not None
        self.assertEqual(poisoned.error.code, "symlink_lock_file")

        actual = self.root / "actual"
        actual.mkdir(mode=0o700)
        ancestor = self.root / "symlink-ancestor"
        ancestor.symlink_to(actual, target_is_directory=True)
        bad_ancestor = acquire_locks(["beta"], operation="symlink-ancestor", lock_root=ancestor / "locks")
        self.assertFalse(bad_ancestor.ok)
        self.assertIsNotNone(bad_ancestor.error)
        assert bad_ancestor.error is not None
        self.assertEqual(bad_ancestor.error.code, "symlink_ancestor")

        writable_ancestor = self.root / "writable-ancestor"
        writable_ancestor.mkdir(mode=0o700)
        writable_ancestor.chmod(0o722)
        bad_mode_ancestor = acquire_locks(["gamma"], operation="bad-ancestor-mode", lock_root=writable_ancestor / "locks")
        self.assertFalse(bad_mode_ancestor.ok)
        self.assertIsNotNone(bad_mode_ancestor.error)
        assert bad_mode_ancestor.error is not None
        self.assertEqual(bad_mode_ancestor.error.code, "unsafe_lock_ancestor_mode")

    def test_rejects_foreign_writable_ancestors_except_root_sticky_temp(self) -> None:
        from modelctl.lifecycle import _FailureSignal, _validate_user_owned_ancestor

        effective_uid = 501
        foreign_writable = os.stat_result((stat.S_IFDIR | 0o777, 0, 0, 0, 777, 0, 0, 0, 0, 0))
        trusted_temp = os.stat_result((stat.S_IFDIR | 0o1777, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        foreign_read_only = os.stat_result((stat.S_IFDIR | 0o755, 0, 0, 0, 777, 0, 0, 0, 0, 0))

        with patch("modelctl.lifecycle.os.geteuid", return_value=effective_uid):
            with self.assertRaises(_FailureSignal) as raised:
                _validate_user_owned_ancestor(foreign_writable, Path("/foreign"), seen_user_owned=False)
            self.assertEqual(raised.exception.failure.code, "unsafe_lock_ancestor_mode")
            self.assertFalse(
                _validate_user_owned_ancestor(trusted_temp, Path("/tmp"), seen_user_owned=False)
            )
            self.assertFalse(
                _validate_user_owned_ancestor(foreign_read_only, Path("/usr"), seen_user_owned=False)
            )

    def test_resource_directory_prevents_normal_cleanup_from_replacing_held_inode(self) -> None:
        from modelctl.lifecycle import acquire_locks

        result = acquire_locks(["inode-guard"], operation="inode-guard", lock_root=self.lock_root)
        self.assertTrue(result.ok, result.error)
        assert result.lock is not None
        try:
            info = result.lock.info
            self.assertEqual(info.lock_root, str(self.lock_root))
            self.assertEqual(len(info.resource_paths), 1)
            resource_path = Path(info.resource_paths[0])
            lock_path = Path(info.paths[0])
            self.assertEqual(lock_path.parent, resource_path)
            self.assertEqual(stat.S_IMODE(resource_path.stat().st_mode), 0o500)
            directory_inode = resource_path.stat()
            file_inode = lock_path.stat()
            if os.geteuid() != 0:
                with self.assertRaises(PermissionError):
                    lock_path.unlink()
                with self.assertRaises(PermissionError):
                    shutil.rmtree(resource_path)
            self.assertEqual((resource_path.stat().st_dev, resource_path.stat().st_ino), (directory_inode.st_dev, directory_inode.st_ino))
            self.assertEqual((lock_path.stat().st_dev, lock_path.stat().st_ino), (file_inode.st_dev, file_inode.st_ino))
            observed: list[object] = []

            def contend() -> None:
                observed.append(
                    acquire_locks(
                        ["inode-guard"],
                        operation="inode-guard-contender",
                        lock_root=self.lock_root,
                        blocking=False,
                    )
                )

            contender_thread = threading.Thread(target=contend)
            contender_thread.start()
            contender_thread.join(3)
            self.assertFalse(contender_thread.is_alive())
            self.assertEqual(len(observed), 1)
            contender = observed[0]
            self.assertFalse(contender.ok)
            self.assertIsNotNone(contender.error)
            assert contender.error is not None
            self.assertEqual(contender.error.code, "contended")
        finally:
            result.lock.release()

    def test_rejects_hardlinks_and_resource_directory_symlinks(self) -> None:
        from modelctl.lifecycle import acquire_locks

        initial = acquire_locks(["hardlink"], operation="hardlink-create", lock_root=self.lock_root)
        self.assertTrue(initial.ok, initial.error)
        assert initial.lock is not None
        lock_path = Path(initial.lock.info.paths[0])
        resource_path = Path(initial.lock.info.resource_paths[0])
        initial.lock.release()

        hardlink = self.root / "second-link"
        hardlink.hardlink_to(lock_path)
        linked = acquire_locks(["hardlink"], operation="hardlink-reject", lock_root=self.lock_root)
        self.assertFalse(linked.ok)
        self.assertIsNotNone(linked.error)
        assert linked.error is not None
        self.assertEqual(linked.error.code, "unsafe_lock_file_links")
        hardlink.unlink()

        target = self.root / "resource-target"
        target.mkdir(mode=0o700)
        symlink_lock_path = self.lock_root / hashlib.sha256(b"directory-symlink").hexdigest() / "lock"
        symlink_lock_path.parent.symlink_to(target, target_is_directory=True)
        symlinked = acquire_locks(
            ["directory-symlink"], operation="directory-symlink-reject", lock_root=self.lock_root
        )
        self.assertFalse(symlinked.ok)
        self.assertIsNotNone(symlinked.error)
        assert symlinked.error is not None
        self.assertEqual(symlinked.error.code, "symlink_resource_directory")
        self.assertEqual(stat.S_IMODE(resource_path.stat().st_mode), 0o500)

    def test_bounded_lock_file_creation_race_is_iterative_and_fail_closed(self) -> None:
        import modelctl.lifecycle as lifecycle

        real_open = lifecycle.os.open
        attempts = 0

        def always_racing_open(path: str | bytes | os.PathLike[str] | os.PathLike[bytes], flags: int, *args: object, **kwargs: object) -> int:
            nonlocal attempts
            if os.fspath(path) == "lock" and flags & os.O_EXCL:
                attempts += 1
                raise FileExistsError(errno.EEXIST, "simulated creation race")
            return real_open(path, flags, *args, **kwargs)

        with patch("modelctl.lifecycle.os.open", side_effect=always_racing_open):
            result = lifecycle.acquire_locks(["creation-race"], operation="creation-race", lock_root=self.lock_root)
        if result.lock is not None:
            result.lock.release()
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertEqual(result.error.code, "lock_file_creation_race")
        self.assertEqual(attempts, lifecycle._LOCK_FILE_CREATE_RETRIES)
        resource_path = self.lock_root / hashlib.sha256(b"creation-race").hexdigest()
        self.assertTrue(resource_path.exists())
        self.assertEqual(stat.S_IMODE(resource_path.stat().st_mode), 0o500)

    def test_reentrant_subset_acquisition_is_immediate_and_widening_fails_explicitly(self) -> None:
        from modelctl.lifecycle import LifecycleLockError, acquire_locks, resource_lock

        with resource_lock(["b", "a"], operation="outer", lock_root=self.lock_root) as outer:
            with resource_lock(["a"], operation="inner", lock_root=self.lock_root) as inner:
                self.assertEqual(outer.resources, ("a", "b"))
                self.assertEqual(inner.resources, ("a",))
            widened = acquire_locks(["a", "b", "c"], operation="widen", lock_root=self.lock_root)
            self.assertFalse(widened.ok)
            self.assertIsNotNone(widened.error)
            assert widened.error is not None
            self.assertEqual(widened.error.code, "nested_scope_widening")
            with self.assertRaises(LifecycleLockError):
                with resource_lock(["a", "b", "c"], operation="widen", lock_root=self.lock_root):
                    pass

    def test_thread_contender_waits_after_holder_has_acquired(self) -> None:
        from modelctl.lifecycle import acquire_locks, resource_lock

        real = self.root / "thread-real"
        real.mkdir()
        alias = self.root / "thread-alias"
        alias.symlink_to(real, target_is_directory=True)
        holder_acquired = threading.Event()
        contender_started = threading.Event()
        contender_acquired = threading.Event()
        release_holder = threading.Event()
        errors: list[BaseException] = []

        def holder() -> None:
            try:
                with resource_lock([f"path:{alias / 'resource'}"], operation="holder", lock_root=self.lock_root):
                    holder_acquired.set()
                    release_holder.wait(5)
            except BaseException as exc:  # Keep assertion failures in the parent thread.
                errors.append(exc)

        def contender() -> None:
            try:
                self.assertTrue(holder_acquired.wait(3))
                contender_started.set()
                with resource_lock([f"path:{real / 'resource'}"], operation="contender", lock_root=self.lock_root):
                    contender_acquired.set()
            except BaseException as exc:
                errors.append(exc)

        first = threading.Thread(target=holder)
        second = threading.Thread(target=contender)
        first.start()
        try:
            self.assertTrue(holder_acquired.wait(3), "holder did not complete lock acquisition")
            nonblocking = acquire_locks(
                [f"path:{real / 'resource'}"],
                operation="nonblocking-contender",
                lock_root=self.lock_root,
                blocking=False,
            )
            self.assertFalse(nonblocking.ok)
            self.assertIsNotNone(nonblocking.error)
            assert nonblocking.error is not None
            self.assertEqual(nonblocking.error.code, "contended")
            second.start()
            self.assertTrue(contender_started.wait(3), "contender did not reach its pre-acquisition handshake")
            self.assertFalse(contender_acquired.wait(0.2), "thread contender acquired while holder still owned the resource")
        finally:
            release_holder.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertFalse(errors, errors)
        self.assertTrue(contender_acquired.is_set())

    def test_opposite_order_multi_resource_threads_do_not_deadlock(self) -> None:
        from modelctl.lifecycle import resource_lock

        holder_acquired = threading.Event()
        contender_started = threading.Event()
        contender_acquired = threading.Event()
        release_holder = threading.Event()
        errors: list[BaseException] = []

        def holder() -> None:
            try:
                with resource_lock(["thread-b", "thread-a"], operation="thread-holder", lock_root=self.lock_root):
                    holder_acquired.set()
                    release_holder.wait(5)
            except BaseException as exc:
                errors.append(exc)

        def contender() -> None:
            try:
                self.assertTrue(holder_acquired.wait(3))
                contender_started.set()
                with resource_lock(["thread-a", "thread-b"], operation="thread-contender", lock_root=self.lock_root):
                    contender_acquired.set()
            except BaseException as exc:
                errors.append(exc)

        first = threading.Thread(target=holder)
        second = threading.Thread(target=contender)
        first.start()
        try:
            self.assertTrue(holder_acquired.wait(3))
            second.start()
            self.assertTrue(contender_started.wait(3))
            self.assertFalse(contender_acquired.wait(0.2))
        finally:
            release_holder.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertFalse(errors, errors)
        self.assertTrue(contender_acquired.is_set())

    def test_partial_nonblocking_failure_releases_earlier_resources_and_thread_entries(self) -> None:
        from modelctl.lifecycle import _thread_locks, acquire_locks

        ready = threading.Event()
        release = threading.Event()
        errors: list[BaseException] = []

        def hold() -> None:
            try:
                result = acquire_locks(["partial-b"], operation="partial-holder", lock_root=self.lock_root)
                self.assertTrue(result.ok, result.error)
                assert result.lock is not None
                try:
                    ready.set()
                    release.wait(5)
                finally:
                    result.lock.release()
            except BaseException as exc:
                errors.append(exc)

        holder = threading.Thread(target=hold)
        holder.start()
        try:
            self.assertTrue(ready.wait(3), "holder did not complete lock acquisition")
            failed = acquire_locks(
                ["partial-a", "partial-b"],
                operation="partial-contender",
                lock_root=self.lock_root,
                blocking=False,
            )
            self.assertFalse(failed.ok)
            self.assertIsNotNone(failed.error)
            assert failed.error is not None
            self.assertEqual(failed.error.code, "contended")
            released = acquire_locks(
                ["partial-a"], operation="partial-cleanup-check", lock_root=self.lock_root, blocking=False
            )
            self.assertTrue(released.ok, released.error)
            assert released.lock is not None
            released.lock.release()
        finally:
            release.set()
            holder.join(5)
        self.assertFalse(holder.is_alive())
        self.assertFalse(errors, errors)
        self.assertFalse(any(root == str(self.lock_root) for root, _resource in _thread_locks))

    def test_out_of_order_lease_release_keeps_scope_until_last_lease(self) -> None:
        from modelctl.lifecycle import acquire_locks

        outer = acquire_locks(["release-a", "release-b"], operation="outer", lock_root=self.lock_root)
        self.assertTrue(outer.ok, outer.error)
        assert outer.lock is not None
        inner = acquire_locks(["release-a"], operation="inner", lock_root=self.lock_root)
        self.assertTrue(inner.ok, inner.error)
        assert inner.lock is not None
        try:
            outer.lock.release()
            blocked = acquire_locks(
                ["release-a"], operation="external-check", lock_root=self.lock_root, blocking=False
            )
            self.assertTrue(blocked.ok, blocked.error)
            # The same thread is reentrant, so use an independent thread for the real contention check.
            assert blocked.lock is not None
            blocked.lock.release()
            observed: list[object] = []

            def contender() -> None:
                observed.append(
                    acquire_locks(
                        ["release-a"], operation="external-thread-check", lock_root=self.lock_root, blocking=False
                    )
                )

            thread = threading.Thread(target=contender)
            thread.start()
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(observed), 1)
            result = observed[0]
            self.assertFalse(result.ok)  # type: ignore[union-attr]
            self.assertEqual(result.error.code, "contended")  # type: ignore[union-attr]
        finally:
            inner.lock.release()
        after = acquire_locks(["release-a"], operation="after-last-release", lock_root=self.lock_root, blocking=False)
        self.assertTrue(after.ok, after.error)
        assert after.lock is not None
        after.lock.release()

    def test_cross_process_opposite_order_multi_resource_contention_uses_fixed_namespace(self) -> None:
        ready = self.root / "holder-ready"
        release = self.root / "holder-release"
        contender_started = self.root / "contender-started"
        contender_acquired = self.root / "contender-acquired"
        holder_resources = ["process-b", "process-a"]
        contender_resources = ["process-a", "process-b"]
        holder_code = """
import json, os, sys, time
from pathlib import Path
from modelctl.lifecycle import resource_lock
os.environ['XDG_STATE_HOME'] = sys.argv[5]
with resource_lock(json.loads(sys.argv[1]), operation='holder', lock_root=sys.argv[2]):
    Path(sys.argv[3]).touch()
    while not Path(sys.argv[4]).exists():
        time.sleep(0.01)
"""
        contender_code = """
import json, os, sys
from pathlib import Path
from modelctl.lifecycle import resource_lock
os.environ['XDG_STATE_HOME'] = sys.argv[5]
Path(sys.argv[3]).touch()
with resource_lock(json.loads(sys.argv[1]), operation='contender', lock_root=sys.argv[2]):
    Path(sys.argv[4]).touch()
"""
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        holder = subprocess.Popen(
            [sys.executable, "-c", holder_code, json.dumps(holder_resources), str(self.lock_root), str(ready), str(release), str(self.root / "xdg-a")],
            cwd=ROOT,
            env=env,
        )
        contender = None
        try:
            self.assertTrue(self.wait_for_path(ready, 5), "holder did not complete lock acquisition")
            contender = subprocess.Popen(
                [sys.executable, "-c", contender_code, json.dumps(contender_resources), str(self.lock_root), str(contender_started), str(contender_acquired), str(self.root / "xdg-b")],
                cwd=ROOT,
                env=env,
            )
            self.assertTrue(self.wait_for_path(contender_started, 5), "contender did not reach its pre-acquisition handshake")
            self.assertFalse(
                self.wait_for_path(contender_acquired, 0.2),
                "cross-process contender acquired while holder still owned the resource",
            )
            release.touch()
            self.assertEqual(contender.wait(timeout=5), 0)
            self.assertTrue(contender_acquired.exists())
            self.assertEqual(holder.wait(timeout=5), 0)
        finally:
            release.touch()
            for process in (contender, holder):
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)

    @staticmethod
    def wait_for_path(path: Path, timeout: float) -> bool:
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                return True
            time.sleep(0.01)
        return path.exists()


if __name__ == "__main__":
    unittest.main()
