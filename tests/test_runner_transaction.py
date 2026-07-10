from __future__ import annotations

import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from modelctl.manifest import load_manifest
from modelctl.system import capture_process_identity, reap_retained_popens, terminate_process_identity

_SLEEP = "import time; time.sleep(60)"


class RunnerTransactionTests(unittest.TestCase):
    def manifest(self, root: Path):
        path = root / "model.toml"
        path.write_text(
            "\n".join(
                [
                    "[model]",
                    'id = "transaction"',
                    'model_id = "transaction-model"',
                    'endpoint = "http://127.0.0.1:9/v1"',
                    "",
                    "[start]",
                    f'command = ["{sys.executable}", "-c", "{_SLEEP}"]',
                    f'cwd = "{root}"',
                    f'pid_path = "{root / "model.pid.json"}"',
                    f'log_path = "{root / "model.log"}"',
                    "startup_timeout_sec = 1",
                ]
            ),
            encoding="utf-8",
        )
        return load_manifest(path)

    def locks(self, root: Path):
        return patch("modelctl.lifecycle.default_lock_root", return_value=root.resolve() / "locks")

    def test_pending_is_private_and_durable_before_popen(self) -> None:
        from modelctl import runner

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = self.manifest(root)
            state_path = root / "model.pid.json"

            def fail_after_asserting_pending(*_args, **_kwargs):
                state = runner.read_pid_state(manifest)
                assert state is not None
                self.assertEqual(state["kind"], "launch_pending")
                self.assertEqual(state_path.stat().st_mode & 0o777, 0o600)
                raise OSError("popen refused")

            with self.locks(root), patch.object(runner.subprocess, "Popen", side_effect=fail_after_asserting_pending):
                result = runner.start(manifest)
            self.assertFalse(result["started"], result)
            self.assertEqual(result["status"], "popen_failed")
            self.assertFalse(state_path.exists())

    def test_identity_capture_failure_reaps_leader_and_keeps_pending_blocker(self) -> None:
        from modelctl import runner

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = self.manifest(root)
            with self.locks(root), patch.object(runner, "capture_process_identity", return_value=None):
                result = runner.start(manifest)
            self.assertTrue(result["started"], result)
            self.assertFalse(result["ok"], result)
            self.assertEqual(result["status"], "identity_capture_failed")
            self.assertTrue(result["durable_pending"])
            state = runner.read_pid_state(manifest)
            assert state is not None
            self.assertEqual(state["kind"], "launch_pending")
            self.assertIsNone(capture_process_identity(result["pid"]))
            with self.locks(root):
                blocked = runner.start(manifest)
                stopped = runner.stop(manifest)
            self.assertEqual(blocked["status"], "pid_state_blocked")
            self.assertFalse(stopped["safe_to_start"])
            self.assertTrue((root / "model.pid.json").exists())

    def test_identity_capture_exception_reaps_leader_and_keeps_pending_blocker(self) -> None:
        from modelctl import runner

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = self.manifest(root)
            with self.locks(root), patch.object(runner, "capture_process_identity", side_effect=OSError("kernel inventory failed")):
                result = runner.start(manifest)
            self.assertEqual(result["status"], "identity_capture_failed")
            self.assertTrue(result["started"], result)
            self.assertTrue(result["durable_pending"])
            self.assertIn("kernel inventory failed", result["error"])
            state = runner.read_pid_state(manifest)
            assert state is not None
            self.assertEqual(state["kind"], "launch_pending")

    def test_uncertified_exact_cleanup_keeps_pending_blocker(self) -> None:
        from modelctl import runner

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = self.manifest(root)
            with (
                self.locks(root),
                patch.object(runner, "_replace_pending", side_effect=OSError("disk failed")),
                patch.object(runner, "terminate_process_identity", return_value=False),
            ):
                result = runner.start(manifest)
            self.assertEqual(result["status"], "activation_failed")
            self.assertFalse(result["cleanup"]["group_death_certified"], result)
            state = runner.read_pid_state(manifest)
            assert state is not None
            self.assertEqual(state["kind"], "launch_pending")
            identity = capture_process_identity(result["pid"])
            assert identity is not None
            self.assertTrue(terminate_process_identity(identity, timeout_sec=1))
            reap_retained_popens()

    def test_success_promotes_pending_to_v2_and_stop_authenticates_exact_group(self) -> None:
        from modelctl import runner

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = self.manifest(root)
            with self.locks(root):
                started = runner.start(manifest)
            self.assertTrue(started["ok"], started)
            state = runner.read_pid_state(manifest)
            assert state is not None
            self.assertEqual(state["schema_version"], 2)
            self.assertEqual(state["kind"], "active")
            self.assertEqual(state["pid"], started["pid"])
            self.assertEqual((root / "model.pid.json").stat().st_mode & 0o777, 0o600)
            with self.locks(root):
                stopped = runner.stop(manifest, timeout_sec=1)
            self.assertTrue(stopped["safe_to_start"], stopped)
            self.assertIsNone(capture_process_identity(started["pid"]))
            self.assertFalse((root / "model.pid.json").exists())

    def test_activation_failure_cleans_exact_group_and_pending_state(self) -> None:
        from modelctl import runner

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = self.manifest(root)
            with self.locks(root), patch.object(runner, "_replace_pending", side_effect=OSError("disk failed")):
                result = runner.start(manifest)
            self.assertTrue(result["started"], result)
            self.assertFalse(result["ok"], result)
            self.assertEqual(result["status"], "activation_failed")
            self.assertTrue(result["cleanup"]["group_death_certified"], result)
            self.assertIsNone(capture_process_identity(result["pid"]))
            self.assertFalse((root / "model.pid.json").exists())

    def test_retain_failure_cleans_promoted_state_and_exact_group(self) -> None:
        from modelctl import runner

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = self.manifest(root)
            with self.locks(root), patch.object(runner, "retain_popen", side_effect=RuntimeError("retain failed")):
                result = runner.start(manifest)
            self.assertEqual(result["status"], "activation_failed")
            self.assertTrue(result["cleanup"]["group_death_certified"], result)
            self.assertIsNone(capture_process_identity(result["pid"]))
            self.assertFalse((root / "model.pid.json").exists())

    def test_readiness_exception_cleans_promoted_state_and_process(self) -> None:
        from modelctl import runner

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = self.manifest(root)
            with self.locks(root), patch.object(runner, "wait_ready", side_effect=RuntimeError("readiness exploded")):
                result = runner.start(manifest, wait=True)
            self.assertEqual(result["status"], "readiness_exception")
            self.assertTrue(result["started"], result)
            self.assertTrue(result["cleanup"]["group_death_certified"], result)
            self.assertIsNone(capture_process_identity(result["pid"]))
            self.assertFalse((root / "model.pid.json").exists())

    def test_tampered_active_endpoint_blocks_stop_without_signaling(self) -> None:
        from modelctl import runner

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = self.manifest(root)
            with self.locks(root):
                started = runner.start(manifest)
            state = runner.read_pid_state(manifest)
            assert state is not None
            original = dict(state)
            state["endpoint"] = {"host": "local", "port": 65535}
            runner.write_pid_state(manifest, state)
            with self.locks(root), patch.object(runner, "terminate_process_identity") as terminate:
                blocked = runner.stop(manifest)
            self.assertFalse(blocked["safe_to_start"], blocked)
            terminate.assert_not_called()
            self.assertIsNotNone(capture_process_identity(started["pid"]))
            runner.write_pid_state(manifest, original)
            with self.locks(root):
                self.assertTrue(runner.stop(manifest, timeout_sec=1)["safe_to_start"])

    def test_stop_unlink_failure_is_structured_and_preserves_blocker(self) -> None:
        from modelctl import runner

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = self.manifest(root)
            with self.locks(root):
                started = runner.start(manifest)
            with self.locks(root), patch.object(runner, "_remove_if", side_effect=OSError("unlink refused")):
                stopped = runner.stop(manifest, timeout_sec=1)
            self.assertTrue(stopped["stopped"], stopped)
            self.assertFalse(stopped["safe_to_start"], stopped)
            self.assertIn("unlink refused", stopped["state_error"])
            self.assertTrue((root / "model.pid.json").exists())
            self.assertIsNone(capture_process_identity(started["pid"]))
            with self.locks(root):
                self.assertTrue(runner.stop(manifest)["safe_to_start"])

    def test_readiness_failure_cleans_promoted_state_and_process(self) -> None:
        from modelctl import runner

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = self.manifest(root)
            with self.locks(root), patch.object(runner, "wait_ready", return_value={"ready": False, "error": "timeout"}):
                result = runner.start(manifest, wait=True)
            self.assertEqual(result["status"], "readiness_failed")
            certified = result["cleanup"]["group_death_certified"]
            if certified:
                self.assertFalse(result["durable_blocker"], result)
                self.assertFalse((root / "model.pid.json").exists())
            else:
                self.assertTrue(result["durable_blocker"], result)
                self.assertTrue((root / "model.pid.json").exists())
                identity = capture_process_identity(result["pid"])
                if identity is not None and not terminate_process_identity(identity, timeout_sec=5):
                    try:
                        os.killpg(identity.pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                reap_retained_popens()
                (root / "model.pid.json").unlink(missing_ok=True)
            deadline = time.monotonic() + 5
            while capture_process_identity(result["pid"]) is not None and time.monotonic() < deadline:
                reap_retained_popens()
                time.sleep(0.01)
            self.assertIsNone(capture_process_identity(result["pid"]))


if __name__ == "__main__":
    unittest.main()
