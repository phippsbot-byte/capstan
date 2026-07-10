from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from modelctl.cli import main


MANIFEST = """
[model]
id = "test-lane"
model_id = "test-model"
endpoint = "http://127.0.0.1:18081/v1"

[start]
command = ["python3", "-c", "pass"]
pid_path = "{pid_path}"
log_path = "{log_path}"
startup_timeout_sec = 1
"""


class CliExitCodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.manifest = root / "manifest.toml"
        self.candidate = root / "candidate.toml"
        content = MANIFEST.format(pid_path=root / "lane.pid", log_path=root / "lane.log")
        self.manifest.write_text(content, encoding="utf-8")
        self.candidate.write_text(content, encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, *args: str) -> int:
        with redirect_stdout(StringIO()):
            return main(["--manifest", str(self.manifest), *args], prog="capstan")

    def test_start_wait_returns_nonzero_when_readiness_fails(self):
        result = {"started": True, "readiness": {"ready": False, "error": "timeout"}}
        with patch("modelctl.runner.start", return_value=result):
            self.assertEqual(self.run_cli("start", "--wait"), 2)

    def test_failed_stop_returns_nonzero(self):
        with patch("modelctl.runner.stop", return_value={"stopped": False, "already_stopped": False}):
            self.assertEqual(self.run_cli("stop"), 2)

    def test_stop_requires_safe_or_exact_known_pid_evidence(self):
        failed = [
            {"ok": True},
            {"stopped": True},
            {"ok": True, "stopped": True},
            {"already_stopped": True},
            {"known_pid_stopped": True, "unexpected_active_pid": 4321},
            {"safe_to_start": True, "unexpected_active_pid": 4321},
            {"known_pid_stopped": False, "safe_to_start": True},
            {"stopped": False, "known_pid_stopped": True, "unexpected_active_pid": None},
        ]
        for result in failed:
            with self.subTest(result=result), patch("modelctl.runner.stop", return_value=result):
                self.assertEqual(self.run_cli("stop"), 2)

        succeeded = [
            {"safe_to_start": True},
            {"known_pid_stopped": True, "unexpected_active_pid": None},
        ]
        for result in succeeded:
            with self.subTest(result=result), patch("modelctl.runner.stop", return_value=result):
                self.assertEqual(self.run_cli("stop"), 0)

    def test_failed_action_payloads_return_nonzero(self):
        cases = [
            (["fleet", "recover"], "modelctl.fleet.fleet_recover"),
            (["rotate", "--to", str(self.candidate)], "modelctl.runner.rotate"),
            (["promote", "--candidate", str(self.candidate)], "modelctl.promote.promote"),
            (["cleanup", "--execute"], "modelctl.ops.cleanup_execute"),
        ]
        for args, target in cases:
            with self.subTest(command=args[0]), patch(target, return_value={"ok": False, "error": "blocked"}):
                self.assertEqual(self.run_cli(*args), 2)

    def test_successful_cleanup_execute_has_explicit_success_evidence(self):
        self.assertEqual(self.run_cli("cleanup", "--execute"), 0)

    def test_malformed_or_contradictory_action_payloads_fail_closed(self):
        failed = [
            {},
            {"ok": 0},
            {"ok": None},
            {"ok": ""},
            {"ok": []},
            {"started": False, "already_running": False, "error": "failed"},
            {"started": True, "already_running": True},
            {"started": 1},
            {"already_running": "yes"},
            {"readiness": {"ready": True}},
            {"ok": True, "readiness": {"ready": False}},
        ]
        for result in failed:
            with self.subTest(result=result), patch("modelctl.runner.start", return_value=result):
                self.assertEqual(self.run_cli("start"), 2)

        for result in ({"started": True}, {"already_running": True}):
            with self.subTest(result=result), patch("modelctl.runner.start", return_value=result):
                self.assertEqual(self.run_cli("start"), 0)

    def test_snapshot_commands_do_not_gate_on_reported_health(self):
        with patch("modelctl.ops.status", return_value={"ok": False, "readiness": {"ready": False}}):
            self.assertEqual(self.run_cli("status"), 0)
        with patch("modelctl.registry.list_registry", return_value={"ok": False, "entries": []}):
            self.assertEqual(self.run_cli("list"), 0)
        unhealthy = {"ok": False, "entry": {"ok": False, "name": "broken", "error": "invalid manifest"}}
        with patch("modelctl.registry.show_registry", return_value=unhealthy):
            self.assertEqual(self.run_cli("registry", "show", "broken"), 0)
        with patch("modelctl.registry.show_registry", return_value={"ok": False, "error": "not found"}):
            self.assertEqual(self.run_cli("registry", "show", "missing"), 2)


if __name__ == "__main__":
    unittest.main()
