from __future__ import annotations

from pathlib import Path
import json
import os
import plistlib
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from modelctl.manifest import load_manifest
from modelctl.ops import cleanup_execute, cleanup_plan, doctor_fix, preflight
from modelctl.system import pid_alive


def write_registry_manifest(registry: Path, name: str, endpoint: str) -> Path:
    path = registry / f"{name}.toml"
    path.write_text(textwrap.dedent(f'''
        [model]
        id = "{name}"
        model_id = "{name}-model"
        endpoint = "{endpoint}"
    '''), encoding="utf-8")
    return path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ModelCtlTests(unittest.TestCase):
    def test_version_command_avoids_runtime_module_imports(self):
        probe = textwrap.dedent('''
            import contextlib, io, json, sys
            from modelctl import cli
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = cli.main(["version"])
            names = ["modelctl.manifest", "modelctl.ops", "modelctl.fleet", "modelctl.runner", "urllib.request"]
            print(json.dumps({"rc": rc, "out": json.loads(out.getvalue()), "modules": {name: name in sys.modules for name in names}}))
        ''')
        result = subprocess.run([sys.executable, "-c", probe], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        body = json.loads(result.stdout)
        self.assertEqual(body["rc"], 0, body)
        self.assertRegex(body["out"]["version"], r"^\d+\.\d+\.\d+$")
        self.assertEqual(body["modules"], {"modelctl.manifest": False, "modelctl.ops": False, "modelctl.fleet": False, "modelctl.runner": False, "urllib.request": False})

    def test_fleet_status_checks_entries_concurrently_and_preserves_float_timeout(self):
        from modelctl import fleet as fleet_mod

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = root / "registry"
            registry.mkdir()
            for idx in range(4):
                write_registry_manifest(registry, f"m{idx}", f"http://127.0.0.1:{9000 + idx}/v1")

            calls: list[tuple[str, float]] = []
            original_readiness = fleet_mod.readiness_check
            original_active_pid = fleet_mod.active_pid
            original_pid_state = fleet_mod.read_pid_state
            original_service = fleet_mod._service_snapshot
            original_swap = fleet_mod.swap_used_gib
            original_xdg = os.environ.get("XDG_CONFIG_HOME")
            original_registry_env = os.environ.get("MODELCTL_REGISTRY")

            def slow_readiness(manifest, timeout=10):
                time.sleep(0.2)
                calls.append((manifest.id, timeout))
                return {"ready": False, "status": 599, "url": manifest.models_url}

            try:
                fleet_mod.readiness_check = slow_readiness
                fleet_mod.active_pid = lambda _manifest: None
                fleet_mod.read_pid_state = lambda _manifest: None
                fleet_mod._service_snapshot = lambda manifest: {"label": f"ai.modelctl.{manifest.id}", "managed": False}
                fleet_mod.swap_used_gib = lambda: 0.0
                os.environ["XDG_CONFIG_HOME"] = str(root / "xdg-config")
                os.environ.pop("MODELCTL_REGISTRY", None)
                started = time.perf_counter()
                result = fleet_mod.fleet_status(registries=[str(registry)], readiness_timeout=0.25)
                elapsed = time.perf_counter() - started
            finally:
                fleet_mod.readiness_check = original_readiness
                fleet_mod.active_pid = original_active_pid
                fleet_mod.read_pid_state = original_pid_state
                fleet_mod._service_snapshot = original_service
                fleet_mod.swap_used_gib = original_swap
                if original_xdg is None:
                    os.environ.pop("XDG_CONFIG_HOME", None)
                else:
                    os.environ["XDG_CONFIG_HOME"] = original_xdg
                if original_registry_env is None:
                    os.environ.pop("MODELCTL_REGISTRY", None)
                else:
                    os.environ["MODELCTL_REGISTRY"] = original_registry_env

            self.assertTrue(result["ok"], result)
            self.assertEqual([row["id"] for row in result["models"]], ["m0", "m1", "m2", "m3"])
            self.assertLess(elapsed, 0.45, f"fleet status should not scan readiness serially; elapsed={elapsed:.3f}s")
            self.assertEqual(sorted(calls), [("m0", 0.25), ("m1", 0.25), ("m2", 0.25), ("m3", 0.25)])

    def test_fleet_health_checks_entries_concurrently_with_jobs_and_timing(self):
        from modelctl import fleet as fleet_mod

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = root / "registry"
            registry.mkdir()
            for idx in range(4):
                write_registry_manifest(registry, f"h{idx}", f"http://127.0.0.1:{9100 + idx}/v1")

            calls: list[str] = []
            active = 0
            max_active = 0
            lock = threading.Lock()
            original_health = fleet_mod.health
            original_xdg = os.environ.get("XDG_CONFIG_HOME")
            original_registry_env = os.environ.get("MODELCTL_REGISTRY")

            def slow_health(manifest, **_kwargs):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.12)
                with lock:
                    active -= 1
                    calls.append(manifest.id)
                return {"ok": True, "status": "ok", "issues": [], "warnings": []}

            try:
                fleet_mod.health = slow_health
                os.environ["XDG_CONFIG_HOME"] = str(root / "xdg-config")
                os.environ.pop("MODELCTL_REGISTRY", None)
                started = time.perf_counter()
                result = fleet_mod.fleet_health(registries=[str(registry)], jobs=2)
                elapsed = time.perf_counter() - started
            finally:
                fleet_mod.health = original_health
                if original_xdg is None:
                    os.environ.pop("XDG_CONFIG_HOME", None)
                else:
                    os.environ["XDG_CONFIG_HOME"] = original_xdg
                if original_registry_env is None:
                    os.environ.pop("MODELCTL_REGISTRY", None)
                else:
                    os.environ["MODELCTL_REGISTRY"] = original_registry_env

            self.assertTrue(result["ok"], result)
            self.assertEqual([row["id"] for row in result["models"]], ["h0", "h1", "h2", "h3"])
            self.assertEqual(result["jobs"], 2)
            self.assertIn("elapsed_sec", result)
            self.assertLess(elapsed, 0.7, f"fleet health should honor --jobs concurrency; elapsed={elapsed:.3f}s")
            self.assertEqual(max_active, 2)
            self.assertEqual(sorted(calls), ["h0", "h1", "h2", "h3"])
            self.assertTrue(all(row["elapsed_sec"] >= 0 for row in result["models"]))

    def write_manifest(self, root: Path, content: str) -> Path:
        path = root / "modelctl.toml"
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        return path

    def test_pid_alive_treats_zombie_processes_as_dead(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pid_file = root / "child.pid"
            script = root / "zombie_parent.py"
            script.write_text(textwrap.dedent('''
                import subprocess, sys, time
                child = subprocess.Popen([sys.executable, "-c", "pass"])
                open(sys.argv[1], "w", encoding="utf-8").write(str(child.pid))
                time.sleep(60)
            '''), encoding="utf-8")
            parent = subprocess.Popen([sys.executable, str(script), str(pid_file)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                deadline = time.time() + 10
                child_pid = None
                while time.time() < deadline:
                    if pid_file.exists():
                        child_pid = int(pid_file.read_text(encoding="utf-8"))
                        stat = subprocess.run(["ps", "-p", str(child_pid), "-o", "stat="], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
                        if stat.returncode == 0 and "Z" in stat.stdout:
                            break
                    time.sleep(0.05)
                self.assertIsNotNone(child_pid)
                assert child_pid is not None
                self.assertFalse(pid_alive(child_pid), "zombie PIDs must not count as active model processes")
            finally:
                parent.terminate()
                try:
                    parent.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    parent.kill()
                    parent.wait(timeout=5)

    def test_manifest_expands_and_validates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            required = root / "required.txt"
            required.write_text("ok")
            manifest_path = self.write_manifest(root, f'''
                [model]
                id = "test"
                model_id = "test-model"
                endpoint = "http://127.0.0.1:9/v1"

                [preflight]
                required_paths = ["{required}"]
                exclusive_ports = []

                [[preflight.disk]]
                path = "{root}"
                min_free_gib = 0
            ''')
            manifest = load_manifest(manifest_path)
            self.assertEqual(manifest.id, "test")
            result = preflight(manifest)
            self.assertTrue(result["ok"], result)

    def test_manifest_health_table_drives_health_wait_daemon_and_service_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            class H(BaseHTTPRequestHandler):
                def log_message(self, format, *args):
                    pass
                def _send(self, body, status=200):
                    data = json.dumps(body).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                def do_GET(self):
                    if self.path == "/v1/models":
                        self._send({"object": "list", "data": [{"id": "nostart-model"}]})
                    else:
                        self._send({"error": "not found"}, 404)
                def do_POST(self):
                    if self.path == "/v1/chat/completions":
                        self._send({"choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}]})
                    else:
                        self._send({"error": "not found"}, 404)

            server = HTTPServer(("127.0.0.1", 0), H)
            port = int(server.server_address[1])
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(server.shutdown)
            self.addCleanup(server.server_close)
            self.addCleanup(lambda: thread.join(timeout=5))

            probe_file = root / "probe.bin"
            probe_file.write_bytes(b"probe" * 1024)
            manifest_path = self.write_manifest(root, f'''
                [model]
                id = "nostart"
                model_id = "nostart-model"
                endpoint = "http://127.0.0.1:{port}/v1"

                [preflight]
                required_paths = ["{probe_file}"]

                [health]
                max_swap_gib = 999999
                max_swap_delta_gib = 999999
                sample_sec = 0.001
                smoke = true
                max_latency_sec = 10
                max_io_latency_sec = 25

                [smoke]
                prompt = "Reply with exactly the word pong."
                expect = "pong"
                max_tokens = 8
            ''')
            manifest = load_manifest(manifest_path)
            self.assertEqual(manifest.health.max_swap_gib, 999999)
            self.assertEqual(manifest.health.max_io_latency_sec, 25)

            cmd = [sys.executable, "-m", "modelctl.cli", "-m", str(manifest_path)]
            wait = subprocess.run(cmd + ["wait", "--timeout", "1"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(wait.returncode, 0, wait.stderr + wait.stdout)
            wait_body = json.loads(wait.stdout)
            self.assertTrue(wait_body["ready"], wait_body)

            health = subprocess.run(cmd + ["health"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(health.returncode, 0, health.stderr + health.stdout)
            health_body = json.loads(health.stdout)
            self.assertTrue(health_body["ok"], health_body)
            self.assertEqual(health_body["checks"]["swap"]["max_swap_delta_gib"], 999999)
            self.assertEqual(health_body["checks"]["swap"]["sample_sec"], 0.001)
            self.assertEqual(health_body["checks"]["smoke"]["status"], "ok")
            self.assertEqual(health_body["checks"]["io"]["status"], "ok")
            self.assertEqual(health_body["checks"]["io"]["max_io_latency_sec"], 25)

            daemon = subprocess.run(cmd + ["daemon", "--iterations", "1", "--interval", "0"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(daemon.returncode, 0, daemon.stderr + daemon.stdout)
            daemon_body = json.loads(daemon.stdout)
            self.assertTrue(daemon_body["health_mode"], daemon_body)
            self.assertTrue(daemon_body["include_smoke"], daemon_body)
            self.assertEqual(daemon_body["max_swap_delta_gib"], 999999)

            env = os.environ.copy()
            env["MODELCTL_LAUNCHD_DIR"] = str(root / "LaunchAgents")
            svc = subprocess.run(cmd + ["service", "install", "--dry-run", "--interval", "7"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(svc.returncode, 0, svc.stderr + svc.stdout)
            svc_body = json.loads(svc.stdout)
            args = svc_body["program_arguments"]
            self.assertIn("--health-mode", args)
            self.assertIn("--max-swap-gib", args)
            self.assertIn("999999", args)
            self.assertIn("--max-swap-delta-gib", args)
            self.assertIn("--sample-sec", args)
            self.assertIn("0.001", args)
            self.assertIn("--smoke", args)
            self.assertIn("--max-latency-sec", args)
            self.assertIn("10", args)

    def test_smoke_custom_prompt_without_expect_is_not_forced_to_pong(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            default_path = self.write_manifest(root, '''
                [model]
                id = "default-smoke"
                model_id = "default-model"
                endpoint = "http://127.0.0.1:9/v1"
            ''')
            self.assertEqual(load_manifest(default_path).smoke.expect, "pong")

            manifest_path = self.write_manifest(root, '''
                [model]
                id = "loose-smoke"
                model_id = "loose-model"
                endpoint = "http://127.0.0.1:9/v1"

                [smoke]
                prompt = "Say hello."
                max_tokens = 8
            ''')
            manifest = load_manifest(manifest_path)
            self.assertIsNone(manifest.smoke.expect)

            strict_path = self.write_manifest(root, '''
                [model]
                id = "strict-smoke"
                model_id = "strict-model"
                endpoint = "http://127.0.0.1:9/v1"

                [smoke]
                prompt = "Reply with exactly the word pong."
                expect = "pong"
            ''')
            strict_manifest = load_manifest(strict_path)
            from modelctl import ops as ops_mod
            original_http_json = ops_mod.http_json
            try:
                ops_mod.http_json = lambda *_args, **_kwargs: (200, {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}, "")
                overridden = ops_mod.smoke(strict_manifest, prompt="Say hello.")
            finally:
                ops_mod.http_json = original_http_json
            self.assertTrue(overridden["ok"], overridden)
            self.assertIsNone(overridden["expect"])
            self.assertIsNone(overridden["exact"])

    def test_ingest_connection_refused_returns_json_failure(self):
        result = subprocess.run([sys.executable, "-m", "modelctl.cli", "ingest", "--endpoint", "http://127.0.0.1:9/v1"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        self.assertEqual(result.returncode, 2, result.stderr + result.stdout)
        body = json.loads(result.stdout)
        self.assertFalse(body["ok"], body)
        self.assertEqual(body["models_url"], "http://127.0.0.1:9/v1/models")
        self.assertIn("URLError", body["error"])

        malformed = subprocess.run([sys.executable, "-m", "modelctl.cli", "ingest", "--endpoint", "not-a-url"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        self.assertEqual(malformed.returncode, 2, malformed.stderr + malformed.stdout)
        malformed_body = json.loads(malformed.stdout)
        self.assertFalse(malformed_body["ok"], malformed_body)
        self.assertIn("ValueError", malformed_body["error"])

    def test_runner_start_closes_parent_log_file_handle(self):
        from modelctl import runner

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest_path = self.write_manifest(root, f'''
                [model]
                id = "fd-test"
                model_id = "fd-model"
                endpoint = "http://127.0.0.1:9/v1"

                [start]
                command = ["noop"]
                log_path = "{root / 'fd.log'}"
                pid_path = "{root / 'fd.pid.json'}"
            ''')
            manifest = load_manifest(manifest_path)
            captured = {}

            class FakeProc:
                pid = 123456

            original_popen = runner.subprocess.Popen
            try:
                def fake_popen(*_args, **kwargs):
                    captured["stdout"] = kwargs["stdout"]
                    return FakeProc()
                runner.subprocess.Popen = fake_popen
                result = runner.start(manifest)
            finally:
                runner.subprocess.Popen = original_popen

            self.assertTrue(result["started"], result)
            self.assertTrue(captured["stdout"].closed, "parent must close log fd after Popen duplicates it into the child")

    def test_rotate_stops_current_starts_target_and_hands_off_pid_state(self):
        if os.environ.get("CI"):
            self.skipTest("subprocess same-port rotation is flaky on GitHub macOS runners; unit rotate paths cover CI")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            server = root / "model_server.py"
            server.write_text(textwrap.dedent(r'''
                import json, sys
                from http.server import BaseHTTPRequestHandler, HTTPServer
                port = int(sys.argv[1])
                model = sys.argv[2]
                class H(BaseHTTPRequestHandler):
                    def log_message(self, *args):
                        pass
                    def _send(self, body, status=200):
                        data = json.dumps(body).encode()
                        self.send_response(status)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Content-Length', str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                    def do_GET(self):
                        if self.path == '/v1/models':
                            self._send({'object':'list','data':[{'id':model}]})
                        else:
                            self._send({'error':'not found'}, 404)
                    def do_POST(self):
                        if self.path == '/v1/chat/completions':
                            self._send({'choices':[{'message':{'content':'pong'},'finish_reason':'stop'}]})
                        else:
                            self._send({'error':'not found'}, 404)
                class ReuseHTTPServer(HTTPServer):
                    allow_reuse_address = True
                ReuseHTTPServer(('127.0.0.1', port), H).serve_forever()
            '''), encoding="utf-8")
            current_port = free_port()
            target_port = current_port
            current_pid = root / "current.pid.json"
            target_pid = root / "target.pid.json"
            current_manifest = root / "current.toml"
            target_manifest = root / "target.toml"
            current_manifest.write_text(textwrap.dedent(f'''
                [model]
                id = "current"
                model_id = "stable-model"
                endpoint = "http://127.0.0.1:{current_port}/v1"

                [start]
                command = ["{sys.executable}", "{server}", "{current_port}", "stable-model"]
                cwd = "{root}"
                log_path = "{root / 'current.log'}"
                pid_path = "{current_pid}"
                startup_timeout_sec = 30
                readiness_url = "http://127.0.0.1:{current_port}/v1/models"
                readiness_contains = "stable-model"
            '''), encoding="utf-8")
            target_manifest.write_text(textwrap.dedent(f'''
                [model]
                id = "target"
                model_id = "stable-model"
                endpoint = "http://127.0.0.1:{target_port}/v1"

                [start]
                command = ["{sys.executable}", "{server}", "{target_port}", "stable-model"]
                cwd = "{root}"
                log_path = "{root / 'target.log'}"
                pid_path = "{target_pid}"
                startup_timeout_sec = 30
                readiness_url = "http://127.0.0.1:{target_port}/v1/models"
                readiness_contains = "stable-model"
            '''), encoding="utf-8")

            cmd = [sys.executable, "-m", "modelctl.cli", "-m", str(current_manifest)]
            started = subprocess.run(cmd + ["start", "--wait"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=75)
            self.assertEqual(started.returncode, 0, started.stderr + started.stdout)
            old_pid = json.loads(current_pid.read_text(encoding="utf-8"))["pid"]
            self.assertTrue(pid_alive(old_pid))

            rotated = subprocess.run(cmd + ["rotate", "--to", str(target_manifest), "--readiness-timeout", "30"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
            self.assertEqual(rotated.returncode, 0, rotated.stderr + rotated.stdout)
            body = json.loads(rotated.stdout)
            self.assertTrue(body["ok"], body)
            self.assertEqual(body["from"]["id"], "current")
            self.assertEqual(body["to"]["id"], "target")
            self.assertTrue(body["readiness"]["ready"], body)
            self.assertEqual(body["handoff"]["current_pid_path"], str(current_pid))
            self.assertEqual(body["handoff"]["target_pid_path"], str(target_pid))
            self.assertTrue(body["handoff"]["target_pid_state_removed"], body)
            self.assertFalse(target_pid.exists(), "target pid state should be moved to the current manifest owner")
            handed_off = json.loads(current_pid.read_text(encoding="utf-8"))
            new_pid = handed_off["pid"]
            self.assertNotEqual(old_pid, new_pid)
            self.assertTrue(pid_alive(new_pid))
            self.assertFalse(pid_alive(old_pid), "old model process should be stopped before target handoff")
            self.assertEqual(handed_off["rotated_from"]["id"], "current")
            self.assertEqual(handed_off["rotated_to"]["id"], "target")
            self.assertEqual(handed_off["manifest"], str(current_manifest.resolve()))
            self.assertEqual(handed_off["source_manifest"], str(target_manifest.resolve()))
            status = subprocess.run(cmd + ["status"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(status.returncode, 0, status.stderr + status.stdout)
            status_body = json.loads(status.stdout)
            self.assertEqual(status_body["pid"], new_pid, status_body)
            self.assertTrue(status_body["readiness"]["ready"], status_body)

            stopped = subprocess.run(cmd + ["stop"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(stopped.returncode, 0, stopped.stderr + stopped.stdout)
            self.assertFalse(pid_alive(new_pid))

    def test_rotate_readiness_failure_rolls_back_current_owner(self):
        if os.environ.get("CI"):
            self.skipTest("subprocess same-port rotation is flaky on GitHub macOS runners; unit rotate paths cover CI")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            server = root / "model_server.py"
            server.write_text(textwrap.dedent(r'''
                import json, sys
                from http.server import BaseHTTPRequestHandler, HTTPServer
                port = int(sys.argv[1])
                model = sys.argv[2]
                class H(BaseHTTPRequestHandler):
                    def log_message(self, *args):
                        pass
                    def _send(self, body, status=200):
                        data = json.dumps(body).encode()
                        self.send_response(status)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Content-Length', str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                    def do_GET(self):
                        if self.path == '/v1/models':
                            self._send({'object':'list','data':[{'id':model}]})
                        else:
                            self._send({'error':'not found'}, 404)
                class ReuseHTTPServer(HTTPServer):
                    allow_reuse_address = True
                ReuseHTTPServer(('127.0.0.1', port), H).serve_forever()
            '''), encoding="utf-8")
            current_port = free_port()
            target_port = current_port
            current_pid = root / "current.pid.json"
            target_pid = root / "target.pid.json"
            current_manifest = root / "current.toml"
            target_manifest = root / "target.toml"
            current_manifest.write_text(textwrap.dedent(f'''
                [model]
                id = "current"
                model_id = "stable-model"
                endpoint = "http://127.0.0.1:{current_port}/v1"

                [start]
                command = ["{sys.executable}", "{server}", "{current_port}", "stable-model"]
                cwd = "{root}"
                log_path = "{root / 'current.log'}"
                pid_path = "{current_pid}"
                startup_timeout_sec = 30
                readiness_url = "http://127.0.0.1:{current_port}/v1/models"
                readiness_contains = "stable-model"
            '''), encoding="utf-8")
            target_manifest.write_text(textwrap.dedent(f'''
                [model]
                id = "target"
                model_id = "stable-model"
                endpoint = "http://127.0.0.1:{target_port}/v1"

                [start]
                command = ["{sys.executable}", "{server}", "{target_port}", "wrong-model"]
                cwd = "{root}"
                log_path = "{root / 'target.log'}"
                pid_path = "{target_pid}"
                startup_timeout_sec = 30
                readiness_url = "http://127.0.0.1:{target_port}/v1/models"
                readiness_contains = "stable-model"
            '''), encoding="utf-8")

            cmd = [sys.executable, "-m", "modelctl.cli", "-m", str(current_manifest)]
            started = subprocess.run(cmd + ["start", "--wait"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=75)
            self.assertEqual(started.returncode, 0, started.stderr + started.stdout)
            original_pid = json.loads(current_pid.read_text(encoding="utf-8"))["pid"]

            rotated = subprocess.run(cmd + ["rotate", "--to", str(target_manifest), "--readiness-timeout", "1"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(rotated.returncode, 2, rotated.stderr + rotated.stdout)
            body = json.loads(rotated.stdout)
            self.assertFalse(body["ok"], body)
            self.assertEqual(body["status"], "target_not_ready")
            self.assertTrue(body["rollback"]["attempted"], body)
            self.assertTrue(body["rollback"]["readiness"]["ready"], body)
            self.assertFalse(target_pid.exists(), "failed target pid state should be cleaned up")
            rolled_back_pid = json.loads(current_pid.read_text(encoding="utf-8"))["pid"]
            self.assertNotEqual(original_pid, rolled_back_pid)
            self.assertTrue(pid_alive(rolled_back_pid))

            stopped = subprocess.run(cmd + ["stop"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(stopped.returncode, 0, stopped.stderr + stopped.stdout)

    def test_rotate_happy_path_hands_off_pid_state_with_mocked_lifecycle(self):
        from modelctl import runner
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            current_path = root / "current.toml"
            target_path = root / "target.toml"
            for path, ident in ((current_path, "current"), (target_path, "target")):
                path.write_text(textwrap.dedent(f'''
                    [model]
                    id = "{ident}"
                    model_id = "stable-model"
                    endpoint = "http://127.0.0.1:9000/v1"

                    [start]
                    command = ["noop"]
                    pid_path = "{root / (ident + '.pid.json')}"
                '''), encoding="utf-8")
            current = load_manifest(current_path)
            target = load_manifest(target_path)
            original_active_pid = runner.active_pid
            original_stop = runner.stop
            original_start = runner.start
            original_wait_ready = runner.wait_ready
            try:
                runner.active_pid = lambda manifest: 111 if manifest.id == "current" else 222
                runner.stop = lambda manifest, timeout_sec=10: {"stopped": True, "pid": 111 if manifest.id == "current" else 222}
                def fake_start(manifest, wait=False):
                    runner.write_pid_state(manifest, {"pid": 222, "manifest": str(manifest.path)})
                    return {"started": True, "pid": 222, "pid_path": str(runner.default_pid_path(manifest))}
                runner.start = fake_start
                runner.wait_ready = lambda _manifest, timeout_sec=None: {"ready": True}
                result = runner.rotate(current, target)
            finally:
                runner.active_pid = original_active_pid
                runner.stop = original_stop
                runner.start = original_start
                runner.wait_ready = original_wait_ready
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["status"], "rotated")
            self.assertFalse((root / "target.pid.json").exists())
            state = json.loads((root / "current.pid.json").read_text(encoding="utf-8"))
            self.assertEqual(state["pid"], 222)
            self.assertEqual(state["manifest"], str(current_path.resolve()))
            self.assertEqual(state["source_manifest"], str(target_path.resolve()))

    def test_promote_dry_run_plans_preflight_and_rotate_without_side_effects(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            current_path = root / "current.toml"
            target_path = root / "target.toml"
            for path, ident in ((current_path, "current"), (target_path, "target")):
                path.write_text(textwrap.dedent(f'''
                    [model]
                    id = "{ident}"
                    model_id = "stable-model"
                    endpoint = "http://127.0.0.1:9000/v1"

                    [start]
                    command = ["noop"]
                    pid_path = "{root / (ident + '.pid.json')}"
                '''), encoding="utf-8")
            cmd = [sys.executable, "-m", "modelctl.cli", "-m", str(current_path), "promote", "--candidate", str(target_path)]
            planned = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(planned.returncode, 0, planned.stderr + planned.stdout)
            body = json.loads(planned.stdout)
            self.assertTrue(body["ok"], body)
            self.assertEqual(body["status"], "planned")
            self.assertFalse(body["execute"])
            self.assertEqual(body["rotate_plan"]["status"], "planned")
            self.assertFalse((root / "current.pid.json").exists())
            self.assertFalse((root / "target.pid.json").exists())
            bad_nan = subprocess.run(cmd + ["--max-swap-gib", "nan"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(bad_nan.returncode, 2)
            self.assertIn("expected a non-negative finite number", bad_nan.stderr)

    def test_promote_tolerates_candidate_port_occupied_by_current_lane(self):
        from modelctl import promote as promote_mod
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            current_path = root / "current.toml"
            target_path = root / "target.toml"
            for path, ident in ((current_path, "current"), (target_path, "target")):
                path.write_text(textwrap.dedent(f'''
                    [model]
                    id = "{ident}"
                    model_id = "stable-model"
                    endpoint = "http://127.0.0.1:9123/v1"

                    [start]
                    command = ["noop"]
                    pid_path = "{root / (ident + '.pid.json')}"
                '''), encoding="utf-8")
            current = load_manifest(current_path)
            target = load_manifest(target_path)
            original_preflight = promote_mod.preflight
            original_rotate = promote_mod.rotate
            try:
                def fake_preflight(manifest):
                    if manifest.id == "current":
                        return {"ok": True, "checks": [{"type": "port", "port": 9123, "ok": True, "free": False, "active_pid": 123}]}
                    return {"ok": False, "checks": [{"type": "port", "port": 9123, "ok": False, "free": False, "active_pid": None}]}
                promote_mod.preflight = fake_preflight
                promote_mod.rotate = lambda *args, **kwargs: {"ok": True, "status": "planned"}
                result = promote_mod.promote(current, target)
            finally:
                promote_mod.preflight = original_preflight
                promote_mod.rotate = original_rotate
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["status"], "planned")
            self.assertEqual(result["tolerated_candidate_preflight"], ["shared_endpoint_port:9123"])

    def test_promote_execute_uses_candidate_health_and_smoke_defaults_after_handoff(self):
        from modelctl import promote as promote_mod
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            current_path = root / "current.toml"
            target_path = root / "target.toml"
            current_path.write_text(textwrap.dedent(f'''
                [model]
                id = "current"
                model_id = "stable-model"
                endpoint = "http://127.0.0.1:9000/v1"

                [start]
                command = ["noop"]
                pid_path = "{root / 'current.pid.json'}"

                [smoke]
                prompt = "current"
                expect = "current-ok"
            '''), encoding="utf-8")
            target_path.write_text(textwrap.dedent(f'''
                [model]
                id = "target"
                model_id = "stable-model"
                endpoint = "http://127.0.0.1:9000/v1"

                [start]
                command = ["noop"]
                pid_path = "{root / 'target.pid.json'}"

                [health]
                smoke = true

                [smoke]
                prompt = "candidate"
                expect = "candidate-ok"
            '''), encoding="utf-8")
            current = load_manifest(current_path)
            target = load_manifest(target_path)
            original_preflight = promote_mod.preflight
            original_rotate = promote_mod.rotate
            original_health = promote_mod.health
            try:
                promote_mod.preflight = lambda manifest: {"ok": True, "id": manifest.id}
                promote_mod.rotate = lambda *args, **kwargs: {"ok": True, "status": "rotated", "new_pid": 222}
                def fake_health(manifest, **_kwargs):
                    return {"ok": manifest.smoke.expect == "candidate-ok" and manifest.start == current.start, "status": "ok" if manifest.smoke.expect == "candidate-ok" else "critical", "expect": manifest.smoke.expect}
                promote_mod.health = fake_health
                result = promote_mod.promote(current, target, execute=True)
            finally:
                promote_mod.preflight = original_preflight
                promote_mod.rotate = original_rotate
                promote_mod.health = original_health
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["status"], "promoted")
            self.assertEqual(result["post_health"]["expect"], "candidate-ok")

    def test_promote_execute_rolls_back_on_post_health_failure(self):
        from modelctl import promote as promote_mod
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            current_path = root / "current.toml"
            target_path = root / "target.toml"
            for path, ident in ((current_path, "current"), (target_path, "target")):
                path.write_text(textwrap.dedent(f'''
                    [model]
                    id = "{ident}"
                    model_id = "stable-model"
                    endpoint = "http://127.0.0.1:9000/v1"

                    [start]
                    command = ["noop"]
                    pid_path = "{root / (ident + '.pid.json')}"
                    startup_timeout_sec = 3
                '''), encoding="utf-8")
            current = load_manifest(current_path)
            target = load_manifest(target_path)
            original_preflight = promote_mod.preflight
            original_rotate = promote_mod.rotate
            original_health = promote_mod.health
            original_stop = promote_mod.stop
            original_start = promote_mod.start
            original_wait_ready = promote_mod.wait_ready
            calls: list[str] = []
            try:
                promote_mod.preflight = lambda manifest: {"ok": True, "id": manifest.id}
                promote_mod.rotate = lambda *args, **kwargs: {"ok": True, "status": "rotated", "new_pid": 222}
                promote_mod.health = lambda *args, **kwargs: {"ok": False, "status": "critical", "issues": ["smoke"], "warnings": []}
                promote_mod.stop = lambda manifest, timeout_sec=10: calls.append(f"stop:{manifest.id}") or {"stopped": True, "pid": 222}
                promote_mod.start = lambda manifest, wait=False: calls.append(f"start:{manifest.id}") or {"started": True, "pid": 333}
                promote_mod.wait_ready = lambda manifest, timeout_sec=None: calls.append(f"wait:{manifest.id}") or {"ready": True}
                result = promote_mod.promote(current, target, execute=True, include_smoke=True)
            finally:
                promote_mod.preflight = original_preflight
                promote_mod.rotate = original_rotate
                promote_mod.health = original_health
                promote_mod.stop = original_stop
                promote_mod.start = original_start
                promote_mod.wait_ready = original_wait_ready
            self.assertFalse(result["ok"], result)
            self.assertEqual(result["status"], "post_health_failed")
            self.assertEqual(result["rotation"]["status"], "rotated")
            self.assertTrue(result["rollback"]["attempted"], result)
            self.assertEqual(calls, ["stop:current", "start:current", "wait:current"])

    def test_rotate_rejects_identity_mismatch_and_bad_timeouts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            current_manifest = root / "current.toml"
            target_manifest = root / "target.toml"
            current_manifest.write_text(textwrap.dedent(f'''
                [model]
                id = "current"
                model_id = "stable-model"
                endpoint = "http://127.0.0.1:9000/v1"

                [start]
                command = ["noop"]
                pid_path = "{root / 'current.pid.json'}"
            '''), encoding="utf-8")
            target_manifest.write_text(textwrap.dedent(f'''
                [model]
                id = "target"
                model_id = "different-model"
                endpoint = "http://127.0.0.1:9001/v1"

                [start]
                command = ["noop"]
                pid_path = "{root / 'target.pid.json'}"
            '''), encoding="utf-8")
            cmd = [sys.executable, "-m", "modelctl.cli", "-m", str(current_manifest), "rotate", "--to", str(target_manifest), "--dry-run"]
            mismatch = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(mismatch.returncode, 2, mismatch.stderr + mismatch.stdout)
            mismatch_body = json.loads(mismatch.stdout)
            self.assertEqual(mismatch_body["status"], "invalid_request")
            self.assertIn("target_identity_mismatch", mismatch_body["issues"])

            bad_timeout = subprocess.run(cmd + ["--readiness-timeout", "-1"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(bad_timeout.returncode, 2)
            self.assertIn("expected a positive finite number", bad_timeout.stderr)
            bad_nan = subprocess.run(cmd + ["--readiness-timeout", "nan"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(bad_nan.returncode, 2)
            self.assertIn("expected a positive finite number", bad_nan.stderr)
            bad_inf = subprocess.run(cmd + ["--readiness-timeout", "inf"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(bad_inf.returncode, 2)
            self.assertIn("expected a positive finite number", bad_inf.stderr)
            bad_stop = subprocess.run(cmd + ["--stop-timeout", "0"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(bad_stop.returncode, 2)
            self.assertIn("expected a positive integer", bad_stop.stderr)

    def test_active_pid_ignores_pid_state_owned_by_other_manifest(self):
        from modelctl import runner
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            current_path = self.write_manifest(root, f'''
                [model]
                id = "current"
                model_id = "stable-model"
                endpoint = "http://127.0.0.1:9/v1"

                [start]
                command = ["noop"]
                pid_path = "{root / 'owned.pid.json'}"
            ''')
            other_path = root / "other.toml"
            other_path.write_text(textwrap.dedent('''
                [model]
                id = "other"
                model_id = "stable-model"
                endpoint = "http://127.0.0.1:9/v1"
            '''), encoding="utf-8")
            current = load_manifest(current_path)
            runner.write_pid_state(current, {"pid": os.getpid(), "manifest": str(other_path)})
            self.assertIsNone(runner.active_pid(current))
            stop_result = runner.stop(current)
            self.assertTrue(stop_result["owner_mismatch"], stop_result)
            self.assertTrue((root / "owned.pid.json").exists())
            with self.assertRaises(RuntimeError):
                runner.start(current)
            fixed = doctor_fix(current)
            self.assertTrue((root / "owned.pid.json").exists())
            self.assertIn("pid_state_owner_mismatch_preserved", [row["code"] for row in fixed["fixes"]])

    def test_rotate_rolls_back_if_target_start_raises(self):
        from modelctl import runner
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            current_path = root / "current.toml"
            target_path = root / "target.toml"
            for path, ident in ((current_path, "current"), (target_path, "target")):
                path.write_text(textwrap.dedent(f'''
                    [model]
                    id = "{ident}"
                    model_id = "stable-model"
                    endpoint = "http://127.0.0.1:9000/v1"

                    [start]
                    command = ["noop"]
                    pid_path = "{root / (ident + '.pid.json')}"
                '''), encoding="utf-8")
            current = load_manifest(current_path)
            target = load_manifest(target_path)
            original_active_pid = runner.active_pid
            original_stop = runner.stop
            original_start = runner.start
            original_wait_ready = runner.wait_ready
            try:
                runner.active_pid = lambda _manifest: 123
                runner.stop = lambda _manifest, timeout_sec=10: {"stopped": True, "pid": 123}
                def fake_start(manifest, wait=False):
                    if manifest.id == "target":
                        raise RuntimeError("boom")
                    return {"started": True, "pid": 456}
                runner.start = fake_start
                runner.wait_ready = lambda _manifest, timeout_sec=None: {"ready": True}
                result = runner.rotate(current, target)
            finally:
                runner.active_pid = original_active_pid
                runner.stop = original_stop
                runner.start = original_start
                runner.wait_ready = original_wait_ready
            self.assertFalse(result["ok"], result)
            self.assertEqual(result["status"], "target_start_failed")
            self.assertTrue(result["rollback"]["attempted"], result)
            self.assertTrue(result["rollback"]["readiness"]["ready"], result)

    def test_rotate_rolls_back_if_handoff_replace_fails(self):
        from modelctl import runner
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            current_path = root / "current.toml"
            target_path = root / "target.toml"
            for path, ident in ((current_path, "current"), (target_path, "target")):
                path.write_text(textwrap.dedent(f'''
                    [model]
                    id = "{ident}"
                    model_id = "stable-model"
                    endpoint = "http://127.0.0.1:9000/v1"

                    [start]
                    command = ["noop"]
                    pid_path = "{root / (ident + '.pid.json')}"
                '''), encoding="utf-8")
            current = load_manifest(current_path)
            target = load_manifest(target_path)
            original_active_pid = runner.active_pid
            original_stop = runner.stop
            original_start = runner.start
            original_wait_ready = runner.wait_ready
            original_replace = runner.os.replace
            stopped: list[str] = []
            try:
                runner.active_pid = lambda manifest: 123 if manifest.id == "current" else 456
                def fake_stop(manifest, timeout_sec=10):
                    stopped.append(manifest.id)
                    return {"stopped": True, "pid": 123 if manifest.id == "current" else 456}
                runner.stop = fake_stop
                def fake_start(manifest, wait=False):
                    if manifest.id == "target":
                        runner.write_pid_state(target, {"pid": 456, "manifest": str(target.path)})
                        runner.os.replace = lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("replace boom"))
                        return {"started": True, "pid": 456}
                    return {"started": True, "pid": 789}
                runner.start = fake_start
                runner.wait_ready = lambda _manifest, timeout_sec=None: {"ready": True}
                result = runner.rotate(current, target)
            finally:
                runner.active_pid = original_active_pid
                runner.stop = original_stop
                runner.start = original_start
                runner.wait_ready = original_wait_ready
                runner.os.replace = original_replace
            self.assertFalse(result["ok"], result)
            self.assertEqual(result["status"], "handoff_failed")
            self.assertIn("target", stopped)
            self.assertTrue(result["rollback"]["attempted"], result)

    def test_cleanup_dry_run_and_safe_execute(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            safe = root / "safe-cache"
            unsafe = root / "unsafe-model"
            safe.mkdir()
            unsafe.mkdir()
            (safe / "x").write_text("x")
            (unsafe / "x").write_text("x")
            manifest_path = self.write_manifest(root, f'''
                [model]
                id = "cleanup"
                model_id = "cleanup-model"
                endpoint = "http://127.0.0.1:9/v1"

                [[cleanup]]
                path = "{safe}"
                description = "safe"
                safe = true

                [[cleanup]]
                path = "{unsafe}"
                description = "unsafe"
                safe = false
            ''')
            manifest = load_manifest(manifest_path)
            plan = cleanup_plan(manifest)
            self.assertEqual(len(plan["candidates"]), 2)
            result = cleanup_execute(manifest, force=False)
            self.assertFalse(safe.exists())
            self.assertTrue(unsafe.exists())
            self.assertEqual(len(result["deleted"]), 1)
            self.assertEqual(len(result["skipped"]), 1)

    def test_cli_start_wait_smoke_stop_against_fake_openai_server(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            port = free_port()
            server = root / "fake_openai_server.py"
            server.write_text(textwrap.dedent(r'''
                import json, sys
                from http.server import BaseHTTPRequestHandler, HTTPServer
                port = int(sys.argv[1])
                class H(BaseHTTPRequestHandler):
                    def log_message(self, *args):
                        pass
                    def _send(self, body, status=200):
                        data=json.dumps(body).encode()
                        self.send_response(status)
                        self.send_header('Content-Type','application/json')
                        self.send_header('Content-Length',str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                    def do_GET(self):
                        if self.path == '/v1/models':
                            self._send({'object':'list','data':[{'id':'fake-model'}]})
                        else:
                            self._send({'error':'not found'}, 404)
                    def do_POST(self):
                        n=int(self.headers.get('Content-Length','0'))
                        body=self.rfile.read(n).decode()
                        if self.path == '/v1/chat/completions':
                            content='BENCH_OK' if 'BENCH_OK' in body else 'pong'
                            self._send({'choices':[{'message':{'content':content},'finish_reason':'stop'}], 'usage': {'completion_tokens': 1}})
                        else:
                            self._send({'error':'not found'}, 404)
                class ReuseHTTPServer(HTTPServer):
                    allow_reuse_address = True
                ReuseHTTPServer(('127.0.0.1', port), H).serve_forever()
            '''), encoding="utf-8")
            manifest_path = self.write_manifest(root, f'''
                [model]
                id = "fake"
                model_id = "fake-model"
                endpoint = "http://127.0.0.1:{port}/v1"

                [start]
                command = ["{sys.executable}", "{server}", "{port}"]
                cwd = "{root}"
                log_path = "{root / 'fake.log'}"
                pid_path = "{root / 'fake.pid.json'}"
                startup_timeout_sec = 20
                readiness_url = "http://127.0.0.1:{port}/v1/models"
                readiness_contains = "fake-model"

                [preflight]
                exclusive_ports = [{port}]

                [smoke]
                prompt = "Reply with exactly the word pong."
                expect = "pong"
                max_tokens = 8
                temperature = 0
            ''')
            cmd = [sys.executable, "-m", "modelctl.cli", "-m", str(manifest_path)]
            start = subprocess.run(cmd + ["start", "--wait"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(start.returncode, 0, start.stderr + start.stdout)
            smoke = subprocess.run(cmd + ["smoke"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(smoke.returncode, 0, smoke.stderr + smoke.stdout)
            body = json.loads(smoke.stdout)
            self.assertTrue(body["exact"], body)
            soak = subprocess.run(cmd + ["soak", "--count", "2"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(soak.returncode, 0, soak.stderr + soak.stdout)
            soak_body = json.loads(soak.stdout)
            self.assertTrue(soak_body["ok"], soak_body)
            self.assertEqual(soak_body["completed_count"], 2)
            doctor = subprocess.run(cmd + ["doctor"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(doctor.returncode, 0, doctor.stderr + doctor.stdout)
            doctor_body = json.loads(doctor.stdout)
            self.assertTrue(doctor_body["ok"], doctor_body)
            bench = subprocess.run(cmd + ["bench", "--prompt-chars", "80,160", "--repeats", "1"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(bench.returncode, 0, bench.stderr + bench.stdout)
            bench_body = json.loads(bench.stdout)
            self.assertTrue(bench_body["ok"], bench_body)
            self.assertEqual(len(bench_body["runs"]), 2)
            bench_md = root / "bench.md"
            bench_out = subprocess.run(cmd + ["bench", "--preset", "tiny", "--output", str(bench_md), "--format", "md"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(bench_out.returncode, 0, bench_out.stderr + bench_out.stdout)
            bench_out_body = json.loads(bench_out.stdout)
            self.assertTrue(bench_out_body["ok"], bench_out_body)
            self.assertTrue(bench_md.exists())
            self.assertIn("modelctl bench", bench_md.read_text())
            watchdog = subprocess.run(cmd + ["watchdog", "--max-swap-gib", "999999", "--duration", "0"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(watchdog.returncode, 0, watchdog.stderr + watchdog.stdout)
            watchdog_body = json.loads(watchdog.stdout)
            self.assertTrue(watchdog_body["ok"], watchdog_body)
            health = subprocess.run(cmd + ["health", "--max-swap-gib", "999999", "--max-swap-delta-gib", "999999", "--sample-sec", "0", "--smoke", "--max-latency-sec", "10"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(health.returncode, 0, health.stderr + health.stdout)
            health_body = json.loads(health.stdout)
            self.assertTrue(health_body["ok"], health_body)
            self.assertEqual(health_body["status"], "ok")
            self.assertEqual(health_body["checks"]["readiness"]["status"], "ok")
            self.assertEqual(health_body["checks"]["smoke"]["status"], "ok")
            slow_health = subprocess.run(cmd + ["health", "--smoke", "--max-latency-sec", "0.000001"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(slow_health.returncode, 2, slow_health.stderr + slow_health.stdout)
            slow_body = json.loads(slow_health.stdout)
            self.assertEqual(slow_body["status"], "warn")
            self.assertIn("smoke_latency", slow_body["warnings"])
            daemon = subprocess.run(cmd + ["daemon", "--iterations", "1", "--interval", "0", "--max-swap-gib", "999999"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(daemon.returncode, 0, daemon.stderr + daemon.stdout)
            daemon_body = json.loads(daemon.stdout)
            self.assertTrue(daemon_body["ok"], daemon_body)
            self.assertEqual(len(daemon_body["iterations"]), 1)
            self.assertTrue(daemon_body["iterations"][0]["sample"]["ready"])
            health_daemon = subprocess.run(cmd + ["daemon", "--iterations", "1", "--interval", "0", "--max-swap-gib", "999999", "--max-swap-delta-gib", "999999", "--sample-sec", "0", "--smoke", "--max-latency-sec", "10"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(health_daemon.returncode, 0, health_daemon.stderr + health_daemon.stdout)
            health_daemon_body = json.loads(health_daemon.stdout)
            self.assertTrue(health_daemon_body["ok"], health_daemon_body)
            self.assertTrue(health_daemon_body["health_mode"])
            self.assertEqual(health_daemon_body["iterations"][0]["sample"]["status"], "ok")
            self.assertEqual(health_daemon_body["iterations"][0]["sample"]["checks"]["smoke"]["status"], "ok")
            report_path = root / "report.json"
            report = subprocess.run(cmd + ["report", "--output", str(report_path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(report.returncode, 0, report.stderr + report.stdout)
            self.assertTrue(report_path.exists())
            report_body = json.loads(report_path.read_text())
            self.assertTrue(report_body["ok"], report_body)
            env = os.environ.copy()
            env["XDG_STATE_HOME"] = str(root / "state")
            saved_report = subprocess.run(cmd + ["reports", "save", "--format", "json"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(saved_report.returncode, 0, saved_report.stderr + saved_report.stdout)
            saved_body = json.loads(saved_report.stdout)
            self.assertTrue(saved_body["ok"], saved_body)
            self.assertTrue(Path(saved_body["path"]).exists())
            reports_list = subprocess.run([sys.executable, "-m", "modelctl.cli", "reports", "list"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(reports_list.returncode, 0, reports_list.stderr + reports_list.stdout)
            reports_body = json.loads(reports_list.stdout)
            self.assertEqual(reports_body["count"], 1)
            reports_show = subprocess.run([sys.executable, "-m", "modelctl.cli", "reports", "show", saved_body["report_id"]], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(reports_show.returncode, 0, reports_show.stderr + reports_show.stdout)
            reports_show_body = json.loads(reports_show.stdout)
            self.assertEqual(reports_show_body["report"]["model"]["id"], "fake")
            ingested = root / "ingested.toml"
            ingest = subprocess.run([sys.executable, "-m", "modelctl.cli", "ingest", "--endpoint", f"http://127.0.0.1:{port}/v1", "--output", str(ingested)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(ingest.returncode, 0, ingest.stderr + ingest.stdout)
            ingest_body = json.loads(ingest.stdout)
            self.assertTrue(ingest_body["ok"], ingest_body)
            self.assertEqual(load_manifest(ingested).model_id, "fake-model")
            stop = subprocess.run(cmd + ["stop"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(stop.returncode, 0, stop.stderr + stop.stdout)

    def test_health_smoke_down_endpoint_returns_structured_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            port = free_port()
            manifest_path = self.write_manifest(root, f'''
                [model]
                id = "down"
                model_id = "down-model"
                endpoint = "http://127.0.0.1:{port}/v1"

                [smoke]
                prompt = "Reply pong."
                expect = "pong"
                timeout_sec = 1
            ''')
            result = subprocess.run([sys.executable, "-m", "modelctl.cli", "-m", str(manifest_path), "health", "--smoke"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(result.returncode, 2, result.stderr + result.stdout)
            self.assertFalse(result.stderr.strip(), result.stderr)
            body = json.loads(result.stdout)
            self.assertEqual(body["status"], "critical")
            self.assertIn("readiness", body["issues"])
            self.assertIn("smoke", body["issues"])
            self.assertIn("error", body["checks"]["smoke"])

    def test_registry_list_command(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = root / "registry"
            registry.mkdir()
            manifest_path = self.write_manifest(registry, '''
                [model]
                id = "registered"
                model_id = "registered-model"
                endpoint = "http://127.0.0.1:9/v1"
            ''')
            source = registry / "registered.toml"
            manifest_path.rename(source)
            env = os.environ.copy()
            env["XDG_CONFIG_HOME"] = str(root / "xdg-config")
            cmd = [sys.executable, "-m", "modelctl.cli", "list", "--registry", str(registry)]
            result = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            body = json.loads(result.stdout)
            self.assertEqual(body["count"], 1)
            self.assertEqual(body["entries"][0]["id"], "registered")
            managed = root / "managed"
            add = subprocess.run([sys.executable, "-m", "modelctl.cli", "registry", "add", "--source", str(source), "--name", "managed-model", "--registry", str(managed)], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(add.returncode, 0, add.stderr + add.stdout)
            show = subprocess.run([sys.executable, "-m", "modelctl.cli", "registry", "show", "managed-model", "--registry", str(managed), "--content"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(show.returncode, 0, show.stderr + show.stdout)
            show_body = json.loads(show.stdout)
            self.assertTrue(show_body["ok"], show_body)
            self.assertIn("content", show_body["entry"])
            used = root / "used.toml"
            use = subprocess.run([sys.executable, "-m", "modelctl.cli", "registry", "use", "managed-model", "--registry", str(managed), "--output", str(used)], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(use.returncode, 0, use.stderr + use.stdout)
            use_body = json.loads(use.stdout)
            self.assertTrue(use_body["ok"], use_body)
            self.assertEqual(load_manifest(used).model_id, "registered-model")
            duplicate = subprocess.run([sys.executable, "-m", "modelctl.cli", "registry", "add", "--source", str(source), "--name", "managed-model", "--registry", str(managed)], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(duplicate.returncode, 2, duplicate.stderr + duplicate.stdout)
            rm = subprocess.run([sys.executable, "-m", "modelctl.cli", "registry", "remove", "managed-model", "--registry", str(managed)], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(rm.returncode, 0, rm.stderr + rm.stdout)
            self.assertFalse((managed / "managed-model.toml").exists())

    def test_fleet_health_scans_registry_entries(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = root / "registry"
            registry.mkdir()
            class H(BaseHTTPRequestHandler):
                def log_message(self, format, *args):
                    pass
                def _send(self, body, status=200):
                    data = json.dumps(body).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                def do_GET(self):
                    if self.path == "/v1/models":
                        self._send({"object": "list", "data": [{"id": "healthy-model"}]})
                    else:
                        self._send({"error": "not found"}, 404)

            server = HTTPServer(("127.0.0.1", 0), H)
            port = int(server.server_address[1])
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                (registry / "healthy.toml").write_text(textwrap.dedent(f'''
                    [model]
                    id = "healthy"
                    model_id = "healthy-model"
                    endpoint = "http://127.0.0.1:{port}/v1"
                '''), encoding="utf-8")
                down_port = free_port()
                (registry / "down.toml").write_text(textwrap.dedent(f'''
                    [model]
                    id = "down"
                    model_id = "down-model"
                    endpoint = "http://127.0.0.1:{down_port}/v1"
                '''), encoding="utf-8")
                env = os.environ.copy()
                env["XDG_CONFIG_HOME"] = str(root / "xdg-config")
                env.pop("MODELCTL_REGISTRY", None)
                empty_registry = root / "empty-registry"
                empty_registry.mkdir()
                empty = subprocess.run([sys.executable, "-m", "modelctl.cli", "fleet", "health", "--registry", str(empty_registry), "--jobs", "2"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                self.assertEqual(empty.returncode, 2, empty.stderr + empty.stdout)
                empty_body = json.loads(empty.stdout)
                self.assertEqual(empty_body["status"], "empty")
                self.assertIn("no_models", empty_body["issues"])

                (registry / "bad.toml").write_text("[model\nthis is not toml", encoding="utf-8")
                result = subprocess.run([sys.executable, "-m", "modelctl.cli", "fleet", "health", "--registry", str(registry), "--jobs", "2", "--max-swap-gib", "999999"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                self.assertEqual(result.returncode, 2, result.stderr + result.stdout)
                self.assertFalse(result.stderr.strip(), result.stderr)
                body = json.loads(result.stdout)
                self.assertFalse(body["ok"], body)
                self.assertEqual(body["count"], 3)
                rows = {row["id"]: row for row in body["models"] if row.get("id")}
                bad_rows = [row for row in body["models"] if row.get("name") == "bad"]
                self.assertEqual(len(bad_rows), 1, body)
                self.assertEqual(bad_rows[0]["status"], "invalid")
                self.assertIn("manifest_invalid", bad_rows[0]["issues"])
                self.assertTrue(rows["healthy"]["ok"], body)
                self.assertEqual(rows["healthy"]["status"], "ok")
                self.assertFalse(rows["down"]["ok"], body)
                self.assertEqual(rows["down"]["status"], "critical")
                self.assertIn("readiness", rows["down"]["issues"])
                pretty = subprocess.run([sys.executable, "-m", "modelctl.cli", "--pretty", "fleet", "health", "--registry", str(registry), "--jobs", "2", "--max-swap-gib", "999999"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                self.assertEqual(pretty.returncode, 2, pretty.stderr + pretty.stdout)
                self.assertIn("healthy", pretty.stdout)
                self.assertIn("critical", pretty.stdout)

                fleet_status = subprocess.run([sys.executable, "-m", "modelctl.cli", "fleet", "status", "--registry", str(registry), "--jobs", "2", "--readiness-timeout", "1"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                self.assertEqual(fleet_status.returncode, 0, fleet_status.stderr + fleet_status.stdout)
                self.assertFalse(fleet_status.stderr.strip(), fleet_status.stderr)
                status_body = json.loads(fleet_status.stdout)
                self.assertTrue(status_body["ok"], status_body)
                self.assertEqual(status_body["count"], 3)
                self.assertEqual(status_body["states"], {"down": 1, "invalid": 1, "ready": 1})
                status_rows = {row["id"]: row for row in status_body["models"] if row.get("id")}
                self.assertTrue(status_rows["healthy"]["ready"], status_body)
                self.assertEqual(status_rows["healthy"]["state"], "ready")
                self.assertFalse(status_rows["down"]["ready"], status_body)
                self.assertEqual(status_rows["down"]["state"], "down")
                status_bad = [row for row in status_body["models"] if row.get("name") == "bad"]
                self.assertEqual(status_bad[0]["state"], "invalid")
                self.assertIn("service", status_rows["healthy"])
                pretty_status = subprocess.run([sys.executable, "-m", "modelctl.cli", "--pretty", "fleet", "status", "--registry", str(registry), "--jobs", "2", "--readiness-timeout", "1"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                self.assertEqual(pretty_status.returncode, 0, pretty_status.stderr + pretty_status.stdout)
                self.assertIn("ready", pretty_status.stdout)
                self.assertIn("invalid", pretty_status.stdout)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_fleet_recover_dry_run_and_execute_starts_down_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = root / "registry"
            registry.mkdir()
            marker = root / "recover.ready"
            starter = root / "mark_ready.py"
            starter.write_text(textwrap.dedent('''
                from pathlib import Path
                import sys
                import time
                Path(sys.argv[1]).write_text("ready", encoding="utf-8")
                time.sleep(60)
            '''), encoding="utf-8")

            class RecoverH(BaseHTTPRequestHandler):
                def log_message(self, format, *args):
                    pass
                def _send(self, body, status=200):
                    data = json.dumps(body).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                def do_GET(self):
                    if self.path == "/v1/models" and marker.exists():
                        self._send({"object": "list", "data": [{"id": "recover-model"}]})
                    elif self.path == "/v1/models":
                        self._send({"object": "list", "data": []})
                    else:
                        self._send({"error": "not found"}, 404)

            readiness_server = HTTPServer(("127.0.0.1", 0), RecoverH)
            port = int(readiness_server.server_address[1])
            readiness_thread = threading.Thread(target=readiness_server.serve_forever, daemon=True)
            readiness_thread.start()
            self.addCleanup(readiness_server.shutdown)
            self.addCleanup(readiness_server.server_close)
            self.addCleanup(lambda: readiness_thread.join(timeout=5))

            (registry / "recover.toml").write_text(textwrap.dedent(f'''
                [model]
                id = "recover"
                model_id = "recover-model"
                endpoint = "http://127.0.0.1:{port}/v1"

                [start]
                command = ["{sys.executable}", "{starter}", "{marker}"]
                cwd = "{root}"
                log_path = "{root / 'recover.log'}"
                pid_path = "{root / 'recover.pid.json'}"
                startup_timeout_sec = 20
                readiness_url = "http://127.0.0.1:{port}/v1/models"
                readiness_contains = "recover-model"
            '''), encoding="utf-8")
            (registry / "nostart.toml").write_text(textwrap.dedent(f'''
                [model]
                id = "nostart"
                model_id = "nostart-model"
                endpoint = "http://127.0.0.1:{free_port()}/v1"
            '''), encoding="utf-8")
            env = os.environ.copy()
            env["XDG_CONFIG_HOME"] = str(root / "xdg-config")
            env.pop("MODELCTL_REGISTRY", None)
            cmd = [sys.executable, "-m", "modelctl.cli", "fleet", "recover", "--registry", str(registry), "--jobs", "1", "--readiness-timeout", "1"]

            dry = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(dry.returncode, 0, dry.stderr + dry.stdout)
            dry_body = json.loads(dry.stdout)
            self.assertTrue(dry_body["ok"], dry_body)
            self.assertFalse(dry_body["executed"], dry_body)
            dry_rows = {row["id"]: row for row in dry_body["models"] if row.get("id")}
            self.assertEqual(dry_rows["recover"]["planned_action"], "start")
            self.assertEqual(dry_rows["recover"]["action"]["type"], "dry_run")
            self.assertEqual(dry_rows["nostart"]["planned_action"], "skip")
            self.assertFalse((root / "recover.pid.json").exists(), "dry-run must not start processes")

            no_wait = subprocess.run(cmd + ["--execute"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(no_wait.returncode, 2, no_wait.stderr + no_wait.stdout)
            no_wait_body = json.loads(no_wait.stdout)
            self.assertEqual(no_wait_body["status"], "invalid_request")
            self.assertIn("execute_requires_wait", no_wait_body["issues"])
            self.assertFalse((root / "recover.pid.json").exists(), "execute without wait must not start processes")

            unsafe_parallel = subprocess.run([sys.executable, "-m", "modelctl.cli", "fleet", "recover", "--registry", str(registry), "--jobs", "2", "--readiness-timeout", "1", "--execute", "--wait"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(unsafe_parallel.returncode, 2, unsafe_parallel.stderr + unsafe_parallel.stdout)
            unsafe_body = json.loads(unsafe_parallel.stdout)
            self.assertEqual(unsafe_body["status"], "invalid_request")
            self.assertIn("execute_requires_serial_jobs", unsafe_body["issues"])
            self.assertFalse((root / "recover.pid.json").exists(), "parallel execute recovery must not start processes")

            executed = subprocess.run(cmd + ["--execute", "--wait"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(executed.returncode, 0, executed.stderr + executed.stdout)
            executed_body = json.loads(executed.stdout)
            self.assertTrue(executed_body["ok"], executed_body)
            self.assertTrue(executed_body["executed"], executed_body)
            rows = {row["id"]: row for row in executed_body["models"] if row.get("id")}
            self.assertEqual(rows["recover"]["action"]["type"], "start")
            self.assertTrue(rows["recover"]["action"]["result"].get("started"), rows["recover"])
            self.assertTrue(rows["recover"]["after"].get("ready"), rows["recover"])

            stop = subprocess.run([sys.executable, "-m", "modelctl.cli", "-m", str(registry / "recover.toml"), "stop"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(stop.returncode, 0, stop.stderr + stop.stdout)

    def test_doctor_fix_and_pretty_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pid_path = root / "state" / "stale.pid.json"
            log_path = root / "logs" / "model.log"
            manifest_path = self.write_manifest(root, f'''
                [model]
                id = "repair"
                model_id = "repair-model"
                endpoint = "http://127.0.0.1:9/v1"

                [start]
                command = ["{sys.executable}", "-c", "import time; time.sleep(60)"]
                cwd = "{root}"
                log_path = "{log_path}"
                pid_path = "{pid_path}"
                startup_timeout_sec = 1
                readiness_url = "http://127.0.0.1:9/v1/models"
                readiness_contains = "repair-model"
            ''')
            pid_path.parent.mkdir(parents=True)
            pid_path.write_text(json.dumps({"pid": 999999, "started_at": "old"}), encoding="utf-8")
            cmd = [sys.executable, "-m", "modelctl.cli", "-m", str(manifest_path)]
            fix = subprocess.run(cmd + ["doctor", "--fix"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(fix.returncode, 0, fix.stderr + fix.stdout)
            fix_body = json.loads(fix.stdout)
            self.assertTrue(fix_body["ok"], fix_body)
            self.assertFalse(pid_path.exists())
            self.assertIn("stale_pid_state_removed", [item["code"] for item in fix_body["fixes"]])
            self.assertTrue(log_path.parent.exists())
            pretty = subprocess.run([sys.executable, "-m", "modelctl.cli", "--pretty", "-m", str(manifest_path), "validate"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(pretty.returncode, 0, pretty.stderr + pretty.stdout)
            self.assertIn("id: repair", pretty.stdout)
            self.assertNotIn('{', pretty.stdout)

    def test_service_launchd_install_preview_and_dry_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            launchd = root / "LaunchAgents"
            manifest_path = self.write_manifest(root, f'''
                [model]
                id = "Fake Model"
                model_id = "fake-model"
                endpoint = "http://127.0.0.1:9191/v1"

                [start]
                command = ["{sys.executable}", "-c", "import time; time.sleep(60)"]
                cwd = "{root}"
                startup_timeout_sec = 5
                readiness_url = "http://127.0.0.1:9191/v1/models"
                readiness_contains = "fake-model"

                [preflight]
                exclusive_ports = [9191]
                max_swap_gib = 8
            ''')
            env = os.environ.copy()
            env["MODELCTL_LAUNCHD_DIR"] = str(launchd)
            cmd = [sys.executable, "-m", "modelctl.cli", "-m", str(manifest_path)]

            bad_label = subprocess.run(cmd + ["service", "install", "--dry-run", "--label", "../escape"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(bad_label.returncode, 2, bad_label.stderr + bad_label.stdout)
            self.assertIn("service error", bad_label.stderr)
            self.assertFalse((root / "escape.plist").exists())

            preview = subprocess.run(cmd + ["service", "install", "--dry-run", "--restart", "--max-swap-gib", "4", "--interval", "5"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(preview.returncode, 0, preview.stderr + preview.stdout)
            preview_body = json.loads(preview.stdout)
            self.assertTrue(preview_body["ok"], preview_body)
            self.assertFalse(preview_body["written"], preview_body)
            self.assertEqual(preview_body["label"], "ai.modelctl.fake-model")
            self.assertIn("daemon", preview_body["program_arguments"])
            self.assertIn("--restart", preview_body["program_arguments"])
            self.assertIn("--max-swap-gib", preview_body["program_arguments"])
            self.assertFalse(launchd.exists(), "dry-run must not create LaunchAgents")

            health_preview = subprocess.run(cmd + ["service", "install", "--dry-run", "--restart", "--max-swap-gib", "48", "--max-swap-delta-gib", "1", "--sample-sec", "5", "--smoke", "--max-latency-sec", "30", "--interval", "120"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(health_preview.returncode, 0, health_preview.stderr + health_preview.stdout)
            health_preview_body = json.loads(health_preview.stdout)
            health_args = health_preview_body["program_arguments"]
            self.assertIn("--health-mode", health_args)
            self.assertIn("--max-swap-delta-gib", health_args)
            self.assertIn("1", health_args)
            self.assertIn("--sample-sec", health_args)
            self.assertIn("5", health_args)
            self.assertIn("--smoke", health_args)
            self.assertIn("--max-latency-sec", health_args)
            self.assertIn("30", health_args)
            self.assertFalse(launchd.exists(), "health dry-run must not create LaunchAgents")

            installed = subprocess.run(cmd + ["service", "install", "--restart", "--max-swap-gib", "4", "--interval", "5"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(installed.returncode, 0, installed.stderr + installed.stdout)
            installed_body = json.loads(installed.stdout)
            self.assertTrue(installed_body["ok"], installed_body)
            self.assertTrue(installed_body["written"], installed_body)
            plist_path = Path(installed_body["plist_path"])
            self.assertTrue(plist_path.exists())
            plist = plistlib.loads(plist_path.read_bytes())
            self.assertEqual(plist["Label"], "ai.modelctl.fake-model")
            self.assertTrue(plist["KeepAlive"])
            self.assertFalse(plist["RunAtLoad"])
            self.assertIn(str(manifest_path.resolve()), plist["ProgramArguments"])
            self.assertIn("daemon", plist["ProgramArguments"])
            self.assertIn("--restart", plist["ProgramArguments"])
            self.assertIn("--max-swap-gib", plist["ProgramArguments"])
            self.assertIn("MODELCTL_MANIFEST", plist["EnvironmentVariables"])

            same_diff = subprocess.run(cmd + ["service", "diff", "--restart", "--max-swap-gib", "4", "--interval", "5"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(same_diff.returncode, 0, same_diff.stderr + same_diff.stdout)
            same_diff_body = json.loads(same_diff.stdout)
            self.assertTrue(same_diff_body["ok"], same_diff_body)
            self.assertFalse(same_diff_body["drift"], same_diff_body)
            self.assertEqual(same_diff_body["differences"], [])

            plist["ProgramArguments"][0] = "/custom/python3.11"
            plist_path.write_bytes(plistlib.dumps(plist, sort_keys=False))
            preserved_python_diff = subprocess.run(cmd + ["service", "diff", "--restart", "--max-swap-gib", "4", "--interval", "5"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(preserved_python_diff.returncode, 0, preserved_python_diff.stderr + preserved_python_diff.stdout)
            explicit_python_diff = subprocess.run(cmd + ["service", "diff", "--restart", "--max-swap-gib", "4", "--interval", "5", "--python", sys.executable], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(explicit_python_diff.returncode, 2, explicit_python_diff.stderr + explicit_python_diff.stdout)
            explicit_python_body = json.loads(explicit_python_diff.stdout)
            self.assertIn("ProgramArguments", [row["key"] for row in explicit_python_body["differences"]])

            drift = subprocess.run(cmd + ["service", "diff", "--restart", "--max-swap-gib", "8", "--interval", "5"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(drift.returncode, 2, drift.stderr + drift.stdout)
            drift_body = json.loads(drift.stdout)
            self.assertFalse(drift_body["ok"], drift_body)
            self.assertTrue(drift_body["drift"], drift_body)
            self.assertIn("ProgramArguments", [row["key"] for row in drift_body["differences"]])

            missing_diff = subprocess.run(cmd + ["service", "diff", "--label", "ai.modelctl.missing"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(missing_diff.returncode, 2, missing_diff.stderr + missing_diff.stdout)
            missing_diff_body = json.loads(missing_diff.stdout)
            self.assertEqual(missing_diff_body["error"], "plist_missing")

            duplicate = subprocess.run(cmd + ["service", "install"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(duplicate.returncode, 2, duplicate.stderr + duplicate.stdout)

            status = subprocess.run(cmd + ["service", "status", "--dry-run"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(status.returncode, 0, status.stderr + status.stdout)
            status_body = json.loads(status.stdout)
            self.assertTrue(status_body["ok"], status_body)
            self.assertEqual(status_body["action"], "status")
            self.assertIn("launchctl", status_body["commands"][0])
            self.assertIn("print", status_body["commands"][0])

            start = subprocess.run(cmd + ["service", "start", "--dry-run"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(start.returncode, 0, start.stderr + start.stdout)
            start_body = json.loads(start.stdout)
            self.assertEqual(start_body["action"], "start")
            self.assertTrue(any("bootstrap" in command for command in start_body["commands"]))
            self.assertTrue(any("kickstart" in command for command in start_body["commands"]))

            uninstall = subprocess.run(cmd + ["service", "uninstall", "--dry-run"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(uninstall.returncode, 0, uninstall.stderr + uninstall.stdout)
            self.assertTrue(plist_path.exists(), "dry-run uninstall must not remove plist")

    def test_mlx_discover_inspect_overlay_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "Qwen-Test-MLX"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "quantization": {"bits": 4}}), encoding="utf-8")
            (model / "tokenizer_config.json").write_text(json.dumps({"chat_template": "external"}), encoding="utf-8")
            (model / "weights.safetensors").write_text("fake", encoding="utf-8")
            (model / "chat_template.jinja").write_text(textwrap.dedent('''
                {% for message in messages %}
                {{ message['content'] }}
                {% endfor %}
                {% if add_generation_prompt %}
                {{- '<|im_start|>assistant\\n<think>\\n' }}
                {% endif %}
            '''), encoding="utf-8")
            cmd = [sys.executable, "-m", "modelctl.cli", "mlx"]
            discovered = subprocess.run(cmd + ["discover", "--root", str(root)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(discovered.returncode, 0, discovered.stderr + discovered.stdout)
            discovered_body = json.loads(discovered.stdout)
            self.assertEqual(discovered_body["count"], 1)
            self.assertEqual(discovered_body["models"][0]["name"], "Qwen-Test-MLX")
            inspected = subprocess.run(cmd + ["inspect", str(model)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(inspected.returncode, 0, inspected.stderr + inspected.stdout)
            inspected_body = json.loads(inspected.stdout)
            self.assertTrue(inspected_body["template"]["bad_think_preamble"], inspected_body)
            self.assertIn("qwen_think_preamble", inspected_body["warnings"])
            inline = root / "Inline-Template-MLX"
            inline.mkdir()
            (inline / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
            (inline / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{% if add_generation_prompt %}{{ '<think>\\n' }}{% endif %}"}), encoding="utf-8")
            inspected_inline = subprocess.run(cmd + ["inspect", str(inline)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(inspected_inline.returncode, 0, inspected_inline.stderr + inspected_inline.stdout)
            inspected_inline_body = json.loads(inspected_inline.stdout)
            self.assertEqual(inspected_inline_body["template"]["source"], "tokenizer_config.json")
            self.assertTrue(inspected_inline_body["template"]["bad_think_preamble"], inspected_inline_body)
            self.assertFalse(inspected_inline_body["template"]["recommended_overlay"], inspected_inline_body)
            overlay = root / "Qwen-Test-MLX-served"
            overlaid = subprocess.run(cmd + ["overlay", str(model), "--output", str(overlay)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(overlaid.returncode, 0, overlaid.stderr + overlaid.stdout)
            overlaid_body = json.loads(overlaid.stdout)
            self.assertTrue(overlaid_body["ok"], overlaid_body)
            self.assertTrue(overlaid_body["patched"])
            self.assertTrue((overlay / "config.json").exists())
            self.assertIn("</think>", (overlay / "chat_template.jinja").read_text(encoding="utf-8"))
            self.assertIn("<think>", (model / "chat_template.jinja").read_text(encoding="utf-8"), "source artifact must stay untouched")
            unsafe = subprocess.run(cmd + ["overlay", str(model), "--output", str(root), "--overwrite"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(unsafe.returncode, 2, unsafe.stderr + unsafe.stdout)
            self.assertTrue(model.exists(), "unsafe overwrite guard must not delete the model root")
            other_served = root / "other-served"
            other_served.mkdir()
            (other_served / "keep.txt").write_text("keep", encoding="utf-8")
            unsafe_served = subprocess.run(cmd + ["overlay", str(model), "--output", str(other_served), "--overwrite"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(unsafe_served.returncode, 2, unsafe_served.stderr + unsafe_served.stdout)
            self.assertTrue((other_served / "keep.txt").exists(), "overwrite guard must not delete arbitrary *-served dirs")
            ancestor = root / "ancestor-served"
            child = ancestor / "Child-MLX"
            child.mkdir(parents=True)
            (child / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
            (child / "chat_template.jinja").write_text("{% if add_generation_prompt %}<think>{% endif %}", encoding="utf-8")
            unsafe_ancestor = subprocess.run(cmd + ["overlay", str(child), "--output", str(ancestor), "--overwrite"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(unsafe_ancestor.returncode, 2, unsafe_ancestor.stderr + unsafe_ancestor.stdout)
            self.assertTrue(child.exists(), "overwrite guard must not delete an ancestor of the source")
            inspected_overlay = subprocess.run(cmd + ["inspect", str(overlay)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(inspected_overlay.returncode, 0, inspected_overlay.stderr + inspected_overlay.stdout)
            inspected_overlay_body = json.loads(inspected_overlay.stdout)
            self.assertFalse(inspected_overlay_body["template"]["bad_think_preamble"], inspected_overlay_body)
            manifest_path = root / "mlx.toml"
            manifest = subprocess.run(cmd + ["manifest", str(overlay), "--output", str(manifest_path), "--id", "qwen-test-served", "--port", "8123"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(manifest.returncode, 0, manifest.stderr + manifest.stdout)
            manifest_body = json.loads(manifest.stdout)
            self.assertTrue(manifest_body["ok"], manifest_body)
            loaded = load_manifest(manifest_path)
            self.assertEqual(loaded.id, "qwen-test-served")
            self.assertEqual(loaded.model_id, "default_model")
            self.assertEqual(loaded.endpoint, "http://127.0.0.1:8123/v1")
            self.assertEqual(loaded.start.readiness_contains, str(overlay.resolve()))
            self.assertIn(str(overlay.resolve()), loaded.start.command)
            self.assertIn("mlx_lm", loaded.start.command)
            self.assertIn('{"enable_thinking":false}', loaded.start.command)
            tuned_manifest_path = root / "mlx-tuned.toml"
            tuned_manifest = subprocess.run(cmd + ["manifest", str(overlay), "--output", str(tuned_manifest_path), "--id", "qwen-test-tuned", "--port", "8124", "--prompt-cache-size", "7", "--prompt-cache-gib", "2"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(tuned_manifest.returncode, 0, tuned_manifest.stderr + tuned_manifest.stdout)
            tuned_loaded = load_manifest(tuned_manifest_path)
            self.assertIsNotNone(tuned_loaded.start)
            assert tuned_loaded.start is not None
            cache_size_idx = tuned_loaded.start.command.index("--prompt-cache-size")
            cache_bytes_idx = tuned_loaded.start.command.index("--prompt-cache-bytes")
            self.assertEqual(tuned_loaded.start.command[cache_size_idx + 1], "7")
            self.assertEqual(tuned_loaded.start.command[cache_bytes_idx + 1], str(2 * 1024 * 1024 * 1024))
            bad_cache_size = subprocess.run(cmd + ["manifest", str(overlay), "--output", str(root / "bad-cache.toml"), "--prompt-cache-size", "0"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(bad_cache_size.returncode, 2, bad_cache_size.stderr + bad_cache_size.stdout)
            self.assertIn("expected a positive integer", bad_cache_size.stderr)
            bad_alias = subprocess.run(cmd + ["manifest", str(overlay), "--output", str(root / "bad.toml"), "--model-id", "qwen-test-served", "--port", "8123"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(bad_alias.returncode, 2, bad_alias.stderr + bad_alias.stdout)

    def test_init_and_version_commands(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "modelctl.toml"
            init = subprocess.run([sys.executable, "-m", "modelctl.cli", "init", "--output", str(out), "--model-id", "init-model"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(init.returncode, 0, init.stderr + init.stdout)
            init_body = json.loads(init.stdout)
            self.assertTrue(init_body["ok"], init_body)
            self.assertEqual(load_manifest(out).model_id, "init-model")
            duplicate = subprocess.run([sys.executable, "-m", "modelctl.cli", "init", "--output", str(out)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(duplicate.returncode, 2, duplicate.stderr + duplicate.stdout)
            version = subprocess.run([sys.executable, "-m", "modelctl.cli", "version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(version.returncode, 0, version.stderr + version.stdout)
            version_body = json.loads(version.stdout)
            self.assertRegex(version_body["version"], r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()
