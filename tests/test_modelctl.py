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
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from modelctl.manifest import load_manifest
from modelctl.ops import cleanup_execute, cleanup_plan, preflight


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ModelCtlTests(unittest.TestCase):
    def write_manifest(self, root: Path, content: str) -> Path:
        path = root / "modelctl.toml"
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        return path

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
                HTTPServer(('127.0.0.1', port), H).serve_forever()
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
                empty = subprocess.run([sys.executable, "-m", "modelctl.cli", "fleet", "health", "--registry", str(empty_registry)], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                self.assertEqual(empty.returncode, 2, empty.stderr + empty.stdout)
                empty_body = json.loads(empty.stdout)
                self.assertEqual(empty_body["status"], "empty")
                self.assertIn("no_models", empty_body["issues"])

                (registry / "bad.toml").write_text("[model\nthis is not toml", encoding="utf-8")
                result = subprocess.run([sys.executable, "-m", "modelctl.cli", "fleet", "health", "--registry", str(registry), "--max-swap-gib", "999999"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
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
                pretty = subprocess.run([sys.executable, "-m", "modelctl.cli", "--pretty", "fleet", "health", "--registry", str(registry), "--max-swap-gib", "999999"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                self.assertEqual(pretty.returncode, 2, pretty.stderr + pretty.stdout)
                self.assertIn("healthy", pretty.stdout)
                self.assertIn("critical", pretty.stdout)

                fleet_status = subprocess.run([sys.executable, "-m", "modelctl.cli", "fleet", "status", "--registry", str(registry), "--readiness-timeout", "1"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
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
                pretty_status = subprocess.run([sys.executable, "-m", "modelctl.cli", "--pretty", "fleet", "status", "--registry", str(registry), "--readiness-timeout", "1"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                self.assertEqual(pretty_status.returncode, 0, pretty_status.stderr + pretty_status.stdout)
                self.assertIn("ready", pretty_status.stdout)
                self.assertIn("invalid", pretty_status.stdout)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

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
