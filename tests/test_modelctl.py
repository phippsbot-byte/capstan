from __future__ import annotations

from pathlib import Path
import json
import socket
import subprocess
import sys
import tempfile
import textwrap
import unittest

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
            watchdog = subprocess.run(cmd + ["watchdog", "--max-swap-gib", "999999", "--duration", "0"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(watchdog.returncode, 0, watchdog.stderr + watchdog.stdout)
            watchdog_body = json.loads(watchdog.stdout)
            self.assertTrue(watchdog_body["ok"], watchdog_body)
            ingested = root / "ingested.toml"
            ingest = subprocess.run([sys.executable, "-m", "modelctl.cli", "ingest", "--endpoint", f"http://127.0.0.1:{port}/v1", "--output", str(ingested)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(ingest.returncode, 0, ingest.stderr + ingest.stdout)
            ingest_body = json.loads(ingest.stdout)
            self.assertTrue(ingest_body["ok"], ingest_body)
            self.assertEqual(load_manifest(ingested).model_id, "fake-model")
            stop = subprocess.run(cmd + ["stop"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(stop.returncode, 0, stop.stderr + stop.stdout)
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
            manifest_path.rename(registry / "registered.toml")
            cmd = [sys.executable, "-m", "modelctl.cli", "list", "--registry", str(registry)]
            result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            body = json.loads(result.stdout)
            self.assertEqual(body["count"], 1)
            self.assertEqual(body["entries"][0]["id"], "registered")


if __name__ == "__main__":
    unittest.main()
