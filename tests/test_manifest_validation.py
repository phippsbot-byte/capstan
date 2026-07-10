from __future__ import annotations

from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from modelctl.manifest import ManifestError, load_manifest


BASE = """
[model]
id = "lane"
model_id = "model"
endpoint = "http://127.0.0.1:18080/v1"

[start]
command = ["python3", "-c", "pass"]
startup_timeout_sec = 30

[preflight]
required_paths = []
exclusive_ports = [18080]
max_swap_gib = 4

[health]
max_swap_gib = 4
max_swap_delta_gib = 1
sample_sec = 1
smoke = false

[fleet]
enabled = true
reason = ""

[smoke]
prompt = "ping"
expect = "pong"
max_tokens = 8
temperature = 0
timeout_sec = 10
"""


class StrictManifestValidationTests(unittest.TestCase):
    def load_text(self, text: str):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.toml"
            path.write_text(textwrap.dedent(text), encoding="utf-8")
            return load_manifest(path)

    def assert_manifest_error(self, text: str, field: str) -> None:
        with self.assertRaisesRegex(ManifestError, field.replace("[", r"\[").replace("]", r"\]")):
            self.load_text(text)

    def replace(self, old: str, new: str) -> str:
        self.assertIn(old, BASE)
        return BASE.replace(old, new)

    def test_rejects_non_boolean_cleanup_health_and_fleet_values(self):
        for value in ('"false"', "0", "1"):
            with self.subTest(cleanup_safe=value):
                text = BASE + f'\n[[cleanup]]\npath = "/tmp/disposable"\nsafe = {value}\n'
                self.assert_manifest_error(text, "cleanup[0].safe")
        self.assert_manifest_error(self.replace("smoke = false", 'smoke = "false"'), "health.smoke")
        self.assert_manifest_error(self.replace("enabled = true", 'enabled = "true"'), "fleet.enabled")

    def test_rejects_non_string_model_endpoint_prompt_and_expect(self):
        cases = [
            ("model_id = \"model\"", "model_id = 42", "model.model_id"),
            ('endpoint = "http://127.0.0.1:18080/v1"', "endpoint = 18080", "model.endpoint"),
            ('prompt = "ping"', "prompt = 7", "smoke.prompt"),
            ('expect = "pong"', "expect = 7", "smoke.expect"),
        ]
        for old, new, field in cases:
            with self.subTest(field=field):
                self.assert_manifest_error(self.replace(old, new), field)

    def test_rejects_path_bearing_model_ids_even_with_explicit_state_paths(self):
        unsafe = [".", "..", "../escape", "nested/lane", r"nested\\lane", r"C:\\tmp\\lane"]
        for ident in unsafe:
            for explicit_paths in (False, True):
                with self.subTest(ident=ident, explicit_paths=explicit_paths):
                    text = self.replace('id = "lane"', f'id = {ident!r}')
                    if explicit_paths:
                        text = text.replace(
                            "startup_timeout_sec = 30",
                            'startup_timeout_sec = 30\npid_path = "/tmp/explicit.pid.json"\nlog_path = "/tmp/explicit.log"',
                        )
                    self.assert_manifest_error(text, "model.id")

        nul = self.replace('id = "lane"', 'id = "bad\\u0000lane"')
        self.assert_manifest_error(nul, "model.id")

    def test_rejects_malformed_endpoint_urls_and_invalid_ports(self):
        endpoints = [
            "127.0.0.1:18080/v1",
            "ftp://127.0.0.1:18080/v1",
            "http://:18080/v1",
            "http://127.0.0.1:0/v1",
            "http://127.0.0.1:65536/v1",
            "http://127.0.0.1:not-a-port/v1",
        ]
        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                text = self.replace('endpoint = "http://127.0.0.1:18080/v1"', f'endpoint = "{endpoint}"')
                self.assert_manifest_error(text, "model.endpoint")
        for port in (0, -1, 65536):
            with self.subTest(exclusive_port=port):
                self.assert_manifest_error(self.replace("exclusive_ports = [18080]", f"exclusive_ports = [{port}]"), "preflight.exclusive_ports")

    def test_endpoint_validation_contains_parser_errors_and_rejects_unsafe_suffixes(self):
        endpoints = [
            "http://[::1",
            "http://::1:18080/v1",
            "http://[::1]extra:18080/v1",
            "http://user@@127.0.0.1:18080/v1",
            "http://127.0.0.1:18080/v1?",
            "http://127.0.0.1:18080/v1?api_key=secret",
            "http://127.0.0.1:18080/v1#",
            "http://127.0.0.1:18080/v1#models",
            "http://127.0.0.1:/v1",
            "http://[::1]:/v1",
            "http://%zz/v1",
            "http://host%00.evil/v1",
            "http://host%7f.evil/v1",
            r"http://host\u0000.evil/v1",
        ]
        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                text = self.replace('endpoint = "http://127.0.0.1:18080/v1"', f'endpoint = "{endpoint}"')
                self.assert_manifest_error(text, "model.endpoint")

    def test_rejects_backslashes_and_malformed_raw_authority_delimiters(self):
        endpoints = [
            "http://127.0.0.1\\evil/v1",
            "http:\\127.0.0.1:18080/v1",
            "http:/\\127.0.0.1:18080/v1",
            "http:////127.0.0.1:18080/v1",
        ]
        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                text = self.replace(
                    'endpoint = "http://127.0.0.1:18080/v1"',
                    f"endpoint = {endpoint!r}",
                )
                self.assert_manifest_error(text, "model.endpoint")

    def test_expands_v024_environment_urls_before_contextual_validation(self):
        text = self.replace(
            'endpoint = "http://127.0.0.1:18080/v1"',
            'endpoint = "$CAPSTAN_TEST_ENDPOINT"',
        ).replace(
            "startup_timeout_sec = 30",
            'startup_timeout_sec = 30\nreadiness_url = "$CAPSTAN_TEST_READINESS"',
        )
        with patch.dict(
            "os.environ",
            {
                "CAPSTAN_TEST_ENDPOINT": "http://127.0.0.1:18080/v1",
                "CAPSTAN_TEST_READINESS": "http://127.0.0.1:18080/ready?lane=stable",
            },
        ):
            manifest = self.load_text(text)
        self.assertEqual(manifest.endpoint, "http://127.0.0.1:18080/v1")
        self.assertEqual(manifest.start.readiness_url, "http://127.0.0.1:18080/ready?lane=stable")

        with patch.dict("os.environ", {"CAPSTAN_TEST_ENDPOINT": "not-a-url"}):
            self.assert_manifest_error(text, "model.endpoint")

        readiness_text = text.replace(
            'endpoint = "$CAPSTAN_TEST_ENDPOINT"',
            'endpoint = "http://127.0.0.1:18080/v1"',
        )
        with patch.dict("os.environ", {"CAPSTAN_TEST_READINESS": "not-a-url"}):
            self.assert_manifest_error(readiness_text, "start.readiness_url")

    def test_start_environment_expansion_preserves_v024_host_semantics(self):
        text = BASE.replace(
            'command = ["python3", "-c", "pass"]',
            'command = ["$CAPSTAN_HOST_BIN", "-c", "pass"]\nenv = { PATH = "$PATH:/custom", HOME = "/child/home" }',
        )
        with patch.dict("os.environ", {"CAPSTAN_HOST_BIN": "/host/python", "PATH": "/host/bin", "HOME": "/host/home"}):
            manifest = self.load_text(text)
        start = manifest.start
        self.assertIsNotNone(start)
        assert start is not None
        self.assertEqual(start.command[0], "/host/python")
        self.assertEqual(start.env["PATH"], "/host/bin:/custom")
        self.assertEqual(start.env["HOME"], "/child/home")

    def test_rejects_paths_that_are_empty_after_environment_expansion(self):
        cases = [
            (
                BASE.replace("required_paths = []", 'required_paths = ["$CAPSTAN_EMPTY"]'),
                "preflight.required_paths[0]",
            ),
            (
                BASE.replace("startup_timeout_sec = 30", 'startup_timeout_sec = 30\ncwd = "$CAPSTAN_EMPTY"'),
                "start.cwd",
            ),
            (
                BASE.replace("startup_timeout_sec = 30", 'startup_timeout_sec = 30\nlog_path = "$CAPSTAN_EMPTY"'),
                "start.log_path",
            ),
            (
                BASE.replace("startup_timeout_sec = 30", 'startup_timeout_sec = 30\npid_path = "$CAPSTAN_EMPTY"'),
                "start.pid_path",
            ),
            (
                BASE + '\n[[preflight.disk]]\npath = "$CAPSTAN_EMPTY"\nmin_free_gib = 0\n',
                "preflight.disk[0].path",
            ),
            (
                BASE + '\n[[cleanup]]\npath = "$CAPSTAN_EMPTY"\nsafe = true\n',
                "cleanup[0].path",
            ),
            (
                BASE.replace('command = ["python3", "-c", "pass"]', 'command = ["$CAPSTAN_EMPTY", "-c", "pass"]'),
                "start.command[0]",
            ),
            (
                BASE.replace("required_paths = []", r'required_paths = ["\u0000"]'),
                "preflight.required_paths[0]",
            ),
        ]
        with patch.dict("os.environ", {"CAPSTAN_EMPTY": ""}):
            for text, field in cases:
                with self.subTest(field=field):
                    self.assert_manifest_error(text, field)

    def test_valid_bracketed_ipv6_endpoint_builds_api_urls(self):
        endpoints = [
            "http://127.0.0.1/v1",
            "HTTP://127.0.0.1/v1",
            "http://127.0.0.1:80/v1",
            "https://127.0.0.1:443/v1",
            "http://[::1]/v1",
            "http://[::1]:80/v1",
            "http://[::1]:18080/v1",
        ]
        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                manifest = self.load_text(
                    self.replace('endpoint = "http://127.0.0.1:18080/v1"', f'endpoint = "{endpoint}"')
                )
                self.assertEqual(manifest.models_url, f"{endpoint}/models")
                self.assertEqual(manifest.chat_url, f"{endpoint}/chat/completions")

    def test_rejects_negative_or_non_finite_limits(self):
        cases = [
            ("max_swap_gib = 4", "max_swap_gib = -1", "preflight.max_swap_gib"),
            ("max_swap_gib = 4", "max_swap_gib = inf", "preflight.max_swap_gib"),
            ("max_swap_delta_gib = 1", "max_swap_delta_gib = nan", "health.max_swap_delta_gib"),
            ("sample_sec = 1", "sample_sec = -0.1", "health.sample_sec"),
            ("temperature = 0", "temperature = -1", "smoke.temperature"),
        ]
        for old, new, field in cases:
            with self.subTest(field=field, value=new):
                self.assert_manifest_error(self.replace(old, new), field)
        disk = BASE + '\n[[preflight.disk]]\npath = "/tmp"\nmin_free_gib = -1\n'
        self.assert_manifest_error(disk, "preflight.disk[0].min_free_gib")

    def test_rejects_non_positive_timeouts_and_token_counts(self):
        cases = [
            ("startup_timeout_sec = 30", "startup_timeout_sec = 0", "start.startup_timeout_sec"),
            ("startup_timeout_sec = 30", "startup_timeout_sec = -1", "start.startup_timeout_sec"),
            ("max_tokens = 8", "max_tokens = 0", "smoke.max_tokens"),
            ("timeout_sec = 10", "timeout_sec = 0", "smoke.timeout_sec"),
        ]
        for old, new, field in cases:
            with self.subTest(field=field, value=new):
                self.assert_manifest_error(self.replace(old, new), field)

    def test_rejects_incompatible_health_sampling(self):
        immediate = self.load_text(self.replace("sample_sec = 1", "sample_sec = 0"))
        self.assertEqual(immediate.health.sample_sec, 0)
        self.assertEqual(immediate.health.max_swap_delta_gib, 1)
        without_delta = BASE.replace("max_swap_delta_gib = 1\n", "")
        self.assert_manifest_error(without_delta, "health.sample_sec")

    def test_preserves_valid_v024_manifests_and_examples(self):
        parsed = self.load_text(BASE)
        self.assertEqual(parsed.model_id, "model")
        self.assertEqual(parsed.smoke.max_tokens, 8)
        root = Path(__file__).resolve().parents[1]
        for path in sorted((root / "examples").glob("*.toml")):
            with self.subTest(example=path.name):
                self.assertEqual(load_manifest(path).path, path.resolve())


if __name__ == "__main__":
    unittest.main()
