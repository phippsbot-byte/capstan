from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from modelctl.manifest import load_manifest
from modelctl.system import effective_launch_environment


ROOT = Path(__file__).resolve().parents[1]
ON = ROOT / "examples" / "omlx-qwen4-ssd-expert-cache-on.example.toml"
OFF = ROOT / "examples" / "omlx-qwen4-ssd-expert-cache-off.example.toml"
CACHE_ENV = "OMLX_QWEN4_SSD_EXPERT_CACHE_BYTES"


class Qwen4CacheProfileTests(unittest.TestCase):
    def load_profiles(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            values = {
                "CAPSTAN_OMLX_RUNTIME_ROOT": str(root / "runtime-v3"),
                "CAPSTAN_OMLX_PYTHON": str(root / "runtime-v3" / "bin" / "python"),
                "CAPSTAN_QWEN4_BASE_PATH": str(root / "omlx-base"),
                "CAPSTAN_QWEN4_MODEL_DIR": str(root / "omlx-base" / "models"),
                "CAPSTAN_QWEN4_MODEL_SETTINGS_PATH": str(root / "omlx-base" / "model_settings.json"),
                "CAPSTAN_QWEN4_PROFILE_PATH": str(root / "evidence" / "profile.json"),
                "CAPSTAN_QWEN4_LOAD_TELEMETRY_PATH": str(root / "evidence" / "load.jsonl"),
                "CAPSTAN_QWEN4_STATE_DIR": str(root / "state"),
            }
            with patch.dict(os.environ, values, clear=False):
                return load_manifest(ON), load_manifest(OFF)

    def test_profiles_are_dormant_explicit_omlx_cache_configs(self):
        on, off = self.load_profiles()
        assert on.start is not None and off.start is not None
        for manifest in (on, off):
            self.assertFalse(manifest.fleet.enabled)
            self.assertEqual(manifest.endpoint, "http://127.0.0.1:18144/v1")
            self.assertEqual(manifest.model_id, "qwen4-ssd-expert-cache-experimental")
            self.assertIsNotNone(manifest.start)
            assert manifest.start is not None
            assert manifest.start.cwd is not None
            self.assertEqual(
                manifest.start.command,
                [
                    manifest.start.command[0], "-m", "omlx.cli", "serve",
                    "--base-path", manifest.start.cwd, "--host", "127.0.0.1",
                    "--port", "18144", "--no-hf-cache", "--no-cache",
                    "--hot-cache-max-size", "0", "--memory-guard", "balanced",
                    "--max-concurrent-requests", "1",
                ],
            )
            self.assertEqual(manifest.start.env["PYTHONPATH"], str(Path(manifest.start.command[0]).parents[1]))
            self.assertEqual(manifest.start.env["OMLX_QWEN4_SSD_EXPERT_LAYERS"], "0,1,46,47")
            self.assertEqual(manifest.start.env["OMLX_QWEN4_MMAP_DEDUPE_MATERIALIZE"], "1")
            self.assertEqual(manifest.start.env["OMLX_QWEN4_MMAP_STAGED_RECLAIM"], "1")
            self.assertIn(str(Path(manifest.start.cwd) / "models"), manifest.preflight.required_paths)
            self.assertIn(str(Path(manifest.start.cwd) / "model_settings.json"), manifest.preflight.required_paths)
            self.assertNotIn("mlx_lm", manifest.start.command)
        self.assertEqual(on.id, "omlx-qwen4-ssd-expert-cache-on")
        self.assertEqual(off.id, "omlx-qwen4-ssd-expert-cache-off")
        self.assertEqual(on.endpoint, off.endpoint)
        self.assertEqual(on.start.command, off.start.command)
        self.assertEqual(on.start.cwd, off.start.cwd)
        self.assertEqual(on.preflight, off.preflight)
        self.assertEqual(
            {key: value for key, value in on.start.env.items() if key != CACHE_ENV},
            {key: value for key, value in off.start.env.items() if key != CACHE_ENV},
        )
        self.assertEqual(on.start.env[CACHE_ENV], "1073741824")
        self.assertEqual(off.start.env[CACHE_ENV], "0")

    def test_real_child_environment_overrides_an_inherited_cache_on_value(self):
        on, off = self.load_profiles()
        assert on.start is not None and off.start is not None
        program = "import json, os; print(json.dumps({k: os.environ[k] for k in sys.argv[1:]}))"
        program = "import sys; " + program
        keys = [CACHE_ENV, "OMLX_QWEN4_SSD_EXPERT_LAYERS", "OMLX_QWEN4_MMAP_DEDUPE_MATERIALIZE", "OMLX_QWEN4_MMAP_STAGED_RECLAIM", "PYTHONPATH"]
        with patch.dict(os.environ, {CACHE_ENV: "inherited-on"}, clear=False):
            for manifest, expected in ((on, "1073741824"), (off, "0")):
                assert manifest.start is not None
                env = effective_launch_environment(manifest.start.env, cwd=manifest.start.cwd)
                child = subprocess.run(
                    [sys.executable, "-c", program, *keys],
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                received = json.loads(child.stdout)
                self.assertEqual(received[CACHE_ENV], expected)
                self.assertEqual(received["OMLX_QWEN4_SSD_EXPERT_LAYERS"], "0,1,46,47")
                self.assertEqual(received["OMLX_QWEN4_MMAP_DEDUPE_MATERIALIZE"], "1")
                self.assertEqual(received["OMLX_QWEN4_MMAP_STAGED_RECLAIM"], "1")


if __name__ == "__main__":
    unittest.main()
