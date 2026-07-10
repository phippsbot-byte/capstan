from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from modelctl import cli
from modelctl.manifest import ManifestError, load_manifest
from modelctl.receipt import RECEIPT_SCHEMA, candidate_binding, validate_promotion_receipt


class PromotionReceiptTests(unittest.TestCase):
    def write_manifest(self, root: Path, *, receipt_sha: str | None = None, command_suffix: str = "stable") -> Path:
        artifact = root / "artifact.bin"
        if not artifact.exists():
            artifact.write_bytes(b"artifact-v1")
        receipt = root / "receipt.json"
        rows = [
            "[model]",
            'id = "candidate"',
            'model_id = "candidate-model"',
            'endpoint = "http://127.0.0.1:65431/v1"',
            "",
            "[start]",
            f'command = ["/usr/bin/env", "python3", "-c", "import time; time.sleep(60)", "{command_suffix}"]',
            f'cwd = "{root}"',
            f'pid_path = "{root / "candidate.pid.json"}"',
            f'log_path = "{root / "candidate.log"}"',
            "startup_timeout_sec = 5",
            "",
            "[preflight]",
            f'required_paths = ["{artifact}"]',
        ]
        if receipt_sha is not None:
            rows.extend([
                "",
                "[promotion.receipt]",
                f'path = "{receipt}"',
                f'sha256 = "{receipt_sha}"',
                "max_age_sec = 3600",
                'require_decision = "promote"',
                'required_gates = ["logit", "quality"]',
            ])
        path = root / "candidate.toml"
        path.write_text("\n".join(rows) + "\n")
        return path

    def issue_receipt(
        self,
        root: Path,
        *,
        decision: str = "promote",
        logit: bool = True,
        quality: bool = True,
        generated_at: datetime | None = None,
    ) -> tuple[Path, dict]:
        manifest_path = self.write_manifest(root)
        binding = candidate_binding(load_manifest(manifest_path))
        body = {
            "schema": RECEIPT_SCHEMA,
            "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
            "decision": decision,
            "candidate_fingerprint": binding["candidate_fingerprint"],
            "gates": {
                "logit": {"pass": logit, "strict_pass": logit},
                "quality": {"pass": quality, "score": 4.2, "blocking": [] if quality else ["hallucination"]},
            },
        }
        receipt_path = root / "receipt.json"
        data = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode()
        receipt_path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        return self.write_manifest(root, receipt_sha=digest), body

    def test_manifest_requires_strict_receipt_fields(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = self.write_manifest(root)
            with path.open("a") as handle:
                handle.write("\n[promotion.receipt]\npath = \"receipt.json\"\n")
            with self.assertRaisesRegex(ManifestError, "requires path and sha256"):
                load_manifest(path)
            path = self.write_manifest(root, receipt_sha="bad")
            with self.assertRaisesRegex(ManifestError, "64-character hexadecimal"):
                load_manifest(path)

    def test_candidate_binding_is_deterministic_and_detects_artifact_or_launch_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = candidate_binding(load_manifest(self.write_manifest(root)))
            second = candidate_binding(load_manifest(self.write_manifest(root)))
            self.assertEqual(first["candidate_fingerprint"], second["candidate_fingerprint"])
            (root / "artifact.bin").write_bytes(b"artifact-v2")
            artifact_changed = candidate_binding(load_manifest(self.write_manifest(root)))
            self.assertNotEqual(first["candidate_fingerprint"], artifact_changed["candidate_fingerprint"])
            launch_changed = candidate_binding(load_manifest(self.write_manifest(root, command_suffix="changed")))
            self.assertNotEqual(artifact_changed["candidate_fingerprint"], launch_changed["candidate_fingerprint"])
            env_path = self.write_manifest(root, command_suffix="changed")
            with env_path.open("a") as handle:
                handle.write('\n[start.env]\nRUNTIME_MODE = "direct"\n')
            env_changed = candidate_binding(load_manifest(env_path))
            self.assertNotEqual(launch_changed["candidate_fingerprint"], env_changed["candidate_fingerprint"])
            self.assertNotIn("RUNTIME_MODE", json.dumps(env_changed))

    def test_valid_receipt_is_hash_age_decision_gate_and_candidate_bound(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            now = datetime.now(timezone.utc)
            manifest_path, _ = self.issue_receipt(root, generated_at=now - timedelta(seconds=30))
            result = validate_promotion_receipt(load_manifest(manifest_path), now=now)
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["status"], "valid")
            self.assertLess(result["age_sec"], 31)
            self.assertTrue(result["gates"]["quality"]["pass"])

    def test_rejected_stale_gate_failed_tampered_and_drifted_receipts_fail_closed(self) -> None:
        cases = [
            ({"decision": "reject"}, "receipt_decision_rejected"),
            ({"quality": False}, "required_gate_failed:quality"),
            ({"generated_at": datetime.now(timezone.utc) - timedelta(hours=2)}, "receipt_stale"),
        ]
        for kwargs, issue in cases:
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                path, _ = self.issue_receipt(root, **kwargs)
                result = validate_promotion_receipt(load_manifest(path))
                self.assertFalse(result["ok"], result)
                self.assertIn(issue, result["issues"])

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path, _ = self.issue_receipt(root)
            with (root / "receipt.json").open("ab") as handle:
                handle.write(b" ")
            result = validate_promotion_receipt(load_manifest(path))
            self.assertEqual(result["issues"], ["receipt_sha256_mismatch"])

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path, _ = self.issue_receipt(root)
            (root / "artifact.bin").write_bytes(b"changed-after-evaluation")
            result = validate_promotion_receipt(load_manifest(path))
            self.assertIn("candidate_fingerprint_mismatch", result["issues"])

    def test_cli_fingerprint_and_validate_are_structured_and_gating(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path, _ = self.issue_receipt(root)
            out = io.StringIO()
            with redirect_stdout(out):
                rc = cli.main(["-m", str(path), "receipt", "fingerprint"])
            fingerprint = json.loads(out.getvalue())
            self.assertEqual(rc, 0)
            self.assertRegex(fingerprint["candidate_fingerprint"], r"^[0-9a-f]{64}$")
            out = io.StringIO()
            with redirect_stdout(out):
                rc = cli.main(["-m", str(path), "receipt", "validate"])
            self.assertEqual(rc, 0, out.getvalue())
            self.assertTrue(json.loads(out.getvalue())["ok"])

    def test_unsafe_link_mode_and_duplicate_json_receipts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path, _ = self.issue_receipt(root)
            receipt = root / "receipt.json"
            receipt.chmod(0o666)
            result = validate_promotion_receipt(load_manifest(path))
            self.assertEqual(result["issues"], ["unsafe_or_unreadable_receipt"])
            receipt.chmod(0o644)
            os.link(receipt, root / "receipt-hardlink.json")
            result = validate_promotion_receipt(load_manifest(path))
            self.assertEqual(result["issues"], ["unsafe_or_unreadable_receipt"])

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path, _ = self.issue_receipt(root)
            receipt = root / "receipt.json"
            linked = root / "receipt-link.json"
            linked.symlink_to(receipt)
            path.write_text(path.read_text().replace(str(receipt), str(linked)))
            result = validate_promotion_receipt(load_manifest(path))
            self.assertEqual(result["issues"], ["unsafe_or_unreadable_receipt"])

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            duplicate = b'{"schema":"first","schema":"second"}\n'
            (root / "receipt.json").write_bytes(duplicate)
            path = self.write_manifest(root, receipt_sha=hashlib.sha256(duplicate).hexdigest())
            result = validate_promotion_receipt(load_manifest(path))
            self.assertEqual(result["issues"], ["receipt_json_invalid"])

    def test_promote_plan_rejects_failed_receipt_before_mutation(self) -> None:
        from modelctl.promote import promote

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate_path, _ = self.issue_receipt(root, decision="reject")
            candidate = load_manifest(candidate_path)
            current_path = root / "current.toml"
            current_text = candidate_path.read_text().replace('id = "candidate"', 'id = "current"').split("\n[promotion.receipt]", 1)[0]
            current_path.write_text(current_text)
            current = load_manifest(current_path)
            result = promote(current, candidate, execute=True)
            self.assertEqual(result["status"], "blocked", result)
            self.assertIn("candidate_receipt_failed", result["issues"])
            self.assertIn("receipt_decision_rejected", result["candidate_receipt"]["issues"])
            self.assertFalse((root / "candidate.pid.json").exists())

    def test_stable_policy_or_cli_can_require_candidate_receipt_presence(self) -> None:
        from modelctl.promote import promote

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate_path = self.write_manifest(root)
            candidate = load_manifest(candidate_path)
            current_path = root / "current.toml"
            current_text = candidate_path.read_text().replace('id = "candidate"', 'id = "current"')
            current_text += '\n[promotion]\nrequire_receipt = true\n'
            current_path.write_text(current_text)
            current = load_manifest(current_path)
            result = promote(current, candidate, execute=True)
            self.assertEqual(result["status"], "blocked", result)
            self.assertIn("candidate_receipt_required", result["issues"])
            self.assertTrue(result["receipt_required"])
            self.assertFalse((root / "candidate.pid.json").exists())

            no_policy = load_manifest(candidate_path)
            cli_required = promote(no_policy, candidate, execute=True, require_receipt=True)
            self.assertEqual(cli_required["status"], "blocked", cli_required)
            self.assertIn("candidate_receipt_required", cli_required["issues"])

    def test_execute_revalidates_receipt_inside_lock_before_rotation(self) -> None:
        from modelctl import promote as promote_mod

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = self.write_manifest(root)
            current = load_manifest(path)
            candidate = load_manifest(path)
            valid = {"ok": True, "configured": True, "status": "valid"}
            invalid = {"ok": False, "configured": True, "status": "invalid", "issues": ["receipt_sha256_mismatch"]}
            with (
                patch.object(promote_mod, "preflight", return_value={"ok": True, "checks": []}),
                patch.object(promote_mod, "rotate", return_value={"ok": True, "status": "planned"}),
                patch.object(promote_mod, "validate_promotion_receipt", side_effect=[valid, invalid]),
                patch.object(promote_mod, "lifecycle_lock"),
                patch.object(promote_mod, "_rotate_locked") as rotate_locked,
            ):
                result = promote_mod.promote(current, candidate, execute=True)
            self.assertEqual(result["status"], "blocked_after_lock", result)
            self.assertIn("candidate_receipt_failed", result["issues"])
            rotate_locked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
