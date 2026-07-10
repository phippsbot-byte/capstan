from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from .manifest import ModelManifest
from .system import command_cwd_fingerprint

RECEIPT_SCHEMA = "capstan-promotion-receipt-v1"
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_ARTIFACT_HASH_BYTES = 128 * 1024 * 1024
FUTURE_SKEW_SEC = 300


class ReceiptError(RuntimeError):
    """A receipt or its candidate binding cannot be trusted."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_binding(raw_path: str) -> dict[str, Any]:
    source = Path(raw_path).expanduser()
    try:
        canonical = source.resolve(strict=True)
        info = canonical.stat()
    except OSError as exc:
        raise ReceiptError(f"required artifact unavailable: {raw_path}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ReceiptError(
            f"receipt-bound artifact must be a regular file, not a directory; list a small digest manifest instead: {canonical}"
        )
    if info.st_size > MAX_ARTIFACT_HASH_BYTES:
        raise ReceiptError(
            f"receipt-bound artifact exceeds {MAX_ARTIFACT_HASH_BYTES} bytes; list a small digest manifest instead: {canonical}"
        )
    return {
        "path": str(canonical),
        "kind": "file",
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "sha256": _sha256_file(canonical),
    }


def candidate_binding(manifest: ModelManifest) -> dict[str, Any]:
    if manifest.start is None:
        raise ReceiptError("candidate manifest has no [start] configuration")
    cwd = manifest.start.cwd or str(manifest.path.parent)
    artifacts = [_artifact_binding(path) for path in sorted(set(manifest.preflight.required_paths))]
    environment_bytes = json.dumps(manifest.start.env, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload = {
        "model_id": manifest.model_id,
        "endpoint": manifest.endpoint,
        "launch_fingerprint": command_cwd_fingerprint(manifest.start.command, cwd),
        "environment_fingerprint": hashlib.sha256(environment_bytes).hexdigest(),
        "artifacts": artifacts,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {
        "ok": True,
        "schema": "capstan-candidate-binding-v1",
        "candidate_fingerprint": hashlib.sha256(encoded).hexdigest(),
        **payload,
    }


def _safe_receipt_bytes(path: Path) -> bytes:
    try:
        original = path.lstat()
    except OSError as exc:
        raise ReceiptError(f"receipt unavailable: {path}: {exc}") from exc
    if stat.S_ISLNK(original.st_mode) or not stat.S_ISREG(original.st_mode):
        raise ReceiptError(f"receipt must be a regular non-symlink file: {path}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ReceiptError(f"receipt could not be opened safely: {path}: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ReceiptError(f"receipt must be a regular file: {path}")
        if info.st_uid != os.getuid():
            raise ReceiptError(f"receipt must be owned by uid {os.getuid()}: {path}")
        if info.st_mode & 0o022:
            raise ReceiptError(f"receipt must not be group/world writable: {path}")
        if info.st_nlink != 1:
            raise ReceiptError(f"receipt must have exactly one hard link: {path}")
        if info.st_size <= 0 or info.st_size > MAX_RECEIPT_BYTES:
            raise ReceiptError(f"receipt size must be between 1 and {MAX_RECEIPT_BYTES} bytes: {path}")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(fd)
        if len(data) != info.st_size or (after.st_dev, after.st_ino, after.st_size) != (info.st_dev, info.st_ino, info.st_size):
            raise ReceiptError(f"receipt changed while being read: {path}")
        return data
    finally:
        os.close(fd)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReceiptError(f"receipt JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ReceiptError("receipt generated_at must be a non-empty RFC3339 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ReceiptError("receipt generated_at must be a valid RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ReceiptError("receipt generated_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def validate_promotion_receipt(manifest: ModelManifest, *, now: datetime | None = None) -> dict[str, Any]:
    config = manifest.promotion_receipt
    if config is None:
        return {"ok": True, "configured": False, "required": False, "status": "not_configured"}

    path = Path(config.path).expanduser()
    base: dict[str, Any] = {
        "ok": False,
        "configured": True,
        "required": True,
        "status": "invalid",
        "path": str(path),
        "expected_sha256": config.sha256,
        "require_decision": config.require_decision,
        "required_gates": list(config.required_gates),
        "max_age_sec": config.max_age_sec,
    }
    try:
        data = _safe_receipt_bytes(path)
    except ReceiptError as exc:
        return {**base, "issues": ["unsafe_or_unreadable_receipt"], "error": str(exc)}

    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != config.sha256:
        return {**base, "actual_sha256": actual_sha256, "issues": ["receipt_sha256_mismatch"]}

    try:
        receipt = json.loads(data, object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ReceiptError) as exc:
        return {**base, "actual_sha256": actual_sha256, "issues": ["receipt_json_invalid"], "error": str(exc)}
    if not isinstance(receipt, dict):
        return {**base, "actual_sha256": actual_sha256, "issues": ["receipt_schema_invalid"]}

    issues: list[str] = []
    if receipt.get("schema") != RECEIPT_SCHEMA:
        issues.append("receipt_schema_invalid")

    generated_at: datetime | None = None
    age_sec: float | None = None
    try:
        generated_at = _parse_timestamp(receipt.get("generated_at"))
        reference = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        age_sec = (reference - generated_at).total_seconds()
        if age_sec < -FUTURE_SKEW_SEC:
            issues.append("receipt_from_future")
        elif age_sec > config.max_age_sec:
            issues.append("receipt_stale")
    except ReceiptError:
        issues.append("receipt_timestamp_invalid")

    decision = receipt.get("decision")
    if not isinstance(decision, str) or decision != config.require_decision:
        issues.append("receipt_decision_rejected")

    try:
        binding = candidate_binding(manifest)
    except ReceiptError as exc:
        return {**base, "actual_sha256": actual_sha256, "issues": ["candidate_binding_failed"], "error": str(exc)}
    if not binding["artifacts"]:
        issues.append("candidate_artifacts_missing")
    if receipt.get("candidate_fingerprint") != binding["candidate_fingerprint"]:
        issues.append("candidate_fingerprint_mismatch")

    gates = receipt.get("gates")
    gate_results: dict[str, Any] = {}
    if not isinstance(gates, dict):
        issues.append("receipt_gates_invalid")
        gates = {}
    for name in config.required_gates:
        gate = gates.get(name)
        passed = isinstance(gate, dict) and type(gate.get("pass")) is bool and gate.get("pass") is True
        gate_results[name] = {"pass": passed, "receipt": gate if isinstance(gate, dict) else None}
        if not passed:
            issues.append(f"required_gate_failed:{name}")

    return {
        **base,
        "ok": not issues,
        "status": "valid" if not issues else "invalid",
        "actual_sha256": actual_sha256,
        "generated_at": generated_at.isoformat() if generated_at is not None else None,
        "age_sec": round(age_sec, 3) if age_sec is not None else None,
        "decision": decision,
        "candidate_fingerprint": receipt.get("candidate_fingerprint"),
        "expected_candidate_fingerprint": binding["candidate_fingerprint"],
        "gates": gate_results,
        "issues": issues,
    }
