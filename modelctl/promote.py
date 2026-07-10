from __future__ import annotations

from dataclasses import replace
from typing import Any
from urllib.parse import urlparse

from .lifecycle import LifecycleLockError, lifecycle_lock
from .manifest import ModelManifest
from .ops import health, preflight
from .receipt import validate_promotion_receipt
from .runner import _rotate_locked, _start_locked, _stop_locked, rotate


def _manifest_ref(manifest: ModelManifest) -> dict[str, Any]:
    return {"id": manifest.id, "model_id": manifest.model_id, "endpoint": manifest.endpoint, "path": str(manifest.path)}


def _rollback_promoted_locked(current: ModelManifest, *, stop_timeout_sec: int, readiness_timeout_sec: float | None) -> dict[str, Any]:
    stopped = _stop_locked(current, stop_timeout_sec)
    if stopped.get("safe_to_start") is not True:
        return {"attempted": True, "ok": False, "stop": stopped, "error": "promoted process could not be stopped safely"}
    started = _start_locked(current, True, readiness_timeout_sec)
    return {"attempted": True, "ok": started.get("ok") is True, "stop": stopped, "start": started}


def _endpoint_port(manifest: ModelManifest) -> int | None:
    parsed = urlparse(manifest.endpoint)
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "http":
        return 80
    if parsed.scheme == "https":
        return 443
    return None


def _candidate_preflight_blocking_issues(current: ModelManifest, candidate: ModelManifest, candidate_preflight: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Separate real candidate preflight blockers from expected stable-port occupancy."""
    if candidate_preflight.get("ok"):
        return [], []
    shared_port = _endpoint_port(current) if current.endpoint == candidate.endpoint and current.model_id == candidate.model_id else None
    blocking: list[str] = []
    tolerated: list[str] = []
    for check in candidate_preflight.get("checks", []):
        if not isinstance(check, dict) or check.get("ok", True):
            continue
        if check.get("type") == "port" and shared_port is not None and check.get("port") == shared_port:
            tolerated.append(f"shared_endpoint_port:{shared_port}")
            continue
        blocking.append("candidate_preflight_failed")
    if not blocking and not tolerated:
        blocking.append("candidate_preflight_failed")
    return sorted(set(blocking)), tolerated


def _promotion_receipt_guard(candidate: ModelManifest, *, required: bool) -> dict[str, Any]:
    receipt = validate_promotion_receipt(candidate)
    issues: list[str] = []
    if receipt.get("ok") is not True:
        issues.append("candidate_receipt_failed")
    if required and receipt.get("configured") is not True:
        issues.append("candidate_receipt_required")
    return {
        "ok": not issues,
        "status": "valid" if not issues else "blocked",
        "candidate_receipt": receipt,
        "issues": issues,
    }


def _post_health_manifest(current: ModelManifest, candidate: ModelManifest) -> ModelManifest:
    """Use candidate health/smoke gates while keeping current PID ownership after handoff."""
    return replace(candidate, path=current.path, id=current.id, start=current.start)


def promote(
    current: ModelManifest,
    candidate: ModelManifest,
    *,
    execute: bool = False,
    readiness_timeout_sec: float | None = None,
    stop_timeout_sec: int = 10,
    rollback: bool = True,
    max_swap_gib: float | None = None,
    max_swap_delta_gib: float | None = None,
    sample_sec: float | None = None,
    include_smoke: bool = False,
    max_latency_sec: float | None = None,
    require_receipt: bool = False,
) -> dict[str, Any]:
    """Promote a candidate manifest through preflight, rotate, and post-health gating.

    Plan-only by default. `execute=True` performs the stop/start rotation.
    """
    receipt_required = bool(require_receipt or current.promotion_requires_receipt)
    base: dict[str, Any] = {
        "ok": False,
        "action": "promote",
        "execute": execute,
        "current": _manifest_ref(current),
        "candidate": _manifest_ref(candidate),
        "readiness_timeout_sec": readiness_timeout_sec,
        "stop_timeout_sec": stop_timeout_sec,
        "rollback_enabled": rollback,
        "receipt_required": receipt_required,
        "health_options": {
            "max_swap_gib": max_swap_gib,
            "max_swap_delta_gib": max_swap_delta_gib,
            "sample_sec": sample_sec,
            "smoke": include_smoke,
            "max_latency_sec": max_latency_sec,
        },
    }

    current_preflight = preflight(current)
    candidate_preflight = preflight(candidate)
    candidate_receipt = validate_promotion_receipt(candidate)
    rotate_plan = rotate(
        current,
        candidate,
        readiness_timeout_sec=readiness_timeout_sec,
        stop_timeout_sec=stop_timeout_sec,
        rollback=rollback,
        dry_run=True,
    )
    issues: list[str] = []
    if not current_preflight.get("ok"):
        issues.append("current_preflight_failed")
    candidate_issues, tolerated_candidate_preflight = _candidate_preflight_blocking_issues(current, candidate, candidate_preflight)
    issues.extend(candidate_issues)
    if not rotate_plan.get("ok"):
        issues.append("rotate_plan_failed")
    if not candidate_receipt.get("ok"):
        issues.append("candidate_receipt_failed")
    if receipt_required and not candidate_receipt.get("configured"):
        issues.append("candidate_receipt_required")

    common = {
        **base,
        "current_preflight": current_preflight,
        "candidate_preflight": candidate_preflight,
        "candidate_receipt": candidate_receipt,
        "tolerated_candidate_preflight": tolerated_candidate_preflight,
        "rotate_plan": rotate_plan,
        "issues": issues,
    }
    if issues:
        return {**common, "status": "blocked"}
    if not execute:
        return {**common, "ok": True, "status": "planned"}

    timeout = readiness_timeout_sec
    if timeout is None:
        timeout = float(candidate.start.startup_timeout_sec) if candidate.start is not None else 120.0
    try:
        with lifecycle_lock("promote", current, candidate):
            locked_current_preflight = preflight(current)
            locked_candidate_preflight = preflight(candidate)
            locked_receipt_guard = _promotion_receipt_guard(candidate, required=receipt_required)
            locked_candidate_receipt = locked_receipt_guard["candidate_receipt"]
            locked_candidate_issues, locked_tolerated = _candidate_preflight_blocking_issues(
                current, candidate, locked_candidate_preflight
            )
            locked_issues: list[str] = []
            if not locked_current_preflight.get("ok"):
                locked_issues.append("current_preflight_failed")
            locked_issues.extend(locked_candidate_issues)
            locked_issues.extend(locked_receipt_guard["issues"])
            if locked_issues:
                return {
                    **common,
                    "status": "blocked_after_lock",
                    "current_preflight": locked_current_preflight,
                    "candidate_preflight": locked_candidate_preflight,
                    "candidate_receipt": locked_candidate_receipt,
                    "tolerated_candidate_preflight": locked_tolerated,
                    "issues": locked_issues,
                }

            rotation = _rotate_locked(
                current,
                candidate,
                readiness_timeout_sec=timeout,
                stop_timeout_sec=stop_timeout_sec,
                rollback=rollback,
                target_pre_spawn_check=lambda: _promotion_receipt_guard(candidate, required=receipt_required),
            )
            if not rotation.get("ok"):
                return {**common, "status": "rotation_failed", "rotation": rotation, "issues": ["rotation_failed"]}

            post_health_manifest = _post_health_manifest(current, candidate)
            try:
                post_health = health(
                    post_health_manifest,
                    max_swap_gib=max_swap_gib,
                    max_swap_delta_gib=max_swap_delta_gib,
                    sample_sec=sample_sec,
                    include_smoke=include_smoke,
                    max_latency_sec=max_latency_sec,
                )
            except Exception as exc:
                post_health = {"ok": False, "status": "health_exception", "error": f"{type(exc).__name__}: {exc}"}
            if post_health.get("ok"):
                return {**common, "ok": True, "status": "promoted", "rotation": rotation, "post_health": post_health}

            rollback_result = {"attempted": False}
            if rollback:
                rollback_result = _rollback_promoted_locked(
                    current,
                    stop_timeout_sec=stop_timeout_sec,
                    readiness_timeout_sec=readiness_timeout_sec,
                )
            return {
                **common,
                "status": "post_health_failed",
                "rotation": rotation,
                "post_health": post_health,
                "rollback": rollback_result,
                "issues": ["post_health_failed"],
            }
    except LifecycleLockError as exc:
        return {**common, "status": "lock_failed", "lock": exc.failure.as_dict(), "issues": ["lock_failed"]}
    except Exception as exc:
        return {**common, "status": "promotion_exception", "error": f"{type(exc).__name__}: {exc}", "issues": ["promotion_exception"]}
