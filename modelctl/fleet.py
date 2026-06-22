from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
import time

from .manifest import ManifestError, load_manifest
from .ops import health
from .registry import list_registry
from .runner import active_pid, default_log_path, default_pid_path, readiness_check, read_pid_state, start as start_model
from .service import default_label, service_plist_path
from .system import swap_used_gib


def _base_row(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": entry.get("name"),
        "path": entry.get("path"),
        "id": entry.get("id"),
        "model_id": entry.get("model_id"),
        "endpoint": entry.get("endpoint"),
    }


def _service_snapshot(manifest) -> dict[str, Any]:
    label = default_label(manifest)
    plist_path = service_plist_path(label)
    exists = plist_path.exists()
    return {"label": label, "plist_path": str(plist_path), "plist_exists": exists, "managed": exists}


def _readiness_timeout(timeout: float) -> float:
    return max(0.001, float(timeout))


def _elapsed_sec(started: float) -> float:
    return round(max(0.0, time.perf_counter() - started), 3)


def _worker_count(count: int, jobs: int | None, *, default_jobs: int = 32) -> int:
    if count <= 0:
        return 0
    requested = default_jobs if jobs is None else jobs
    return max(1, min(count, int(requested)))


def _map_entries(entries: list[dict[str, Any]], fn, *, jobs: int | None, default_jobs: int = 32) -> tuple[list[dict[str, Any]], int]:
    workers = _worker_count(len(entries), jobs, default_jobs=default_jobs)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, entries)), workers
    return [fn(entry) for entry in entries], workers


def _fleet_status_row(entry: dict[str, Any], *, swap: float | None, readiness_timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    row = _base_row(entry)
    if not entry.get("ok"):
        return {
            **row,
            "ok": False,
            "valid": False,
            "state": "invalid",
            "ready": None,
            "error": entry.get("error"),
            "elapsed_sec": _elapsed_sec(started),
        }
    try:
        manifest = load_manifest(Path(str(entry["path"])))
        pid = active_pid(manifest)
        try:
            readiness = readiness_check(manifest, timeout=_readiness_timeout(readiness_timeout))
            readiness_error = None
        except Exception as exc:
            readiness = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
            readiness_error = readiness["error"]
        ready = bool(readiness.get("ready"))
        return {
            **row,
            "ok": True,
            "valid": True,
            "state": "ready" if ready else "down",
            "ready": ready,
            "pid": pid,
            "pid_path": str(default_pid_path(manifest)),
            "pid_state": read_pid_state(manifest),
            "log_path": str(default_log_path(manifest)),
            "has_start": manifest.start is not None,
            "readiness": readiness,
            "readiness_error": readiness_error,
            "swap_used_gib": None if swap is None else round(swap, 3),
            "service": _service_snapshot(manifest),
            "elapsed_sec": _elapsed_sec(started),
        }
    except ManifestError as exc:
        return {**row, "ok": False, "valid": False, "state": "invalid", "ready": None, "error": str(exc), "elapsed_sec": _elapsed_sec(started)}
    except Exception as exc:
        return {**row, "ok": False, "valid": False, "state": "error", "ready": None, "error": f"{type(exc).__name__}: {exc}", "elapsed_sec": _elapsed_sec(started)}


def fleet_status(
    *,
    registries: list[str] | None = None,
    limit: int | None = None,
    readiness_timeout: float = 1.0,
    jobs: int | None = None,
) -> dict[str, Any]:
    """Return an operator snapshot across registered model manifests."""
    listing = list_registry(registries)
    entries = listing.get("entries", [])
    if limit is not None:
        entries = entries[: max(0, limit)]

    started = time.perf_counter()
    swap = swap_used_gib()
    rows, workers = _map_entries(
        entries,
        lambda entry: _fleet_status_row(entry, swap=swap, readiness_timeout=readiness_timeout),
        jobs=jobs,
    )

    states = Counter(str(row.get("state") or "unknown") for row in rows)
    return {
        "ok": True,
        "status": "ok",
        "count": len(rows),
        "jobs": workers,
        "elapsed_sec": _elapsed_sec(started),
        "registry_dirs": listing.get("registry_dirs", []),
        "states": dict(sorted(states.items())),
        "models": rows,
    }


def _fleet_health_row(
    entry: dict[str, Any],
    *,
    max_swap_gib: float | None,
    max_swap_delta_gib: float | None,
    sample_sec: float | None,
    include_smoke: bool,
    max_latency_sec: float | None,
    max_prompt_latency_sec: float | None,
    max_completion_latency_sec: float | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    row = _base_row(entry)
    if not entry.get("ok"):
        return {
            **row,
            "ok": False,
            "status": "invalid",
            "issues": ["manifest_invalid"],
            "warnings": [],
            "error": entry.get("error"),
            "elapsed_sec": _elapsed_sec(started),
        }
    try:
        manifest = load_manifest(Path(str(entry["path"])))
        verdict = health(
            manifest,
            max_swap_gib=max_swap_gib,
            max_swap_delta_gib=max_swap_delta_gib,
            sample_sec=sample_sec,
            include_smoke=include_smoke,
            max_latency_sec=max_latency_sec,
            max_prompt_latency_sec=max_prompt_latency_sec,
            max_completion_latency_sec=max_completion_latency_sec,
        )
        return {
            **row,
            "ok": bool(verdict.get("ok")),
            "status": verdict.get("status"),
            "issues": verdict.get("issues", []),
            "warnings": verdict.get("warnings", []),
            "health": verdict,
            "elapsed_sec": _elapsed_sec(started),
        }
    except ManifestError as exc:
        return {**row, "ok": False, "status": "invalid", "issues": ["manifest_invalid"], "warnings": [], "error": str(exc), "elapsed_sec": _elapsed_sec(started)}
    except Exception as exc:
        return {**row, "ok": False, "status": "critical", "issues": ["health_exception"], "warnings": [], "error": f"{type(exc).__name__}: {exc}", "elapsed_sec": _elapsed_sec(started)}


def fleet_health(
    *,
    registries: list[str] | None = None,
    max_swap_gib: float | None = None,
    max_swap_delta_gib: float | None = None,
    sample_sec: float | None = None,
    include_smoke: bool = False,
    max_latency_sec: float | None = None,
    max_prompt_latency_sec: float | None = None,
    max_completion_latency_sec: float | None = None,
    limit: int | None = None,
    jobs: int | None = None,
) -> dict[str, Any]:
    """Run structured health checks across registered model manifests."""
    listing = list_registry(registries)
    entries = listing.get("entries", [])
    if limit is not None:
        entries = entries[: max(0, limit)]

    started = time.perf_counter()
    rows, workers = _map_entries(
        entries,
        lambda entry: _fleet_health_row(
            entry,
            max_swap_gib=max_swap_gib,
            max_swap_delta_gib=max_swap_delta_gib,
            sample_sec=sample_sec,
            include_smoke=include_smoke,
            max_latency_sec=max_latency_sec,
            max_prompt_latency_sec=max_prompt_latency_sec,
            max_completion_latency_sec=max_completion_latency_sec,
        ),
        jobs=jobs,
    )

    counts = Counter(str(row.get("status") or "unknown") for row in rows)
    if not rows:
        return {
            "ok": False,
            "status": "empty",
            "issues": ["no_models"],
            "count": 0,
            "jobs": workers,
            "elapsed_sec": _elapsed_sec(started),
            "registry_dirs": listing.get("registry_dirs", []),
            "statuses": {},
            "models": [],
        }
    unhealthy = [row for row in rows if not row.get("ok")]
    return {
        "ok": not unhealthy,
        "status": "ok" if not unhealthy else "critical",
        "issues": [str(row.get("id") or row.get("name") or row.get("path")) for row in unhealthy],
        "count": len(rows),
        "jobs": workers,
        "elapsed_sec": _elapsed_sec(started),
        "registry_dirs": listing.get("registry_dirs", []),
        "statuses": dict(sorted(counts.items())),
        "models": rows,
    }


def _readiness_or_error(manifest, timeout: float) -> tuple[bool, dict[str, Any]]:
    try:
        readiness = readiness_check(manifest, timeout=_readiness_timeout(timeout))
        return bool(readiness.get("ready")), readiness
    except Exception as exc:
        return False, {"ready": False, "error": f"{type(exc).__name__}: {exc}"}


def fleet_recover(
    *,
    registries: list[str] | None = None,
    limit: int | None = None,
    readiness_timeout: float = 1.0,
    execute: bool = False,
    wait: bool = False,
    jobs: int | None = None,
) -> dict[str, Any]:
    """Plan or execute safe recovery for down registered model manifests.

    Recovery is intentionally narrow: start manifests that are down and have a
    [start] section. It does not restart already-ready models or mutate invalid
    registry entries. Dry-run is the default; callers must pass execute=True for
    side effects.
    """
    listing = list_registry(registries)
    if execute and not wait:
        return {
            "ok": False,
            "status": "invalid_request",
            "executed": False,
            "wait": wait,
            "error": "--execute requires --wait so recovery is readiness-verified",
            "issues": ["execute_requires_wait"],
            "count": 0,
            "jobs": 0,
            "elapsed_sec": 0.0,
            "registry_dirs": listing.get("registry_dirs", []),
            "planned": {},
            "models": [],
        }
    if execute and jobs is not None and jobs > 1:
        return {
            "ok": False,
            "status": "invalid_request",
            "executed": False,
            "wait": wait,
            "error": "--execute recovery is serial only; omit --jobs or pass --jobs 1",
            "issues": ["execute_requires_serial_jobs"],
            "count": 0,
            "jobs": jobs,
            "elapsed_sec": 0.0,
            "registry_dirs": listing.get("registry_dirs", []),
            "planned": {},
            "models": [],
        }
    entries = listing.get("entries", [])
    if limit is not None:
        entries = entries[: max(0, limit)]

    def recover_one(entry: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        row_started = time.perf_counter()
        row = _base_row(entry)
        if not entry.get("ok"):
            return ({
                **row,
                "ok": False,
                "valid": False,
                "planned_action": "skip",
                "reason": "manifest_invalid",
                "error": entry.get("error"),
                "action": {"type": "skip", "reason": "manifest_invalid"},
                "elapsed_sec": _elapsed_sec(row_started),
            }, None)
        try:
            manifest = load_manifest(Path(str(entry["path"])))
            before_ready, before_readiness = _readiness_or_error(manifest, readiness_timeout)
            base = {
                **row,
                "ok": True,
                "valid": True,
                "pid": active_pid(manifest),
                "before": before_readiness,
                "has_start": manifest.start is not None,
            }
            if before_ready:
                return ({
                    **base,
                    "state": "ready",
                    "planned_action": "none",
                    "reason": "already_ready",
                    "action": {"type": "none", "reason": "already_ready"},
                    "elapsed_sec": _elapsed_sec(row_started),
                }, None)
            if manifest.start is None:
                return ({
                    **base,
                    "state": "down",
                    "planned_action": "skip",
                    "reason": "no_start_section",
                    "action": {"type": "skip", "reason": "no_start_section"},
                    "elapsed_sec": _elapsed_sec(row_started),
                }, None)

            if not execute:
                return ({
                    **base,
                    "state": "down",
                    "planned_action": "start",
                    "reason": "not_ready",
                    "action": {"type": "dry_run", "would": "start", "wait": wait},
                    "elapsed_sec": _elapsed_sec(row_started),
                }, None)

            result = start_model(manifest, wait=wait)
            after_ready, after_readiness = _readiness_or_error(manifest, readiness_timeout)
            action_ok = bool(result.get("already_running") or result.get("started"))
            if wait:
                result_readiness = result.get("readiness")
                readiness = result_readiness if isinstance(result_readiness, dict) else after_readiness
                action_ok = action_ok and bool(readiness.get("ready"))
            ident = str(entry.get("id") or entry.get("name") or entry.get("path"))
            return ({
                **base,
                "state": "ready" if after_ready else "down",
                "planned_action": "start",
                "reason": "not_ready",
                "action": {"type": "start", "ok": action_ok, "result": result},
                "after": after_readiness,
                "pid_after": active_pid(manifest),
                "elapsed_sec": _elapsed_sec(row_started),
            }, None if action_ok else ident)
        except ManifestError as exc:
            ident = str(entry.get("id") or entry.get("name") or entry.get("path"))
            return ({**row, "ok": False, "valid": False, "planned_action": "skip", "reason": "manifest_invalid", "error": str(exc), "action": {"type": "skip", "reason": "manifest_invalid"}, "elapsed_sec": _elapsed_sec(row_started)}, ident)
        except Exception as exc:
            ident = str(entry.get("id") or entry.get("name") or entry.get("path"))
            return ({**row, "ok": False, "valid": True, "planned_action": "error", "reason": "recover_exception", "error": f"{type(exc).__name__}: {exc}", "action": {"type": "error"}, "elapsed_sec": _elapsed_sec(row_started)}, ident)

    started = time.perf_counter()
    default_jobs = 1 if execute else 32
    workers = _worker_count(len(entries), jobs, default_jobs=default_jobs)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(recover_one, entries))
    else:
        results = [recover_one(entry) for entry in entries]
    rows = [row for row, _failure in results]
    failures = [failure for _row, failure in results if failure is not None]

    planned = Counter(str(row.get("planned_action") or "unknown") for row in rows)
    return {
        "ok": not failures,
        "status": "ok" if not failures else "critical",
        "executed": execute,
        "wait": wait,
        "count": len(rows),
        "jobs": workers,
        "elapsed_sec": _elapsed_sec(started),
        "registry_dirs": listing.get("registry_dirs", []),
        "planned": dict(sorted(planned.items())),
        "issues": failures,
        "models": rows,
    }
