from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .manifest import ManifestError, load_manifest
from .ops import health
from .registry import list_registry


def _base_row(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": entry.get("name"),
        "path": entry.get("path"),
        "id": entry.get("id"),
        "model_id": entry.get("model_id"),
        "endpoint": entry.get("endpoint"),
    }


def fleet_health(
    *,
    registries: list[str] | None = None,
    max_swap_gib: float | None = None,
    max_swap_delta_gib: float | None = None,
    sample_sec: float = 0.0,
    include_smoke: bool = False,
    max_latency_sec: float | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run structured health checks across registered model manifests."""
    listing = list_registry(registries)
    entries = listing.get("entries", [])
    if limit is not None:
        entries = entries[: max(0, limit)]

    rows: list[dict[str, Any]] = []
    for entry in entries:
        row = _base_row(entry)
        if not entry.get("ok"):
            rows.append({
                **row,
                "ok": False,
                "status": "invalid",
                "issues": ["manifest_invalid"],
                "warnings": [],
                "error": entry.get("error"),
            })
            continue
        try:
            manifest = load_manifest(Path(str(entry["path"])))
            verdict = health(
                manifest,
                max_swap_gib=max_swap_gib,
                max_swap_delta_gib=max_swap_delta_gib,
                sample_sec=sample_sec,
                include_smoke=include_smoke,
                max_latency_sec=max_latency_sec,
            )
            rows.append({
                **row,
                "ok": bool(verdict.get("ok")),
                "status": verdict.get("status"),
                "issues": verdict.get("issues", []),
                "warnings": verdict.get("warnings", []),
                "health": verdict,
            })
        except ManifestError as exc:
            rows.append({**row, "ok": False, "status": "invalid", "issues": ["manifest_invalid"], "warnings": [], "error": str(exc)})
        except Exception as exc:
            rows.append({**row, "ok": False, "status": "critical", "issues": ["health_exception"], "warnings": [], "error": f"{type(exc).__name__}: {exc}"})

    counts = Counter(str(row.get("status") or "unknown") for row in rows)
    if not rows:
        return {
            "ok": False,
            "status": "empty",
            "issues": ["no_models"],
            "count": 0,
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
        "registry_dirs": listing.get("registry_dirs", []),
        "statuses": dict(sorted(counts.items())),
        "models": rows,
    }
