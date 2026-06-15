from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import time

from .manifest import ModelManifest
from .report import build_report, report_markdown


def _safe(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value.strip())
    return cleaned.strip(".-_") or "unknown"


def reports_dir() -> Path:
    state = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return state / "modelctl" / "reports"


def _report_path(report_id_or_path: str) -> Path:
    candidate = Path(report_id_or_path).expanduser()
    if candidate.exists():
        return candidate
    return reports_dir() / report_id_or_path


def save_report(manifest: ModelManifest, fmt: str = "json", include_smoke: bool = False) -> dict[str, Any]:
    if fmt not in {"json", "md"}:
        raise ValueError("fmt must be json or md")
    payload = build_report(manifest, include_smoke=include_smoke)
    model_dir = reports_dir() / _safe(manifest.id)
    model_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    path = model_dir / f"{stamp}.{fmt}"
    content = json.dumps(payload, indent=2, sort_keys=True) if fmt == "json" else report_markdown(payload)
    path.write_text(content, encoding="utf-8")
    report_id = f"{model_dir.name}/{path.name}"
    return {"ok": payload.get("ok"), "report_id": report_id, "path": str(path), "format": fmt, "model": manifest.id, "generated_at": payload.get("generated_at")}


def _entry(path: Path) -> dict[str, Any]:
    rel = path.relative_to(reports_dir())
    entry: dict[str, Any] = {
        "report_id": str(rel),
        "path": str(path),
        "format": path.suffix.lstrip("."),
        "size_bytes": path.stat().st_size,
        "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(path.stat().st_mtime)),
        "model": rel.parts[0] if rel.parts else None,
    }
    if path.suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entry["ok"] = payload.get("ok")
            entry["generated_at"] = payload.get("generated_at")
            model = payload.get("model") if isinstance(payload, dict) else None
            if isinstance(model, dict):
                entry["id"] = model.get("id")
                entry["model_id"] = model.get("model_id")
        except Exception as exc:
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def list_reports(model: str | None = None) -> dict[str, Any]:
    base = reports_dir()
    if not base.exists():
        return {"ok": True, "reports_dir": str(base), "count": 0, "entries": []}
    if model:
        roots = [base / _safe(model)]
    else:
        roots = [base]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(p for p in root.rglob("*") if p.is_file() and p.suffix in {".json", ".md"})
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    entries = [_entry(path) for path in files]
    return {"ok": True, "reports_dir": str(base), "count": len(entries), "entries": entries}


def show_report(report_id_or_path: str) -> dict[str, Any]:
    path = _report_path(report_id_or_path)
    if not path.exists() or not path.is_file():
        return {"ok": False, "error": f"report not found: {report_id_or_path}"}
    fmt = path.suffix.lstrip(".")
    if fmt == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {"ok": True, "report_id": str(path.relative_to(reports_dir())) if reports_dir() in path.parents else str(path), "path": str(path), "format": fmt, "report": payload}
    content = path.read_text(encoding="utf-8")
    return {"ok": True, "report_id": str(path.relative_to(reports_dir())) if reports_dir() in path.parents else str(path), "path": str(path), "format": fmt, "content": content}
