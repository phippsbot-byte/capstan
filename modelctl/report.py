from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

from .manifest import ModelManifest
from .ops import doctor, smoke, validate


def build_report(manifest: ModelManifest, include_smoke: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": validate(manifest),
        "doctor": doctor(manifest),
    }
    if include_smoke:
        payload["smoke"] = smoke(manifest)
    payload["ok"] = bool(payload["doctor"].get("ok")) and ("smoke" not in payload or bool(payload["smoke"].get("ok")))
    return payload


def report_markdown(report: dict[str, Any]) -> str:
    model = report.get("model", {})
    doctor_payload = report.get("doctor", {})
    status = doctor_payload.get("status", {})
    readiness = status.get("readiness", {})
    warnings = doctor_payload.get("warnings", [])
    issues = doctor_payload.get("issues", [])
    cleanup = doctor_payload.get("cleanup", {})
    lines = [
        f"# Capstan report: {model.get('id', 'unknown')}",
        "",
        f"Generated: `{report.get('generated_at')}`",
        f"Overall OK: `{report.get('ok')}`",
        "",
        "## Model",
        "",
        f"- id: `{model.get('id')}`",
        f"- model_id: `{model.get('model_id')}`",
        f"- endpoint: `{model.get('endpoint')}`",
        f"- manifest: `{model.get('manifest')}`",
        "",
        "## Runtime status",
        "",
        f"- pid: `{status.get('pid')}`",
        f"- ready: `{readiness.get('ready')}`",
        f"- swap_used_gib: `{status.get('swap_used_gib')}`",
        f"- log_path: `{status.get('log_path')}`",
        "",
        "## Issues",
        "",
    ]
    if issues:
        lines.extend(f"- `{item.get('code', 'issue')}`" for item in issues)
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- `{item.get('code', 'warning')}`" for item in warnings)
    else:
        lines.append("- none")
    lines.extend(["", "## Cleanup", ""])
    lines.append(f"- candidates: `{len(cleanup.get('candidates', []))}`")
    lines.append(f"- total: `{cleanup.get('total')}`")
    if "smoke" in report:
        smoke_payload = report["smoke"]
        lines.extend([
            "",
            "## Smoke",
            "",
            f"- ok: `{smoke_payload.get('ok')}`",
            f"- exact: `{smoke_payload.get('exact')}`",
            f"- status: `{smoke_payload.get('status')}`",
            f"- content: `{smoke_payload.get('content')}`",
        ])
    lines.append("")
    return "\n".join(lines)


def write_report(manifest: ModelManifest, output: str | None = None, fmt: str = "json", include_smoke: bool = False) -> dict[str, Any]:
    payload = build_report(manifest, include_smoke=include_smoke)
    if fmt not in {"json", "md"}:
        raise ValueError("fmt must be json or md")
    content = json.dumps(payload, indent=2, sort_keys=True) if fmt == "json" else report_markdown(payload)
    written = None
    if output:
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written = str(path)
    return {"ok": payload.get("ok"), "format": fmt, "output": written, "report": None if output else payload, "content": None if output or fmt == "json" else content}
