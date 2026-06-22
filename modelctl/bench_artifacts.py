from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .manifest import ModelManifest


def bench_markdown(manifest: ModelManifest, result: dict[str, Any]) -> str:
    lines = [
        f"# Capstan bench: {manifest.id}",
        "",
        f"- model_id: `{manifest.model_id}`",
        f"- endpoint: `{manifest.endpoint}`",
        f"- ok: `{result.get('ok')}`",
        f"- prompt_chars: `{result.get('prompt_chars')}`",
        f"- repeats: `{result.get('repeats')}`",
        f"- total_elapsed_s: `{result.get('total_elapsed_s')}`",
        f"- swap_before_gib: `{result.get('swap_before_gib')}`",
        f"- swap_after_gib: `{result.get('swap_after_gib')}`",
        f"- swap_delta_gib: `{result.get('swap_delta_gib')}`",
        "",
        "## Runs",
        "",
    ]
    for run in result.get("runs", []):
        lines.extend([
            f"### prompt_chars={run.get('prompt_chars')} repeat={run.get('repeat')}",
            "",
            f"- ok: `{run.get('ok')}`",
            f"- exact: `{run.get('exact')}`",
            f"- status: `{run.get('status')}`",
            f"- elapsed_s: `{run.get('elapsed_s')}`",
            f"- swap_delta_gib: `{run.get('swap_delta_gib')}`",
            f"- usage: `{run.get('usage')}`",
            f"- content_preview: `{run.get('content_preview')}`",
            "",
        ])
        timings = run.get("timings")
        if timings:
            lines.extend(["Timings:", ""])
            for key in sorted(timings):
                lines.append(f"- {key}: `{timings[key]}`")
            lines.append("")
    return "\n".join(lines)


def write_bench_artifact(manifest: ModelManifest, result: dict[str, Any], output: str | None = None, fmt: str = "json") -> dict[str, Any]:
    if fmt not in {"json", "md"}:
        raise ValueError("fmt must be json or md")
    content = json.dumps(result, indent=2, sort_keys=True) if fmt == "json" else bench_markdown(manifest, result)
    written = None
    if output:
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written = str(path)
    return {"ok": result.get("ok"), "format": fmt, "output": written, "result": None if output else result, "content": None if output or fmt == "json" else content}
