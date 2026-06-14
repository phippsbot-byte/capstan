from __future__ import annotations

from pathlib import Path
from typing import Any
import shutil

from .http import http_json
from .manifest import ModelManifest
from .runner import active_pid, default_log_path, default_pid_path, readiness_check, read_pid_state
from .system import disk_free_gib, human_bytes, path_size_bytes, port_is_free, swap_used_gib


def validate(manifest: ModelManifest) -> dict[str, Any]:
    return {"id": manifest.id, "model_id": manifest.model_id, "endpoint": manifest.endpoint, "manifest": str(manifest.path), "has_start": manifest.start is not None, "required_paths": manifest.preflight.required_paths, "exclusive_ports": manifest.preflight.exclusive_ports, "cleanup_candidates": len(manifest.cleanup)}


def preflight(manifest: ModelManifest) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    ok = True
    for p in manifest.preflight.required_paths:
        exists = Path(p).exists()
        checks.append({"type": "path", "path": p, "ok": exists})
        ok = ok and exists
    current_pid = active_pid(manifest)
    for port in manifest.preflight.exclusive_ports:
        free = port_is_free(port)
        port_ok = free or current_pid is not None
        checks.append({"type": "port", "port": port, "free": free, "ok": port_ok, "active_pid": current_pid})
        ok = ok and port_ok
    for disk in manifest.preflight.disk:
        try:
            free_gib = disk_free_gib(disk.path)
            disk_ok = free_gib >= disk.min_free_gib
            checks.append({"type": "disk", "path": disk.path, "free_gib": round(free_gib, 2), "min_free_gib": disk.min_free_gib, "ok": disk_ok})
            ok = ok and disk_ok
        except Exception as exc:
            checks.append({"type": "disk", "path": disk.path, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            ok = False
    if manifest.preflight.max_swap_gib is not None:
        used = swap_used_gib()
        swap_ok = used is None or used <= manifest.preflight.max_swap_gib
        checks.append({"type": "swap", "used_gib": None if used is None else round(used, 2), "max_swap_gib": manifest.preflight.max_swap_gib, "ok": swap_ok})
        ok = ok and swap_ok
    return {"ok": ok, "checks": checks}


def status(manifest: ModelManifest) -> dict[str, Any]:
    used = swap_used_gib()
    result: dict[str, Any] = {"id": manifest.id, "model_id": manifest.model_id, "endpoint": manifest.endpoint, "pid": active_pid(manifest), "pid_state": read_pid_state(manifest), "pid_path": str(default_pid_path(manifest)), "log_path": str(default_log_path(manifest)), "swap_used_gib": None if used is None else round(used, 2)}
    try:
        result["readiness"] = readiness_check(manifest, timeout=5)
    except Exception as exc:
        result["readiness"] = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
    return result


def smoke(manifest: ModelManifest, prompt: str | None = None, expect: str | None = None, max_tokens: int | None = None, temperature: float | None = None) -> dict[str, Any]:
    prompt = prompt if prompt is not None else manifest.smoke.prompt
    expect = expect if expect is not None else manifest.smoke.expect
    payload = {"model": manifest.model_id, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens if max_tokens is not None else manifest.smoke.max_tokens, "temperature": temperature if temperature is not None else manifest.smoke.temperature}
    status_code, body, _text = http_json("POST", manifest.chat_url, payload=payload, timeout=manifest.smoke.timeout_sec)
    content = ""
    finish = None
    usage = None
    if isinstance(body, dict):
        choices = body.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
            finish = choices[0].get("finish_reason")
        usage = body.get("usage")
    exact = None if expect is None else content.strip() == expect
    return {"ok": 200 <= status_code < 300 and (exact is not False), "status": status_code, "content": content, "expect": expect, "exact": exact, "finish_reason": finish, "usage": usage, "raw": body}


def cleanup_plan(manifest: ModelManifest) -> dict[str, Any]:
    rows = []
    total = 0
    for c in manifest.cleanup:
        exists = Path(c.path).exists() or Path(c.path).is_symlink()
        size = path_size_bytes(c.path) if exists else 0
        total += size
        rows.append({"path": c.path, "exists": exists, "size_bytes": size, "size": human_bytes(size), "safe": c.safe, "description": c.description})
    return {"total_bytes": total, "total": human_bytes(total), "candidates": rows}


def cleanup_execute(manifest: ModelManifest, force: bool = False) -> dict[str, Any]:
    plan = cleanup_plan(manifest)
    deleted = []
    skipped = []
    for row, candidate in zip(plan["candidates"], manifest.cleanup):
        if not row["exists"]:
            skipped.append({**row, "reason": "missing"})
            continue
        if not candidate.safe and not force:
            skipped.append({**row, "reason": "unsafe_without_force"})
            continue
        p = Path(candidate.path)
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
        deleted.append(row)
    return {"deleted": deleted, "skipped": skipped}
