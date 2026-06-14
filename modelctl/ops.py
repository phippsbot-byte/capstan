from __future__ import annotations

from pathlib import Path
from typing import Any
import shutil
import time

from .http import http_json
from .manifest import ModelManifest
from .runner import active_pid, default_log_path, default_pid_path, readiness_check, read_pid_state, stop
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


def soak(manifest: ModelManifest, count: int = 3, delay_sec: float = 0.0, fail_fast: bool = True) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    swap_before = swap_used_gib()
    started = time.time()
    for idx in range(1, count + 1):
        before = swap_used_gib()
        t0 = time.time()
        result = smoke(manifest)
        elapsed = time.time() - t0
        after = swap_used_gib()
        row = {
            "index": idx,
            "ok": result.get("ok"),
            "exact": result.get("exact"),
            "status": result.get("status"),
            "elapsed_s": round(elapsed, 3),
            "swap_before_gib": None if before is None else round(before, 3),
            "swap_after_gib": None if after is None else round(after, 3),
            "swap_delta_gib": None if before is None or after is None else round(after - before, 3),
            "usage": result.get("usage"),
            "content_preview": (result.get("content") or "")[:200],
        }
        runs.append(row)
        if fail_fast and not row["ok"]:
            break
        if delay_sec > 0 and idx < count:
            time.sleep(delay_sec)
    swap_after = swap_used_gib()
    elapsed_values = [r["elapsed_s"] for r in runs]
    return {
        "ok": bool(runs) and all(bool(r.get("ok")) for r in runs) and len(runs) == count,
        "requested_count": count,
        "completed_count": len(runs),
        "total_elapsed_s": round(time.time() - started, 3),
        "min_elapsed_s": min(elapsed_values) if elapsed_values else None,
        "max_elapsed_s": max(elapsed_values) if elapsed_values else None,
        "swap_before_gib": None if swap_before is None else round(swap_before, 3),
        "swap_after_gib": None if swap_after is None else round(swap_after, 3),
        "swap_delta_gib": None if swap_before is None or swap_after is None else round(swap_after - swap_before, 3),
        "runs": runs,
    }


def _tail_text(path: Path, max_bytes: int = 4096) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    with path.open("rb") as fh:
        try:
            fh.seek(max(0, path.stat().st_size - max_bytes))
        except OSError:
            return None
        return fh.read().decode("utf-8", "replace")


def doctor(manifest: ModelManifest) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    pf = preflight(manifest)
    st = status(manifest)
    plan = cleanup_plan(manifest)

    if not pf.get("ok"):
        issues.append({"code": "preflight_failed", "detail": pf})
    if manifest.start is None:
        warnings.append({"code": "no_start_section", "message": "Manifest can inspect/smoke an existing endpoint but cannot start it."})
    pid_state = read_pid_state(manifest)
    if pid_state and active_pid(manifest) is None:
        warnings.append({"code": "stale_pid_state", "pid_state": pid_state})
    readiness = st.get("readiness") or {}
    if not readiness.get("ready"):
        warnings.append({"code": "endpoint_not_ready", "detail": readiness})
    unsafe = [c for c in plan["candidates"] if c.get("exists") and not c.get("safe")]
    if unsafe:
        warnings.append({"code": "unsafe_cleanup_candidates", "count": len(unsafe), "candidates": unsafe})
    if manifest.start:
        log_path = default_log_path(manifest)
        if not log_path.exists():
            warnings.append({"code": "log_missing", "path": str(log_path)})
        else:
            st["log_tail"] = _tail_text(log_path)

    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "preflight": pf,
        "status": st,
        "cleanup": plan,
    }


def _synthetic_prompt(prompt_chars: int) -> str:
    prefix = "Read the filler text, then reply exactly BENCH_OK.\n\nFILLER:\n"
    suffix = "\n\nReply exactly BENCH_OK."
    filler_len = max(0, prompt_chars - len(prefix) - len(suffix))
    pattern = "alpha beta gamma delta epsilon zeta eta theta "
    filler = (pattern * ((filler_len // len(pattern)) + 1))[:filler_len]
    return prefix + filler + suffix


def bench(manifest: ModelManifest, prompt_chars: list[int], repeats: int = 1, max_tokens: int = 16) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    swap_before = swap_used_gib()
    started = time.time()
    for chars in prompt_chars:
        for repeat in range(1, repeats + 1):
            prompt = _synthetic_prompt(chars)
            before = swap_used_gib()
            t0 = time.time()
            result = smoke(manifest, prompt=prompt, expect="BENCH_OK", max_tokens=max_tokens, temperature=0)
            elapsed = time.time() - t0
            after = swap_used_gib()
            raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
            rows.append({
                "prompt_chars": chars,
                "repeat": repeat,
                "ok": result.get("ok"),
                "exact": result.get("exact"),
                "status": result.get("status"),
                "elapsed_s": round(elapsed, 3),
                "usage": result.get("usage"),
                "timings": raw.get("timings") if isinstance(raw, dict) else None,
                "swap_before_gib": None if before is None else round(before, 3),
                "swap_after_gib": None if after is None else round(after, 3),
                "swap_delta_gib": None if before is None or after is None else round(after - before, 3),
                "content_preview": (result.get("content") or "")[:200],
            })
    swap_after = swap_used_gib()
    return {
        "ok": bool(rows) and all(bool(r.get("ok")) for r in rows),
        "prompt_chars": prompt_chars,
        "repeats": repeats,
        "total_elapsed_s": round(time.time() - started, 3),
        "swap_before_gib": None if swap_before is None else round(swap_before, 3),
        "swap_after_gib": None if swap_after is None else round(swap_after, 3),
        "swap_delta_gib": None if swap_before is None or swap_after is None else round(swap_after - swap_before, 3),
        "runs": rows,
    }


def watchdog(manifest: ModelManifest, max_swap_gib: float | None = None, duration_sec: float = 0.0, interval_sec: float = 10.0, stop_on_breach: bool = False) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    deadline = time.time() + max(0.0, duration_sec)
    breached = False
    stop_result: dict[str, Any] | None = None
    while True:
        used = swap_used_gib()
        ready = False
        readiness_error = None
        try:
            ready = bool(readiness_check(manifest, timeout=5).get("ready"))
        except Exception as exc:
            readiness_error = f"{type(exc).__name__}: {exc}"
        sample = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pid": active_pid(manifest),
            "ready": ready,
            "swap_used_gib": None if used is None else round(used, 3),
            "readiness_error": readiness_error,
        }
        if max_swap_gib is not None and used is not None and used > max_swap_gib:
            sample["breach"] = "swap"
            breached = True
        if readiness_error is not None or not ready:
            sample["breach"] = sample.get("breach") or "readiness"
            breached = True
        samples.append(sample)
        if breached:
            if stop_on_breach:
                stop_result = stop(manifest)
            break
        if duration_sec <= 0 or time.time() >= deadline:
            break
        time.sleep(max(0.1, interval_sec))
    return {"ok": not breached, "breached": breached, "max_swap_gib": max_swap_gib, "duration_sec": duration_sec, "interval_sec": interval_sec, "stop_on_breach": stop_on_breach, "stop_result": stop_result, "samples": samples}
