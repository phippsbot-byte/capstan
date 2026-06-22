from __future__ import annotations

from pathlib import Path
from typing import Any
import shutil
import time

from .http import http_json
from .manifest import ModelManifest
from .runner import active_pid, default_log_path, default_pid_path, pid_state_owner_mismatch, readiness_check, read_pid_state, start as start_model, stop
from .system import disk_free_gib, human_bytes, path_size_bytes, port_is_free, swap_used_gib


def validate(manifest: ModelManifest) -> dict[str, Any]:
    return {"id": manifest.id, "model_id": manifest.model_id, "endpoint": manifest.endpoint, "manifest": str(manifest.path), "has_start": manifest.start is not None, "required_paths": manifest.preflight.required_paths, "exclusive_ports": manifest.preflight.exclusive_ports, "cleanup_candidates": len(manifest.cleanup), "fleet": {"enabled": manifest.fleet.enabled, "reason": manifest.fleet.reason}, "health": {"max_swap_gib": manifest.health.max_swap_gib, "max_swap_delta_gib": manifest.health.max_swap_delta_gib, "sample_sec": manifest.health.sample_sec, "smoke": manifest.health.smoke, "max_latency_sec": manifest.health.max_latency_sec, "max_prompt_latency_sec": manifest.health.max_prompt_latency_sec, "max_completion_latency_sec": manifest.health.max_completion_latency_sec, "max_io_latency_sec": manifest.health.max_io_latency_sec}}


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


def _rounded(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def _effective_health_options(
    manifest: ModelManifest,
    *,
    max_swap_gib: float | None = None,
    max_swap_delta_gib: float | None = None,
    sample_sec: float | None = None,
    include_smoke: bool = False,
    max_latency_sec: float | None = None,
    max_prompt_latency_sec: float | None = None,
    max_completion_latency_sec: float | None = None,
) -> dict[str, Any]:
    return {
        "max_swap_gib": max_swap_gib if max_swap_gib is not None else (manifest.health.max_swap_gib if manifest.health.max_swap_gib is not None else manifest.preflight.max_swap_gib),
        "max_swap_delta_gib": max_swap_delta_gib if max_swap_delta_gib is not None else manifest.health.max_swap_delta_gib,
        "sample_sec": sample_sec if sample_sec is not None else manifest.health.sample_sec,
        "include_smoke": bool(include_smoke or manifest.health.smoke),
        "max_latency_sec": max_latency_sec if max_latency_sec is not None else manifest.health.max_latency_sec,
        "max_prompt_latency_sec": max_prompt_latency_sec if max_prompt_latency_sec is not None else manifest.health.max_prompt_latency_sec,
        "max_completion_latency_sec": max_completion_latency_sec if max_completion_latency_sec is not None else manifest.health.max_completion_latency_sec,
        "max_io_latency_sec": manifest.health.max_io_latency_sec,
    }


def _io_probe(manifest: ModelManifest, *, max_bytes: int = 4 * 1024 * 1024) -> dict[str, Any]:
    for raw_path in manifest.preflight.required_paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        started = time.perf_counter()
        bytes_read = 0
        with path.open("rb") as fh:
            data = fh.read(max_bytes)
            bytes_read = len(data)
        elapsed = time.perf_counter() - started
        return {"ok": True, "path": str(path), "bytes_read": bytes_read, "elapsed_s": round(elapsed, 6)}
    return {"ok": False, "error": "no file in preflight.required_paths to probe"}


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _timing_ms(timings: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number(timings.get(key))
        if value is not None:
            return value
    return None


def _latency_summary(body: Any, *, elapsed_s: float) -> tuple[dict[str, Any], dict[str, Any] | None]:
    raw_timings = body.get("timings") if isinstance(body, dict) and isinstance(body.get("timings"), dict) else None
    timings = dict(raw_timings) if raw_timings else None
    raw_usage = body.get("usage") if isinstance(body, dict) else None
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    prompt_ms = _timing_ms(timings or {}, "prompt_ms", "prompt_eval_ms")
    completion_ms = _timing_ms(timings or {}, "predicted_ms", "completion_ms", "eval_ms")
    latency: dict[str, Any] = {
        "wall_s": round(elapsed_s, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "server_prompt_s": None if prompt_ms is None else round(prompt_ms / 1000.0, 3),
        "server_completion_s": None if completion_ms is None else round(completion_ms / 1000.0, 3),
    }
    if timings:
        if "prompt_per_second" in timings:
            latency["prompt_tokens_per_sec"] = timings.get("prompt_per_second")
        if "predicted_per_second" in timings:
            latency["completion_tokens_per_sec"] = timings.get("predicted_per_second")
    return latency, timings


def health(
    manifest: ModelManifest,
    *,
    max_swap_gib: float | None = None,
    max_swap_delta_gib: float | None = None,
    sample_sec: float | None = None,
    include_smoke: bool = False,
    max_latency_sec: float | None = None,
    max_prompt_latency_sec: float | None = None,
    max_completion_latency_sec: float | None = None,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    warnings: list[str] = []
    options = _effective_health_options(
        manifest,
        max_swap_gib=max_swap_gib,
        max_swap_delta_gib=max_swap_delta_gib,
        sample_sec=sample_sec,
        include_smoke=include_smoke,
        max_latency_sec=max_latency_sec,
        max_prompt_latency_sec=max_prompt_latency_sec,
        max_completion_latency_sec=max_completion_latency_sec,
    )
    effective_max_swap_gib = options["max_swap_gib"]
    effective_max_swap_delta_gib = options["max_swap_delta_gib"]
    effective_sample_sec = float(options["sample_sec"] or 0.0)
    effective_include_smoke = bool(options["include_smoke"])
    effective_max_latency_sec = options["max_latency_sec"]
    effective_max_prompt_latency_sec = options["max_prompt_latency_sec"]
    effective_max_completion_latency_sec = options["max_completion_latency_sec"]
    effective_max_io_latency_sec = options["max_io_latency_sec"]

    def record(name: str, ok: bool, *, severity: str = "critical", **detail: Any) -> None:
        nonlocal checks, issues, warnings
        checks[name] = {**detail, "status": "ok" if ok else severity, "ok": ok}
        if not ok:
            if severity == "warn":
                warnings.append(name)
            else:
                issues.append(name)

    pid = active_pid(manifest)
    if manifest.start is not None:
        record("pid", pid is not None, pid=pid, pid_path=str(default_pid_path(manifest)))
    else:
        record("pid", True, severity="warn", pid=pid, message="manifest has no [start] section")

    try:
        ready = readiness_check(manifest, timeout=5)
        record("readiness", bool(ready.get("ready")), **ready)
    except Exception as exc:
        record("readiness", False, error=f"{type(exc).__name__}: {exc}")

    ceiling = effective_max_swap_gib
    swap_before = swap_used_gib()
    swap_after = swap_before
    if effective_sample_sec > 0:
        time.sleep(effective_sample_sec)
        swap_after = swap_used_gib()
    elif effective_max_swap_delta_gib is not None:
        swap_after = swap_used_gib()
    swap_delta = None if swap_before is None or swap_after is None else swap_after - swap_before
    swap_detail: dict[str, Any] = {
        "used_gib": _rounded(swap_after),
        "before_gib": _rounded(swap_before),
        "after_gib": _rounded(swap_after),
        "delta_gib": _rounded(swap_delta),
        "max_swap_gib": ceiling,
        "max_swap_delta_gib": effective_max_swap_delta_gib,
        "sample_sec": effective_sample_sec,
    }
    swap_ok = True
    if ceiling is not None and swap_after is not None and swap_after > ceiling:
        swap_ok = False
        swap_detail["breach"] = "absolute"
    if effective_max_swap_delta_gib is not None and swap_delta is not None and swap_delta > effective_max_swap_delta_gib:
        swap_ok = False
        swap_detail["breach"] = "delta" if swap_detail.get("breach") is None else "absolute_and_delta"
    record("swap", swap_ok, **swap_detail)

    if effective_max_io_latency_sec is not None:
        try:
            io_result = _io_probe(manifest)
            io_ok = bool(io_result.get("ok")) and float(io_result.get("elapsed_s", 0.0)) <= float(effective_max_io_latency_sec)
            io_detail = {k: v for k, v in io_result.items() if k != "ok"}
            record("io", io_ok, severity="warn", max_io_latency_sec=effective_max_io_latency_sec, **io_detail)
        except Exception as exc:
            record("io", False, severity="warn", max_io_latency_sec=effective_max_io_latency_sec, error=f"{type(exc).__name__}: {exc}")

    if effective_include_smoke:
        t0 = time.time()
        try:
            result = smoke(manifest)
            elapsed = time.time() - t0
            smoke_ok = bool(result.get("ok"))
            detail = {"elapsed_s": round(elapsed, 3), "result": {k: v for k, v in result.items() if k != "raw"}}
            record("smoke", smoke_ok, **detail)
            if effective_max_latency_sec is not None and elapsed > effective_max_latency_sec:
                checks["smoke_latency"] = {"status": "warn", "ok": False, "elapsed_s": round(elapsed, 3), "max_latency_sec": effective_max_latency_sec}
                warnings.append("smoke_latency")
            latency_candidate = result.get("latency")
            latency: dict[str, Any] = latency_candidate if isinstance(latency_candidate, dict) else {}
            server_prompt_s = latency.get("server_prompt_s")
            if effective_max_prompt_latency_sec is not None and isinstance(server_prompt_s, (int, float)) and server_prompt_s > effective_max_prompt_latency_sec:
                checks["smoke_prompt_latency"] = {
                    "status": "warn",
                    "ok": False,
                    "server_prompt_s": server_prompt_s,
                    "max_prompt_latency_sec": effective_max_prompt_latency_sec,
                    "latency": latency,
                }
                warnings.append("smoke_prompt_latency")
            server_completion_s = latency.get("server_completion_s")
            if effective_max_completion_latency_sec is not None and isinstance(server_completion_s, (int, float)) and server_completion_s > effective_max_completion_latency_sec:
                checks["smoke_completion_latency"] = {
                    "status": "warn",
                    "ok": False,
                    "server_completion_s": server_completion_s,
                    "max_completion_latency_sec": effective_max_completion_latency_sec,
                    "latency": latency,
                }
                warnings.append("smoke_completion_latency")
        except Exception as exc:
            elapsed = time.time() - t0
            record("smoke", False, elapsed_s=round(elapsed, 3), error=f"{type(exc).__name__}: {exc}")
    else:
        checks["smoke"] = {"status": "skipped", "ok": None}

    status_name = "critical" if issues else "warn" if warnings else "ok"
    return {
        "ok": status_name == "ok",
        "status": status_name,
        "id": manifest.id,
        "model_id": manifest.model_id,
        "endpoint": manifest.endpoint,
        "issues": issues,
        "warnings": warnings,
        "thresholds": {
            "max_swap_gib": effective_max_swap_gib,
            "max_swap_delta_gib": effective_max_swap_delta_gib,
            "sample_sec": effective_sample_sec,
            "smoke": effective_include_smoke,
            "max_latency_sec": effective_max_latency_sec,
            "max_prompt_latency_sec": effective_max_prompt_latency_sec,
            "max_completion_latency_sec": effective_max_completion_latency_sec,
            "max_io_latency_sec": effective_max_io_latency_sec,
        },
        "checks": checks,
    }


def smoke(manifest: ModelManifest, prompt: str | None = None, expect: str | None = None, max_tokens: int | None = None, temperature: float | None = None) -> dict[str, Any]:
    prompt_overridden = prompt is not None
    prompt = prompt if prompt is not None else manifest.smoke.prompt
    expect = expect if expect is not None else (None if prompt_overridden else manifest.smoke.expect)
    payload = {"model": manifest.model_id, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens if max_tokens is not None else manifest.smoke.max_tokens, "temperature": temperature if temperature is not None else manifest.smoke.temperature}
    started = time.perf_counter()
    status_code, body, _text = http_json("POST", manifest.chat_url, payload=payload, timeout=manifest.smoke.timeout_sec)
    elapsed_s = time.perf_counter() - started
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
    latency, timings = _latency_summary(body, elapsed_s=elapsed_s)
    return {"ok": 200 <= status_code < 300 and (exact is not False), "status": status_code, "content": content, "expect": expect, "exact": exact, "finish_reason": finish, "usage": usage, "elapsed_s": round(elapsed_s, 3), "latency": latency, "timings": timings, "raw": body}


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


def doctor_fix(manifest: ModelManifest) -> dict[str, Any]:
    fixes: list[dict[str, Any]] = []
    pid_path = default_pid_path(manifest)
    pid_state = read_pid_state(manifest)
    if pid_path.exists() and pid_state is None:
        pid_path.unlink()
        fixes.append({"code": "invalid_pid_state_removed", "path": str(pid_path)})
    elif pid_state and active_pid(manifest) is None:
        if pid_state_owner_mismatch(manifest, pid_state):
            fixes.append({"code": "pid_state_owner_mismatch_preserved", "path": str(pid_path), "pid_state": pid_state})
        else:
            pid_path.unlink()
            fixes.append({"code": "stale_pid_state_removed", "path": str(pid_path), "pid_state": pid_state})

    if manifest.start:
        for code, path in (("pid_dir_created", default_pid_path(manifest).parent), ("log_dir_created", default_log_path(manifest).parent)):
            existed = path.exists()
            path.mkdir(parents=True, exist_ok=True)
            if not existed:
                fixes.append({"code": code, "path": str(path)})

    result = doctor(manifest)
    result["fixes"] = fixes
    result["fixed"] = bool(fixes)
    return result


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
    ceiling = max_swap_gib if max_swap_gib is not None else (manifest.health.max_swap_gib if manifest.health.max_swap_gib is not None else manifest.preflight.max_swap_gib)
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
        if ceiling is not None and used is not None and used > ceiling:
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
    return {"ok": not breached, "breached": breached, "max_swap_gib": ceiling, "duration_sec": duration_sec, "interval_sec": interval_sec, "stop_on_breach": stop_on_breach, "stop_result": stop_result, "samples": samples}


def daemon(
    manifest: ModelManifest,
    max_swap_gib: float | None = None,
    interval_sec: float = 30.0,
    iterations: int | None = None,
    restart: bool = False,
    wait: bool = True,
    *,
    max_swap_delta_gib: float | None = None,
    sample_sec: float | None = None,
    include_smoke: bool = False,
    max_latency_sec: float | None = None,
    max_prompt_latency_sec: float | None = None,
    max_completion_latency_sec: float | None = None,
    health_mode: bool = False,
) -> dict[str, Any]:
    options = _effective_health_options(
        manifest,
        max_swap_gib=max_swap_gib,
        max_swap_delta_gib=max_swap_delta_gib,
        sample_sec=sample_sec,
        include_smoke=include_smoke,
        max_latency_sec=max_latency_sec,
        max_prompt_latency_sec=max_prompt_latency_sec,
        max_completion_latency_sec=max_completion_latency_sec,
    )
    effective_max_swap_gib = options["max_swap_gib"]
    effective_max_swap_delta_gib = options["max_swap_delta_gib"]
    effective_sample_sec = float(options["sample_sec"] or 0.0)
    effective_include_smoke = bool(options["include_smoke"])
    effective_max_latency_sec = options["max_latency_sec"]
    effective_max_prompt_latency_sec = options["max_prompt_latency_sec"]
    effective_max_completion_latency_sec = options["max_completion_latency_sec"]
    effective_health_mode = bool(
        health_mode
        or effective_max_swap_delta_gib is not None
        or effective_sample_sec > 0
        or effective_include_smoke
        or effective_max_latency_sec is not None
        or effective_max_prompt_latency_sec is not None
        or effective_max_completion_latency_sec is not None
        or options["max_io_latency_sec"] is not None
    )
    rows: list[dict[str, Any]] = []
    index = 0
    ok = True
    while True:
        index += 1
        if effective_health_mode:
            sample = health(
                manifest,
                max_swap_gib=effective_max_swap_gib,
                max_swap_delta_gib=effective_max_swap_delta_gib,
                sample_sec=effective_sample_sec,
                include_smoke=effective_include_smoke,
                max_latency_sec=effective_max_latency_sec,
                max_prompt_latency_sec=effective_max_prompt_latency_sec,
                max_completion_latency_sec=effective_max_completion_latency_sec,
            )
            first = sample
            breached = not bool(sample.get("ok"))
            row_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        else:
            sample = watchdog(manifest, max_swap_gib=effective_max_swap_gib, duration_sec=0, interval_sec=interval_sec, stop_on_breach=False)
            first = sample["samples"][0] if sample.get("samples") else {}
            breached = bool(sample.get("breached"))
            row_time = first.get("time")
        action: dict[str, Any] | None = None
        if breached:
            ok = False
            if restart:
                if manifest.start is None:
                    action = {"type": "restart_skipped", "reason": "manifest_has_no_start"}
                else:
                    stopped = stop(manifest)
                    started = start_model(manifest, wait=wait)
                    action = {"type": "restart", "stop": stopped, "start": started}
                    # If restart restored readiness, keep the daemon result green.
                    readiness = started.get("readiness") if isinstance(started, dict) else None
                    if isinstance(readiness, dict) and readiness.get("ready"):
                        ok = True
        rows.append({"index": index, "time": row_time, "breached": breached, "sample": first, "action": action})
        if iterations is not None and index >= iterations:
            break
        time.sleep(max(0.1, interval_sec))
    return {"ok": ok, "iterations_requested": iterations, "iterations_completed": len(rows), "restart": restart, "health_mode": effective_health_mode, "max_swap_gib": effective_max_swap_gib, "max_swap_delta_gib": effective_max_swap_delta_gib, "sample_sec": effective_sample_sec, "include_smoke": effective_include_smoke, "max_latency_sec": effective_max_latency_sec, "max_prompt_latency_sec": effective_max_prompt_latency_sec, "max_completion_latency_sec": effective_max_completion_latency_sec, "interval_sec": interval_sec, "iterations": rows}
