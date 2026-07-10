from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import subprocess
import time

from .http import http_json
from .manifest import ModelManifest
from .system import pid_alive, terminate_process_group


def default_pid_path(manifest: ModelManifest) -> Path:
    if manifest.start and manifest.start.pid_path:
        return Path(manifest.start.pid_path)
    state_dir = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "modelctl"
    return state_dir / f"{manifest.id}.pid.json"


def default_log_path(manifest: ModelManifest) -> Path:
    if manifest.start and manifest.start.log_path:
        return Path(manifest.start.log_path)
    state_dir = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "modelctl"
    return state_dir / f"{manifest.id}.log"


def read_pid_state(manifest: ModelManifest) -> dict[str, Any] | None:
    path = default_pid_path(manifest)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def pid_state_owner_mismatch(manifest: ModelManifest, state: dict[str, Any] | None = None) -> bool:
    state = read_pid_state(manifest) if state is None else state
    if not state:
        return False
    state_manifest = state.get("manifest")
    return isinstance(state_manifest, str) and state_manifest != str(manifest.path)


def write_pid_state(manifest: ModelManifest, state: dict[str, Any]) -> Path:
    path = default_pid_path(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, indent=2)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def active_pid(manifest: ModelManifest) -> int | None:
    state = read_pid_state(manifest)
    if not state:
        return None
    if pid_state_owner_mismatch(manifest, state):
        return None
    pid = state.get("pid")
    if isinstance(pid, int) and pid_alive(pid):
        return pid
    return None


def readiness_check(manifest: ModelManifest, timeout: float = 10.0) -> dict[str, Any]:
    url = manifest.start.readiness_url if manifest.start and manifest.start.readiness_url else manifest.models_url
    contains = manifest.start.readiness_contains if manifest.start else manifest.model_id
    status, body, text = http_json("GET", url, timeout=timeout)
    ready = 200 <= status < 300 and (not contains or contains in text)
    return {"ready": ready, "status": status, "url": url, "contains": contains, "body": body if isinstance(body, dict) else text[:500]}


def wait_ready(manifest: ModelManifest, timeout_sec: float | None = None) -> dict[str, Any]:
    if timeout_sec is None:
        timeout_sec = manifest.start.startup_timeout_sec if manifest.start else 120
    deadline = time.time() + timeout_sec
    last: dict[str, Any] | None = None
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        pid = active_pid(manifest)
        if manifest.start and pid is None:
            return {"ready": False, "error": "process exited before readiness", "last": last}
        try:
            last = readiness_check(manifest, timeout=max(0.001, min(5.0, remaining)))
            if last.get("ready"):
                return last
        except Exception as exc:
            last = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(2.0, max(0.05, remaining)))
    return {"ready": False, "error": "timeout", "last": last}


def start(manifest: ModelManifest, wait: bool = False) -> dict[str, Any]:
    if not manifest.start:
        raise RuntimeError("manifest has no [start] section")
    state = read_pid_state(manifest)
    if pid_state_owner_mismatch(manifest, state):
        raise RuntimeError(f"pid state at {default_pid_path(manifest)} is owned by another manifest")
    existing = active_pid(manifest)
    if existing is not None:
        result: dict[str, Any] = {"started": False, "already_running": True, "pid": existing, "pid_path": str(default_pid_path(manifest))}
        if wait:
            result["readiness"] = wait_ready(manifest)
        return result

    log_path = default_log_path(manifest)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(manifest.start.env)
    cwd = manifest.start.cwd or str(manifest.path.parent)
    with log_path.open("ab", buffering=0) as log:
        proc = subprocess.Popen(manifest.start.command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    state = {"pid": proc.pid, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "command": manifest.start.command, "cwd": cwd, "log_path": str(log_path), "manifest": str(manifest.path)}
    try:
        pid_path = write_pid_state(manifest, state)
    except Exception:
        terminate_process_group(proc.pid, timeout_sec=5)
        raise
    result = {"started": True, "pid": proc.pid, "pid_path": str(pid_path), "log_path": str(log_path)}
    if wait:
        result["readiness"] = wait_ready(manifest)
    return result


def stop(manifest: ModelManifest, timeout_sec: int = 10) -> dict[str, Any]:
    state = read_pid_state(manifest)
    pid = active_pid(manifest)
    pid_path = default_pid_path(manifest)
    if pid is None:
        if pid_path.exists():
            if pid_state_owner_mismatch(manifest, state):
                return {"ok": False, "stopped": False, "already_stopped": False, "owner_mismatch": True, "safe_to_start": False, "pid_path": str(pid_path), "pid_state": state}
            pid_path.unlink()
        return {"ok": True, "stopped": False, "already_stopped": True, "safe_to_start": True, "pid_path_removed": str(pid_path)}
    ok = terminate_process_group(pid, timeout_sec=timeout_sec)
    if ok and pid_path.exists():
        pid_path.unlink()
    return {"ok": ok, "stopped": ok, "pid": pid, "known_pid_stopped": ok, "safe_to_start": ok, "unexpected_active_pid": None if ok else pid, "pid_path": str(pid_path)}


def _manifest_ref(manifest: ModelManifest) -> dict[str, Any]:
    return {"id": manifest.id, "model_id": manifest.model_id, "endpoint": manifest.endpoint, "path": str(manifest.path)}


def rotate(
    current: ModelManifest,
    target: ModelManifest,
    *,
    readiness_timeout_sec: float | None = None,
    stop_timeout_sec: int = 10,
    rollback: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rotate from current manifest process to target with readiness-gated PID ownership handoff."""
    base: dict[str, Any] = {
        "ok": False,
        "action": "rotate",
        "from": _manifest_ref(current),
        "to": _manifest_ref(target),
        "readiness_timeout_sec": readiness_timeout_sec,
        "stop_timeout_sec": stop_timeout_sec,
        "rollback_enabled": rollback,
        "dry_run": dry_run,
    }
    if current.start is None:
        return {**base, "status": "invalid_request", "error": "current manifest has no [start] section", "issues": ["current_missing_start"]}
    if target.start is None:
        return {**base, "status": "invalid_request", "error": "target manifest has no [start] section", "issues": ["target_missing_start"]}
    if current.endpoint != target.endpoint or current.model_id != target.model_id:
        return {
            **base,
            "status": "invalid_request",
            "error": "target must preserve current endpoint and model_id for stable-lane rotation",
            "issues": ["target_identity_mismatch"],
        }
    timeout = readiness_timeout_sec if readiness_timeout_sec is not None else target.start.startup_timeout_sec
    base["readiness_timeout_sec"] = timeout
    current_pid_path = default_pid_path(current)
    target_pid_path = default_pid_path(target)
    plan = {
        "steps": ["stop_current", "start_target", "verify_target_readiness", "atomically_handoff_pid_state"],
        "current_pid_path": str(current_pid_path),
        "target_pid_path": str(target_pid_path),
    }
    if dry_run:
        return {**base, "ok": True, "status": "planned", "plan": plan}

    old_pid = active_pid(current)
    stop_current = stop(current, timeout_sec=stop_timeout_sec)
    if stop_current.get("owner_mismatch") or (old_pid is not None and not stop_current.get("stopped")):
        return {
            **base,
            "status": "current_stop_failed",
            "old_pid": old_pid,
            "stop_current": stop_current,
            "issues": ["current_stop_failed"],
        }

    def rollback_current() -> dict[str, Any]:
        if not rollback:
            return {"attempted": False}
        try:
            rollback_start = start(current, wait=False)
            rollback_timeout = current.start.startup_timeout_sec if current.start else timeout
            rollback_readiness = wait_ready(current, timeout_sec=rollback_timeout)
            return {"attempted": True, "start": rollback_start, "readiness": rollback_readiness}
        except Exception as exc:
            return {"attempted": True, "error": f"{type(exc).__name__}: {exc}"}

    try:
        target_start = start(target, wait=False)
    except Exception as exc:
        rollback_result = rollback_current()
        return {
            **base,
            "status": "target_start_failed",
            "old_pid": old_pid,
            "stop_current": stop_current,
            "error": f"{type(exc).__name__}: {exc}",
            "rollback": rollback_result,
            "issues": ["target_start_failed"],
        }
    readiness = wait_ready(target, timeout_sec=timeout)
    if not readiness.get("ready"):
        target_stop = stop(target, timeout_sec=stop_timeout_sec)
        rollback_result = rollback_current()
        return {
            **base,
            "status": "target_not_ready",
            "old_pid": old_pid,
            "stop_current": stop_current,
            "target_start": target_start,
            "readiness": readiness,
            "target_stop": target_stop,
            "rollback": rollback_result,
            "issues": ["target_not_ready"],
        }

    target_state = read_pid_state(target)
    expected_pid = target_start.get("pid") if isinstance(target_start, dict) else None
    target_state_valid = bool(target_state) and isinstance(target_state.get("pid") if target_state else None, int)
    if target_state_valid and isinstance(expected_pid, int) and target_state and target_state.get("pid") != expected_pid:
        target_state_valid = False
    if target_state_valid and target_state and target_state.get("manifest") != str(target.path):
        target_state_valid = False
    if not target_state_valid or target_state is None:
        target_stop = stop(target, timeout_sec=stop_timeout_sec)
        rollback_result = rollback_current()
        return {
            **base,
            "status": "target_pid_state_missing",
            "old_pid": old_pid,
            "stop_current": stop_current,
            "target_start": target_start,
            "readiness": readiness,
            "target_stop": target_stop,
            "rollback": rollback_result,
            "issues": ["target_pid_state_missing"],
        }

    handoff_state = dict(target_state)
    handoff_state["manifest"] = str(current.path)
    handoff_state["source_manifest"] = str(target.path)
    handoff_state["rotated_from"] = _manifest_ref(current)
    handoff_state["rotated_to"] = _manifest_ref(target)
    handoff_state["rotated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    handoff_state["source_pid_path"] = str(target_pid_path)
    handoff_state["owner_pid_path"] = str(current_pid_path)
    try:
        if target_pid_path == current_pid_path:
            write_pid_state(current, handoff_state)
            removed = False
        else:
            current_pid_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = current_pid_path.with_name(f".{current_pid_path.name}.{os.getpid()}.rotate")
            try:
                tmp.write_text(json.dumps(handoff_state, indent=2), encoding="utf-8")
                os.replace(tmp, current_pid_path)
            finally:
                if tmp.exists():
                    tmp.unlink()
            target_pid_path.unlink()
            removed = True
    except Exception as exc:
        target_stop = stop(target, timeout_sec=stop_timeout_sec)
        rollback_result = rollback_current()
        return {
            **base,
            "status": "handoff_failed",
            "old_pid": old_pid,
            "stop_current": stop_current,
            "target_start": target_start,
            "readiness": readiness,
            "target_stop": target_stop,
            "rollback": rollback_result,
            "error": f"{type(exc).__name__}: {exc}",
            "issues": ["handoff_failed"],
        }

    return {
        **base,
        "ok": True,
        "status": "rotated",
        "old_pid": old_pid,
        "new_pid": handoff_state.get("pid"),
        "stop_current": stop_current,
        "target_start": target_start,
        "readiness": readiness,
        "handoff": {
            "current_pid_path": str(current_pid_path),
            "target_pid_path": str(target_pid_path),
            "target_pid_state_removed": removed,
            "atomic": True,
        },
    }
