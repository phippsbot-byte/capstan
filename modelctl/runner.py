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


def write_pid_state(manifest: ModelManifest, state: dict[str, Any]) -> Path:
    path = default_pid_path(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return path


def active_pid(manifest: ModelManifest) -> int | None:
    state = read_pid_state(manifest)
    if not state:
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


def wait_ready(manifest: ModelManifest, timeout_sec: int | None = None) -> dict[str, Any]:
    if timeout_sec is None:
        timeout_sec = manifest.start.startup_timeout_sec if manifest.start else 120
    deadline = time.time() + timeout_sec
    last: dict[str, Any] | None = None
    while time.time() < deadline:
        pid = active_pid(manifest)
        if manifest.start and pid is None:
            return {"ready": False, "error": "process exited before readiness", "last": last}
        try:
            last = readiness_check(manifest, timeout=5)
            if last.get("ready"):
                return last
        except Exception as exc:
            last = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
        time.sleep(2)
    return {"ready": False, "error": "timeout", "last": last}


def start(manifest: ModelManifest, wait: bool = False) -> dict[str, Any]:
    if not manifest.start:
        raise RuntimeError("manifest has no [start] section")
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
    log = log_path.open("ab", buffering=0)
    proc = subprocess.Popen(manifest.start.command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    state = {"pid": proc.pid, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "command": manifest.start.command, "cwd": cwd, "log_path": str(log_path), "manifest": str(manifest.path)}
    pid_path = write_pid_state(manifest, state)
    result = {"started": True, "pid": proc.pid, "pid_path": str(pid_path), "log_path": str(log_path)}
    if wait:
        result["readiness"] = wait_ready(manifest)
    return result


def stop(manifest: ModelManifest, timeout_sec: int = 10) -> dict[str, Any]:
    pid = active_pid(manifest)
    pid_path = default_pid_path(manifest)
    if pid is None:
        if pid_path.exists():
            pid_path.unlink()
        return {"stopped": False, "already_stopped": True, "pid_path_removed": str(pid_path)}
    ok = terminate_process_group(pid, timeout_sec=timeout_sec)
    if ok and pid_path.exists():
        pid_path.unlink()
    return {"stopped": ok, "pid": pid, "pid_path": str(pid_path)}
