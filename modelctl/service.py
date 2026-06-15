from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import plistlib
import platform
import re
import subprocess
import sys

from .manifest import ModelManifest

DEFAULT_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ServiceError(ValueError):
    """Raised for invalid service configuration or service action input."""


def slug(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-.")
    return text or "model"


def default_label(manifest: ModelManifest) -> str:
    return f"ai.modelctl.{slug(manifest.id)}"


def validate_label(label: str) -> str:
    if not label or not LABEL_RE.fullmatch(label) or ".." in label:
        raise ServiceError("launchd label must match [A-Za-z0-9][A-Za-z0-9_.-]* and must not contain '..'")
    return label


def resolve_label(manifest: ModelManifest, label: str | None = None) -> str:
    return validate_label(label or default_label(manifest))


def launchd_dir() -> Path:
    override = os.environ.get("MODELCTL_LAUNCHD_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "LaunchAgents"


def default_service_log_path(label: str) -> Path:
    label = validate_label(label)
    state_dir = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "modelctl"
    return state_dir / f"{label}.service.log"


def service_plist_path(label: str) -> Path:
    label = validate_label(label)
    return launchd_dir() / f"{label}.plist"


def _fmt_number(value: float | int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return str(value)


def daemon_program_arguments(
    manifest: ModelManifest,
    *,
    restart: bool = False,
    max_swap_gib: float | None = None,
    interval_sec: float = 30.0,
    python: str | None = None,
    wait: bool = True,
) -> list[str]:
    if restart and manifest.start is None:
        raise ServiceError("--restart requires a manifest [start] section")
    args = [
        python or sys.executable,
        "-m",
        "modelctl.cli",
        "-m",
        str(manifest.path),
        "daemon",
        "--interval",
        _fmt_number(interval_sec),
    ]
    ceiling = max_swap_gib if max_swap_gib is not None else manifest.preflight.max_swap_gib
    if ceiling is not None:
        args.extend(["--max-swap-gib", _fmt_number(ceiling)])
    if restart:
        args.append("--restart")
    if not wait:
        args.append("--no-wait")
    return args


def render_launchd_plist(
    manifest: ModelManifest,
    *,
    label: str | None = None,
    restart: bool = False,
    max_swap_gib: float | None = None,
    interval_sec: float = 30.0,
    python: str | None = None,
    keep_alive: bool = True,
    run_at_load: bool = False,
    service_log_path: str | None = None,
    wait: bool = True,
) -> dict[str, Any]:
    service_label = resolve_label(manifest, label)
    log_path = Path(service_log_path).expanduser() if service_log_path else default_service_log_path(service_label)
    err_path = log_path.with_suffix(log_path.suffix + ".err") if log_path.suffix else Path(str(log_path) + ".err")
    program_args = daemon_program_arguments(
        manifest,
        restart=restart,
        max_swap_gib=max_swap_gib,
        interval_sec=interval_sec,
        python=python,
        wait=wait,
    )
    plist: dict[str, Any] = {
        "Label": service_label,
        "ProgramArguments": program_args,
        "WorkingDirectory": str(manifest.path.parent),
        "StandardOutPath": str(log_path),
        "StandardErrorPath": str(err_path),
        "RunAtLoad": bool(run_at_load),
        "KeepAlive": bool(keep_alive),
        "ThrottleInterval": 10,
        "ProcessType": "Background",
        "EnvironmentVariables": {
            "PATH": os.environ.get("MODELCTL_SERVICE_PATH", DEFAULT_PATH),
            "MODELCTL_MANIFEST": str(manifest.path),
        },
    }
    xml = plistlib.dumps(plist, sort_keys=False).decode("utf-8")
    return {
        "label": service_label,
        "plist": plist,
        "plist_xml": xml,
        "program_arguments": program_args,
        "service_log_path": str(log_path),
        "service_error_log_path": str(err_path),
    }


def install_service(
    manifest: ModelManifest,
    *,
    label: str | None = None,
    restart: bool = False,
    max_swap_gib: float | None = None,
    interval_sec: float = 30.0,
    python: str | None = None,
    keep_alive: bool = True,
    run_at_load: bool = False,
    service_log_path: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    wait: bool = True,
) -> dict[str, Any]:
    rendered = render_launchd_plist(
        manifest,
        label=label,
        restart=restart,
        max_swap_gib=max_swap_gib,
        interval_sec=interval_sec,
        python=python,
        keep_alive=keep_alive,
        run_at_load=run_at_load,
        service_log_path=service_log_path,
        wait=wait,
    )
    service_label = rendered["label"]
    plist_path = service_plist_path(service_label)
    result = {
        "ok": True,
        "action": "install",
        "label": service_label,
        "plist_path": str(plist_path),
        "written": False,
        "dry_run": dry_run,
        "program_arguments": rendered["program_arguments"],
        "service_log_path": rendered["service_log_path"],
        "service_error_log_path": rendered["service_error_log_path"],
        "plist": rendered["plist"],
    }
    if plist_path.exists() and not overwrite:
        return {**result, "ok": False, "error": "plist_exists", "hint": "pass --overwrite to replace it"}
    if dry_run:
        return result
    if platform.system() != "Darwin" and not os.environ.get("MODELCTL_LAUNCHD_DIR"):
        return {**result, "ok": False, "error": "launchd_only", "platform": platform.system()}
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    Path(rendered["service_log_path"]).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(rendered["service_error_log_path"]).expanduser().parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(rendered["plist_xml"], encoding="utf-8")
    return {**result, "written": True}


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _target(label: str) -> str:
    return f"{_domain()}/{label}"


def _commands(action: str, label: str, plist_path: Path) -> list[list[str]]:
    label = validate_label(label)
    target = _target(label)
    if action == "status":
        return [["launchctl", "print", target]]
    if action == "start":
        return [["launchctl", "bootstrap", _domain(), str(plist_path)], ["launchctl", "kickstart", "-k", target]]
    if action == "stop":
        return [["launchctl", "bootout", target]]
    if action == "restart":
        return [["launchctl", "bootout", target], ["launchctl", "bootstrap", _domain(), str(plist_path)], ["launchctl", "kickstart", "-k", target]]
    if action == "uninstall":
        return [["launchctl", "bootout", target]]
    raise ServiceError(f"unknown service action: {action}")


def _benign_bootout_failure(row: dict[str, Any]) -> bool:
    if row.get("returncode") == 0:
        return True
    text = f"{row.get('stdout', '')}\n{row.get('stderr', '')}".lower()
    markers = ("not found", "could not find", "no such process", "no such service", "service is not loaded", "does not exist")
    return any(marker in text for marker in markers)


def service_action(manifest: ModelManifest, action: str, *, label: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    service_label = resolve_label(manifest, label)
    plist_path = service_plist_path(service_label)
    commands = _commands(action, service_label, plist_path)
    base: dict[str, Any] = {
        "ok": True,
        "action": action,
        "label": service_label,
        "plist_path": str(plist_path),
        "loaded_target": _target(service_label),
        "dry_run": dry_run,
        "commands": commands,
        "plist_exists": plist_path.exists(),
    }
    if dry_run:
        if action == "uninstall":
            base["would_remove"] = str(plist_path)
        return base
    if platform.system() != "Darwin":
        return {**base, "ok": False, "error": "launchd_only", "platform": platform.system()}
    if action in {"start", "restart"} and not plist_path.exists():
        return {**base, "ok": False, "error": "plist_missing"}

    rows: list[dict[str, Any]] = []
    ok = True
    for cmd in commands:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        row = {"command": cmd, "returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
        rows.append(row)
        tolerated = False
        # bootout is allowed to fail only when launchd clearly says the service is absent.
        if cmd[1] == "bootout" and action in {"restart", "uninstall"} and _benign_bootout_failure(row):
            tolerated = True
        # bootstrap can fail if the service is already loaded; kickstart below is the real start signal.
        if cmd[1] == "bootstrap" and action == "start":
            tolerated = True
        if proc.returncode != 0 and not tolerated:
            ok = False
            break
    removed = False
    if action == "uninstall" and plist_path.exists():
        plist_path.unlink()
        removed = True
    return {**base, "ok": ok, "results": rows, "removed": removed, "plist_exists": plist_path.exists()}
