from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os
import tomllib


class ManifestError(ValueError):
    """Raised when a manifest is invalid."""


def expand(value: str | None) -> str | None:
    if value is None:
        return None
    return os.path.expandvars(os.path.expanduser(value))


def expand_list(values: list[str]) -> list[str]:
    return [expand(v) or "" for v in values]


@dataclass(slots=True)
class DiskCheck:
    path: str
    min_free_gib: float


@dataclass(slots=True)
class CleanupCandidate:
    path: str
    description: str = ""
    safe: bool = False


@dataclass(slots=True)
class StartConfig:
    command: list[str]
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    log_path: str | None = None
    pid_path: str | None = None
    startup_timeout_sec: int = 120
    readiness_url: str | None = None
    readiness_contains: str | None = None


@dataclass(slots=True)
class SmokeConfig:
    prompt: str = "Reply with exactly the word pong."
    expect: str | None = "pong"
    max_tokens: int = 32
    temperature: float = 0.0
    timeout_sec: int = 300


@dataclass(slots=True)
class PreflightConfig:
    required_paths: list[str] = field(default_factory=list)
    exclusive_ports: list[int] = field(default_factory=list)
    max_swap_gib: float | None = None
    disk: list[DiskCheck] = field(default_factory=list)


@dataclass(slots=True)
class ModelManifest:
    path: Path
    id: str
    model_id: str
    endpoint: str
    description: str = ""
    start: StartConfig | None = None
    preflight: PreflightConfig = field(default_factory=PreflightConfig)
    smoke: SmokeConfig = field(default_factory=SmokeConfig)
    cleanup: list[CleanupCandidate] = field(default_factory=list)

    @property
    def models_url(self) -> str:
        return self.endpoint.rstrip("/") + "/models"

    @property
    def chat_url(self) -> str:
        return self.endpoint.rstrip("/") + "/chat/completions"


def _as_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ManifestError(f"[{key}] must be a TOML table")
    return value


def _as_list(value: Any, key: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ManifestError(f"{key} must be a list")
    return value


def _as_str_list(value: Any, key: str) -> list[str]:
    items = _as_list(value, key)
    if not all(isinstance(x, str) for x in items):
        raise ManifestError(f"{key} must contain only strings")
    return list(items)


def _as_int_list(value: Any, key: str) -> list[int]:
    items = _as_list(value, key)
    if not all(isinstance(x, int) for x in items):
        raise ManifestError(f"{key} must contain only integers")
    return list(items)


def load_manifest(path: str | Path) -> ModelManifest:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ManifestError(f"manifest not found: {p}")
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"invalid TOML in {p}: {exc}") from exc
    model = _as_table(data, "model")
    try:
        model_id = str(model["model_id"])
        endpoint = str(model["endpoint"])
    except KeyError as exc:
        raise ManifestError(f"[model].{exc.args[0]} is required") from exc
    ident = str(model.get("id") or model_id)

    start_cfg: StartConfig | None = None
    if "start" in data:
        start = _as_table(data, "start")
        command_raw = start.get("command")
        if isinstance(command_raw, str):
            command = ["bash", "-lc", command_raw]
        elif isinstance(command_raw, list) and all(isinstance(x, str) for x in command_raw):
            command = list(command_raw)
        else:
            raise ManifestError("[start].command must be a string or list of strings")
        env_raw = start.get("env", {})
        if not isinstance(env_raw, dict):
            raise ManifestError("[start.env] must be a TOML table")
        env = {str(k): expand(str(v)) or "" for k, v in env_raw.items()}
        start_cfg = StartConfig(
            command=expand_list(command),
            cwd=expand(start.get("cwd")),
            env=env,
            log_path=expand(start.get("log_path")),
            pid_path=expand(start.get("pid_path")),
            startup_timeout_sec=int(start.get("startup_timeout_sec", 120)),
            readiness_url=expand(start.get("readiness_url")),
            readiness_contains=start.get("readiness_contains"),
        )

    pre = _as_table(data, "preflight") if "preflight" in data else {}
    disk_checks: list[DiskCheck] = []
    for idx, row in enumerate(_as_list(pre.get("disk"), "preflight.disk")):
        if not isinstance(row, dict):
            raise ManifestError(f"[[preflight.disk]] row {idx} must be a table")
        if "path" not in row or "min_free_gib" not in row:
            raise ManifestError("[[preflight.disk]] requires path and min_free_gib")
        disk_checks.append(DiskCheck(path=expand(str(row["path"])) or "", min_free_gib=float(row["min_free_gib"])))
    preflight = PreflightConfig(
        required_paths=expand_list(_as_str_list(pre.get("required_paths"), "preflight.required_paths")),
        exclusive_ports=_as_int_list(pre.get("exclusive_ports"), "preflight.exclusive_ports"),
        max_swap_gib=float(pre["max_swap_gib"]) if "max_swap_gib" in pre else None,
        disk=disk_checks,
    )

    smoke_raw = _as_table(data, "smoke") if "smoke" in data else {}
    smoke = SmokeConfig(
        prompt=str(smoke_raw.get("prompt", SmokeConfig.prompt)),
        expect=(None if smoke_raw.get("expect") is None else str(smoke_raw.get("expect", SmokeConfig.expect))),
        max_tokens=int(smoke_raw.get("max_tokens", 32)),
        temperature=float(smoke_raw.get("temperature", 0.0)),
        timeout_sec=int(smoke_raw.get("timeout_sec", 300)),
    )

    cleanup: list[CleanupCandidate] = []
    for idx, row in enumerate(_as_list(data.get("cleanup"), "cleanup")):
        if not isinstance(row, dict):
            raise ManifestError(f"[[cleanup]] row {idx} must be a table")
        if "path" not in row:
            raise ManifestError("[[cleanup]] requires path")
        cleanup.append(CleanupCandidate(path=expand(str(row["path"])) or "", description=str(row.get("description", "")), safe=bool(row.get("safe", False))))

    return ModelManifest(path=p, id=ident, model_id=model_id, endpoint=endpoint, description=str(model.get("description", "")), start=start_cfg, preflight=preflight, smoke=smoke, cleanup=cleanup)
