from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
import math
import os
import re
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
class HealthConfig:
    max_swap_gib: float | None = None
    max_swap_delta_gib: float | None = None
    sample_sec: float = 0.0
    smoke: bool = False
    max_latency_sec: float | None = None
    max_prompt_latency_sec: float | None = None
    max_completion_latency_sec: float | None = None
    max_io_latency_sec: float | None = None


@dataclass(slots=True)
class FleetConfig:
    enabled: bool = True
    reason: str = ""


@dataclass(slots=True)
class PreflightConfig:
    required_paths: list[str] = field(default_factory=list)
    exclusive_ports: list[int] = field(default_factory=list)
    max_swap_gib: float | None = None
    disk: list[DiskCheck] = field(default_factory=list)


@dataclass(slots=True)
class PromotionReceiptConfig:
    path: str
    sha256: str
    max_age_sec: int = 86400
    require_decision: str = "promote"
    required_gates: list[str] = field(default_factory=lambda: ["logit", "quality"])


@dataclass(slots=True)
class ModelManifest:
    path: Path
    id: str
    model_id: str
    endpoint: str
    description: str = ""
    start: StartConfig | None = None
    preflight: PreflightConfig = field(default_factory=PreflightConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    smoke: SmokeConfig = field(default_factory=SmokeConfig)
    cleanup: list[CleanupCandidate] = field(default_factory=list)
    promotion_requires_receipt: bool = False
    promotion_receipt: PromotionReceiptConfig | None = None

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
    if not all(type(x) is int for x in items):
        raise ManifestError(f"{key} must contain only integers")
    return list(items)


def _as_bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{key} must be a boolean")
    return value


def _as_str(value: Any, key: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{key} must be a string")
    if not allow_empty and not value:
        raise ManifestError(f"{key} must not be empty")
    return value


def _has_control(value: str) -> bool:
    return any(ord(char) < 32 or ord(char) == 127 for char in value)


def _manifest_id(value: Any) -> str:
    ident = _as_str(value, "model.id", allow_empty=False)
    if ident in {".", ".."} or "/" in ident or "\\" in ident or _has_control(ident):
        raise ManifestError("model.id must be a filename-safe identifier without path components")
    return ident


def _optional_str(table: dict[str, Any], name: str, key: str) -> str | None:
    if name not in table:
        return None
    return _as_str(table[name], key)


def _expanded_nonempty_str(value: Any, key: str) -> str:
    raw = _as_str(value, key, allow_empty=False)
    expanded = expand(raw)
    if not expanded:
        raise ManifestError(f"{key} must not be empty after environment expansion")
    if _has_control(expanded):
        raise ManifestError(f"{key} must not contain control characters")
    return expanded


def _optional_expanded_path(table: dict[str, Any], name: str, key: str) -> str | None:
    if name not in table:
        return None
    return _expanded_nonempty_str(table[name], key)


def _expanded_path_list(value: Any, key: str) -> list[str]:
    values = _as_str_list(value, key)
    return [_expanded_nonempty_str(item, f"{key}[{idx}]") for idx, item in enumerate(values)]


def _finite_nonnegative(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{key} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ManifestError(f"{key} must be a finite non-negative number")
    return result


def _optional_nonnegative(table: dict[str, Any], name: str, key: str) -> float | None:
    if name not in table:
        return None
    return _finite_nonnegative(table[name], key)


def _positive_int(value: Any, key: str) -> int:
    if type(value) is not int or value <= 0:
        raise ManifestError(f"{key} must be a positive integer")
    return value


def _port(value: Any, key: str) -> int:
    result = _positive_int(value, key)
    if result > 65535:
        raise ManifestError(f"{key} must be between 1 and 65535")
    return result


def _url(value: Any, key: str, *, allow_query_fragment: bool = False) -> str:
    result = _as_str(value, key, allow_empty=False)
    if any(char.isspace() for char in result) or "\\" in result or _has_control(result):
        raise ManifestError(f"{key} must be a valid http(s) URL")
    if re.search(r"%(?![0-9A-Fa-f]{2})", result) or _has_control(unquote(result)):
        raise ManifestError(f"{key} must be a valid http(s) URL")
    try:
        parsed = urlparse(result)
        hostname = parsed.hostname
        username = parsed.username
        password = parsed.password
        port = parsed.port
    except ValueError as exc:
        raise ManifestError(f"{key} must be a valid http(s) URL: {exc}") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not result.lower().startswith(f"{parsed.scheme}://")
        or not hostname
        or username is not None
        or password is not None
        or "@" in parsed.netloc
    ):
        raise ManifestError(f"{key} must be a valid http(s) URL without userinfo")
    if not allow_query_fragment and ("?" in result or "#" in result):
        raise ManifestError(f"{key} must not contain a query or fragment")
    if parsed.netloc.endswith(":"):
        raise ManifestError(f"{key} must not contain an empty explicit port")
    if ":" in hostname:
        closing = parsed.netloc.find("]")
        suffix = parsed.netloc[closing + 1 :] if closing >= 0 else ""
        if not parsed.netloc.startswith("[") or closing < 0 or (suffix and not suffix.startswith(":")):
            raise ManifestError(f"{key} must use brackets around an IPv6 host")
    elif "[" in parsed.netloc or "]" in parsed.netloc:
        raise ManifestError(f"{key} must be a valid http(s) URL")
    if port is not None and not 1 <= port <= 65535:
        raise ManifestError(f"{key} port must be between 1 and 65535")
    return result


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
        model_id = _as_str(model["model_id"], "model.model_id", allow_empty=False)
        endpoint = _url(expand(_as_str(model["endpoint"], "model.endpoint", allow_empty=False)), "model.endpoint")
    except KeyError as exc:
        raise ManifestError(f"[model].{exc.args[0]} is required") from exc
    ident = _manifest_id(model.get("id", model_id))
    description = _as_str(model.get("description", ""), "model.description")

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
        env = {k: expand(_as_str(v, f"start.env.{k}")) or "" for k, v in env_raw.items()}
        if any("\x00" in value for value in env.values()):
            raise ManifestError("start.env values must not contain NUL")
        command = [expand(value) or "" for value in command]
        if any("\x00" in value for value in command):
            raise ManifestError("start.command values must not contain NUL")
        if not command or not command[0]:
            raise ManifestError("start.command[0] must not be empty after environment expansion")
        cwd = _optional_expanded_path(start, "cwd", "start.cwd")
        log_path = _optional_expanded_path(start, "log_path", "start.log_path")
        pid_path = _optional_expanded_path(start, "pid_path", "start.pid_path")
        readiness_url = _url(expand(_as_str(start["readiness_url"], "start.readiness_url", allow_empty=False)), "start.readiness_url", allow_query_fragment=True) if "readiness_url" in start else None
        start_cfg = StartConfig(
            command=command,
            cwd=cwd,
            env=env,
            log_path=log_path,
            pid_path=pid_path,
            startup_timeout_sec=_positive_int(start.get("startup_timeout_sec", 120), "start.startup_timeout_sec"),
            readiness_url=readiness_url,
            readiness_contains=_optional_str(start, "readiness_contains", "start.readiness_contains"),
        )

    pre = _as_table(data, "preflight") if "preflight" in data else {}
    disk_checks: list[DiskCheck] = []
    for idx, row in enumerate(_as_list(pre.get("disk"), "preflight.disk")):
        if not isinstance(row, dict):
            raise ManifestError(f"[[preflight.disk]] row {idx} must be a table")
        if "path" not in row or "min_free_gib" not in row:
            raise ManifestError("[[preflight.disk]] requires path and min_free_gib")
        disk_checks.append(
            DiskCheck(
                path=_expanded_nonempty_str(row["path"], f"preflight.disk[{idx}].path"),
                min_free_gib=_finite_nonnegative(row["min_free_gib"], f"preflight.disk[{idx}].min_free_gib"),
            )
        )
    exclusive_ports = [_port(value, f"preflight.exclusive_ports[{idx}]") for idx, value in enumerate(_as_int_list(pre.get("exclusive_ports"), "preflight.exclusive_ports"))]
    preflight = PreflightConfig(
        required_paths=_expanded_path_list(pre.get("required_paths"), "preflight.required_paths"),
        exclusive_ports=exclusive_ports,
        max_swap_gib=_optional_nonnegative(pre, "max_swap_gib", "preflight.max_swap_gib"),
        disk=disk_checks,
    )

    health_raw = _as_table(data, "health") if "health" in data else {}
    health = HealthConfig(
        max_swap_gib=_optional_nonnegative(health_raw, "max_swap_gib", "health.max_swap_gib"),
        max_swap_delta_gib=_optional_nonnegative(health_raw, "max_swap_delta_gib", "health.max_swap_delta_gib"),
        sample_sec=_finite_nonnegative(health_raw.get("sample_sec", 0.0), "health.sample_sec"),
        smoke=_as_bool(health_raw.get("smoke", False), "health.smoke"),
        max_latency_sec=_optional_nonnegative(health_raw, "max_latency_sec", "health.max_latency_sec"),
        max_prompt_latency_sec=_optional_nonnegative(health_raw, "max_prompt_latency_sec", "health.max_prompt_latency_sec"),
        max_completion_latency_sec=_optional_nonnegative(health_raw, "max_completion_latency_sec", "health.max_completion_latency_sec"),
        max_io_latency_sec=_optional_nonnegative(health_raw, "max_io_latency_sec", "health.max_io_latency_sec"),
    )
    if health.sample_sec > 0 and health.max_swap_delta_gib is None:
        raise ManifestError("health.sample_sec requires health.max_swap_delta_gib")

    fleet_raw = _as_table(data, "fleet") if "fleet" in data else {}
    fleet = FleetConfig(
        enabled=_as_bool(fleet_raw.get("enabled", True), "fleet.enabled"),
        reason=_as_str(fleet_raw.get("reason", ""), "fleet.reason"),
    )

    has_smoke = "smoke" in data
    smoke_raw = _as_table(data, "smoke") if has_smoke else {}
    smoke_defaults = SmokeConfig()
    smoke_expect = "pong" if not has_smoke else None
    if "expect" in smoke_raw and smoke_raw.get("expect") is not None:
        smoke_expect = _as_str(smoke_raw["expect"], "smoke.expect")
    smoke = SmokeConfig(
        prompt=_as_str(smoke_raw.get("prompt", smoke_defaults.prompt), "smoke.prompt"),
        expect=smoke_expect,
        max_tokens=_positive_int(smoke_raw.get("max_tokens", smoke_defaults.max_tokens), "smoke.max_tokens"),
        temperature=_finite_nonnegative(smoke_raw.get("temperature", smoke_defaults.temperature), "smoke.temperature"),
        timeout_sec=_positive_int(smoke_raw.get("timeout_sec", smoke_defaults.timeout_sec), "smoke.timeout_sec"),
    )

    promotion_receipt: PromotionReceiptConfig | None = None
    promotion_requires_receipt = False
    if "promotion" in data:
        promotion_raw = _as_table(data, "promotion")
        unknown_promotion = sorted(set(promotion_raw) - {"require_receipt", "receipt"})
        if unknown_promotion:
            raise ManifestError(f"[promotion] contains unknown keys: {', '.join(unknown_promotion)}")
        promotion_requires_receipt = _as_bool(promotion_raw.get("require_receipt", False), "promotion.require_receipt")
        if "receipt" in promotion_raw:
            receipt_raw = promotion_raw["receipt"]
            if not isinstance(receipt_raw, dict):
                raise ManifestError("[promotion.receipt] must be a TOML table")
            unknown_receipt = sorted(set(receipt_raw) - {"path", "sha256", "max_age_sec", "require_decision", "required_gates"})
            if unknown_receipt:
                raise ManifestError(f"[promotion.receipt] contains unknown keys: {', '.join(unknown_receipt)}")
            if "path" not in receipt_raw or "sha256" not in receipt_raw:
                raise ManifestError("[promotion.receipt] requires path and sha256")
            receipt_sha256 = _as_str(receipt_raw["sha256"], "promotion.receipt.sha256", allow_empty=False).lower()
            if not re.fullmatch(r"[0-9a-f]{64}", receipt_sha256):
                raise ManifestError("promotion.receipt.sha256 must be a 64-character hexadecimal SHA-256 digest")
            required_gates = _as_str_list(receipt_raw.get("required_gates", ["logit", "quality"]), "promotion.receipt.required_gates")
            if not required_gates or any(not gate or _has_control(gate) for gate in required_gates):
                raise ManifestError("promotion.receipt.required_gates must contain non-empty gate names")
            if len(set(required_gates)) != len(required_gates):
                raise ManifestError("promotion.receipt.required_gates must not contain duplicates")
            require_decision = _as_str(receipt_raw.get("require_decision", "promote"), "promotion.receipt.require_decision", allow_empty=False)
            if _has_control(require_decision):
                raise ManifestError("promotion.receipt.require_decision must not contain control characters")
            promotion_receipt = PromotionReceiptConfig(
                path=_expanded_nonempty_str(receipt_raw["path"], "promotion.receipt.path"),
                sha256=receipt_sha256,
                max_age_sec=_positive_int(receipt_raw.get("max_age_sec", 86400), "promotion.receipt.max_age_sec"),
                require_decision=require_decision,
                required_gates=required_gates,
            )

    cleanup: list[CleanupCandidate] = []
    for idx, row in enumerate(_as_list(data.get("cleanup"), "cleanup")):
        if not isinstance(row, dict):
            raise ManifestError(f"[[cleanup]] row {idx} must be a table")
        if "path" not in row:
            raise ManifestError("[[cleanup]] requires path")
        cleanup.append(
            CleanupCandidate(
                path=_expanded_nonempty_str(row["path"], f"cleanup[{idx}].path"),
                description=_as_str(row.get("description", ""), f"cleanup[{idx}].description"),
                safe=_as_bool(row.get("safe", False), f"cleanup[{idx}].safe"),
            )
        )

    return ModelManifest(path=p, id=ident, model_id=model_id, endpoint=endpoint, description=description, start=start_cfg, preflight=preflight, health=health, fleet=fleet, smoke=smoke, promotion_requires_receipt=promotion_requires_receipt, promotion_receipt=promotion_receipt, cleanup=cleanup)
