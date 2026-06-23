from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import json
import re
import subprocess
import urllib.error
import http.client

from .http import http_json
from .ingest import _endpoint_from_models_url, _model_names, _models_url, _q
from .manifest import load_manifest
from .registry import list_registry, safe_stem, selected_registry_dir


LOCAL_HOST = "127.0.0.1"
DORMANT_REASON = "intake candidate; review before enabling"


def _normalize_endpoint(endpoint: str) -> str:
    text = endpoint.strip().rstrip("/")
    if text.endswith("/models"):
        text = _endpoint_from_models_url(text)
    return text


def _endpoint_key(endpoint: str) -> str:
    return _normalize_endpoint(endpoint).rstrip("/")


def _endpoint_port(endpoint: str) -> int | None:
    try:
        return urlparse(_normalize_endpoint(endpoint)).port
    except ValueError:
        return None


def _endpoint_for_port(port: int, host: str = LOCAL_HOST) -> str:
    return f"http://{host}:{int(port)}/v1"


def _candidate_id(endpoint: str, model_id: str) -> str:
    port = _endpoint_port(endpoint)
    prefix = f"intake-{port}-" if port is not None else "intake-"
    return safe_stem(prefix + safe_stem(model_id))


def _toml_int_list(values: list[int]) -> str:
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


def intake_manifest_text(endpoint: str, model_id: str, ident: str | None = None, reason: str = DORMANT_REASON) -> str:
    endpoint = _normalize_endpoint(endpoint)
    ident = ident or _candidate_id(endpoint, model_id)
    port = _endpoint_port(endpoint)
    exclusive_ports = [] if port is None else [port]
    description = f"Intake candidate for OpenAI-compatible endpoint {endpoint}"
    return f'''[model]
id = {_q(ident)}
model_id = {_q(model_id)}
endpoint = {_q(endpoint)}
description = {_q(description)}

[fleet]
enabled = false
reason = {_q(reason)}

[preflight]
required_paths = []
exclusive_ports = {_toml_int_list(exclusive_ports)}

[smoke]
prompt = "Reply with exactly the word pong."
expect = "pong"
max_tokens = 16
temperature = 0
'''


def _parse_lsof_ports(output: str) -> list[int]:
    ports: set[int] = set()
    for line in output.splitlines():
        match = re.search(r":(\d+)\s*\(LISTEN\)\s*$", line)
        if match:
            ports.add(int(match.group(1)))
    return sorted(ports)


def _parse_ss_ports(output: str) -> list[int]:
    ports: set[int] = set()
    for line in output.splitlines():
        parts = line.split()
        if not parts:
            continue
        # ss output usually has Local Address:Port near the end.
        for token in reversed(parts):
            token = token.strip()
            match = re.search(r":(\d+)$", token)
            if match:
                ports.add(int(match.group(1)))
                break
    return sorted(ports)


def discover_listening_ports() -> list[int]:
    """Best-effort local TCP listener discovery. No probing, just OS inventory."""
    commands = [
        (["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"], _parse_lsof_ports),
        (["ss", "-H", "-ltn"], _parse_ss_ports),
    ]
    for command, parser in commands:
        try:
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            ports = parser(result.stdout)
            if ports:
                return ports
    return []


def _registered_endpoints(registries: list[str] | None = None) -> set[str]:
    endpoints: set[str] = set()
    for entry in list_registry(registries).get("entries", []):
        endpoint = entry.get("endpoint")
        if entry.get("ok") and isinstance(endpoint, str):
            endpoints.add(_endpoint_key(endpoint))
    return endpoints


def _dedupe_endpoints(endpoints: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for endpoint in endpoints:
        normalized = _normalize_endpoint(endpoint)
        key = _endpoint_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _probe_endpoint(endpoint: str, timeout: float) -> dict[str, Any]:
    models_url = _models_url(endpoint)
    try:
        status, body, text = http_json("GET", models_url, timeout=timeout)
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError, TimeoutError) as exc:
        return {"ok": False, "endpoint": endpoint, "models_url": models_url, "status": None, "error": f"{type(exc).__name__}: {exc}"}
    if not (200 <= status < 300):
        return {"ok": False, "endpoint": endpoint, "models_url": models_url, "status": status, "error": body}
    models = _model_names(body)
    if not models:
        return {"ok": False, "endpoint": endpoint, "models_url": models_url, "status": status, "error": "no model ids found", "body": body if isinstance(body, dict) else text[:500]}
    return {"ok": True, "endpoint": endpoint, "models_url": models_url, "status": status, "models": models, "model_id": models[0]}


def fleet_intake(
    *,
    registries: list[str] | None = None,
    endpoints: list[str] | None = None,
    ports: list[int] | None = None,
    host: str = LOCAL_HOST,
    timeout: float = 1.0,
    output_dir: str | None = None,
    execute: bool = False,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Discover live OpenAI-compatible endpoints and draft dormant manifests.

    Dry-run by default. The only network operation is GET /models against candidate
    endpoints. No readiness, health, smoke, start, or service actions are called.
    """
    explicit_endpoints = [_normalize_endpoint(e) for e in (endpoints or [])]
    explicit_ports = [int(p) for p in (ports or [])]
    discovered_ports = [] if explicit_endpoints or explicit_ports else discover_listening_ports()
    all_ports = sorted(set(explicit_ports + discovered_ports))
    all_endpoints = explicit_endpoints + [_endpoint_for_port(port, host=host) for port in all_ports]
    if limit is not None:
        all_endpoints = all_endpoints[: max(0, limit)]
    candidates_to_check = _dedupe_endpoints(all_endpoints)

    registered = _registered_endpoints(registries)
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    unreachable: list[dict[str, Any]] = []
    written: list[str] = []
    errors: list[dict[str, Any]] = []

    for endpoint in candidates_to_check:
        key = _endpoint_key(endpoint)
        port = _endpoint_port(endpoint)
        if key in registered:
            skipped.append({"reason": "already_registered", "endpoint": endpoint, "port": port})
            continue
        probe = _probe_endpoint(endpoint, timeout=timeout)
        if not probe.get("ok"):
            row = {"endpoint": endpoint, "port": port, **probe}
            unreachable.append(row)
            continue
        model_id = str(probe["model_id"])
        ident = _candidate_id(endpoint, model_id)
        manifest = intake_manifest_text(endpoint, model_id, ident=ident)
        row: dict[str, Any] = {
            "ok": True,
            "id": ident,
            "model_id": model_id,
            "models": probe.get("models", []),
            "endpoint": endpoint,
            "port": port,
            "fleet": {"enabled": False, "reason": DORMANT_REASON},
            "manifest": manifest,
            "output": None,
        }
        candidates.append(row)

    if execute:
        directory = Path(output_dir).expanduser() if output_dir else selected_registry_dir(None)
        directory.mkdir(parents=True, exist_ok=True)
        for row in candidates:
            path = directory / f"{safe_stem(str(row['id']))}.toml"
            if path.exists() and not overwrite:
                errors.append({"code": "output_exists", "id": row["id"], "path": str(path)})
                continue
            path.write_text(str(row["manifest"]), encoding="utf-8")
            load_manifest(path)
            row["output"] = str(path)
            written.append(str(path))

    ok = not errors
    return {
        "ok": ok,
        "status": "ok" if ok else "error",
        "execute": execute,
        "host": host,
        "timeout": timeout,
        "registry_dirs": list_registry(registries).get("registry_dirs", []),
        "discovered_ports": discovered_ports,
        "ports": all_ports,
        "checked_count": len(candidates_to_check),
        "candidate_count": len(candidates),
        "registered_skipped": len(skipped),
        "unreachable_count": len(unreachable),
        "written_count": len(written),
        "written": written,
        "errors": errors,
        "skipped": skipped,
        "unreachable": unreachable,
        "candidates": candidates,
    }
