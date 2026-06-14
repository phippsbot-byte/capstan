from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .http import http_json
from .manifest import load_manifest


def _q(value: str) -> str:
    # TOML basic strings are JSON-string compatible for our simple scalar values.
    return json.dumps(value)


def _models_url(endpoint: str) -> str:
    e = endpoint.rstrip("/")
    if e.endswith("/models"):
        return e
    return e + "/models"


def _endpoint_from_models_url(endpoint: str) -> str:
    e = endpoint.rstrip("/")
    return e[:-7] if e.endswith("/models") else e


def _model_names(body: Any) -> list[str]:
    names: list[str] = []
    if isinstance(body, dict):
        for key in ("data", "models"):
            rows = body.get(key)
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        for field in ("id", "model", "name"):
                            value = row.get(field)
                            if isinstance(value, str) and value and value not in names:
                                names.append(value)
                    elif isinstance(row, str) and row not in names:
                        names.append(row)
    return names


def manifest_text(endpoint: str, model_id: str, ident: str | None = None, description: str | None = None) -> str:
    ident = ident or model_id.replace("/", "-").replace(":", "-")
    description = description or f"Ingested OpenAI-compatible endpoint for {model_id}"
    return f'''[model]
id = {_q(ident)}
model_id = {_q(model_id)}
endpoint = {_q(_endpoint_from_models_url(endpoint))}
description = {_q(description)}

[preflight]
required_paths = []
exclusive_ports = []

[smoke]
prompt = "Reply with exactly the word pong."
expect = "pong"
max_tokens = 16
temperature = 0
'''


def ingest(endpoint: str, output: str | None = None, model_id: str | None = None, ident: str | None = None, overwrite: bool = False, timeout: int = 30) -> dict[str, Any]:
    models_url = _models_url(endpoint)
    status, body, text = http_json("GET", models_url, timeout=timeout)
    if not (200 <= status < 300):
        return {"ok": False, "status": status, "models_url": models_url, "error": body}
    names = _model_names(body)
    chosen = model_id or (names[0] if names else None)
    if not chosen:
        return {"ok": False, "status": status, "models_url": models_url, "error": "no model ids found", "body": body if isinstance(body, dict) else text[:500]}
    if model_id and model_id not in names:
        # Not fatal; some servers hide aliases. Tell the user though.
        warning = f"requested model_id {model_id!r} was not present in /models"
    else:
        warning = None
    content = manifest_text(endpoint, chosen, ident=ident)
    written = None
    if output:
        path = Path(output).expanduser()
        if path.exists() and not overwrite:
            return {"ok": False, "status": status, "models_url": models_url, "error": f"output exists: {path}; pass --overwrite", "models": names}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        # Validate the generated file before claiming success. Trust, but verify. Mostly verify.
        load_manifest(path)
        written = str(path)
    return {"ok": True, "status": status, "models_url": models_url, "endpoint": _endpoint_from_models_url(endpoint), "models": names, "model_id": chosen, "output": written, "warning": warning, "manifest": content if output is None else None}
