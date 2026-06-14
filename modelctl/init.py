from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .manifest import load_manifest


def _q(value: str) -> str:
    return json.dumps(value)


def minimal_template(model_id: str, endpoint: str, ident: str) -> str:
    return f'''[model]
id = {_q(ident)}
model_id = {_q(model_id)}
endpoint = {_q(endpoint.rstrip('/'))}
description = "OpenAI-compatible local model endpoint"

[preflight]
required_paths = []
exclusive_ports = []

[smoke]
prompt = "Reply with exactly the word pong."
expect = "pong"
max_tokens = 16
temperature = 0
'''


def llama_cpp_template(model_id: str, endpoint: str, ident: str, port: int) -> str:
    return f'''[model]
id = {_q(ident)}
model_id = {_q(model_id)}
endpoint = {_q(endpoint.rstrip('/'))}
description = "llama.cpp OpenAI-compatible server"

[start]
command = ["llama-server", "-m", "/path/to/model.gguf", "--host", "127.0.0.1", "--port", "{port}", "-a", {_q(model_id)}]
cwd = "."
log_path = "~/.local/state/modelctl/{ident}.log"
pid_path = "~/.local/state/modelctl/{ident}.pid.json"
startup_timeout_sec = 120
readiness_url = "http://127.0.0.1:{port}/v1/models"
readiness_contains = {_q(model_id)}

[preflight]
required_paths = []
exclusive_ports = [{port}]

[[preflight.disk]]
path = "."
min_free_gib = 5

[smoke]
prompt = "Reply with exactly the word pong."
expect = "pong"
max_tokens = 16
temperature = 0

[[cleanup]]
path = "~/.cache/llama.cpp"
description = "Optional llama.cpp cache; review before deleting."
safe = false
'''


def template_text(template: str, model_id: str = "local-model", endpoint: str = "http://127.0.0.1:8080/v1", ident: str | None = None, port: int = 8080) -> str:
    ident = ident or model_id.replace("/", "-").replace(":", "-")
    if template == "minimal":
        return minimal_template(model_id, endpoint, ident)
    if template == "llama-cpp":
        return llama_cpp_template(model_id, endpoint, ident, port)
    raise ValueError(f"unknown template: {template}")


def init_manifest(output: str = "modelctl.toml", template: str = "minimal", model_id: str = "local-model", endpoint: str = "http://127.0.0.1:8080/v1", ident: str | None = None, port: int = 8080, overwrite: bool = False) -> dict[str, Any]:
    path = Path(output).expanduser()
    if path.exists() and not overwrite:
        return {"ok": False, "error": f"output exists: {path}; pass --overwrite", "output": str(path)}
    content = template_text(template, model_id=model_id, endpoint=endpoint, ident=ident, port=port)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    manifest = load_manifest(path)
    return {"ok": True, "output": str(path), "template": template, "id": manifest.id, "model_id": manifest.model_id, "endpoint": manifest.endpoint}
