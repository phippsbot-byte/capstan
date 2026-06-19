from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import re
import shutil

from .manifest import load_manifest


DEFAULT_MLX_ROOT = "~/.cache/mlx-models"
DEFAULT_CHAT_TEMPLATE_ARGS = {"enable_thinking": False}


def _q(value: str) -> str:
    return json.dumps(value)


def _safe_id(value: str) -> str:
    ident = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return ident or "mlx-model"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {"_invalid_json": True}
    return data if isinstance(data, dict) else {"_invalid_json": True}


def _model_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ValueError(f"model path not found: {p}")
    if not p.is_dir():
        raise ValueError(f"model path must be a directory: {p}")
    if not (p / "config.json").exists():
        raise ValueError(f"not an MLX model directory; missing config.json: {p}")
    return p


def _chat_template_path(model_path: Path) -> Path | None:
    template = model_path / "chat_template.jinja"
    return template if template.exists() else None


def _template_tail(text: str) -> str:
    idx = text.rfind("add_generation_prompt")
    return text[idx:] if idx >= 0 else text[-2000:]


def has_bad_think_preamble(text: str) -> bool:
    tail = _template_tail(text)
    return "<think>" in tail and "add_generation_prompt" in text


def patch_think_preamble(text: str) -> tuple[str, bool]:
    if not has_bad_think_preamble(text):
        return text, False
    marker = text.rfind("add_generation_prompt")
    start = marker if marker >= 0 else 0
    tail = text[start:]
    rel = tail.rfind("<think>")
    if rel < 0:
        return text, False
    idx = start + rel
    return text[:idx] + "</think>" + text[idx + len("<think>"):], True


def inspect_mlx_model(model_path: str | Path) -> dict[str, Any]:
    model = _model_dir(model_path)
    config = _read_json(model / "config.json")
    tokenizer_config = _read_json(model / "tokenizer_config.json")
    template_path = _chat_template_path(model)
    template_source = "chat_template.jinja" if template_path else None
    template_text = template_path.read_text(encoding="utf-8") if template_path else ""
    inline_template = tokenizer_config.get("chat_template")
    if not template_text and isinstance(inline_template, str):
        template_text = inline_template
        template_source = "tokenizer_config.json"
    bad_think = has_bad_think_preamble(template_text) if template_text else False
    warnings: list[str] = []
    if bad_think:
        warnings.append("qwen_think_preamble")
    if not template_text:
        warnings.append("no_chat_template")
    if config.get("_invalid_json"):
        warnings.append("invalid_config_json")
    if tokenizer_config.get("_invalid_json"):
        warnings.append("invalid_tokenizer_config_json")
    quantization = config.get("quantization") or config.get("quantization_config") or {}
    return {
        "ok": True,
        "path": str(model),
        "name": model.name,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "quantization": quantization if isinstance(quantization, dict) else quantization,
        "template": {
            "path": str(template_path) if template_path else None,
            "source": template_source,
            "exists": bool(template_text),
            "bad_think_preamble": bad_think,
            "recommended_overlay": bad_think and template_source == "chat_template.jinja",
        },
        "warnings": warnings,
    }


def discover_mlx_models(root: str | Path = DEFAULT_MLX_ROOT, limit: int = 200) -> dict[str, Any]:
    base = Path(root).expanduser().resolve()
    if not base.exists():
        return {"ok": True, "root": str(base), "count": 0, "models": [], "warnings": ["root_missing"]}
    if limit <= 0:
        return {"ok": True, "root": str(base), "count": 0, "models": []}
    found: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for config_path in base.rglob("config.json"):
        model = config_path.parent.resolve()
        if model in seen:
            continue
        seen.add(model)
        try:
            info = inspect_mlx_model(model)
        except Exception as exc:  # keep discovery robust across half-written downloads
            found.append({"path": str(model), "name": model.name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        else:
            found.append({
                "path": info["path"],
                "name": info["name"],
                "model_type": info.get("model_type"),
                "architectures": info.get("architectures", []),
                "bad_think_preamble": info["template"]["bad_think_preamble"],
                "warnings": info.get("warnings", []),
            })
        if len(found) >= limit:
            break
    return {"ok": True, "root": str(base), "count": len(found), "models": found}


def _remove_existing_output(output: Path) -> None:
    if output.is_symlink() or output.is_file():
        output.unlink()
    elif output.exists():
        shutil.rmtree(output)


def _safe_overwrite_target(source: Path, target: Path) -> bool:
    if target in source.parents:
        return False
    return target == source.with_name(f"{source.name}-served").resolve()


def create_overlay(model_path: str | Path, output: str | Path | None = None, overwrite: bool = False) -> dict[str, Any]:
    source = _model_dir(model_path)
    target = Path(output).expanduser().resolve() if output else source.with_name(f"{source.name}-served")
    if target == source:
        return {"ok": False, "error": "overlay output must not equal source", "source": str(source), "output": str(target)}
    if source in target.parents:
        return {"ok": False, "error": "overlay output must not be inside source", "source": str(source), "output": str(target)}
    template = _chat_template_path(source)
    if template is None:
        return {"ok": False, "error": "source has no chat_template.jinja to patch", "source": str(source), "output": str(target)}
    if target.exists() or target.is_symlink():
        if not overwrite:
            return {"ok": False, "error": f"output exists: {target}; pass --overwrite", "source": str(source), "output": str(target)}
        if not _safe_overwrite_target(source, target):
            return {"ok": False, "error": "refusing to overwrite anything except the default sibling NAME-served overlay", "source": str(source), "output": str(target)}
        _remove_existing_output(target)
    target.mkdir(parents=True, exist_ok=True)
    symlinked: list[str] = []
    copied: list[str] = []
    for child in sorted(source.iterdir(), key=lambda p: p.name):
        if child.name == "chat_template.jinja":
            continue
        os.symlink(child, target / child.name, target_is_directory=child.is_dir())
        symlinked.append(child.name)
    original = template.read_text(encoding="utf-8")
    patched_text, patched = patch_think_preamble(original)
    (target / "chat_template.jinja").write_text(patched_text, encoding="utf-8")
    copied.append("chat_template.jinja")
    return {
        "ok": True,
        "source": str(source),
        "output": str(target),
        "patched": patched,
        "symlinked_count": len(symlinked),
        "copied": copied,
        "warnings": ([] if patched else ["no_qwen_think_preamble_patched"]),
    }


def _default_python() -> str:
    homebrew = Path("/opt/homebrew/bin/python3.11")
    return str(homebrew) if homebrew.exists() else "python3.11"


def mlx_manifest_text(model_path: str | Path, model_id: str | None = None, ident: str | None = None, port: int = 8080, python: str | None = None, max_tokens: int = 4096, temp: float = 0.23, top_p: float = 0.9, prompt_cache_gib: float = 4.0, prompt_cache_size: int = 4) -> str:
    model = _model_dir(model_path)
    request_model_id = model_id or "default_model"
    allowed_request_ids = {"default_model", str(model)}
    if request_model_id not in allowed_request_ids:
        raise ValueError("stock mlx_lm server request model id must be 'default_model' or the absolute model path; use --id for a friendly manifest id")
    ident = ident or _safe_id(model.name)
    endpoint = f"http://127.0.0.1:{port}/v1"
    py = python or _default_python()
    chat_args = json.dumps(DEFAULT_CHAT_TEMPLATE_ARGS, separators=(",", ":"))
    prompt_cache_bytes = int(max(0.0, prompt_cache_gib) * 1024 * 1024 * 1024)
    command = [
        py,
        "-m",
        "mlx_lm",
        "server",
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-tokens",
        str(max_tokens),
        "--temp",
        str(temp),
        "--top-p",
        str(top_p),
        "--chat-template-args",
        chat_args,
        "--decode-concurrency",
        "1",
        "--prompt-concurrency",
        "1",
        "--prompt-cache-size",
        str(prompt_cache_size),
        "--prompt-cache-bytes",
        str(prompt_cache_bytes),
        "--log-level",
        "INFO",
    ]
    command_toml = ", ".join(_q(item) for item in command)
    return f'''[model]
id = {_q(ident)}
model_id = {_q(request_model_id)}
endpoint = {_q(endpoint)}
description = {_q(f"MLX mlx_lm server for {model.name}")}

[start]
command = [{command_toml}]
cwd = {_q(str(model))}
log_path = {_q(f"~/.local/state/modelctl/{ident}.log")}
pid_path = {_q(f"~/.local/state/modelctl/{ident}.pid.json")}
startup_timeout_sec = 300
readiness_url = {_q(f"http://127.0.0.1:{port}/v1/models")}
readiness_contains = {_q(str(model))}

[preflight]
required_paths = [{_q(str(model))}]
exclusive_ports = [{port}]
max_swap_gib = 8

[health]
max_swap_gib = 8
max_swap_delta_gib = 1
sample_sec = 5

[[preflight.disk]]
path = {_q(str(model.parent))}
min_free_gib = 5

[smoke]
prompt = "Reply with exactly the word pong."
expect = "pong"
max_tokens = 32
temperature = 0
'''


def write_mlx_manifest(model_path: str | Path, output: str | Path, overwrite: bool = False, **kwargs: Any) -> dict[str, Any]:
    path = Path(output).expanduser()
    if path.exists() and not overwrite:
        return {"ok": False, "error": f"output exists: {path}; pass --overwrite", "output": str(path)}
    try:
        content = mlx_manifest_text(model_path, **kwargs)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "output": str(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    manifest = load_manifest(path)
    return {"ok": True, "output": str(path), "id": manifest.id, "model_id": manifest.model_id, "endpoint": manifest.endpoint}
