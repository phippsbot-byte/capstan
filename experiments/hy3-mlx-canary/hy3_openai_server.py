#!/usr/bin/env python3
"""Tiny OpenAI-compatible HTTP server for the Hy3 lazy sidecar canary.

This is intentionally single-threaded. The canary model/runtime is not production
safe; this exists so we can smoke Phipps-style harness calls against the exact
lazy-sidecar path without pretending it is done.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from transformers import AutoTokenizer

WORKDIR = Path(__file__).resolve().parent
SMOKE_FILE = WORKDIR / "hy3_lazy_smoke.py"
MODEL_DIR = Path("/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX")
PACKED_MANIFEST = Path("/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/manifest.json")

MODEL_ID = "hy3-lazy-sidecar-canary"


def import_smoke_module():
    spec = importlib.util.spec_from_file_location("hy3_lazy_smoke_import", SMOKE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SMOKE_FILE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def now_s() -> int:
    return int(time.time())


class Hy3Runtime:
    def __init__(self, max_default_tokens: int):
        self.max_default_tokens = max_default_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
        smoke = import_smoke_module()
        self.model, self.lazy_mod, self.config, self.meta = smoke.load_lazy_model(eval_params=True)
        self.eos_id = int(self.config.get("eos_token_id", -1))

    def render_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": False,
            "reasoning_effort": "no_think",
        }
        if tools:
            kwargs["tools"] = tools
        rendered = self.tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(rendered, list):
            rendered = "".join(rendered)
        return str(rendered)

    @staticmethod
    def _strip_reasoning(text: str) -> tuple[str, str | None]:
        reasoning = None
        m = re.match(r"\s*<think>(.*?)</think>(.*)\Z", text, flags=re.S)
        if m:
            reasoning = m.group(1)
            text = m.group(2)
        return text.strip(), reasoning

    @staticmethod
    def _parse_hy3_tool_calls(text: str) -> tuple[str, list[dict[str, Any]] | None]:
        if "<tool_calls>" not in text:
            return text, None
        body_match = re.search(r"<tool_calls>\s*(.*?)\s*</tool_calls>", text, flags=re.S)
        if not body_match:
            return text, None
        body = body_match.group(1)
        calls = []
        for block in re.findall(r"<tool_call>(.*?)</tool_call>", body, flags=re.S):
            if "<tool_sep>" in block:
                name, rest = block.split("<tool_sep>", 1)
            else:
                lines = block.strip().splitlines()
                name = lines[0] if lines else "tool"
                rest = "\n".join(lines[1:])
            args: dict[str, Any] = {}
            key_iter = list(re.finditer(r"<arg_key>(.*?)</arg_key>", rest, flags=re.S))
            val_iter = list(re.finditer(r"<arg_value>(.*?)</arg_value>", rest, flags=re.S))
            for key_m, val_m in zip(key_iter, val_iter):
                key = key_m.group(1).strip()
                raw_val = val_m.group(1).strip()
                try:
                    val: Any = json.loads(raw_val)
                except Exception:
                    val = raw_val
                args[key] = val
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": name.strip(), "arguments": json.dumps(args, ensure_ascii=False)},
                }
            )
        visible = text[: body_match.start()].strip() + text[body_match.end() :].strip()
        return visible.strip(), calls or None

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int | None) -> dict[str, Any]:
        max_new_tokens = int(max_tokens or self.max_default_tokens)
        prompt = self.render_prompt(messages, tools)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        cache = make_prompt_cache(self.model)
        generated: list[int] = []
        step_timings: list[float] = []

        t_step = time.time()
        logits = self.model(mx.array([prompt_ids], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        next_id = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        generated.append(next_id)
        step_timings.append(round(time.time() - t_step, 3))

        while len(generated) < max_new_tokens and generated[-1] != self.eos_id:
            t_step = time.time()
            logits = self.model(mx.array([[generated[-1]]], dtype=mx.int32), cache=cache)
            mx.eval(logits)
            next_id = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            generated.append(next_id)
            step_timings.append(round(time.time() - t_step, 3))

        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        clean = self.tokenizer.decode(generated, skip_special_tokens=True)
        clean, reasoning = self._strip_reasoning(clean)
        content, tool_calls = self._parse_hy3_tool_calls(clean)
        finish_reason = "stop" if generated and generated[-1] == self.eos_id else "length"
        if tool_calls:
            finish_reason = "tool_calls"
        return {
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(generated),
            "raw_text": raw,
            "step_timings_s": step_timings,
            "sidecar_store": self.lazy_mod.get_sidecar_store().stats(),
        }


RUNTIME: Hy3Runtime | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "hy3-canary/0.1"

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/health", "/healthz"}:
            assert RUNTIME is not None
            self._json(200, {"ok": True, "model": MODEL_ID, "sidecar_store": RUNTIME.lazy_mod.get_sidecar_store().stats()})
            return
        if self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": now_s(), "owned_by": "local"}]})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            assert RUNTIME is not None
            messages = body.get("messages") or []
            tools = body.get("tools") or None
            max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
            t0 = time.time()
            gen = RUNTIME.generate(messages, tools, max_tokens)
            msg: dict[str, Any] = {"role": "assistant", "content": gen["content"]}
            if gen.get("reasoning_content"):
                msg["reasoning_content"] = gen["reasoning_content"]
            if gen.get("tool_calls"):
                msg["tool_calls"] = gen["tool_calls"]
                msg["content"] = gen["content"] or None
            payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": now_s(),
                "model": body.get("model") or MODEL_ID,
                "choices": [{"index": 0, "message": msg, "finish_reason": gen["finish_reason"]}],
                "usage": {
                    "prompt_tokens": gen["prompt_tokens"],
                    "completion_tokens": gen["completion_tokens"],
                    "total_tokens": gen["prompt_tokens"] + gen["completion_tokens"],
                },
                "hy3_canary": {
                    "elapsed_s": round(time.time() - t0, 3),
                    "raw_text": gen["raw_text"],
                    "step_timings_s": gen["step_timings_s"],
                    "sidecar_store": gen["sidecar_store"],
                    "experiment": {
                        "slot_bank": int(os.environ.get("HY3_SLOT_BANK", "16")),
                        "retain_policy": os.environ.get("HY3_RETAIN_POLICY", "freq"),
                        "topk_cap": int(os.environ["HY3_TOPK_CAP"]) if os.environ.get("HY3_TOPK_CAP") else None,
                    },
                },
            }
            self._json(200, payload)
        except Exception as exc:  # keep canary debuggable
            self._json(500, {"error": repr(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8133)
    parser.add_argument("--slot-bank", type=int, default=32)
    parser.add_argument("--retain-policy", default="freq_last", choices=["id", "freq", "last", "freq_last", "last_freq"])
    parser.add_argument("--topk-cap", type=int, default=4)
    parser.add_argument("--max-default-tokens", type=int, default=256)
    args = parser.parse_args()

    os.environ.setdefault("HY3_SIDECAR_LAYOUT", str(PACKED_MANIFEST))
    os.environ["HY3_SLOT_BANK"] = str(args.slot_bank)
    os.environ["HY3_RETAIN_POLICY"] = args.retain_policy
    if args.topk_cap:
        os.environ["HY3_TOPK_CAP"] = str(args.topk_cap)

    global RUNTIME
    t0 = time.time()
    RUNTIME = Hy3Runtime(max_default_tokens=args.max_default_tokens)
    print(
        json.dumps(
            {
                "event": "ready",
                "model": MODEL_ID,
                "host": args.host,
                "port": args.port,
                "load_s": round(time.time() - t0, 3),
                "experiment": {"slot_bank": args.slot_bank, "retain_policy": args.retain_policy, "topk_cap": args.topk_cap},
            },
            sort_keys=True,
        ),
        flush=True,
    )
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
