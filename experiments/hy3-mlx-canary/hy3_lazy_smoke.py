#!/usr/bin/env python3
"""Guarded Hy3 lazy-sidecar smoke tests.

Modes:
- expert-read: load a few routed experts from the sidecar layout and verify MLX qmm works.
- resident-load: instantiate lazy Hy3 and load/eval only non-expert resident weights.
- forward-one: run a one-token full-model forward through lazy experts.

This is deliberately not an OpenAI server. First prove the memory substrate.
"""
from __future__ import annotations

import argparse
import gc
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

WORKDIR = Path(__file__).resolve().parent
MODEL_DIR = Path("/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX")
LAYOUT_PATH = WORKDIR / "hy3-sidecar-layout.json"
LAZY_MODEL_FILE = WORKDIR / "hy_v3_mlx_lazy.py"


def swap_used_gib() -> float:
    out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    # total = 3072.00M  used = 1964.88M  free = ...
    match = re.search(r"used = ([0-9.]+)([MGT])", out)
    if not match:
        return -1.0
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "M":
        return value / 1024.0
    if unit == "G":
        return value
    if unit == "T":
        return value * 1024.0
    return value


def start_swap_guard(max_swap_gib: float, max_delta_gib: float, sample_sec: float = 2.0) -> dict[str, float]:
    start = swap_used_gib()
    state = {"start_gib": start, "last_gib": start, "max_gib": start}

    def watch() -> None:
        while True:
            used = swap_used_gib()
            state["last_gib"] = used
            state["max_gib"] = max(state["max_gib"], used)
            if used > max_swap_gib or used - start > max_delta_gib:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "kill_reason": "swap_guard",
                            "swap_start_gib": round(start, 3),
                            "swap_used_gib": round(used, 3),
                            "max_swap_gib": max_swap_gib,
                            "max_delta_gib": max_delta_gib,
                        },
                        indent=2,
                    ),
                    flush=True,
                )
                os._exit(99)
            time.sleep(sample_sec)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    return state


def import_lazy_module():
    spec = importlib.util.spec_from_file_location("hy_v3_mlx_lazy", LAZY_MODEL_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {LAZY_MODEL_FILE}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hy_v3_mlx_lazy"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_config() -> dict[str, Any]:
    with (MODEL_DIR / "config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_resident_weight_dict() -> dict[str, mx.array]:
    """Load only non-switch_mlp arrays from MLX safetensors.

    mx.load returns lazy arrays; dropping switch_mlp references prevents expert eval.
    """
    weights: dict[str, mx.array] = {}
    for wf in sorted(glob.glob(str(MODEL_DIR / "model*.safetensors"))):
        shard = mx.load(wf)
        kept = {k: v for k, v in shard.items() if ".mlp.switch_mlp." not in k}
        weights.update(kept)
        del shard
        gc.collect()
    return weights


def apply_mlx_lm_quantization(model: nn.Module, config: dict[str, Any], weights: dict[str, mx.array]) -> None:
    quantization = config.get("quantization")
    if not quantization:
        raise RuntimeError("Hy3 preview config did not contain MLX quantization block")

    def class_predicate(path: str, module: nn.Module):
        if path in quantization:
            return quantization[path]
        if not hasattr(module, "to_quantized"):
            return False
        return f"{path}.scales" in weights

    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=quantization["bits"],
        mode=quantization.get("mode", "affine"),
        class_predicate=class_predicate,
    )


def load_lazy_model(eval_params: bool = True):
    mod = import_lazy_module()
    config = load_config()
    args = mod.ModelArgs.from_dict(config)
    model = mod.Model(args)
    model.eval()

    t0 = time.time()
    weights = load_resident_weight_dict()
    load_weight_s = time.time() - t0
    expert_keys = [k for k in weights if ".mlp.switch_mlp." in k]
    if expert_keys:
        raise RuntimeError(f"resident loader leaked expert tensors: {expert_keys[:3]}")

    apply_mlx_lm_quantization(model, config, weights)
    model.load_weights(list(weights.items()), strict=False)
    if eval_params:
        mx.eval(model.parameters())
    return model, mod, config, {"resident_tensors": len(weights), "load_weight_s": load_weight_s}


def mode_expert_read(args) -> dict[str, Any]:
    os.environ.setdefault("HY3_SIDECAR_LAYOUT", str(LAYOUT_PATH))
    os.environ["HY3_SLOT_BANK"] = str(args.slot_bank)
    mod = import_lazy_module()
    store = mod.Hy3SidecarStore(os.environ["HY3_SIDECAR_LAYOUT"], slot_bank=args.slot_bank)
    t0 = time.time()
    packed = store.get_experts(args.layer, list(range(args.experts)))
    # Tiny qmm sanity: x shape like SwitchGLU after expand, indices select each loaded expert once.
    x = mx.zeros((1, 1, 4096), dtype=mx.bfloat16)
    x = mx.expand_dims(x, (-2, -3))
    idx = mx.array(np.arange(args.experts, dtype=np.int32).reshape(1, 1, args.experts))
    up = mx.gather_qmm(
        x,
        packed["up_proj"]["weight"],
        packed["up_proj"]["scales"],
        packed["up_proj"]["biases"],
        rhs_indices=idx,
        transpose=True,
        group_size=64,
        bits=4,
        mode="affine",
    )
    gate = mx.gather_qmm(
        x,
        packed["gate_proj"]["weight"],
        packed["gate_proj"]["scales"],
        packed["gate_proj"]["biases"],
        rhs_indices=idx,
        transpose=True,
        group_size=64,
        bits=4,
        mode="affine",
    )
    hidden = mod.swiglu(gate, up)
    down = mx.gather_qmm(
        hidden,
        packed["down_proj"]["weight"],
        packed["down_proj"]["scales"],
        packed["down_proj"]["biases"],
        rhs_indices=idx,
        transpose=True,
        group_size=64,
        bits=4,
        mode="affine",
    )
    mx.eval(down)
    elapsed = time.time() - t0
    return {
        "ok": True,
        "mode": "expert-read",
        "layer": args.layer,
        "experts": args.experts,
        "elapsed_s": round(elapsed, 3),
        "down_shape": list(down.shape),
        "store": store.stats(),
    }


def mode_resident_load(args) -> dict[str, Any]:
    os.environ.setdefault("HY3_SIDECAR_LAYOUT", str(LAYOUT_PATH))
    os.environ["HY3_SLOT_BANK"] = str(args.slot_bank)
    t0 = time.time()
    model, mod, config, meta = load_lazy_model(eval_params=True)
    elapsed = time.time() - t0
    params = model.parameters()
    flat_count = 0
    expert_param_leaks = []
    from mlx.utils import tree_flatten

    for key, _ in tree_flatten(params):
        flat_count += 1
        if ".mlp.switch_mlp." in key:
            expert_param_leaks.append(key)
    return {
        "ok": not expert_param_leaks,
        "mode": "resident-load",
        "elapsed_s": round(elapsed, 3),
        "resident_tensors": meta["resident_tensors"],
        "load_weight_s": round(meta["load_weight_s"], 3),
        "parameter_leaves": flat_count,
        "expert_param_leaks": expert_param_leaks[:10],
        "sidecar_store": mod.get_sidecar_store().stats(),
    }


def mode_forward_one(args) -> dict[str, Any]:
    os.environ.setdefault("HY3_SIDECAR_LAYOUT", str(LAYOUT_PATH))
    os.environ["HY3_SLOT_BANK"] = str(args.slot_bank)
    t0 = time.time()
    model, mod, config, meta = load_lazy_model(eval_params=True)
    load_s = time.time() - t0
    token_id = int(args.token_id if args.token_id is not None else config.get("bos_token_id", 120000))
    ids = mx.array([[token_id]], dtype=mx.int32)
    if getattr(args, "profile_layers", False) and hasattr(mod, "reset_profile"):
        mod.reset_profile()
    t1 = time.time()
    logits = model(ids)
    mx.eval(logits)
    forward_s = time.time() - t1
    next_id = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    return {
        "ok": True,
        "mode": "forward-one",
        "token_id": token_id,
        "next_id": next_id,
        "logits_shape": list(logits.shape),
        "load_s": round(load_s, 3),
        "forward_s": round(forward_s, 3),
        "resident_tensors": meta["resident_tensors"],
        "disable_shared_mlp": bool(getattr(args, "disable_shared_mlp", False)),
        "disable_routed_mlp": bool(getattr(args, "disable_routed_mlp", False)),
        "sidecar_store": mod.get_sidecar_store().stats(),
        "profile": mod.get_profile_stats() if getattr(args, "profile_layers", False) and hasattr(mod, "get_profile_stats") else None,
    }


def mode_generate_raw(args) -> dict[str, Any]:
    from transformers import AutoTokenizer

    os.environ.setdefault("HY3_SIDECAR_LAYOUT", str(LAYOUT_PATH))
    os.environ["HY3_SLOT_BANK"] = str(args.slot_bank)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    t0 = time.time()
    model, mod, config, meta = load_lazy_model(eval_params=True)
    load_s = time.time() - t0

    ids = list(prompt_ids)
    step_timings = []
    generated = []
    for step in range(args.max_new_tokens):
        x = mx.array([ids], dtype=mx.int32)
        t_step = time.time()
        logits = model(x)
        mx.eval(logits)
        next_id = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        step_timings.append(round(time.time() - t_step, 3))
        generated.append(next_id)
        ids.append(next_id)
        if next_id == int(config.get("eos_token_id", -1)):
            break
    text = tokenizer.decode(generated, skip_special_tokens=False)
    clean_text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "ok": True,
        "mode": "generate-raw",
        "prompt": args.prompt,
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": args.max_new_tokens,
        "generated_ids": generated,
        "generated_text": text,
        "generated_text_clean": clean_text,
        "exact_pong": clean_text.strip() == "pong",
        "load_s": round(load_s, 3),
        "step_timings_s": step_timings,
        "resident_tensors": meta["resident_tensors"],
        "sidecar_store": mod.get_sidecar_store().stats(),
    }


def mode_generate_cache(args) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from mlx_lm.models.cache import make_prompt_cache

    os.environ.setdefault("HY3_SIDECAR_LAYOUT", str(LAYOUT_PATH))
    os.environ["HY3_SLOT_BANK"] = str(args.slot_bank)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    t0 = time.time()
    model, mod, config, meta = load_lazy_model(eval_params=True)
    load_s = time.time() - t0

    cache = make_prompt_cache(model)
    generated = []
    step_timings = []
    t_step = time.time()
    logits = model(mx.array([prompt_ids], dtype=mx.int32), cache=cache)
    mx.eval(logits)
    next_id = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    generated.append(next_id)
    step_timings.append(round(time.time() - t_step, 3))

    eos_id = int(config.get("eos_token_id", -1))
    while len(generated) < args.max_new_tokens and generated[-1] != eos_id:
        t_step = time.time()
        logits = model(mx.array([[generated[-1]]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        next_id = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        generated.append(next_id)
        step_timings.append(round(time.time() - t_step, 3))

    text = tokenizer.decode(generated, skip_special_tokens=False)
    clean_text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "ok": True,
        "mode": "generate-cache",
        "prompt": args.prompt,
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": args.max_new_tokens,
        "generated_ids": generated,
        "generated_text": text,
        "generated_text_clean": clean_text,
        "exact_pong": clean_text.strip() == "pong",
        "load_s": round(load_s, 3),
        "step_timings_s": step_timings,
        "resident_tensors": meta["resident_tensors"],
        "sidecar_store": mod.get_sidecar_store().stats(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["expert-read", "resident-load", "forward-one", "generate-raw", "generate-cache"])
    parser.add_argument("--slot-bank", type=int, default=8, help="per-layer expert LRU size for prototype")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--token-id", type=int)
    parser.add_argument("--prompt", default="Reply with exactly pong.")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--max-swap-gib", type=float, default=48.0)
    parser.add_argument("--max-swap-delta-gib", type=float, default=16.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--profile-layers", action="store_true", help="force per-layer evals and emit attention/MLP timing profile")
    parser.add_argument("--disable-shared-mlp", action="store_true", help="diagnostic only: skip Hy3 shared expert/resident MLP contribution")
    parser.add_argument("--disable-routed-mlp", action="store_true", help="diagnostic only: skip Hy3 routed expert contribution")
    parser.add_argument("--sync-timers", action="store_true", help="call mx.synchronize() after timed evals for honest MLX timing")
    parser.add_argument("--retain-policy", choices=["id", "freq", "last", "freq-last", "last-freq"], help="expert cache retention policy after prompt prefill")
    parser.add_argument("--topk-cap", type=int, help="experimental cap for routed experts per token; default uses model top-k")
    args = parser.parse_args()

    if args.profile_layers:
        os.environ["HY3_PROFILE_LAYERS"] = "1"
    if args.disable_shared_mlp:
        os.environ["HY3_DISABLE_SHARED_MLP"] = "1"
    if args.disable_routed_mlp:
        os.environ["HY3_DISABLE_ROUTED_MLP"] = "1"
    if args.sync_timers:
        os.environ["HY3_SYNC_TIMERS"] = "1"
    if args.retain_policy:
        os.environ["HY3_RETAIN_POLICY"] = args.retain_policy.replace("-", "_")
    if args.topk_cap is not None:
        os.environ["HY3_TOPK_CAP"] = str(args.topk_cap)

    guard = start_swap_guard(args.max_swap_gib, args.max_swap_delta_gib)
    result: dict[str, Any]
    try:
        if args.mode == "expert-read":
            result = mode_expert_read(args)
        elif args.mode == "resident-load":
            result = mode_resident_load(args)
        elif args.mode == "forward-one":
            result = mode_forward_one(args)
        elif args.mode == "generate-raw":
            result = mode_generate_raw(args)
        elif args.mode == "generate-cache":
            result = mode_generate_cache(args)
        else:
            raise AssertionError(args.mode)
    except Exception as exc:
        result = {"ok": False, "mode": args.mode, "error": repr(exc)}
        raise
    finally:
        # Give watchdog one last sample window in long runs.
        time.sleep(0.1)

    result["experiment"] = {
        "slot_bank": args.slot_bank,
        "retain_policy": os.environ.get("HY3_RETAIN_POLICY") or ("id" if os.environ.get("HY3_RETAIN_FREQUENT_EXPERTS", "1").lower() in {"0", "false", "no", "off"} else "freq"),
        "topk_cap": int(os.environ["HY3_TOPK_CAP"]) if os.environ.get("HY3_TOPK_CAP") else None,
    }
    result["swap"] = {
        "start_gib": round(guard["start_gib"], 3),
        "last_gib": round(guard["last_gib"], 3),
        "max_gib": round(guard["max_gib"], 3),
        "delta_gib": round(guard["max_gib"] - guard["start_gib"], 3),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
