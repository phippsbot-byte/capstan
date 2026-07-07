#!/usr/bin/env python3
"""Export a compact real Hy3 routed-layer parity fixture.

This runs only through the requested decoder layer, captures the hidden state
entering that layer's routed MoE, the router-selected experts/weights, and the
Python/MLX routed output before the shared MLP is added. The fixture is small
JSON so native code can validate expert-bank layout and q4 math without loading
the full Python model path.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from hy3_lazy_smoke import MODEL_DIR, load_lazy_model, start_swap_guard

DEFAULT_OUT = Path("/Volumes/ModelSSD/logs/hy3-mlx-canary/parity-fixtures/hy3-layer1-top5-bos.json")
PACKED_MANIFEST = Path("/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/manifest.json")


def run_to_layer(model: Any, mod: Any, ids: list[int], target_layer: int) -> None:
    x = mx.array([ids], dtype=mx.int32)
    h = model.model.embed_tokens(x)
    cache = [None] * len(model.model.layers)
    mask = mod.create_attention_mask(h, cache[0])
    for idx, layer in enumerate(model.model.layers[: target_layer + 1]):
        h = layer(h, mask, cache[idx])
        mx.eval(h)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--layout", type=Path, default=PACKED_MANIFEST)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--token", type=int, default=0)
    parser.add_argument("--slot-bank", type=int, default=16)
    parser.add_argument("--topk-cap", type=int, default=5)
    parser.add_argument("--token-id", type=int, help="single token id; defaults to model BOS")
    parser.add_argument("--prompt", help="optional raw prompt to tokenize instead of --token-id/BOS")
    parser.add_argument("--max-swap-gib", type=float, default=64.0)
    parser.add_argument("--max-swap-delta-gib", type=float, default=12.0)
    args = parser.parse_args()

    os.environ["HY3_SIDECAR_LAYOUT"] = str(args.layout)
    os.environ["HY3_SLOT_BANK"] = str(args.slot_bank)
    os.environ["HY3_TOPK_CAP"] = str(args.topk_cap)
    os.environ["HY3_PARITY_LAYER"] = str(args.layer)
    os.environ["HY3_PARITY_TOKEN"] = str(args.token)
    os.environ["HY3_PARITY_BATCH"] = "0"

    guard = start_swap_guard(args.max_swap_gib, args.max_swap_delta_gib)
    t0 = time.time()
    model, mod, config, meta = load_lazy_model(eval_params=True)
    if hasattr(mod, "reset_parity_fixture"):
        mod.reset_parity_fixture()
    load_s = time.time() - t0

    if args.prompt:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
        ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    else:
        token_id = int(args.token_id if args.token_id is not None else config.get("bos_token_id", 120000))
        ids = [token_id]

    t1 = time.time()
    run_to_layer(model, mod, ids, args.layer)
    forward_s = time.time() - t1

    fixture = mod.get_parity_fixture() if hasattr(mod, "get_parity_fixture") else None
    if not fixture:
        raise RuntimeError("no parity fixture captured; check --layer/--token and HY3_PARITY_* support")

    metadata = {
        "model_dir": str(MODEL_DIR),
        "layout": str(args.layout),
        "slot_bank": args.slot_bank,
        "topk_cap": args.topk_cap,
        "input_ids": ids,
        "load_s": round(load_s, 3),
        "forward_to_layer_s": round(forward_s, 3),
        "resident_tensors": meta["resident_tensors"],
        "load_weight_s": round(meta["load_weight_s"], 3),
        "sidecar_store": mod.get_sidecar_store().stats(),
        "swap": {
            "start_gib": round(guard["start_gib"], 3),
            "last_gib": round(guard["last_gib"], 3),
            "max_gib": round(guard["max_gib"], 3),
            "delta_gib": round(guard["max_gib"] - guard["start_gib"], 3),
        },
    }
    path = mod.write_parity_fixture(str(args.out), metadata=metadata)
    summary = {
        "ok": True,
        "fixture": path,
        "layer": fixture["layer"],
        "token": fixture["token"],
        "topk": fixture["topk"],
        "experts": fixture["experts"],
        "load_s": round(load_s, 3),
        "forward_to_layer_s": round(forward_s, 3),
        "sidecar_store": metadata["sidecar_store"],
        "swap": metadata["swap"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
