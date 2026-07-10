#!/usr/bin/env python3
"""Hy3 routed expert microbench.

Purpose: isolate the slow routed path into:
- up/gate/down `mx.gather_qmm`
- swiglu
- post-qmm top-k weighting/sum

This does NOT instantiate the full model. It benchmarks one layer's 8 selected experts,
either loaded from the packed sidecar or generated as synthetic in-memory banks.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

WORKDIR = Path(__file__).resolve().parent
LAZY_MODEL_FILE = WORKDIR / "hy_v3_mlx_lazy.py"
PACKED_MANIFEST = Path("/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/manifest.json")
LOG_DIR = Path("/Volumes/ModelSSD/logs/hy3-mlx-canary")


def swap_used_gib() -> float:
    out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    import re

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


def import_lazy_module():
    spec = importlib.util.spec_from_file_location("hy_v3_mlx_lazy", LAZY_MODEL_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {LAZY_MODEL_FILE}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hy_v3_mlx_lazy"] = mod
    spec.loader.exec_module(mod)
    return mod


def sync_eval(*arrays: Any) -> None:
    mx.eval(*arrays)
    mx.synchronize()


def timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    t0 = time.perf_counter()
    out = fn()
    if isinstance(out, tuple):
        sync_eval(*out)
    else:
        sync_eval(out)
    return time.perf_counter() - t0, out


def load_real_bank(mod: Any, layer: int, experts: int, slot_bank: int, manifest: Path) -> tuple[dict[str, dict[str, mx.array]], dict[str, Any]]:
    store = mod.Hy3SidecarStore(str(manifest), slot_bank=slot_bank)
    t0 = time.perf_counter()
    bank = store.get_experts(layer, list(range(experts)))
    load_s = time.perf_counter() - t0
    return bank, {"source": "real", "load_s": round(load_s, 6), "store": store.stats()}


def synthetic_bank(experts: int, seed: int = 13) -> tuple[dict[str, dict[str, mx.array]], dict[str, Any]]:
    rng = np.random.default_rng(seed)

    def u32(shape):
        # q4 packed words. Random data avoids any chance of a zero fast path.
        arr = rng.integers(0, np.iinfo(np.uint32).max, size=shape, dtype=np.uint32)
        return mx.array(arr, dtype=mx.uint32)

    def bf16(shape):
        # Small nonzero values; shape matches MLX affine q4 scales/biases.
        arr = rng.standard_normal(size=shape).astype(np.float32) * 0.01
        return mx.array(arr, dtype=mx.float32).astype(mx.bfloat16)

    t0 = time.perf_counter()
    bank = {
        "up_proj": {
            "weight": u32((experts, 1536, 512)),
            "scales": bf16((experts, 1536, 64)),
            "biases": bf16((experts, 1536, 64)),
        },
        "gate_proj": {
            "weight": u32((experts, 1536, 512)),
            "scales": bf16((experts, 1536, 64)),
            "biases": bf16((experts, 1536, 64)),
        },
        "down_proj": {
            "weight": u32((experts, 4096, 192)),
            "scales": bf16((experts, 4096, 24)),
            "biases": bf16((experts, 4096, 24)),
        },
    }
    sync_eval(bank)
    return bank, {"source": "synthetic", "create_s": round(time.perf_counter() - t0, 6)}


def qlinear(x: mx.array, indices: mx.array, packed: dict[str, mx.array]) -> mx.array:
    return mx.gather_qmm(
        x,
        packed["weight"],
        packed["scales"],
        packed["biases"],
        rhs_indices=indices,
        transpose=True,
        group_size=64,
        bits=4,
        mode="affine",
        sorted_indices=False,
    )


def run_once(mod: Any, bank: dict[str, dict[str, mx.array]], experts: int, x_source: str, seed: int) -> dict[str, Any]:
    if x_source == "zeros":
        x = mx.zeros((1, 1, 4096), dtype=mx.bfloat16)
    else:
        rng = np.random.default_rng(seed)
        x = mx.array(rng.standard_normal((1, 1, 4096)).astype(np.float32), dtype=mx.float32).astype(mx.bfloat16)
    xq = mx.expand_dims(x, (-2, -3))
    idx = mx.array(np.arange(experts, dtype=np.int32).reshape(1, 1, experts))
    scores = mx.ones((1, 1, experts), dtype=mx.float32) / float(experts)
    sync_eval(xq, idx, scores)

    up_s, up = timed(lambda: qlinear(xq, idx, bank["up_proj"]))
    gate_s, gate = timed(lambda: qlinear(xq, idx, bank["gate_proj"]))
    swiglu_s, hidden = timed(lambda: mod.swiglu(gate, up))
    down_s, down = timed(lambda: qlinear(hidden, idx, bank["down_proj"]))
    squeeze_s, y = timed(lambda: down.squeeze(-2))
    weight_s, weighted = timed(lambda: (y * scores[..., None].astype(mx.float32)).sum(axis=-2).astype(y.dtype))

    full_s, full_weighted = timed(
        lambda: (
            qlinear(
                mod.swiglu(
                    qlinear(xq, idx, bank["gate_proj"]),
                    qlinear(xq, idx, bank["up_proj"]),
                ),
                idx,
                bank["down_proj"],
            ).squeeze(-2)
            * scores[..., None].astype(mx.float32)
        ).sum(axis=-2)
    )

    return {
        "up_qmm_s": round(up_s, 6),
        "gate_qmm_s": round(gate_s, 6),
        "swiglu_s": round(swiglu_s, 6),
        "down_qmm_s": round(down_s, 6),
        "squeeze_s": round(squeeze_s, 6),
        "weight_sum_s": round(weight_s, 6),
        "separated_total_s": round(up_s + gate_s + swiglu_s + down_s + squeeze_s + weight_s, 6),
        "full_fused_expr_s": round(full_s, 6),
        "shapes": {
            "up": list(up.shape),
            "gate": list(gate.shape),
            "hidden": list(hidden.shape),
            "down": list(down.shape),
            "y": list(y.shape),
            "weighted": list(weighted.shape),
            "full_weighted": list(full_weighted.shape),
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "up_qmm_s",
        "gate_qmm_s",
        "swiglu_s",
        "down_qmm_s",
        "squeeze_s",
        "weight_sum_s",
        "separated_total_s",
        "full_fused_expr_s",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows]
        out[key] = {
            "mean": round(statistics.mean(vals), 6),
            "median": round(statistics.median(vals), 6),
            "min": round(min(vals), 6),
            "max": round(max(vals), 6),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--manifest", type=Path, default=PACKED_MANIFEST)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--slot-bank", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--x-source", choices=["zeros", "random"], default="random")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    mod = import_lazy_module()
    swap_start = swap_used_gib()
    if args.source == "real":
        bank, source_meta = load_real_bank(mod, args.layer, args.experts, args.slot_bank, args.manifest)
    else:
        bank, source_meta = synthetic_bank(args.experts)

    warmup_rows = [run_once(mod, bank, args.experts, args.x_source, 10_000 + i) for i in range(args.warmup)]
    rows = [run_once(mod, bank, args.experts, args.x_source, 20_000 + i) for i in range(args.repeat)]
    swap_end = swap_used_gib()
    result = {
        "ok": True,
        "source": args.source,
        "layer": args.layer,
        "experts": args.experts,
        "x_source": args.x_source,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "source_meta": source_meta,
        "summary": summarize(rows),
        "rows": rows,
        "warmup_rows": warmup_rows,
        "swap": {
            "start_gib": round(swap_start, 3),
            "end_gib": round(swap_end, 3),
            "delta_gib": round(swap_end - swap_start, 3),
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
