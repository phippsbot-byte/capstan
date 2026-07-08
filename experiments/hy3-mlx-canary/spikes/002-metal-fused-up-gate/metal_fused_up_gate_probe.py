#!/usr/bin/env python3
"""Spike: fused on-the-fly q4 up+gate Metal kernel for one Hy3 expert.

This tests the smallest useful fused-MoE primitive:
  hidden = silu(gate_q4(x)) * up_q4(x)

If this cannot beat MLX gather_qmm + swiglu, a full routed-MoE Metal kernel is
not worth building in this shape yet.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

WORKDIR = Path(__file__).resolve().parents[2]
LAZY_MODEL_FILE = WORKDIR / "hy_v3_mlx_lazy.py"
PACKED_MANIFEST = Path("/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/manifest.json")
OUT_DIM = 1536
IN_DIM = 4096
PACKED_WORDS = 512
GROUPS = 64
DOWN_OUT_DIM = 4096
DOWN_IN_DIM = 1536
DOWN_PACKED_WORDS = 192
DOWN_GROUPS = 24
THREADS = 256


def import_lazy_module():
    spec = importlib.util.spec_from_file_location("hy_v3_mlx_lazy", LAZY_MODEL_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {LAZY_MODEL_FILE}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hy_v3_mlx_lazy"] = mod
    spec.loader.exec_module(mod)
    return mod


def sync(*arrays: Any) -> None:
    mx.eval(*arrays)
    if hasattr(mx, "synchronize"):
        mx.synchronize()


def timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    t0 = time.perf_counter()
    out = fn()
    sync(out)
    return time.perf_counter() - t0, out


def summarize(vals: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(vals), 6),
        "median": round(statistics.median(vals), 6),
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
    }


def build_single_projection_kernel(name: str):
    source = f"""
        uint row = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        threadgroup float partial[{THREADS}];
        float acc = 0.0f;
        if (row < {OUT_DIM}u) {{
            for (uint i = tid; i < {IN_DIM}u; i += {THREADS}u) {{
                uint word = w[row * {PACKED_WORDS}u + (i >> 3)];
                uint q = (word >> ((i & 7u) * 4u)) & 0xFu;
                uint group = i >> 6;
                float scale = scales[row * {GROUPS}u + group];
                float bias = biases[row * {GROUPS}u + group];
                acc += x[i] * (float(q) * scale + bias);
            }}
        }}
        partial[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = {THREADS // 2}u; stride > 0u; stride >>= 1u) {{
            if (tid < stride) partial[tid] += partial[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid == 0u && row < {OUT_DIM}u) out[row] = partial[0];
    """
    return mx.fast.metal_kernel(
        name=f"hy3_single_q4_{name}_{OUT_DIM}_{IN_DIM}_{THREADS}",
        input_names=["x", "w", "scales", "biases"],
        output_names=["out"],
        source=source,
    )


def build_fused_up_gate_kernel():
    source = f"""
        uint row = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        threadgroup float partial_up[{THREADS}];
        threadgroup float partial_gate[{THREADS}];
        float acc_up = 0.0f;
        float acc_gate = 0.0f;
        if (row < {OUT_DIM}u) {{
            for (uint i = tid; i < {IN_DIM}u; i += {THREADS}u) {{
                float xv = x[i];
                uint word_up = up_w[row * {PACKED_WORDS}u + (i >> 3)];
                uint q_up = (word_up >> ((i & 7u) * 4u)) & 0xFu;
                uint word_gate = gate_w[row * {PACKED_WORDS}u + (i >> 3)];
                uint q_gate = (word_gate >> ((i & 7u) * 4u)) & 0xFu;
                uint group = i >> 6;
                float up_scale = up_scales[row * {GROUPS}u + group];
                float up_bias = up_biases[row * {GROUPS}u + group];
                float gate_scale = gate_scales[row * {GROUPS}u + group];
                float gate_bias = gate_biases[row * {GROUPS}u + group];
                acc_up += xv * (float(q_up) * up_scale + up_bias);
                acc_gate += xv * (float(q_gate) * gate_scale + gate_bias);
            }}
        }}
        partial_up[tid] = acc_up;
        partial_gate[tid] = acc_gate;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = {THREADS // 2}u; stride > 0u; stride >>= 1u) {{
            if (tid < stride) {{
                partial_up[tid] += partial_up[tid + stride];
                partial_gate[tid] += partial_gate[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid == 0u && row < {OUT_DIM}u) {{
            float g = partial_gate[0];
            out[row] = (g / (1.0f + exp(-g))) * partial_up[0];
        }}
    """
    return mx.fast.metal_kernel(
        name=f"hy3_fused_q4_up_gate_swiglu_{OUT_DIM}_{IN_DIM}_{THREADS}",
        input_names=["x", "up_w", "up_scales", "up_biases", "gate_w", "gate_scales", "gate_biases"],
        output_names=["out"],
        source=source,
    )


def qlinear_gather_prepacked(xq: mx.array, idx: mx.array, packed: dict[str, mx.array], out_dim: int) -> mx.array:
    out = mx.gather_qmm(
        xq,
        packed["weight"],
        packed["scales"],
        packed["biases"],
        rhs_indices=idx,
        transpose=True,
        group_size=64,
        bits=4,
        mode="affine",
        sorted_indices=False,
    )
    return out.reshape(out_dim).astype(mx.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PACKED_MANIFEST)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    mod = import_lazy_module()
    store = mod.Hy3SidecarStore(str(args.manifest), slot_bank=8)
    bank = store.get_experts(args.layer, [args.expert])
    up = bank["up_proj"]
    gate = bank["gate_proj"]
    down = bank["down_proj"]

    up_w = up["weight"][0]
    up_scales = up["scales"][0].astype(mx.float32)
    up_biases = up["biases"][0].astype(mx.float32)
    gate_w = gate["weight"][0]
    gate_scales = gate["scales"][0].astype(mx.float32)
    gate_biases = gate["biases"][0].astype(mx.float32)
    down_w = down["weight"][0]
    down_scales = down["scales"][0].astype(mx.float32)
    down_biases = down["biases"][0].astype(mx.float32)

    rng = np.random.default_rng(123)
    x = mx.array(rng.standard_normal((IN_DIM,)).astype(np.float32))
    xq = mx.expand_dims(mx.expand_dims(mx.expand_dims(x.astype(mx.bfloat16), 0), 0), 0)
    idx = mx.array([[[0]]], dtype=mx.int32)
    up_packed = {
        "weight": mx.expand_dims(up_w, 0),
        "scales": mx.expand_dims(up_scales.astype(mx.bfloat16), 0),
        "biases": mx.expand_dims(up_biases.astype(mx.bfloat16), 0),
    }
    gate_packed = {
        "weight": mx.expand_dims(gate_w, 0),
        "scales": mx.expand_dims(gate_scales.astype(mx.bfloat16), 0),
        "biases": mx.expand_dims(gate_biases.astype(mx.bfloat16), 0),
    }
    down_packed = {
        "weight": mx.expand_dims(down_w, 0),
        "scales": mx.expand_dims(down_scales.astype(mx.bfloat16), 0),
        "biases": mx.expand_dims(down_biases.astype(mx.bfloat16), 0),
    }
    sync(x, xq, idx, up_packed, gate_packed, down_packed)

    single_up_kernel = build_single_projection_kernel("up")
    single_gate_kernel = build_single_projection_kernel("gate")
    fused_kernel = build_fused_up_gate_kernel()

    def single_up_call() -> mx.array:
        return single_up_kernel(
            inputs=[x, up_w, up_scales, up_biases],
            output_shapes=[(OUT_DIM,)],
            output_dtypes=[mx.float32],
            grid=(OUT_DIM * THREADS, 1, 1),
            threadgroup=(THREADS, 1, 1),
        )[0]

    def single_gate_call() -> mx.array:
        return single_gate_kernel(
            inputs=[x, gate_w, gate_scales, gate_biases],
            output_shapes=[(OUT_DIM,)],
            output_dtypes=[mx.float32],
            grid=(OUT_DIM * THREADS, 1, 1),
            threadgroup=(THREADS, 1, 1),
        )[0]

    def separate_metal_call() -> mx.array:
        u = single_up_call()
        g = single_gate_call()
        return (g / (1.0 + mx.exp(-g))) * u

    def fused_metal_call() -> mx.array:
        return fused_kernel(
            inputs=[x, up_w, up_scales, up_biases, gate_w, gate_scales, gate_biases],
            output_shapes=[(OUT_DIM,)],
            output_dtypes=[mx.float32],
            grid=(OUT_DIM * THREADS, 1, 1),
            threadgroup=(THREADS, 1, 1),
        )[0]

    def gather_swiglu_call() -> mx.array:
        u = qlinear_gather_prepacked(xq, idx, up_packed, OUT_DIM)
        g = qlinear_gather_prepacked(xq, idx, gate_packed, OUT_DIM)
        return (g / (1.0 + mx.exp(-g))) * u

    def down_from_hidden(hidden: mx.array) -> mx.array:
        hiddenq = mx.expand_dims(mx.expand_dims(mx.expand_dims(hidden.astype(mx.bfloat16), 0), 0), 0)
        return qlinear_gather_prepacked(hiddenq, idx, down_packed, DOWN_OUT_DIM)

    def gather_full_expert_call() -> mx.array:
        return down_from_hidden(gather_swiglu_call())

    def fused_full_expert_call() -> mx.array:
        return down_from_hidden(fused_metal_call())

    gather_out = gather_swiglu_call()
    separate_out = separate_metal_call()
    fused_out = fused_metal_call()
    gather_full_out = gather_full_expert_call()
    fused_full_out = fused_full_expert_call()
    sync(gather_out, separate_out, fused_out, gather_full_out, fused_full_out)
    gather_np = np.array(gather_out, copy=False)
    separate_np = np.array(separate_out, copy=False)
    fused_np = np.array(fused_out, copy=False)
    gather_full_np = np.array(gather_full_out, copy=False)
    fused_full_np = np.array(fused_full_out, copy=False)
    diff_fused = np.abs(fused_np - gather_np)
    diff_separate = np.abs(separate_np - gather_np)
    diff_full = np.abs(fused_full_np - gather_full_np)

    for _ in range(args.warmup):
        sync(gather_swiglu_call())
        sync(separate_metal_call())
        sync(fused_metal_call())
        sync(gather_full_expert_call())
        sync(fused_full_expert_call())

    gather_times: list[float] = []
    separate_times: list[float] = []
    fused_times: list[float] = []
    gather_full_times: list[float] = []
    fused_full_times: list[float] = []
    for _ in range(args.repeat):
        dt, _ = timed(gather_swiglu_call)
        gather_times.append(dt)
        dt, _ = timed(separate_metal_call)
        separate_times.append(dt)
        dt, _ = timed(fused_metal_call)
        fused_times.append(dt)
        dt, _ = timed(gather_full_expert_call)
        gather_full_times.append(dt)
        dt, _ = timed(fused_full_expert_call)
        fused_full_times.append(dt)

    gather_summary = summarize(gather_times)
    separate_summary = summarize(separate_times)
    fused_summary = summarize(fused_times)
    gather_full_summary = summarize(gather_full_times)
    fused_full_summary = summarize(fused_full_times)
    result = {
        "ok": True,
        "schema": "hy3-metal-fused-up-gate-probe-v1",
        "layer": args.layer,
        "expert": args.expert,
        "shape": {
            "up_gate": {"out_dim": OUT_DIM, "in_dim": IN_DIM, "packed_words": PACKED_WORDS, "groups": GROUPS},
            "down": {"out_dim": DOWN_OUT_DIM, "in_dim": DOWN_IN_DIM, "packed_words": DOWN_PACKED_WORDS, "groups": DOWN_GROUPS},
            "threads": THREADS,
        },
        "max_abs_diff_fused_vs_gather_swiglu": float(diff_fused.max()),
        "mean_abs_diff_fused_vs_gather_swiglu": float(diff_fused.mean()),
        "max_abs_diff_separate_metal_vs_gather_swiglu": float(diff_separate.max()),
        "mean_abs_diff_separate_metal_vs_gather_swiglu": float(diff_separate.mean()),
        "max_abs_diff_fused_full_vs_gather_full": float(diff_full.max()),
        "mean_abs_diff_fused_full_vs_gather_full": float(diff_full.mean()),
        "gather_qmm_swiglu_s": gather_summary,
        "separate_metal_swiglu_s": separate_summary,
        "fused_metal_swiglu_s": fused_summary,
        "gather_qmm_full_expert_s": gather_full_summary,
        "fused_up_gate_plus_gather_down_full_expert_s": fused_full_summary,
        "fused_speedup_vs_gather_median": round(gather_summary["median"] / max(fused_summary["median"], 1e-12), 4),
        "fused_speedup_vs_separate_metal_median": round(separate_summary["median"] / max(fused_summary["median"], 1e-12), 4),
        "full_expert_speedup_vs_gather_median": round(gather_full_summary["median"] / max(fused_full_summary["median"], 1e-12), 4),
        "sidecar_store": store.stats(),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
