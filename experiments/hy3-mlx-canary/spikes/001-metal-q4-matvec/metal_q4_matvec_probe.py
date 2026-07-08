#!/usr/bin/env python3
"""Spike: direct MLX custom Metal q4 affine matvec for one Hy3 expert tensor.

This is deliberately tiny: one projection, one expert, one token. It answers
whether a custom on-the-fly q4 matvec can beat MLX gather_qmm before we build a
real routed-MoE Metal path.
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


def build_kernel(out_dim: int, in_dim: int, packed_words: int, groups: int, mode: str):
    if mode == "naive":
        source = f"""
            uint row = thread_position_in_grid.x;
            if (row >= {out_dim}u) return;
            float acc = 0.0f;
            for (uint i = 0; i < {in_dim}u; ++i) {{
                uint word = w[row * {packed_words}u + (i >> 3)];
                uint q = (word >> ((i & 7u) * 4u)) & 0xFu;
                uint group = i >> 6;
                float scale = scales[row * {groups}u + group];
                float bias = biases[row * {groups}u + group];
                acc += x[i] * (float(q) * scale + bias);
            }}
            out[row] = acc;
        """
        grid_scale = 1
    elif mode == "row-reduce":
        source = f"""
            uint row = threadgroup_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x;
            threadgroup float partial[256];
            float acc = 0.0f;
            if (row < {out_dim}u) {{
                for (uint i = tid; i < {in_dim}u; i += 256u) {{
                    uint word = w[row * {packed_words}u + (i >> 3)];
                    uint q = (word >> ((i & 7u) * 4u)) & 0xFu;
                    uint group = i >> 6;
                    float scale = scales[row * {groups}u + group];
                    float bias = biases[row * {groups}u + group];
                    acc += x[i] * (float(q) * scale + bias);
                }}
            }}
            partial[tid] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 128u; stride > 0u; stride >>= 1u) {{
                if (tid < stride) partial[tid] += partial[tid + stride];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            if (tid == 0u && row < {out_dim}u) out[row] = partial[0];
        """
        grid_scale = 256
    else:
        raise ValueError(f"unknown kernel mode: {mode}")
    kernel = mx.fast.metal_kernel(
        name=f"hy3_q4_matvec_{mode.replace('-', '_')}_{out_dim}_{in_dim}_{packed_words}_{groups}",
        input_names=["x", "w", "scales", "biases"],
        output_names=["out"],
        source=source,
    )
    return kernel, grid_scale


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


def summarize(vals: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(vals), 6),
        "median": round(statistics.median(vals), 6),
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PACKED_MANIFEST)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--family", choices=["up_proj", "gate_proj", "down_proj"], default="up_proj")
    parser.add_argument("--kernel", choices=["naive", "row-reduce"], default="row-reduce")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    mod = import_lazy_module()
    store = mod.Hy3SidecarStore(str(args.manifest), slot_bank=8)
    bank = store.get_experts(args.layer, [args.expert])
    proj = bank[args.family]
    w = proj["weight"][0]
    scales = proj["scales"][0].astype(mx.float32)
    biases = proj["biases"][0].astype(mx.float32)
    if args.family in {"up_proj", "gate_proj"}:
        out_dim, in_dim, packed_words, groups = 1536, 4096, 512, 64
    else:
        out_dim, in_dim, packed_words, groups = 4096, 1536, 192, 24
    rng = np.random.default_rng(123)
    x = mx.array(rng.standard_normal((in_dim,)).astype(np.float32))
    xq = mx.expand_dims(mx.expand_dims(mx.expand_dims(x.astype(mx.bfloat16), 0), 0), 0)
    idx = mx.array([[[0]]], dtype=mx.int32)
    packed = {
        "weight": mx.expand_dims(w, 0),
        "scales": mx.expand_dims(scales.astype(mx.bfloat16), 0),
        "biases": mx.expand_dims(biases.astype(mx.bfloat16), 0),
    }
    sync(x, xq, idx, packed)
    kernel, grid_scale = build_kernel(out_dim, in_dim, packed_words, groups, args.kernel)

    def metal_call() -> mx.array:
        return kernel(
            inputs=[x, w, scales, biases],
            output_shapes=[(out_dim,)],
            output_dtypes=[mx.float32],
            grid=(out_dim * grid_scale, 1, 1),
            threadgroup=(256, 1, 1),
        )[0]

    metal_out = metal_call()
    gather_out = qlinear_gather_prepacked(xq, idx, packed, out_dim)
    sync(metal_out, gather_out)
    metal_np = np.array(metal_out, copy=False)
    gather_np = np.array(gather_out, copy=False)
    diff = np.abs(metal_np - gather_np)

    for _ in range(args.warmup):
        sync(metal_call())
        sync(qlinear_gather_prepacked(xq, idx, packed, out_dim))

    metal_times: list[float] = []
    gather_times: list[float] = []
    for _ in range(args.repeat):
        dt, _ = timed(metal_call)
        metal_times.append(dt)
        dt, _ = timed(lambda: qlinear_gather_prepacked(xq, idx, packed, out_dim))
        gather_times.append(dt)

    result = {
        "ok": True,
        "schema": "hy3-metal-q4-matvec-probe-v1",
        "layer": args.layer,
        "expert": args.expert,
        "family": args.family,
        "kernel": args.kernel,
        "shape": {"out_dim": out_dim, "in_dim": in_dim, "packed_words": packed_words, "groups": groups},
        "max_abs_diff_vs_gather_qmm": float(diff.max()),
        "mean_abs_diff_vs_gather_qmm": float(diff.mean()),
        "metal_s": summarize(metal_times),
        "gather_qmm_s": summarize(gather_times),
        "speedup_vs_gather_median": round(summarize(gather_times)["median"] / max(summarize(metal_times)["median"], 1e-12), 4),
        "sidecar_store": store.stats(),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
