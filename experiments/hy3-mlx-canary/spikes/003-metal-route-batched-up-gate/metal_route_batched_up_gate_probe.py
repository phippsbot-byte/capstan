#!/usr/bin/env python3
"""Spike: route-batched fused q4 up+gate Metal kernel for real Hy3 fixtures.

This probes the next useful Capstan/Hy3 kernel shape: not one expert / one
projection, but all selected routes for one layer fixture in one Metal launch.

The kernel emits weighted hidden routes:
  route_weight * silu(gate_q4(hidden[token])) * up_q4(hidden[token])

Then normal MLX gather_qmm handles down_proj and we sum routes per token. This
keeps the probe scoped while testing a real routed-MoE integration lever:
fewer launches and route weights carried before down.
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
DEFAULT_FIXTURE = Path(
    "/Volumes/ModelSSD/logs/hy3-mlx-canary/parity-fixtures/20260707-140203-prefill4-all-layers/hy3-layer1-top5-seq4.json"
)
UP_GATE_OUT_DIM = 1536
HIDDEN_DIM = 4096
UP_GATE_PACKED_WORDS = 512
UP_GATE_GROUPS = 64
DOWN_OUT_DIM = 4096
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


def silu(x: mx.array) -> mx.array:
    return x / (1.0 + mx.exp(-x))


def qlinear_gather(xq: mx.array, idx: mx.array, packed: dict[str, mx.array]) -> mx.array:
    return mx.gather_qmm(
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


def build_route_batched_kernel(route_count: int):
    source = f"""
        uint linear = threadgroup_position_in_grid.x;
        uint route = linear / {UP_GATE_OUT_DIM}u;
        uint row = linear - route * {UP_GATE_OUT_DIM}u;
        uint tid = thread_position_in_threadgroup.x;
        threadgroup float partial_up[{THREADS}];
        threadgroup float partial_gate[{THREADS}];
        float acc_up = 0.0f;
        float acc_gate = 0.0f;
        if (route < {route_count}u && row < {UP_GATE_OUT_DIM}u) {{
            uint token = route_token_idx[route];
            uint expert = route_bank_idx[route];
            uint weight_base = (expert * {UP_GATE_OUT_DIM}u + row) * {UP_GATE_PACKED_WORDS}u;
            uint scale_base = (expert * {UP_GATE_OUT_DIM}u + row) * {UP_GATE_GROUPS}u;
            uint x_base = token * {HIDDEN_DIM}u;
            for (uint i = tid; i < {HIDDEN_DIM}u; i += {THREADS}u) {{
                float xv = hidden_tokens[x_base + i];
                uint word_up = up_w[weight_base + (i >> 3)];
                uint q_up = (word_up >> ((i & 7u) * 4u)) & 0xFu;
                uint word_gate = gate_w[weight_base + (i >> 3)];
                uint q_gate = (word_gate >> ((i & 7u) * 4u)) & 0xFu;
                uint group = i >> 6;
                float up_scale = up_scales[scale_base + group];
                float up_bias = up_biases[scale_base + group];
                float gate_scale = gate_scales[scale_base + group];
                float gate_bias = gate_biases[scale_base + group];
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
        if (tid == 0u && route < {route_count}u && row < {UP_GATE_OUT_DIM}u) {{
            float g = partial_gate[0];
            out[route * {UP_GATE_OUT_DIM}u + row] = route_weights[route] * (g / (1.0f + exp(-g))) * partial_up[0];
        }}
    """
    return mx.fast.metal_kernel(
        name=f"hy3_route_batched_weighted_up_gate_{route_count}_{UP_GATE_OUT_DIM}_{HIDDEN_DIM}_{THREADS}",
        input_names=[
            "hidden_tokens",
            "route_token_idx",
            "route_bank_idx",
            "route_weights",
            "up_w",
            "up_scales",
            "up_biases",
            "gate_w",
            "gate_scales",
            "gate_biases",
        ],
        output_names=["out"],
        source=source,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PACKED_MANIFEST)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--repeat", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text())
    seq_len = int(fixture["seq_len"])
    topk = int(fixture["topk"])
    route_count = seq_len * topk
    layer = int(fixture["layer"])
    hidden_np = np.array(fixture["hidden_tokens"], dtype=np.float32).reshape(seq_len, HIDDEN_DIM)
    experts_flat = [int(x) for x in fixture["experts_flat"]]
    route_weights_np = np.array(fixture["route_weights_flat"], dtype=np.float32)
    expected_np = np.array(fixture["expected_routed_tokens"], dtype=np.float32).reshape(seq_len, DOWN_OUT_DIM)
    if len(experts_flat) != route_count or route_weights_np.shape != (route_count,):
        raise RuntimeError("fixture route shape mismatch")

    unique_experts = sorted(set(experts_flat))
    expert_pos = {expert: i for i, expert in enumerate(unique_experts)}
    route_bank_idx_np = np.array([expert_pos[x] for x in experts_flat], dtype=np.uint32)
    route_token_idx_np = np.array([route // topk for route in range(route_count)], dtype=np.uint32)

    mod = import_lazy_module()
    store = mod.Hy3SidecarStore(str(args.manifest), slot_bank=max(32, len(unique_experts)))
    bank = store.get_experts(layer, unique_experts)
    up = bank["up_proj"]
    gate = bank["gate_proj"]
    down = bank["down_proj"]

    hidden_tokens = mx.array(hidden_np.reshape(-1))
    route_token_idx = mx.array(route_token_idx_np, dtype=mx.uint32)
    route_bank_idx = mx.array(route_bank_idx_np, dtype=mx.uint32)
    route_weights = mx.array(route_weights_np)
    x_routes_np = np.stack([hidden_np[route // topk] for route in range(route_count)]).astype(np.float32)
    x_routes = mx.array(x_routes_np.reshape(1, route_count, 1, 1, HIDDEN_DIM)).astype(mx.bfloat16)
    route_idx = mx.array(route_bank_idx_np.astype(np.int32).reshape(1, route_count, 1), dtype=mx.int32)
    route_weights_broadcast = mx.array(route_weights_np.reshape(route_count, 1))
    expected = mx.array(expected_np)

    up_packed = {
        "weight": up["weight"],
        "scales": up["scales"],
        "biases": up["biases"],
    }
    gate_packed = {
        "weight": gate["weight"],
        "scales": gate["scales"],
        "biases": gate["biases"],
    }
    down_packed = {
        "weight": down["weight"],
        "scales": down["scales"],
        "biases": down["biases"],
    }
    sync(hidden_tokens, route_token_idx, route_bank_idx, route_weights, x_routes, route_idx, expected)

    kernel = build_route_batched_kernel(route_count)

    def gather_weighted_hidden() -> mx.array:
        u = qlinear_gather(x_routes, route_idx, up_packed).reshape(route_count, UP_GATE_OUT_DIM).astype(mx.float32)
        g = qlinear_gather(x_routes, route_idx, gate_packed).reshape(route_count, UP_GATE_OUT_DIM).astype(mx.float32)
        return (silu(g) * u) * route_weights_broadcast

    def metal_weighted_hidden() -> mx.array:
        return kernel(
            inputs=[
                hidden_tokens,
                route_token_idx,
                route_bank_idx,
                route_weights,
                up["weight"],
                up["scales"].astype(mx.float32),
                up["biases"].astype(mx.float32),
                gate["weight"],
                gate["scales"].astype(mx.float32),
                gate["biases"].astype(mx.float32),
            ],
            output_shapes=[(route_count, UP_GATE_OUT_DIM)],
            output_dtypes=[mx.float32],
            grid=(route_count * UP_GATE_OUT_DIM * THREADS, 1, 1),
            threadgroup=(THREADS, 1, 1),
        )[0]

    def down_from_weighted_hidden(weighted_hidden: mx.array) -> mx.array:
        hiddenq = weighted_hidden.astype(mx.bfloat16).reshape(1, route_count, 1, 1, UP_GATE_OUT_DIM)
        down_routes = qlinear_gather(hiddenq, route_idx, down_packed).reshape(route_count, DOWN_OUT_DIM).astype(mx.float32)
        return down_routes.reshape(seq_len, topk, DOWN_OUT_DIM).sum(axis=1)

    def gather_preweighted_full() -> mx.array:
        return down_from_weighted_hidden(gather_weighted_hidden())

    def metal_preweighted_full() -> mx.array:
        return down_from_weighted_hidden(metal_weighted_hidden())

    def gather_standard_full() -> mx.array:
        u = qlinear_gather(x_routes, route_idx, up_packed).reshape(route_count, UP_GATE_OUT_DIM).astype(mx.float32)
        g = qlinear_gather(x_routes, route_idx, gate_packed).reshape(route_count, UP_GATE_OUT_DIM).astype(mx.float32)
        hidden = silu(g) * u
        hiddenq = hidden.astype(mx.bfloat16).reshape(1, route_count, 1, 1, UP_GATE_OUT_DIM)
        down_routes = qlinear_gather(hiddenq, route_idx, down_packed).reshape(route_count, DOWN_OUT_DIM).astype(mx.float32)
        weighted = down_routes * route_weights_broadcast
        return weighted.reshape(seq_len, topk, DOWN_OUT_DIM).sum(axis=1)

    gather_hidden_out = gather_weighted_hidden()
    metal_hidden_out = metal_weighted_hidden()
    gather_preweighted_out = gather_preweighted_full()
    metal_preweighted_out = metal_preweighted_full()
    standard_out = gather_standard_full()
    sync(gather_hidden_out, metal_hidden_out, gather_preweighted_out, metal_preweighted_out, standard_out)

    gather_hidden_np = np.array(gather_hidden_out, copy=False)
    metal_hidden_np = np.array(metal_hidden_out, copy=False)
    standard_np = np.array(standard_out, copy=False)
    gather_preweighted_np = np.array(gather_preweighted_out, copy=False)
    metal_preweighted_np = np.array(metal_preweighted_out, copy=False)
    expected_np_eval = np.array(expected, copy=False)

    for _ in range(args.warmup):
        sync(gather_weighted_hidden())
        sync(metal_weighted_hidden())
        sync(gather_preweighted_full())
        sync(metal_preweighted_full())
        sync(gather_standard_full())

    gather_hidden_times: list[float] = []
    metal_hidden_times: list[float] = []
    gather_preweighted_times: list[float] = []
    metal_preweighted_times: list[float] = []
    standard_times: list[float] = []
    for _ in range(args.repeat):
        dt, _ = timed(gather_weighted_hidden)
        gather_hidden_times.append(dt)
        dt, _ = timed(metal_weighted_hidden)
        metal_hidden_times.append(dt)
        dt, _ = timed(gather_preweighted_full)
        gather_preweighted_times.append(dt)
        dt, _ = timed(metal_preweighted_full)
        metal_preweighted_times.append(dt)
        dt, _ = timed(gather_standard_full)
        standard_times.append(dt)

    gather_hidden_summary = summarize(gather_hidden_times)
    metal_hidden_summary = summarize(metal_hidden_times)
    gather_preweighted_summary = summarize(gather_preweighted_times)
    metal_preweighted_summary = summarize(metal_preweighted_times)
    standard_summary = summarize(standard_times)

    def diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
        d = np.abs(a - b)
        return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "rmse": float(np.sqrt(np.mean(d * d)))}

    result = {
        "ok": True,
        "schema": "hy3-metal-route-batched-up-gate-probe-v1",
        "fixture": str(args.fixture),
        "layer": layer,
        "seq_len": seq_len,
        "topk": topk,
        "route_count": route_count,
        "unique_experts": len(unique_experts),
        "shape": {
            "hidden_dim": HIDDEN_DIM,
            "up_gate_out_dim": UP_GATE_OUT_DIM,
            "down_out_dim": DOWN_OUT_DIM,
            "threads": THREADS,
        },
        "gather_weighted_hidden_s": gather_hidden_summary,
        "metal_weighted_hidden_s": metal_hidden_summary,
        "gather_preweighted_full_s": gather_preweighted_summary,
        "metal_preweighted_full_s": metal_preweighted_summary,
        "gather_standard_full_s": standard_summary,
        "metal_hidden_speedup_vs_gather_median": round(gather_hidden_summary["median"] / max(metal_hidden_summary["median"], 1e-12), 4),
        "metal_preweighted_full_speedup_vs_gather_preweighted_median": round(gather_preweighted_summary["median"] / max(metal_preweighted_summary["median"], 1e-12), 4),
        "metal_preweighted_full_speedup_vs_standard_median": round(standard_summary["median"] / max(metal_preweighted_summary["median"], 1e-12), 4),
        "preweighted_gather_speedup_vs_standard_median": round(standard_summary["median"] / max(gather_preweighted_summary["median"], 1e-12), 4),
        "diff_metal_hidden_vs_gather_hidden": diff_stats(metal_hidden_np, gather_hidden_np),
        "diff_preweighted_gather_vs_standard": diff_stats(gather_preweighted_np, standard_np),
        "diff_metal_preweighted_vs_gather_preweighted": diff_stats(metal_preweighted_np, gather_preweighted_np),
        "diff_standard_vs_fixture_expected": diff_stats(standard_np, expected_np_eval),
        "diff_metal_preweighted_vs_fixture_expected": diff_stats(metal_preweighted_np, expected_np_eval),
        "sidecar_store": store.stats(),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
