#!/usr/bin/env python3
"""Build a metadata-only Hy3 sidecar layout from MLX safetensors.

This does not copy the 150 GiB expert payload. It reads safetensors headers,
classifies resident vs routed-expert tensors, and writes a JSON manifest with
absolute shard byte offsets suitable for a future pread/mmap sidecar runtime.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import struct
from pathlib import Path
from typing import Any

GIB = 1024 ** 3
MIB = 1024 ** 2

EXPERT_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.mlp\.switch_mlp\.(?P<family>down_proj|gate_proj|up_proj)\.(?P<kind>weight|scales|biases)$")
LAYER_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.")


def gib(n: int | float) -> float:
    return float(n) / GIB


def mib(n: int | float) -> float:
    return float(n) / MIB


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: not enough bytes for safetensors header length")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = json.loads(handle.read(header_len))
    return header_len, header


def tensor_nbytes(meta: dict[str, Any]) -> int:
    start, end = meta["data_offsets"]
    return int(end) - int(start)


def classify(name: str) -> str:
    if EXPERT_RE.match(name):
        return "routed_experts_sidecar"
    if re.search(r"(embed_tokens|wte|tok_embeddings|lm_head)", name):
        return "embeddings_lm_head"
    if re.search(r"(mtp|multi_token|draft)", name, re.I):
        return "mtp"
    if re.search(r"(self_attn|attention|attn)", name):
        return "attention_resident"
    if re.search(r"(router|gating|e_score|expert_bias)", name):
        return "router_gate_resident"
    if "shared_mlp" in name:
        return "shared_mlp_resident"
    if re.search(r"(input_layernorm|post_attention_layernorm|norm|layernorm|rms_norm)", name):
        return "norms_resident"
    if ".mlp." in name or "feed_forward" in name:
        return "dense_mlp_resident"
    return "other_resident"


def load_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_layout(model_dir: Path) -> dict[str, Any]:
    model_dir = model_dir.expanduser().resolve()
    config = load_config(model_dir)
    rows: list[dict[str, Any]] = []

    for shard in sorted(model_dir.glob("model-*.safetensors")):
        header_len, header = read_safetensors_header(shard)
        payload_base = 8 + header_len
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            rel_start, rel_end = meta["data_offsets"]
            nbytes = int(rel_end) - int(rel_start)
            category = classify(name)
            layer_match = LAYER_RE.match(name)
            expert_match = EXPERT_RE.match(name)
            row = {
                "name": name,
                "category": category,
                "shard": shard.name,
                "shard_path": str(shard),
                "dtype": meta.get("dtype"),
                "shape": meta.get("shape", []),
                "nbytes": nbytes,
                "file_offset": payload_base + int(rel_start),
                "data_offsets": [int(rel_start), int(rel_end)],
            }
            if layer_match:
                row["layer"] = int(layer_match.group("layer"))
            if expert_match:
                row["expert_family"] = expert_match.group("family")
                row["tensor_kind"] = expert_match.group("kind")
            rows.append(row)

    categories = collections.defaultdict(lambda: {"nbytes": 0, "tensors": 0})
    layers: dict[int, dict[str, Any]] = {}
    expert_by_layer = collections.Counter()
    all_by_layer = collections.Counter()

    for row in rows:
        cat = row["category"]
        categories[cat]["nbytes"] += row["nbytes"]
        categories[cat]["tensors"] += 1
        if "layer" in row:
            all_by_layer[int(row["layer"])] += row["nbytes"]
        if cat == "routed_experts_sidecar" and "layer" in row:
            expert_by_layer[int(row["layer"])] += row["nbytes"]

    for layer in sorted(set(all_by_layer) | set(expert_by_layer)):
        layers[layer] = {
            "all_nbytes": all_by_layer[layer],
            "expert_nbytes": expert_by_layer[layer],
            "all_gib": round(gib(all_by_layer[layer]), 6),
            "expert_gib": round(gib(expert_by_layer[layer]), 6),
        }

    total = sum(row["nbytes"] for row in rows)
    expert_total = categories["routed_experts_sidecar"]["nbytes"]
    resident_total = total - expert_total
    expert_layers = sorted(layer for layer, n in expert_by_layer.items() if n)
    layer_count = len(expert_layers)
    experts_per_layer = int(config.get("num_experts") or config.get("n_routed_experts") or 192)
    top_k = int(config.get("num_experts_per_tok") or config.get("moe_topk") or 8)
    expert_bank_per_layer = expert_total / layer_count if layer_count else 0
    per_expert_payload = expert_bank_per_layer / experts_per_layer if experts_per_layer else 0
    cold_active_per_layer = per_expert_payload * top_k
    cold_active_per_token = cold_active_per_layer * layer_count

    slot_bank_estimates = []
    for slots in [16, 24, 32, 48, 64, 80, 96, 128]:
        nbytes = int(per_expert_payload * slots * layer_count)
        slot_bank_estimates.append({"slots": slots, "nbytes": nbytes, "gib": round(gib(nbytes), 3)})

    sidecar_entries = [row for row in rows if row["category"] == "routed_experts_sidecar"]
    resident_entries = [row for row in rows if row["category"] != "routed_experts_sidecar"]

    return {
        "schema": "hy3-sidecar-layout-v0",
        "model_dir": str(model_dir),
        "config": {
            "model_type": config.get("model_type"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "first_k_dense_replace": config.get("first_k_dense_replace"),
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "moe_intermediate_size": config.get("moe_intermediate_size"),
            "num_experts": experts_per_layer,
            "num_experts_per_tok": top_k,
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "vocab_size": config.get("vocab_size"),
            "rope_theta": config.get("rope_theta") or (config.get("rope_parameters") or {}).get("rope_theta"),
        },
        "totals": {
            "tensors": len(rows),
            "total_nbytes": total,
            "total_gib": round(gib(total), 3),
            "resident_nbytes": resident_total,
            "resident_gib": round(gib(resident_total), 3),
            "expert_sidecar_nbytes": expert_total,
            "expert_sidecar_gib": round(gib(expert_total), 3),
            "expert_sidecar_pct": round(100.0 * expert_total / total, 2) if total else 0,
        },
        "categories": {
            key: {
                "nbytes": value["nbytes"],
                "gib": round(gib(value["nbytes"]), 3),
                "tensors": value["tensors"],
            }
            for key, value in sorted(categories.items())
        },
        "derived": {
            "expert_layers": expert_layers,
            "expert_layer_count": layer_count,
            "expert_bank_per_layer_gib": round(gib(expert_bank_per_layer), 3),
            "per_expert_payload_mib": round(mib(per_expert_payload), 3),
            "top_k": top_k,
            "cold_active_read_per_layer_mib": round(mib(cold_active_per_layer), 3),
            "cold_active_read_per_token_gib": round(gib(cold_active_per_token), 3),
            "slot_bank_estimates": slot_bank_estimates,
        },
        "layers": {str(k): v for k, v in layers.items()},
        "resident_entries": resident_entries,
        "sidecar_entries": sidecar_entries,
    }


def write_summary(layout: dict[str, Any], path: Path) -> None:
    totals = layout["totals"]
    derived = layout["derived"]
    categories = layout["categories"]
    slot_lines = "\n".join(
        f"| {item['slots']} | {item['gib']:.3f} GiB |" for item in derived["slot_bank_estimates"]
    )
    cat_lines = "\n".join(
        f"| `{name}` | {data['gib']:.3f} GiB | {data['tensors']} |" for name, data in categories.items()
    )
    text = f"""# Hy3 preview MLX sidecar layout summary

Source: `{layout['model_dir']}`

## Split verdict

| Bucket | Size |
|---|---:|
| Total MLX safetensor payload | {totals['total_gib']:.3f} GiB |
| Resident non-expert core | {totals['resident_gib']:.3f} GiB |
| Routed expert sidecar | {totals['expert_sidecar_gib']:.3f} GiB |
| Expert share | {totals['expert_sidecar_pct']:.2f}% |

## Derived runtime pressure

| Metric | Value |
|---|---:|
| Expert MoE layers | {derived['expert_layer_count']} |
| Experts/layer | {layout['config']['num_experts']} |
| Native top-k | {derived['top_k']} |
| Expert bank/layer | {derived['expert_bank_per_layer_gib']:.3f} GiB |
| Per-expert payload | {derived['per_expert_payload_mib']:.3f} MiB |
| Cold active read/layer | {derived['cold_active_read_per_layer_mib']:.3f} MiB |
| Cold active read/token | {derived['cold_active_read_per_token_gib']:.3f} GiB |

## Slot-bank estimates

| Slots/layer | Host/cache footprint |
|---:|---:|
{slot_lines}

## Categories

| Category | Size | Tensors |
|---|---:|---:|
{cat_lines}

## Operator note

This layout is metadata-only. It proves the MLX artifact is almost perfectly split for a Capstan-style runtime: keep ~4.6 GiB resident, stream/cache ~150 GiB of routed experts. It does **not** make stock MLX load the model safely; runtime work is still required.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX"))
    parser.add_argument("--out", type=Path, default=Path("/Users/nb/LLM/hy3-mlx-canary/hy3-sidecar-layout.json"))
    parser.add_argument("--summary", type=Path, default=Path("/Users/nb/LLM/hy3-mlx-canary/hy3-sidecar-layout-summary.md"))
    args = parser.parse_args()

    layout = build_layout(args.model_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(layout, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(layout, args.summary)

    totals = layout["totals"]
    derived = layout["derived"]
    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")
    print(
        "split: "
        f"resident={totals['resident_gib']:.3f}GiB "
        f"experts={totals['expert_sidecar_gib']:.3f}GiB "
        f"expert_pct={totals['expert_sidecar_pct']:.2f}%"
    )
    print(
        "runtime: "
        f"layers={derived['expert_layer_count']} "
        f"topk={derived['top_k']} "
        f"per_expert={derived['per_expert_payload_mib']:.3f}MiB "
        f"cold_token={derived['cold_active_read_per_token_gib']:.3f}GiB"
    )


if __name__ == "__main__":
    main()
