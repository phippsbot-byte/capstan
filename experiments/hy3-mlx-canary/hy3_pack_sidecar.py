#!/usr/bin/env python3
"""Pack Hy3 routed experts into a layer-major/expert-major raw sidecar.

Input is the metadata-only layout from hy3_sidecar_layout.py. Output shape:

  OUT/
    manifest.json
    layers/layer_001.bin
    layers/layer_002.bin
    ...

Each layer file is expert-major: for expert 0..191, write all 9 payloads
(up/gate/down × weight/scales/biases). This makes one expert load a tight
~10MiB region instead of bouncing across safetensor tensor banks/shards.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

FAMILIES = ("up_proj", "gate_proj", "down_proj")
KINDS = ("weight", "scales", "biases")
GIB = 1024**3


def parse_layers(spec: str | None, default_layers: list[int]) -> list[int]:
    if not spec:
        return default_layers
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def dtype_nbytes(dtype: str) -> int:
    if dtype == "U32":
        return 4
    if dtype == "BF16":
        return 2
    if dtype == "F32":
        return 4
    raise ValueError(f"unsupported dtype: {dtype}")


def per_expert_nbytes(entry: dict[str, Any]) -> int:
    shape = [int(x) for x in entry["shape"]]
    n = dtype_nbytes(entry["dtype"])
    for dim in shape[1:]:
        n *= dim
    return n


def copy_range(src_path: Path, offset: int, nbytes: int, dst) -> None:
    with src_path.open("rb", buffering=0) as src:
        src.seek(offset)
        remaining = nbytes
        while remaining:
            chunk = src.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise IOError(f"short read from {src_path} at {offset} wanted {nbytes}")
            dst.write(chunk)
            remaining -= len(chunk)


def load_layout(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_source_index(layout: dict[str, Any]) -> dict[tuple[int, str, str], dict[str, Any]]:
    index: dict[tuple[int, str, str], dict[str, Any]] = {}
    for entry in layout["sidecar_entries"]:
        key = (int(entry["layer"]), entry["expert_family"], entry["tensor_kind"])
        index[key] = entry
    return index


def pack_sidecar(args: argparse.Namespace) -> dict[str, Any]:
    layout_path = args.layout.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"{out_dir} exists; pass --force to replace")
        shutil.rmtree(out_dir)
    layers_dir = out_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    layout = load_layout(layout_path)
    source = build_source_index(layout)
    all_layers = [int(x) for x in layout["derived"]["expert_layers"]]
    layers = parse_layers(args.layers, all_layers)
    num_experts = int(layout["config"]["num_experts"])
    if args.max_experts is not None:
        num_experts = min(num_experts, int(args.max_experts))

    total_expected = 0
    packed_entries: list[dict[str, Any]] = []
    layer_summaries: dict[str, Any] = {}
    t0 = time.time()

    for layer in layers:
        layer_path = layers_dir / f"layer_{layer:03d}.bin"
        tmp_path = layer_path.with_suffix(".bin.tmp")
        layer_start = time.time()
        layer_bytes = 0
        with tmp_path.open("wb", buffering=0) as dst:
            for expert in range(num_experts):
                for family in FAMILIES:
                    for kind in KINDS:
                        src_entry = source[(layer, family, kind)]
                        src_shape = [int(x) for x in src_entry["shape"]]
                        nbytes = per_expert_nbytes(src_entry)
                        src_offset = int(src_entry["file_offset"]) + expert * nbytes
                        dst_offset = layer_bytes
                        copy_range(Path(src_entry["shard_path"]), src_offset, nbytes, dst)
                        layer_bytes += nbytes
                        packed_entries.append(
                            {
                                "layer": layer,
                                "expert": expert,
                                "expert_family": family,
                                "tensor_kind": kind,
                                "dtype": src_entry["dtype"],
                                "shape": [1, *src_shape[1:]],
                                "nbytes": nbytes,
                                "file": f"layers/{layer_path.name}",
                                "file_offset": dst_offset,
                                "source_shard": src_entry["shard"],
                                "source_offset": src_offset,
                            }
                        )
        os.replace(tmp_path, layer_path)
        total_expected += layer_bytes
        elapsed = time.time() - layer_start
        layer_summaries[str(layer)] = {
            "file": f"layers/{layer_path.name}",
            "nbytes": layer_bytes,
            "gib": round(layer_bytes / GIB, 6),
            "experts": num_experts,
            "elapsed_s": round(elapsed, 3),
        }
        print(
            f"packed layer {layer:03d}: {layer_bytes / GIB:.3f} GiB in {elapsed:.1f}s",
            flush=True,
        )

    manifest = {
        "schema": "hy3-packed-sidecar-v1",
        "source_layout": str(layout_path),
        "source_model_dir": layout.get("model_dir"),
        "layout": "layer-major-expert-major",
        "families": list(FAMILIES),
        "kinds": list(KINDS),
        "layers": layers,
        "num_experts": num_experts,
        "totals": {
            "entries": len(packed_entries),
            "nbytes": total_expected,
            "gib": round(total_expected / GIB, 6),
            "elapsed_s": round(time.time() - t0, 3),
        },
        "layer_summaries": layer_summaries,
        "packed_entries": packed_entries,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def verify_manifest(out_dir: Path, samples: int) -> dict[str, Any]:
    manifest = json.loads((out_dir / "manifest.json").read_text())
    checked = 0
    mismatches = []
    for entry in manifest["packed_entries"]:
        if checked >= samples:
            break
        packed_path = out_dir / entry["file"]
        with packed_path.open("rb", buffering=0) as handle:
            handle.seek(int(entry["file_offset"]))
            packed = handle.read(int(entry["nbytes"]))
        source_path = Path(manifest["source_model_dir"]) / entry["source_shard"]
        with source_path.open("rb", buffering=0) as handle:
            handle.seek(int(entry["source_offset"]))
            source = handle.read(int(entry["nbytes"]))
        if packed != source:
            mismatches.append(entry)
        checked += 1
    return {"checked": checked, "mismatches": len(mismatches), "ok": not mismatches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, default=Path("/Users/nb/LLM/hy3-mlx-canary/hy3-sidecar-layout.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar"))
    parser.add_argument("--layers", help="optional layer list/range, e.g. 1 or 1-3")
    parser.add_argument("--max-experts", type=int, help="debug only: pack first N experts")
    parser.add_argument("--verify-samples", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = pack_sidecar(args)
    verify = verify_manifest(args.out_dir.expanduser().resolve(), args.verify_samples)
    result = {
        "ok": verify["ok"],
        "out_dir": str(args.out_dir.expanduser().resolve()),
        "totals": manifest["totals"],
        "verify": verify,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not verify["ok"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
