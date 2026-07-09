# Local Hy3 / hy_v3 MLX lazy-expert prototype.
# Built for mlx-community/Hy3-preview-4bit on the Studio.
# Goal: keep resident weights in MLX, stream/cache selected routed experts from safetensors.

from __future__ import annotations

import gc
import json
import math
import os
import struct
import subprocess
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


DEFAULT_LAYOUT = "/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/manifest.json"
WORKDIR = Path(__file__).resolve().parent
DEFAULT_CPP_ROUTE_DAEMON_BIN = str(WORKDIR / "build/hy3-sidecar-io/hy3_route_mlp_daemon")
DEFAULT_CPP_ROUTE_INDEX = "/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/compact-index.tsv"
DEFAULT_CPP_ROUTE_ROOT = "/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar"
CPP_ROUTE_MAX_SEQ = 4096
CPP_ROUTE_MAX_TOPK = 64
CPP_ROUTE_MAX_LAYER = 1000
CPP_ROUTE_MAX_DENSE_CACHE_GIB = 16.0
CPP_ROUTE_MAX_PACKED_CACHE_GIB = 16.0
CPP_ROUTE_MAX_COMBINED_CACHE_GIB = 16.0
CPP_ROUTE_DENSE_EXPERT_GIB = (3 * 1536 * 4096 * 4) / (1024 ** 3)
CPP_ROUTE_PACKED_EXPERT_GIB = 10_616_832 / (1024 ** 3)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int = 4096
    num_hidden_layers: int = 80
    intermediate_size: int = 13312
    moe_intermediate_size: Optional[int] = None
    expert_hidden_dim: Optional[int] = None
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_parameters: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False

    # Hy3 MoE
    num_experts: int = 192
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    first_k_dense_replace: int = 1
    route_norm: bool = True
    router_scaling_factor: float = 2.826
    moe_router_use_sigmoid: bool = True
    moe_router_enable_expert_bias: bool = True
    qk_norm: bool = True

    # ignored compatibility knobs
    attention_bias: bool = False
    num_nextn_predict_layers: int = 0

    def __post_init__(self):
        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.expert_hidden_dim or self.intermediate_size
        if self.rope_parameters and self.rope_parameters.get("rope_theta"):
            self.rope_theta = float(self.rope_parameters["rope_theta"])


class Hy3SidecarStore:
    """Layer/expert LRU cache backed by raw sidecar byte offsets.

    Supports two manifests:
    - hy3-sidecar-layout-v0: metadata-only offsets into original safetensor shards
    - hy3-packed-sidecar-v1: packed layer-major/expert-major files
    """

    def __init__(self, layout_path: str | os.PathLike[str], slot_bank: int = 32):
        self.layout_path = str(layout_path)
        self.manifest_dir = Path(layout_path).expanduser().resolve().parent
        with open(layout_path, "r", encoding="utf-8") as handle:
            self.layout = json.load(handle)
        self.schema = self.layout.get("schema", "hy3-sidecar-layout-v0")
        self.slot_bank = int(slot_bank)
        self.entries: dict[tuple[int, str, str], dict[str, Any]] = {}
        self.packed_entries: dict[tuple[int, int, str, str], dict[str, Any]] = {}
        if self.schema == "hy3-packed-sidecar-v1":
            for entry in self.layout["packed_entries"]:
                self.packed_entries[(int(entry["layer"]), int(entry["expert"]), entry["expert_family"], entry["tensor_kind"])] = entry
        else:
            for entry in self.layout["sidecar_entries"]:
                self.entries[(int(entry["layer"]), entry["expert_family"], entry["tensor_kind"])] = entry
        self.cache: dict[int, OrderedDict[int, dict[str, dict[str, mx.array]]]] = {}
        self._fd_cache: dict[Path, int] = {}
        self.packed_coalesce_max_bytes = self._env_gib_to_bytes("HY3_PACKED_COALESCE_MAX_GIB", 0.032)
        if self.packed_coalesce_max_bytes > 0:
            self.packed_coalesce_max_overread_ratio = self._env_float("HY3_PACKED_COALESCE_MAX_OVERREAD_RATIO", 2.0)
            if self.packed_coalesce_max_overread_ratio < 1.0:
                raise ValueError("HY3_PACKED_COALESCE_MAX_OVERREAD_RATIO must be >= 1.0")
        else:
            self.packed_coalesce_max_overread_ratio = 0.0
        self.loads = 0
        self.hits = 0
        self.evictions = 0
        self.read_calls = 0
        self.read_bytes = 0
        self.get_calls = 0
        self.requested_unique_total = 0
        self.requested_unique_max = 0
        self.packed_batch_loads = 0
        self.packed_read_groups = 0
        self.packed_read_experts = 0
        self.packed_multi_expert_groups = 0
        self.packed_multi_expert_experts = 0
        self.packed_coalesced_extra_bytes = 0
        self.load_time_s = 0.0
        self.pack_time_s = 0.0
        self.remap_time_s = 0.0
        self.qmm_time_s = 0.0
        self.gc_calls = 0
        self.gc_time_s = 0.0

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {raw!r}")
        return value

    @classmethod
    def _env_gib_to_bytes(cls, name: str, default_gib: float) -> int:
        value = cls._env_float(name, default_gib)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}")
        return int(value * (1024 ** 3))

    @staticmethod
    def _dtype_nbytes(dtype: str) -> int:
        if dtype == "U32":
            return 4
        if dtype == "BF16":
            return 2
        if dtype == "F32":
            return 4
        raise ValueError(f"Unsupported sidecar dtype {dtype}")

    @staticmethod
    def _mx_from_raw(raw: bytes, dtype: str, shape: tuple[int, ...]) -> mx.array:
        if dtype == "U32":
            arr = np.frombuffer(raw, dtype=np.uint32).reshape(shape)
            return mx.array(arr, dtype=mx.uint32)
        if dtype == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            return mx.array(arr, dtype=mx.float32)
        if dtype == "BF16":
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
            return mx.array(arr, dtype=mx.uint16).view(mx.bfloat16)
        raise ValueError(f"Unsupported sidecar dtype {dtype}")

    def _packed_path(self, entry: dict[str, Any]) -> Path:
        path = Path(entry["file"])
        if not path.is_absolute():
            path = self.manifest_dir / path
        return path

    def _pread(self, path: Path, offset: int, nbytes: int) -> bytes:
        fd = self._fd_cache.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fd_cache[path] = fd
        raw = os.pread(fd, nbytes, offset)
        self.read_calls += 1
        self.read_bytes += len(raw)
        return raw

    def _packed_expert_entries(self, layer: int, expert_id: int) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for family in ("up_proj", "gate_proj", "down_proj"):
            for kind in ("weight", "scales", "biases"):
                entries.append(self.packed_entries[(layer, expert_id, family, kind)])
        return entries

    def _decode_packed_expert(self, entries: list[dict[str, Any]], raw_view: memoryview, base_offset: int) -> dict[str, dict[str, mx.array]]:
        out: dict[str, dict[str, mx.array]] = {"up_proj": {}, "gate_proj": {}, "down_proj": {}}
        for entry in entries:
            rel = int(entry["file_offset"]) - base_offset
            nbytes = int(entry["nbytes"])
            shape = tuple(int(x) for x in entry["shape"])
            out[entry["expert_family"]][entry["tensor_kind"]] = self._mx_from_raw(raw_view[rel : rel + nbytes], entry["dtype"], shape)
        return out

    def _load_packed_expert(self, layer: int, expert_id: int) -> dict[str, dict[str, mx.array]]:
        entries = self._packed_expert_entries(layer, expert_id)
        paths = {self._packed_path(e) for e in entries}
        if len(paths) != 1:
            raise RuntimeError(f"packed expert L{layer} E{expert_id} spans multiple files: {paths}")
        path = next(iter(paths))
        start = min(int(e["file_offset"]) for e in entries)
        end = max(int(e["file_offset"]) + int(e["nbytes"]) for e in entries)
        raw = self._pread(path, start, end - start)
        if len(raw) != end - start:
            raise IOError(f"short packed read for L{layer} E{expert_id}: {len(raw)} != {end - start}")
        out = self._decode_packed_expert(entries, memoryview(raw), start)
        mx.eval(out)
        self.loads += 1
        return out

    def _load_packed_experts_batch(self, layer: int, expert_ids: list[int]) -> dict[int, dict[str, dict[str, mx.array]]]:
        unique_ids = list(dict.fromkeys(expert_ids))
        if not unique_ids:
            return {}
        records: list[dict[str, Any]] = []
        for expert_id in unique_ids:
            entries = self._packed_expert_entries(layer, expert_id)
            paths = {self._packed_path(e) for e in entries}
            if len(paths) != 1:
                raise RuntimeError(f"packed expert L{layer} E{expert_id} spans multiple files: {paths}")
            start = min(int(e["file_offset"]) for e in entries)
            end = max(int(e["file_offset"]) + int(e["nbytes"]) for e in entries)
            records.append({"expert": expert_id, "entries": entries, "path": next(iter(paths)), "start": start, "end": end, "needed": end - start})
        records.sort(key=lambda r: (str(r["path"]), int(r["start"])))

        groups: list[list[dict[str, Any]]] = []
        for record in records:
            if not groups or groups[-1][-1]["path"] != record["path"]:
                groups.append([record])
                continue
            group = groups[-1]
            new_start = int(group[0]["start"])
            new_end = max(int(group[-1]["end"]), int(record["end"]))
            needed = sum(int(r["needed"]) for r in group) + int(record["needed"])
            coalesced = new_end - new_start
            ratio = coalesced / max(needed, 1)
            if coalesced <= self.packed_coalesce_max_bytes and ratio <= self.packed_coalesce_max_overread_ratio:
                group.append(record)
            else:
                groups.append([record])

        out: dict[int, dict[str, dict[str, mx.array]]] = {}
        arrays: list[Any] = []
        for group in groups:
            path = group[0]["path"]
            start = min(int(r["start"]) for r in group)
            end = max(int(r["end"]) for r in group)
            needed = sum(int(r["needed"]) for r in group)
            raw = self._pread(path, start, end - start)
            if len(raw) != end - start:
                experts = [int(r["expert"]) for r in group]
                raise IOError(f"short packed coalesced read for L{layer} experts={experts}: {len(raw)} != {end - start}")
            raw_view = memoryview(raw)
            self.packed_read_groups += 1
            self.packed_read_experts += len(group)
            if len(group) > 1:
                self.packed_multi_expert_groups += 1
                self.packed_multi_expert_experts += len(group)
            self.packed_coalesced_extra_bytes += (end - start) - needed
            for record in group:
                expert_out = self._decode_packed_expert(record["entries"], raw_view, start)
                out[int(record["expert"])] = expert_out
                arrays.append(expert_out)
        mx.eval(*arrays)
        self.loads += len(unique_ids)
        self.packed_batch_loads += 1
        return out

    def _read_one(self, layer: int, family: str, kind: str, expert_id: int) -> mx.array:
        if self.schema == "hy3-packed-sidecar-v1":
            entry = self.packed_entries[(layer, expert_id, family, kind)]
            shape = tuple(int(x) for x in entry["shape"])
            nbytes = int(entry["nbytes"])
            path = Path(entry["file"])
            if not path.is_absolute():
                path = self.manifest_dir / path
            offset = int(entry["file_offset"])
        else:
            entry = self.entries[(layer, family, kind)]
            full_shape = tuple(int(x) for x in entry["shape"])
            if not full_shape or full_shape[0] <= expert_id:
                raise IndexError(f"expert {expert_id} out of range for {layer=} {family=} {kind=}")
            per_expert_shape = full_shape[1:]
            elem_count = 1
            for dim in per_expert_shape:
                elem_count *= dim
            nbytes = elem_count * self._dtype_nbytes(entry["dtype"])
            offset = int(entry["file_offset"]) + expert_id * nbytes
            path = Path(entry["shard_path"])
            shape = (1, *per_expert_shape)
        with path.open("rb", buffering=0) as handle:
            handle.seek(offset)
            raw = handle.read(nbytes)
        self.read_calls += 1
        self.read_bytes += len(raw)
        if len(raw) != nbytes:
            raise IOError(f"short read for L{layer} E{expert_id} {family}.{kind}: {len(raw)} != {nbytes}")
        return self._mx_from_raw(raw, entry["dtype"], shape)

    def _load_expert(self, layer: int, expert_id: int) -> dict[str, dict[str, mx.array]]:
        if self.schema == "hy3-packed-sidecar-v1":
            return self._load_packed_expert(layer, expert_id)
        out: dict[str, dict[str, mx.array]] = {}
        for family in ("up_proj", "gate_proj", "down_proj"):
            out[family] = {
                "weight": self._read_one(layer, family, "weight", expert_id),
                "scales": self._read_one(layer, family, "scales", expert_id),
                "biases": self._read_one(layer, family, "biases", expert_id),
            }
        mx.eval(out)
        self.loads += 1
        return out

    def _trim_layer_cache(self, layer_cache: OrderedDict[int, dict[str, dict[str, mx.array]]], protected: set[int] | None = None) -> bool:
        protected = protected or set()
        evicted = False
        while len(layer_cache) > self.slot_bank:
            victim = None
            for candidate in layer_cache:
                if candidate not in protected:
                    victim = candidate
                    break
            if victim is None:
                return evicted
            del layer_cache[victim]
            self.evictions += 1
            evicted = True
        return evicted

    def get_experts(self, layer: int, expert_ids: list[int]) -> dict[str, dict[str, mx.array]]:
        self.get_calls += 1
        self.requested_unique_total += len(expert_ids)
        self.requested_unique_max = max(self.requested_unique_max, len(expert_ids))
        layer_cache = self.cache.setdefault(layer, OrderedDict())
        requested_set = set(int(expert_id) for expert_id in expert_ids)
        missing: list[int] = []
        seen_missing: set[int] = set()
        for expert_id in expert_ids:
            expert_id = int(expert_id)
            if expert_id in layer_cache:
                layer_cache.move_to_end(expert_id)
                self.hits += 1
            elif expert_id not in seen_missing:
                missing.append(expert_id)
                seen_missing.add(expert_id)

        if missing:
            t_load = time.time()
            if self.schema == "hy3-packed-sidecar-v1" and self.packed_coalesce_max_bytes > 0:
                loaded = self._load_packed_experts_batch(layer, missing)
            else:
                loaded = {expert_id: self._load_expert(layer, expert_id) for expert_id in missing}
            for expert_id in missing:
                layer_cache[expert_id] = loaded[expert_id]
                layer_cache.move_to_end(expert_id)
            self.load_time_s += time.time() - t_load

        evicted = self._trim_layer_cache(layer_cache, protected=requested_set)
        if evicted and env_truthy("HY3_GC_ON_EVICT"):
            t_gc = time.time()
            gc.collect()
            self.gc_calls += 1
            self.gc_time_s += time.time() - t_gc

        selected = [layer_cache[int(expert_id)] for expert_id in expert_ids]

        t_pack = time.time()
        packed: dict[str, dict[str, mx.array]] = {}
        for family in ("up_proj", "gate_proj", "down_proj"):
            packed[family] = {}
            for kind in ("weight", "scales", "biases"):
                packed[family][kind] = mx.concatenate([expert[family][kind] for expert in selected], axis=0)
        mx.eval(packed)
        self.pack_time_s += time.time() - t_pack
        post_pack_evicted = self._trim_layer_cache(layer_cache)
        if post_pack_evicted and env_truthy("HY3_GC_ON_EVICT"):
            t_gc = time.time()
            gc.collect()
            self.gc_calls += 1
            self.gc_time_s += time.time() - t_gc
        return packed

    def clear_cached_experts(self) -> dict[str, Any]:
        cached = sum(len(v) for v in self.cache.values())
        layers = len(self.cache)
        self.cache.clear()
        t_gc = time.time()
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
        self.gc_calls += 1
        self.gc_time_s += time.time() - t_gc
        return {"cleared_experts": cached, "cleared_layers": layers, "gc_s": round(time.time() - t_gc, 3)}

    def stats(self) -> dict[str, Any]:
        cached = sum(len(v) for v in self.cache.values())
        return {
            "layout_path": self.layout_path,
            "schema": self.schema,
            "slot_bank": self.slot_bank,
            "layers_with_cache": len(self.cache),
            "cached_experts": cached,
            "loads": self.loads,
            "hits": self.hits,
            "evictions": self.evictions,
            "read_calls": self.read_calls,
            "read_gib": round(self.read_bytes / (1024 ** 3), 6),
            "packed_coalesce_max_gib": round(self.packed_coalesce_max_bytes / (1024 ** 3), 6),
            "packed_coalesce_max_overread_ratio": round(self.packed_coalesce_max_overread_ratio, 3),
            "packed_batch_loads": self.packed_batch_loads,
            "packed_read_groups": self.packed_read_groups,
            "packed_read_experts": self.packed_read_experts,
            "packed_multi_expert_groups": self.packed_multi_expert_groups,
            "packed_multi_expert_experts": self.packed_multi_expert_experts,
            "packed_coalesced_extra_gib": round(self.packed_coalesced_extra_bytes / (1024 ** 3), 6),
            "get_calls": self.get_calls,
            "requested_unique_total": self.requested_unique_total,
            "requested_unique_max": self.requested_unique_max,
            "load_time_s": round(self.load_time_s, 3),
            "pack_time_s": round(self.pack_time_s, 3),
            "remap_time_s": round(self.remap_time_s, 3),
            "qmm_time_s": round(self.qmm_time_s, 3),
            "gc_calls": self.gc_calls,
            "gc_time_s": round(self.gc_time_s, 3),
        }


_STORE: Hy3SidecarStore | None = None
_CPP_ROUTE_CLIENT: "CppRouteMlpClient | None" = None
_PROFILE: list[dict[str, Any]] = []
_ROUTE_TRACE: list[dict[str, Any]] = []
_ROUTE_EVENT = 0
_PARITY_FIXTURES: list[dict[str, Any]] = []
_PARITY_CAPTURED_LAYERS: set[int] = set()
_MOE_TIMES: dict[str, float] = {
    "router_s": 0.0,
    "selection_s": 0.0,
    "switch_s": 0.0,
    "weight_s": 0.0,
    "shared_s": 0.0,
    "cpp_route_s": 0.0,
}


def env_truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def profile_enabled() -> bool:
    return env_truthy("HY3_PROFILE_LAYERS")


def route_trace_enabled() -> bool:
    return env_truthy("HY3_TRACE_ROUTES")


def record_route_trace(layer: int, indices: mx.array) -> None:
    global _ROUTE_EVENT
    if not route_trace_enabled():
        return
    mx.eval(indices)
    idx_np = np.array(indices, copy=False)
    if idx_np.ndim != 3:
        idx_np = idx_np.reshape(1, -1, idx_np.shape[-1])
    batch, tokens, _ = idx_np.shape
    for b in range(batch):
        for token in range(tokens):
            _ROUTE_TRACE.append(
                {
                    "event": _ROUTE_EVENT,
                    "layer": int(layer),
                    "batch": int(b),
                    "token": int(token),
                    "experts": [int(x) for x in idx_np[b, token].tolist()],
                }
            )
            _ROUTE_EVENT += 1


def reset_route_trace() -> None:
    global _ROUTE_EVENT
    _ROUTE_TRACE.clear()
    _ROUTE_EVENT = 0


def get_route_trace_stats() -> dict[str, Any]:
    layers = sorted({int(row["layer"]) for row in _ROUTE_TRACE})
    total_selected = sum(len(row["experts"]) for row in _ROUTE_TRACE)
    return {
        "enabled": route_trace_enabled(),
        "events": len(_ROUTE_TRACE),
        "layers": len(layers),
        "first_layer": layers[0] if layers else None,
        "last_layer": layers[-1] if layers else None,
        "total_selected": total_selected,
        "avg_k": round(total_selected / len(_ROUTE_TRACE), 3) if _ROUTE_TRACE else 0.0,
    }


def write_route_trace_tsv(path: str | os.PathLike[str], metadata: Optional[dict[str, Any]] = None) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write("# schema=hy3-route-trace-v1\n")
        if metadata:
            handle.write("# metadata=" + json.dumps(metadata, sort_keys=True) + "\n")
        handle.write("event\tlayer\tbatch\ttoken\texperts\n")
        for row in _ROUTE_TRACE:
            handle.write(
                f"{row['event']}\t{row['layer']}\t{row['batch']}\t{row['token']}\t"
                + ",".join(str(x) for x in row["experts"])
                + "\n"
            )
    return str(out)


def parity_capture_enabled() -> bool:
    return os.environ.get("HY3_PARITY_LAYER") not in {None, ""} or os.environ.get("HY3_PARITY_LAYERS") not in {None, ""}


def parse_parity_layers() -> Optional[set[int]]:
    raw = os.environ.get("HY3_PARITY_LAYERS")
    if raw:
        if raw.lower() == "all":
            return None
        out: set[int] = set()
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                out.update(range(int(start), int(end) + 1))
            else:
                out.add(int(part))
        return out
    raw = os.environ.get("HY3_PARITY_LAYER")
    if raw:
        return {int(raw)}
    return set()


def reset_parity_fixture() -> None:
    _PARITY_FIXTURES.clear()
    _PARITY_CAPTURED_LAYERS.clear()


def capture_parity_fixture(layer: int, x: mx.array, indices: mx.array, route_weights: mx.array, routed_output: mx.array) -> None:
    if not parity_capture_enabled():
        return
    layer = int(layer)
    target_layers = parse_parity_layers()
    if target_layers is not None and layer not in target_layers:
        return
    if layer in _PARITY_CAPTURED_LAYERS:
        return
    token = int(os.environ.get("HY3_PARITY_TOKEN", "0"))
    batch = int(os.environ.get("HY3_PARITY_BATCH", "0"))
    mx.eval(x, indices, route_weights, routed_output)
    idx_np = np.array(indices, copy=False)
    if idx_np.ndim != 3 or batch >= idx_np.shape[0]:
        raise ValueError(f"cannot capture parity batch={batch} from indices shape {idx_np.shape}")
    if env_truthy("HY3_PARITY_ALL_TOKENS"):
        token_indices = list(range(idx_np.shape[1]))
    else:
        if token >= idx_np.shape[1]:
            raise ValueError(f"cannot capture parity token batch={batch} token={token} from indices shape {idx_np.shape}")
        token_indices = [token]
    hidden_np = np.array(x[batch, token_indices].astype(mx.float32))
    weights_np = np.array(route_weights[batch, token_indices].astype(mx.float32))
    output_np = np.array(routed_output[batch, token_indices].astype(mx.float32))
    experts_by_token = [[int(v) for v in idx_np[batch, tok].tolist()] for tok in token_indices]
    weights_by_token = [[float(v) for v in row.tolist()] for row in weights_np]
    _PARITY_FIXTURES.append(
        {
            "schema": "hy3-routed-layer-parity-v2",
            "execution_backend": "cpp_route_mlp" if cpp_route_mlp_enabled() else "python_mlx",
            "layer": layer,
            "batch": batch,
            "token": int(token_indices[0]),
            "tokens": [int(tok) for tok in token_indices],
            "seq_len": len(token_indices),
            "hidden_size": int(hidden_np.shape[-1]),
            "output_size": int(output_np.shape[-1]),
            "topk": int(idx_np.shape[-1]),
            "experts": experts_by_token[0],
            "route_weights": weights_by_token[0],
            "hidden": [float(v) for v in hidden_np[0].tolist()],
            "expected_routed": [float(v) for v in output_np[0].tolist()],
            "experts_by_token": experts_by_token,
            "route_weights_by_token": weights_by_token,
            "experts_flat": [int(v) for row in experts_by_token for v in row],
            "route_weights_flat": [float(v) for row in weights_by_token for v in row],
            "hidden_tokens": [float(v) for v in hidden_np.reshape(-1).tolist()],
            "expected_routed_tokens": [float(v) for v in output_np.reshape(-1).tolist()],
        }
    )
    _PARITY_CAPTURED_LAYERS.add(layer)


def get_parity_fixture() -> dict[str, Any] | None:
    return _PARITY_FIXTURES[0] if _PARITY_FIXTURES else None


def get_parity_fixtures() -> list[dict[str, Any]]:
    return list(_PARITY_FIXTURES)


def write_parity_fixture(path: str | os.PathLike[str], metadata: Optional[dict[str, Any]] = None) -> str:
    fixture = get_parity_fixture()
    if fixture is None:
        raise RuntimeError("no parity fixture captured")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(fixture)
    if metadata:
        payload["metadata"] = metadata
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(out)


def write_parity_fixtures(out_dir: str | os.PathLike[str], metadata: Optional[dict[str, Any]] = None) -> list[str]:
    if not _PARITY_FIXTURES:
        raise RuntimeError("no parity fixtures captured")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for fixture in sorted(_PARITY_FIXTURES, key=lambda row: int(row["layer"])):
        payload = dict(fixture)
        if metadata:
            payload["metadata"] = metadata
        suffix = f"seq{fixture.get('seq_len', 1)}" if int(fixture.get("seq_len", 1)) > 1 else f"token{fixture['token']}"
        path = root / f"hy3-layer{fixture['layer']}-top{fixture['topk']}-{suffix}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(str(path))
    (root / "fixtures.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    return paths


def shared_mlp_disabled() -> bool:
    return env_truthy("HY3_DISABLE_SHARED_MLP")


def routed_mlp_disabled() -> bool:
    return env_truthy("HY3_DISABLE_ROUTED_MLP")


def cpp_route_mlp_enabled() -> bool:
    return env_truthy("HY3_CPP_ROUTE_MLP")


def sync_timers_enabled() -> bool:
    return env_truthy("HY3_SYNC_TIMERS")


def effective_top_k(default: int) -> int:
    raw = os.environ.get("HY3_TOPK_CAP")
    if raw is None or raw == "":
        return default
    cap = int(raw)
    if cap < 1:
        raise ValueError(f"HY3_TOPK_CAP must be >= 1, got {cap}")
    return min(default, cap)


def eval_for_timing(*arrays: Any) -> None:
    mx.eval(*arrays)
    if sync_timers_enabled() and hasattr(mx, "synchronize"):
        mx.synchronize()


def parse_cpp_route_dense_cache_gib(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"HY3_CPP_ROUTE_DENSE_CACHE_GIB must be a number, got {raw!r}") from exc
    if not math.isfinite(value) or value < 0.0 or value > CPP_ROUTE_MAX_DENSE_CACHE_GIB:
        raise ValueError(
            f"HY3_CPP_ROUTE_DENSE_CACHE_GIB must be finite and in [0, {CPP_ROUTE_MAX_DENSE_CACHE_GIB:g}], got {raw!r}; "
            f"one dense expert bank is ~{CPP_ROUTE_DENSE_EXPERT_GIB:.3f}GiB"
        )
    return value


def parse_cpp_route_packed_cache_gib(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"HY3_CPP_ROUTE_PACKED_CACHE_GIB must be a number, got {raw!r}") from exc
    if not math.isfinite(value) or value < 0.0 or value > CPP_ROUTE_MAX_PACKED_CACHE_GIB:
        raise ValueError(
            f"HY3_CPP_ROUTE_PACKED_CACHE_GIB must be finite and in [0, {CPP_ROUTE_MAX_PACKED_CACHE_GIB:g}], got {raw!r}; "
            f"one packed expert bank is ~{CPP_ROUTE_PACKED_EXPERT_GIB:.3f}GiB"
        )
    return value


def parse_cpp_route_q4_mode(raw: str | None) -> str:
    mode = (raw or "dense").strip().lower()
    if mode not in {"dense", "direct", "hybrid"}:
        raise ValueError(f"HY3_CPP_ROUTE_Q4_MODE must be dense, direct, or hybrid, got {raw!r}")
    return mode


class CppRouteMlpClient:
    """Persistent binary client for cpp/hy3_route_mlp_daemon.

    This is intentionally opt-in via HY3_CPP_ROUTE_MLP=1. It executes only the
    routed MoE branch from already-computed hidden states, expert ids, and route
    weights. Router/attention/shared MLP stay in Python/MLX.
    """

    REQ = struct.Struct("<4sIIII")
    RESP = struct.Struct("<4sIIIIdQ")

    def __init__(self) -> None:
        binary = os.environ.get("HY3_CPP_ROUTE_DAEMON_BIN", DEFAULT_CPP_ROUTE_DAEMON_BIN)
        index = os.environ.get("HY3_CPP_ROUTE_INDEX", DEFAULT_CPP_ROUTE_INDEX)
        root = os.environ.get("HY3_CPP_ROUTE_ROOT", DEFAULT_CPP_ROUTE_ROOT)
        cmd = [binary, "--index", index, "--root", root]
        dense_cache_gib = os.environ.get("HY3_CPP_ROUTE_DENSE_CACHE_GIB")
        self.dense_cache_gib = 0.0
        if dense_cache_gib:
            self.dense_cache_gib = parse_cpp_route_dense_cache_gib(dense_cache_gib)
            cmd.extend(["--dense-cache-gib", f"{self.dense_cache_gib:g}"])
        packed_cache_gib = os.environ.get("HY3_CPP_ROUTE_PACKED_CACHE_GIB")
        self.packed_cache_gib = 0.0
        if packed_cache_gib:
            self.packed_cache_gib = parse_cpp_route_packed_cache_gib(packed_cache_gib)
            cmd.extend(["--packed-cache-gib", f"{self.packed_cache_gib:g}"])
        combined_cache_gib = self.dense_cache_gib + self.packed_cache_gib
        if combined_cache_gib > CPP_ROUTE_MAX_COMBINED_CACHE_GIB:
            raise ValueError(
                f"combined HY3 C++ dense and packed cache budget must be <= {CPP_ROUTE_MAX_COMBINED_CACHE_GIB:g}GiB, "
                f"got {combined_cache_gib:g}GiB"
            )
        self.q4_mode = parse_cpp_route_q4_mode(os.environ.get("HY3_CPP_ROUTE_Q4_MODE"))
        cmd.extend(["--q4-mode", self.q4_mode])
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if self.proc.poll() is not None:
            self._fail("C++ route daemon exited during startup")
        self.calls = 0
        self.wall_s = 0.0
        self.compute_s = 0.0
        self.read_calls = 0
        self.bytes_read = 0
        self.packed_cache_hits = 0

    def _stderr_excerpt(self) -> str:
        if self.proc.stderr is None or self.proc.poll() is None:
            return ""
        try:
            data = self.proc.stderr.read() or b""
        except Exception:
            return ""
        text = data.decode("utf-8", "replace").strip()
        return f"; daemon stderr: {text[-1000:]}" if text else ""

    def _fail(self, message: str) -> None:
        detail = self._stderr_excerpt()
        self.close()
        global _CPP_ROUTE_CLIENT
        if _CPP_ROUTE_CLIENT is self:
            _CPP_ROUTE_CLIENT = None
        raise RuntimeError(message + detail)

    def close(self) -> None:
        proc = self.proc
        if proc.stdin:
            try:
                proc.stdin.close()
            except Exception:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    def compute(self, layer: int, x: mx.array, indices: mx.array, route_weights: mx.array) -> mx.array:
        if self.proc.stdin is None or self.proc.stdout is None:
            self._fail("C++ route daemon pipes are closed")
        if self.proc.poll() is not None:
            self._fail("C++ route daemon exited before request")
        mx.eval(x, indices, route_weights)
        x_np = np.ascontiguousarray(np.array(x.astype(mx.float32), copy=False), dtype=np.float32)
        idx_np = np.ascontiguousarray(np.array(indices, copy=False), dtype=np.int32)
        weight_np = np.ascontiguousarray(np.array(route_weights.astype(mx.float32), copy=False), dtype=np.float32)
        if x_np.ndim != 3 or idx_np.ndim != 3 or weight_np.ndim != 3:
            raise ValueError(f"C++ route daemon expects [batch, seq, ...], got x={x_np.shape} idx={idx_np.shape} weights={weight_np.shape}")
        batch, seq_len, hidden = x_np.shape
        if hidden != 4096:
            raise ValueError(f"C++ route daemon only supports hidden=4096, got {hidden}")
        if idx_np.shape[:2] != (batch, seq_len) or weight_np.shape != idx_np.shape:
            raise ValueError(f"C++ route daemon shape mismatch x={x_np.shape} idx={idx_np.shape} weights={weight_np.shape}")
        topk = idx_np.shape[-1]
        flat_seq = batch * seq_len
        if not (0 <= int(layer) <= CPP_ROUTE_MAX_LAYER):
            raise ValueError(f"C++ route daemon layer out of range 0..{CPP_ROUTE_MAX_LAYER}: {layer}")
        if flat_seq < 1 or flat_seq > CPP_ROUTE_MAX_SEQ:
            raise ValueError(f"C++ route daemon supports flattened seq 1..{CPP_ROUTE_MAX_SEQ}, got batch*seq={flat_seq}")
        if topk < 1 or topk > CPP_ROUTE_MAX_TOPK:
            raise ValueError(f"C++ route daemon supports topk 1..{CPP_ROUTE_MAX_TOPK}, got {topk}")
        payload = (
            self.REQ.pack(b"HY3R", int(layer), int(flat_seq), int(topk), 0)
            + x_np.reshape(flat_seq, 4096).tobytes()
            + idx_np.reshape(flat_seq, topk).tobytes()
            + weight_np.reshape(flat_seq, topk).tobytes()
        )
        t0 = time.time()
        try:
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._fail(f"C++ route daemon write failed: {exc}")
        hdr = self.proc.stdout.read(self.RESP.size)
        if len(hdr) != self.RESP.size:
            self._fail(f"C++ route daemon returned short response header ({len(hdr)} bytes)")
        magic, status, payload_floats, read_calls, packed_cache_hits, compute_s, bytes_read = self.RESP.unpack(hdr)
        if magic == b"HY3E" or status != 0:
            msg = self.proc.stdout.read(payload_floats).decode("utf-8", "replace") if payload_floats else "unknown daemon error"
            self._fail(f"C++ route daemon error: {msg}")
        if magic != b"HY3O":
            self._fail(f"C++ route daemon returned bad magic: {magic!r}")
        expected = flat_seq * 4096
        if payload_floats != expected:
            self._fail(f"C++ route daemon payload size mismatch: {payload_floats} != {expected}")
        raw = self.proc.stdout.read(payload_floats * 4)
        if len(raw) != payload_floats * 4:
            self._fail("C++ route daemon returned short output payload")
        wall = time.time() - t0
        self.calls += 1
        self.wall_s += wall
        self.compute_s += float(compute_s)
        self.read_calls += int(read_calls)
        self.bytes_read += int(bytes_read)
        self.packed_cache_hits += int(packed_cache_hits)
        out = np.frombuffer(raw, dtype="<f4").reshape(batch, seq_len, 4096).copy()
        return mx.array(out).astype(x.dtype)

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "calls": self.calls,
            "wall_s": round(self.wall_s, 3),
            "compute_s": round(self.compute_s, 3),
            "dense_cache_gib": self.dense_cache_gib,
            "dense_expert_gib": round(CPP_ROUTE_DENSE_EXPERT_GIB, 6),
            "packed_cache_gib": self.packed_cache_gib,
            "packed_expert_gib": round(CPP_ROUTE_PACKED_EXPERT_GIB, 6),
            "packed_cache_hits": self.packed_cache_hits,
            "q4_mode": self.q4_mode,
            "read_calls": self.read_calls,
            "read_gib": round(self.bytes_read / (1024 ** 3), 6),
        }


def get_cpp_route_client() -> CppRouteMlpClient:
    global _CPP_ROUTE_CLIENT
    if _CPP_ROUTE_CLIENT is None:
        _CPP_ROUTE_CLIENT = CppRouteMlpClient()
    return _CPP_ROUTE_CLIENT


def reset_cpp_route_client() -> None:
    global _CPP_ROUTE_CLIENT
    if _CPP_ROUTE_CLIENT is not None:
        _CPP_ROUTE_CLIENT.close()
    _CPP_ROUTE_CLIENT = None


def get_cpp_route_stats() -> dict[str, Any]:
    return _CPP_ROUTE_CLIENT.stats() if _CPP_ROUTE_CLIENT is not None else {"enabled": cpp_route_mlp_enabled(), "calls": 0}


def reset_profile() -> None:
    _PROFILE.clear()
    reset_route_trace()
    reset_parity_fixture()
    for key in _MOE_TIMES:
        _MOE_TIMES[key] = 0.0


def get_profile_stats() -> dict[str, Any]:
    attn_total = sum(float(row.get("attn_s", 0.0)) for row in _PROFILE)
    mlp_total = sum(float(row.get("mlp_s", 0.0)) for row in _PROFILE)
    residual_total = sum(float(row.get("residual_s", 0.0)) for row in _PROFILE)
    total = sum(float(row.get("total_s", 0.0)) for row in _PROFILE)
    top_mlp = sorted(_PROFILE, key=lambda row: float(row.get("mlp_s", 0.0)), reverse=True)[:8]
    top_attn = sorted(_PROFILE, key=lambda row: float(row.get("attn_s", 0.0)), reverse=True)[:8]
    return {
        "enabled": profile_enabled(),
        "sync_timers": sync_timers_enabled(),
        "shared_mlp_disabled": shared_mlp_disabled(),
        "routed_mlp_disabled": routed_mlp_disabled(),
        "layers": len(_PROFILE),
        "attn_total_s": round(attn_total, 3),
        "mlp_total_s": round(mlp_total, 3),
        "residual_total_s": round(residual_total, 3),
        "profiled_total_s": round(total, 3),
        "moe_times": {key: round(value, 3) for key, value in _MOE_TIMES.items()},
        "cpp_route_mlp": get_cpp_route_stats(),
        "top_mlp_layers": top_mlp,
        "top_attn_layers": top_attn,
    }


def get_sidecar_store() -> Hy3SidecarStore:
    global _STORE
    if _STORE is None:
        layout = os.environ.get("HY3_SIDECAR_LAYOUT", DEFAULT_LAYOUT)
        slot_bank = int(os.environ.get("HY3_SLOT_BANK", "32"))
        _STORE = Hy3SidecarStore(layout, slot_bank=slot_bank)
    return _STORE


def reset_sidecar_store() -> None:
    global _STORE
    _STORE = None
    gc.collect()


class RoPE(nn.Module):
    def __init__(self, dims: int, base: float):
        super().__init__()
        self.dims = dims
        self.base = base

    def __call__(self, x, offset: int = 0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=False,
            base=self.base,
            scale=1.0,
            offset=offset,
        )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or (dim // self.n_heads)
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, args.rms_norm_eps) if args.qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, args.rms_norm_eps) if args.qk_norm else None
        self.rope = RoPE(self.head_dim, args.rope_theta)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.q_norm is not None:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        offset = cache.offset if cache else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Router(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = mx.zeros((args.num_experts,), dtype=mx.float32) if args.moe_router_enable_expert_bias else None

    def __call__(self, x: mx.array) -> mx.array:
        return self.gate(x)


class LazySwitchGLU(nn.Module):
    def __init__(self, layer_idx: int, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

    def _remap_indices(self, indices: mx.array) -> tuple[list[int], mx.array]:
        mx.eval(indices)
        idx_np = np.array(indices, copy=False)
        flat = idx_np.reshape(-1)
        values, counts = np.unique(flat, return_counts=True)
        counts_by_id = {int(value): int(count) for value, count in zip(values, counts)}
        last_by_id: dict[int, int] = {}
        for pos_idx, value in enumerate(flat):
            last_by_id[int(value)] = pos_idx

        # Load less valuable experts first so a bounded LRU retains the policy's
        # preferred experts after prompt prefill. Set HY3_RETAIN_FREQUENT_EXPERTS=0
        # or HY3_RETAIN_POLICY=id to restore stable expert-id order.
        policy = os.environ.get("HY3_RETAIN_POLICY", "").lower().replace("-", "_")
        if os.environ.get("HY3_RETAIN_FREQUENT_EXPERTS", "1").lower() in {"0", "false", "no", "off"}:
            policy = "id"
        if not policy:
            policy = "freq"

        expert_ids = [int(value) for value in values]
        if policy == "id":
            unique = sorted(expert_ids)
        elif policy == "last":
            unique = sorted(expert_ids, key=lambda expert_id: (last_by_id[expert_id], expert_id))
        elif policy == "freq_last":
            unique = sorted(expert_ids, key=lambda expert_id: (counts_by_id[expert_id], last_by_id[expert_id], expert_id))
        elif policy == "last_freq":
            unique = sorted(expert_ids, key=lambda expert_id: (last_by_id[expert_id], counts_by_id[expert_id], expert_id))
        elif policy == "freq":
            unique = sorted(expert_ids, key=lambda expert_id: (counts_by_id[expert_id], expert_id))
        else:
            raise ValueError(f"unsupported HY3_RETAIN_POLICY={policy!r}")

        pos = {expert_id: i for i, expert_id in enumerate(unique)}
        remapped_np = np.vectorize(pos.__getitem__, otypes=[np.int32])(idx_np)
        return unique, mx.array(remapped_np, dtype=mx.int32)

    def _qlinear(self, x: mx.array, indices: mx.array, packed: dict[str, mx.array]) -> mx.array:
        return mx.gather_qmm(
            x,
            packed["weight"],
            packed["scales"],
            packed["biases"],
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=False,
        )

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        store = get_sidecar_store()
        t_remap = time.time()
        expert_ids, remapped = self._remap_indices(indices)
        store.remap_time_s += time.time() - t_remap
        packed = store.get_experts(self.layer_idx, expert_ids)
        t_qmm = time.time()
        x = mx.expand_dims(x, (-2, -3))
        x_up = self._qlinear(x, remapped, packed["up_proj"])
        x_gate = self._qlinear(x, remapped, packed["gate_proj"])
        x = self._qlinear(swiglu(x_gate, x_up), remapped, packed["down_proj"])
        out = x.squeeze(-2)
        eval_for_timing(out)
        store.qmm_time_s += time.time() - t_qmm
        return out


class Hy3MoE(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = int(layer_idx)
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.route_norm = args.route_norm
        self.router_scaling_factor = args.router_scaling_factor
        self.use_sigmoid = args.moe_router_use_sigmoid

        self.router = Router(args)
        self.switch_mlp = LazySwitchGLU(layer_idx)
        if args.num_shared_experts > 0:
            self.shared_mlp = MLP(args.hidden_size, args.moe_intermediate_size * args.num_shared_experts)
        else:
            self.shared_mlp = None

    def __call__(self, x: mx.array):
        if routed_mlp_disabled():
            if self.shared_mlp is not None and not shared_mlp_disabled():
                if profile_enabled():
                    t = time.time()
                    y = self.shared_mlp(x)
                    eval_for_timing(y)
                    _MOE_TIMES["shared_s"] += time.time() - t
                    return y
                return self.shared_mlp(x)
            return mx.zeros_like(x)

        if not profile_enabled():
            gates = self.router(x)
            if self.use_sigmoid:
                scores = mx.sigmoid(gates.astype(mx.float32))
            else:
                scores = mx.softmax(gates.astype(mx.float32), axis=-1)

            selection_scores = scores
            if self.router.expert_bias is not None:
                selection_scores = selection_scores + self.router.expert_bias

            k = effective_top_k(self.top_k)
            inds = mx.stop_gradient(mx.argpartition(-selection_scores, kth=k - 1, axis=-1)[..., :k])
            record_route_trace(self.layer_idx, inds)
            selected_scores = mx.take_along_axis(scores, inds, axis=-1)
            if self.route_norm and k > 1:
                selected_scores = selected_scores / selected_scores.sum(axis=-1, keepdims=True)
            selected_scores = selected_scores * self.router_scaling_factor

            if cpp_route_mlp_enabled():
                y = get_cpp_route_client().compute(self.layer_idx, x, inds, selected_scores)
            else:
                y = self.switch_mlp(x, inds)
                y = (y * selected_scores[..., None].astype(mx.float32)).sum(axis=-2).astype(y.dtype)
            capture_parity_fixture(self.layer_idx, x, inds, selected_scores, y)
            if self.shared_mlp is not None and not shared_mlp_disabled():
                y = y + self.shared_mlp(x)
            return y

        t = time.time()
        gates = self.router(x)
        eval_for_timing(gates)
        _MOE_TIMES["router_s"] += time.time() - t

        t = time.time()
        if self.use_sigmoid:
            scores = mx.sigmoid(gates.astype(mx.float32))
        else:
            scores = mx.softmax(gates.astype(mx.float32), axis=-1)
        selection_scores = scores
        if self.router.expert_bias is not None:
            selection_scores = selection_scores + self.router.expert_bias
        k = effective_top_k(self.top_k)
        inds = mx.stop_gradient(mx.argpartition(-selection_scores, kth=k - 1, axis=-1)[..., :k])
        record_route_trace(self.layer_idx, inds)
        selected_scores = mx.take_along_axis(scores, inds, axis=-1)
        if self.route_norm and k > 1:
            selected_scores = selected_scores / selected_scores.sum(axis=-1, keepdims=True)
        selected_scores = selected_scores * self.router_scaling_factor
        eval_for_timing(inds, selected_scores)
        _MOE_TIMES["selection_s"] += time.time() - t

        t = time.time()
        if cpp_route_mlp_enabled():
            y = get_cpp_route_client().compute(self.layer_idx, x, inds, selected_scores)
            eval_for_timing(y)
            _MOE_TIMES["cpp_route_s"] += time.time() - t
        else:
            y = self.switch_mlp(x, inds)
            eval_for_timing(y)
            _MOE_TIMES["switch_s"] += time.time() - t

            t = time.time()
            y = (y * selected_scores[..., None].astype(mx.float32)).sum(axis=-2).astype(y.dtype)
            eval_for_timing(y)
            _MOE_TIMES["weight_s"] += time.time() - t
        capture_parity_fixture(self.layer_idx, x, inds, selected_scores, y)

        if self.shared_mlp is not None and not shared_mlp_disabled():
            t = time.time()
            y = y + self.shared_mlp(x)
            eval_for_timing(y)
            _MOE_TIMES["shared_s"] += time.time() - t
        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.self_attn = Attention(args)
        if layer_idx < args.first_k_dense_replace:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        else:
            self.mlp = Hy3MoE(args, layer_idx)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None):
        if not profile_enabled():
            r = self.self_attn(self.input_layernorm(x), mask, cache)
            h = x + r
            r = self.mlp(self.post_attention_layernorm(h))
            return h + r

        t_total = time.time()
        t = time.time()
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        eval_for_timing(r)
        attn_s = time.time() - t

        t = time.time()
        h = x + r
        eval_for_timing(h)
        residual_s = time.time() - t

        t = time.time()
        r = self.mlp(self.post_attention_layernorm(h))
        eval_for_timing(r)
        mlp_s = time.time() - t

        out = h + r
        eval_for_timing(out)
        total_s = time.time() - t_total
        _PROFILE.append(
            {
                "layer": self.layer_idx,
                "kind": "moe" if isinstance(self.mlp, Hy3MoE) else "dense",
                "attn_s": round(attn_s, 3),
                "mlp_s": round(mlp_s, 3),
                "residual_s": round(residual_s, 3),
                "total_s": round(total_s, 3),
            }
        )
        return out


class Hy3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Hy3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k
        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if "router.gate" in path:
                return {"group_size": 64, "bits": 8}
            return True
        return predicate
