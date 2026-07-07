# Local Hy3 / hy_v3 MLX lazy-expert prototype.
# Built for mlx-community/Hy3-preview-4bit on the Studio.
# Goal: keep resident weights in MLX, stream/cache selected routed experts from safetensors.

from __future__ import annotations

import gc
import json
import os
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


DEFAULT_LAYOUT = "/Users/nb/LLM/hy3-mlx-canary/hy3-sidecar-layout.json"


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
        self.loads = 0
        self.hits = 0
        self.evictions = 0
        self.read_calls = 0
        self.read_bytes = 0
        self.get_calls = 0
        self.requested_unique_total = 0
        self.requested_unique_max = 0
        self.load_time_s = 0.0
        self.pack_time_s = 0.0
        self.remap_time_s = 0.0
        self.qmm_time_s = 0.0
        self.gc_calls = 0
        self.gc_time_s = 0.0

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

    def _load_packed_expert(self, layer: int, expert_id: int) -> dict[str, dict[str, mx.array]]:
        entries: list[dict[str, Any]] = []
        for family in ("up_proj", "gate_proj", "down_proj"):
            for kind in ("weight", "scales", "biases"):
                entries.append(self.packed_entries[(layer, expert_id, family, kind)])
        paths = {self._packed_path(e) for e in entries}
        if len(paths) != 1:
            raise RuntimeError(f"packed expert L{layer} E{expert_id} spans multiple files: {paths}")
        path = next(iter(paths))
        start = min(int(e["file_offset"]) for e in entries)
        end = max(int(e["file_offset"]) + int(e["nbytes"]) for e in entries)
        raw = self._pread(path, start, end - start)
        if len(raw) != end - start:
            raise IOError(f"short packed read for L{layer} E{expert_id}: {len(raw)} != {end - start}")
        out: dict[str, dict[str, mx.array]] = {"up_proj": {}, "gate_proj": {}, "down_proj": {}}
        raw_view = memoryview(raw)
        for entry in entries:
            rel = int(entry["file_offset"]) - start
            nbytes = int(entry["nbytes"])
            shape = tuple(int(x) for x in entry["shape"])
            out[entry["expert_family"]][entry["tensor_kind"]] = self._mx_from_raw(raw_view[rel : rel + nbytes], entry["dtype"], shape)
        mx.eval(out)
        self.loads += 1
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

    def get_experts(self, layer: int, expert_ids: list[int]) -> dict[str, dict[str, mx.array]]:
        self.get_calls += 1
        self.requested_unique_total += len(expert_ids)
        self.requested_unique_max = max(self.requested_unique_max, len(expert_ids))
        layer_cache = self.cache.setdefault(layer, OrderedDict())
        selected: list[dict[str, dict[str, mx.array]]] = []
        for expert_id in expert_ids:
            if expert_id in layer_cache:
                layer_cache.move_to_end(expert_id)
                self.hits += 1
            else:
                t_load = time.time()
                layer_cache[expert_id] = self._load_expert(layer, expert_id)
                self.load_time_s += time.time() - t_load
                evicted = False
                while len(layer_cache) > self.slot_bank:
                    layer_cache.popitem(last=False)
                    self.evictions += 1
                    evicted = True
                if evicted and env_truthy("HY3_GC_ON_EVICT"):
                    t_gc = time.time()
                    gc.collect()
                    self.gc_calls += 1
                    self.gc_time_s += time.time() - t_gc
            selected.append(layer_cache[expert_id])

        t_pack = time.time()
        packed: dict[str, dict[str, mx.array]] = {}
        for family in ("up_proj", "gate_proj", "down_proj"):
            packed[family] = {}
            for kind in ("weight", "scales", "biases"):
                packed[family][kind] = mx.concatenate([expert[family][kind] for expert in selected], axis=0)
        mx.eval(packed)
        self.pack_time_s += time.time() - t_pack
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
_PROFILE: list[dict[str, Any]] = []
_ROUTE_TRACE: list[dict[str, Any]] = []
_ROUTE_EVENT = 0
_PARITY_FIXTURE: dict[str, Any] | None = None
_MOE_TIMES: dict[str, float] = {
    "router_s": 0.0,
    "selection_s": 0.0,
    "switch_s": 0.0,
    "weight_s": 0.0,
    "shared_s": 0.0,
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
    return os.environ.get("HY3_PARITY_LAYER") not in {None, ""}


def reset_parity_fixture() -> None:
    global _PARITY_FIXTURE
    _PARITY_FIXTURE = None


def capture_parity_fixture(layer: int, x: mx.array, indices: mx.array, route_weights: mx.array, routed_output: mx.array) -> None:
    global _PARITY_FIXTURE
    if not parity_capture_enabled() or _PARITY_FIXTURE is not None:
        return
    target_layer = int(os.environ["HY3_PARITY_LAYER"])
    if int(layer) != target_layer:
        return
    token = int(os.environ.get("HY3_PARITY_TOKEN", "0"))
    batch = int(os.environ.get("HY3_PARITY_BATCH", "0"))
    mx.eval(x, indices, route_weights, routed_output)
    idx_np = np.array(indices, copy=False)
    if idx_np.ndim != 3 or batch >= idx_np.shape[0] or token >= idx_np.shape[1]:
        raise ValueError(f"cannot capture parity token batch={batch} token={token} from indices shape {idx_np.shape}")
    hidden_np = np.array(x[batch, token].astype(mx.float32))
    weights_np = np.array(route_weights[batch, token].astype(mx.float32))
    output_np = np.array(routed_output[batch, token].astype(mx.float32))
    _PARITY_FIXTURE = {
        "schema": "hy3-routed-layer-parity-v1",
        "layer": int(layer),
        "batch": batch,
        "token": token,
        "hidden_size": int(hidden_np.shape[-1]),
        "output_size": int(output_np.shape[-1]),
        "topk": int(idx_np.shape[-1]),
        "experts": [int(v) for v in idx_np[batch, token].tolist()],
        "route_weights": [float(v) for v in weights_np.tolist()],
        "hidden": [float(v) for v in hidden_np.tolist()],
        "expected_routed": [float(v) for v in output_np.tolist()],
    }


def get_parity_fixture() -> dict[str, Any] | None:
    return _PARITY_FIXTURE


def write_parity_fixture(path: str | os.PathLike[str], metadata: Optional[dict[str, Any]] = None) -> str:
    if _PARITY_FIXTURE is None:
        raise RuntimeError("no parity fixture captured")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(_PARITY_FIXTURE)
    if metadata:
        payload["metadata"] = metadata
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(out)


def shared_mlp_disabled() -> bool:
    return env_truthy("HY3_DISABLE_SHARED_MLP")


def routed_mlp_disabled() -> bool:
    return env_truthy("HY3_DISABLE_ROUTED_MLP")


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


def reset_profile() -> None:
    _PROFILE.clear()
    reset_route_trace()
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
        y = self.switch_mlp(x, inds)
        eval_for_timing(y)
        _MOE_TIMES["switch_s"] += time.time() - t

        t = time.time()
        y = (y * selected_scores[..., None].astype(mx.float32)).sum(axis=-2).astype(y.dtype)
        capture_parity_fixture(self.layer_idx, x, inds, selected_scores, y)
        eval_for_timing(y)
        _MOE_TIMES["weight_s"] += time.time() - t

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
