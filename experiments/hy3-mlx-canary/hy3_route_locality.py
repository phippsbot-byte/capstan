#!/usr/bin/env python3
"""Analyze Hy3 route traces and simulate per-layer expert cache policies.

The TSV route trace records one row per routed layer/token. The live Python runtime
loads unique experts per routed-layer call, then packs all requested token/top-k
positions from that per-call unique set. This analyzer models that call-level
shape rather than the older selection-by-selection C++ IO replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_TRACE = Path("/Volumes/ModelSSD/logs/hy3-mlx-canary/route-traces/20260707-102115/top5-slot16-pong-3tok-trace.tsv")
DEFAULT_MANIFEST = Path("/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/manifest.json")
POLICIES = ("trace", "id", "freq", "last", "freq_last", "last_freq")


@dataclass(frozen=True)
class TraceRow:
    event: int
    layer: int
    batch: int
    token: int
    experts: tuple[int, ...]
    pass_id: int
    phase: str


@dataclass(frozen=True)
class RouteCall:
    call_index: int
    pass_id: int
    phase: str
    layer: int
    tokens: tuple[int, ...]
    experts_flat: tuple[int, ...]


def parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def parse_str_list(raw: str, allowed: Iterable[str] | None = None) -> list[str]:
    out = [part.strip().replace("-", "_") for part in raw.split(",") if part.strip()]
    if allowed is not None:
        allowed_set = set(allowed)
        bad = sorted(set(out) - allowed_set)
        if bad:
            raise ValueError(f"unsupported value(s): {bad}; allowed={sorted(allowed_set)}")
    return out


def percentile(values: list[int] | list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(ordered[lo])
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_trace(path: Path) -> tuple[dict[str, Any], list[TraceRow], list[RouteCall]]:
    metadata: dict[str, Any] = {}
    raw_rows: list[tuple[int, int, int, int, tuple[int, ...]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("# metadata="):
                metadata = json.loads(line.split("=", 1)[1])
                continue
            if line.startswith("#") or line.startswith("event\t"):
                continue
            parts = line.split("\t")
            if len(parts) != 5:
                raise ValueError(f"{path}:{line_no}: expected 5 tab-separated fields, got {len(parts)}")
            event, layer, batch, token = (int(parts[i]) for i in range(4))
            experts = tuple(parse_int_list(parts[4]))
            if not experts:
                raise ValueError(f"{path}:{line_no}: empty experts field")
            raw_rows.append((event, layer, batch, token, experts))

    raw_rows.sort(key=lambda row: row[0])
    rows: list[TraceRow] = []
    pass_id = 0
    prev_layer: int | None = None
    prompt_tokens = int(metadata.get("prompt_tokens") or 0)
    for event, layer, batch, token, experts in raw_rows:
        if prev_layer is not None and layer < prev_layer:
            pass_id += 1
        phase = "prefill" if pass_id == 0 else "decode"
        # Some traces do not carry prompt_tokens. The first pass is still the prefill.
        if prompt_tokens and pass_id == 0 and token >= prompt_tokens:
            raise ValueError(f"trace token {token} exceeds metadata prompt_tokens={prompt_tokens} in prefill pass")
        rows.append(TraceRow(event=event, layer=layer, batch=batch, token=token, experts=experts, pass_id=pass_id, phase=phase))
        prev_layer = layer

    calls: list[RouteCall] = []
    grouped: OrderedDict[tuple[int, str, int], list[TraceRow]] = OrderedDict()
    for row in rows:
        grouped.setdefault((row.pass_id, row.phase, row.layer), []).append(row)
    for call_index, ((pid, phase, layer), group) in enumerate(grouped.items()):
        experts_flat: list[int] = []
        tokens: list[int] = []
        for row in group:
            tokens.append(row.token)
            experts_flat.extend(row.experts)
        calls.append(
            RouteCall(
                call_index=call_index,
                pass_id=pid,
                phase=phase,
                layer=layer,
                tokens=tuple(tokens),
                experts_flat=tuple(experts_flat),
            )
        )
    return metadata, rows, calls


def load_span_bytes(manifest_path: Path | None) -> tuple[dict[tuple[int, int], int], int]:
    if manifest_path is None:
        return {}, 0
    manifest = json.loads(manifest_path.read_text())
    schema = manifest.get("schema")
    spans: dict[tuple[int, int], list[int]] = {}
    if schema == "hy3-packed-sidecar-v1":
        for entry in manifest["packed_entries"]:
            layer = int(entry["layer"])
            expert = int(entry["expert"])
            start = int(entry["file_offset"])
            end = start + int(entry["nbytes"])
            current = spans.setdefault((layer, expert), [start, end])
            current[0] = min(current[0], start)
            current[1] = max(current[1], end)
    elif "sidecar_entries" in manifest:
        # v0 metadata-only layout: each entry has an expert-major first dimension.
        per_expert: dict[tuple[int, str, str], int] = {}
        for entry in manifest["sidecar_entries"]:
            layer = int(entry["layer"])
            shape = [int(x) for x in entry["shape"]]
            if not shape or shape[0] <= 0:
                continue
            dtype = entry["dtype"]
            dtype_bytes = {"U32": 4, "BF16": 2, "F32": 4}[dtype]
            elems = 1
            for dim in shape[1:]:
                elems *= dim
            per_expert[(layer, entry["expert_family"], entry["tensor_kind"])] = elems * dtype_bytes
        layers = sorted({key[0] for key in per_expert})
        # Expert count is not directly needed for missing-byte estimates; fill rows lazily below.
        for layer in layers:
            one = sum(size for (l, _, _), size in per_expert.items() if l == layer)
            for expert in range(int(manifest.get("num_experts", 192))):
                spans[(layer, expert)] = [0, one]
    else:
        raise ValueError(f"unsupported manifest schema {schema!r} in {manifest_path}")

    span_bytes = {key: end - start for key, (start, end) in spans.items()}
    default = int(statistics.median(span_bytes.values())) if span_bytes else 0
    return span_bytes, default


def ordered_unique_for_policy(experts_flat: tuple[int, ...], policy: str) -> list[int]:
    counts = Counter(experts_flat)
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for pos, expert in enumerate(experts_flat):
        first.setdefault(expert, pos)
        last[expert] = pos
    experts = list(counts)
    if policy == "trace":
        return sorted(experts, key=lambda expert: (first[expert], expert))
    if policy == "id":
        return sorted(experts)
    if policy == "freq":
        return sorted(experts, key=lambda expert: (counts[expert], expert))
    if policy == "last":
        return sorted(experts, key=lambda expert: (last[expert], expert))
    if policy == "freq_last":
        return sorted(experts, key=lambda expert: (counts[expert], last[expert], expert))
    if policy == "last_freq":
        return sorted(experts, key=lambda expert: (last[expert], counts[expert], expert))
    raise ValueError(f"unsupported policy {policy!r}")


def trim_cache(cache: OrderedDict[int, None], slot_bank: int, protected: set[int] | None = None) -> int:
    protected = protected or set()
    evictions = 0
    while len(cache) > slot_bank:
        victim = None
        for candidate in cache:
            if candidate not in protected:
                victim = candidate
                break
        if victim is None:
            return evictions
        del cache[victim]
        evictions += 1
    return evictions


def phase_template() -> dict[str, Any]:
    return {"calls": 0, "unique_requests": 0, "hits": 0, "misses": 0, "evictions": 0, "bytes_read": 0}


def simulate_policy(
    calls: list[RouteCall],
    *,
    slot_bank: int,
    policy: str,
    span_bytes: dict[tuple[int, int], int],
    default_span_bytes: int,
) -> dict[str, Any]:
    if slot_bank < 0:
        raise ValueError("slot_bank must be non-negative")
    caches: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
    phase_stats: dict[str, dict[str, Any]] = defaultdict(phase_template)
    layer_stats: dict[int, dict[str, Any]] = defaultdict(lambda: {
        "calls": 0,
        "selections": 0,
        "unique_requests": 0,
        "hits": 0,
        "misses": 0,
        "evictions": 0,
        "bytes_read": 0,
        "max_unique_per_call": 0,
    })
    total_hits = total_misses = total_evictions = total_bytes = 0
    total_unique_requests = 0
    total_selections = 0
    oversized_calls = 0
    max_unique_per_call = 0

    for call in calls:
        ordered_unique = ordered_unique_for_policy(call.experts_flat, policy)
        unique_count = len(ordered_unique)
        requested = set(ordered_unique)
        if unique_count > slot_bank:
            oversized_calls += 1
        max_unique_per_call = max(max_unique_per_call, unique_count)
        total_unique_requests += unique_count
        total_selections += len(call.experts_flat)

        pstats = phase_stats[call.phase]
        lstats = layer_stats[call.layer]
        pstats["calls"] += 1
        pstats["unique_requests"] += unique_count
        lstats["calls"] += 1
        lstats["selections"] += len(call.experts_flat)
        lstats["unique_requests"] += unique_count
        lstats["max_unique_per_call"] = max(lstats["max_unique_per_call"], unique_count)

        cache = caches[call.layer]
        missing: list[int] = []
        for expert in ordered_unique:
            if expert in cache:
                cache.move_to_end(expert)
                total_hits += 1
                pstats["hits"] += 1
                lstats["hits"] += 1
            else:
                missing.append(expert)
                total_misses += 1
                pstats["misses"] += 1
                lstats["misses"] += 1
        for expert in missing:
            cache[expert] = None
            cache.move_to_end(expert)
            nbytes = span_bytes.get((call.layer, expert), default_span_bytes)
            total_bytes += nbytes
            pstats["bytes_read"] += nbytes
            lstats["bytes_read"] += nbytes

        # Mirrors Hy3SidecarStore.get_experts(): protect the current request until
        # the packed tensors are materialized, then enforce the hard cap afterward.
        evicted = trim_cache(cache, slot_bank, protected=requested)
        post_evicted = trim_cache(cache, slot_bank)
        total_evictions += evicted + post_evicted
        pstats["evictions"] += evicted + post_evicted
        lstats["evictions"] += evicted + post_evicted

    final_cache_experts = sum(len(cache) for cache in caches.values())
    final_cache_bytes = sum(span_bytes.get((layer, expert), default_span_bytes) for layer, cache in caches.items() for expert in cache)
    layer_rows = []
    for layer, stats in sorted(layer_stats.items()):
        unique_requests = int(stats["unique_requests"])
        misses = int(stats["misses"])
        layer_rows.append(
            {
                "layer": layer,
                **stats,
                "hit_rate": round(float(stats["hits"]) / unique_requests, 6) if unique_requests else 0.0,
                "gib_read": round(float(stats["bytes_read"]) / (1024 ** 3), 6),
            }
        )
    layer_rows_by_misses = sorted(layer_rows, key=lambda row: (row["misses"], row["evictions"], row["unique_requests"]), reverse=True)
    phase_out = {}
    for phase, stats in sorted(phase_stats.items()):
        unique_requests = int(stats["unique_requests"])
        phase_out[phase] = {
            **stats,
            "hit_rate": round(float(stats["hits"]) / unique_requests, 6) if unique_requests else 0.0,
            "gib_read": round(float(stats["bytes_read"]) / (1024 ** 3), 6),
        }
    return {
        "slot_bank": slot_bank,
        "policy": policy,
        "calls": len(calls),
        "selections": total_selections,
        "unique_requests": total_unique_requests,
        "selection_dedup_saved": total_selections - total_unique_requests,
        "hits": total_hits,
        "misses": total_misses,
        "evictions": total_evictions,
        "hit_rate": round(total_hits / total_unique_requests, 6) if total_unique_requests else 0.0,
        "miss_rate": round(total_misses / total_unique_requests, 6) if total_unique_requests else 0.0,
        "bytes_read": total_bytes,
        "gib_read": round(total_bytes / (1024 ** 3), 6),
        "oversized_calls": oversized_calls,
        "max_unique_per_call": max_unique_per_call,
        "final_cache_experts": final_cache_experts,
        "final_cache_bytes": final_cache_bytes,
        "final_cache_gib": round(final_cache_bytes / (1024 ** 3), 6),
        "phase": phase_out,
        "top_miss_layers": layer_rows_by_misses[:12],
    }


def locality_by_layer(calls: list[RouteCall], span_bytes: dict[tuple[int, int], int], default_span_bytes: int) -> list[dict[str, Any]]:
    per_layer_calls: dict[int, list[RouteCall]] = defaultdict(list)
    for call in calls:
        per_layer_calls[call.layer].append(call)
    rows: list[dict[str, Any]] = []
    for layer, layer_calls in sorted(per_layer_calls.items()):
        counts: Counter[int] = Counter()
        prefill: Counter[int] = Counter()
        decode: Counter[int] = Counter()
        unique_per_call: list[int] = []
        reuse_gaps: list[int] = []
        last_call: dict[int, int] = {}
        for local_call_idx, call in enumerate(layer_calls):
            counts.update(call.experts_flat)
            if call.phase == "prefill":
                prefill.update(call.experts_flat)
            else:
                decode.update(call.experts_flat)
            unique = set(call.experts_flat)
            unique_per_call.append(len(unique))
            for expert in unique:
                if expert in last_call:
                    reuse_gaps.append(local_call_idx - last_call[expert])
                last_call[expert] = local_call_idx
        unique_experts = len(counts)
        total_selections = sum(counts.values())
        top = counts.most_common(8)
        prefill_set = set(prefill)
        decode_set = set(decode)
        rows.append(
            {
                "layer": layer,
                "calls": len(layer_calls),
                "selections": total_selections,
                "unique_experts": unique_experts,
                "unique_fraction_of_192": round(unique_experts / 192.0, 6),
                "trace_reuse_rate": round(1.0 - (unique_experts / total_selections), 6) if total_selections else 0.0,
                "unique_per_call_avg": round(statistics.mean(unique_per_call), 3) if unique_per_call else 0.0,
                "unique_per_call_max": max(unique_per_call) if unique_per_call else 0,
                "prefill_unique": len(prefill_set),
                "decode_unique": len(decode_set),
                "decode_overlap_prefill": len(decode_set & prefill_set),
                "decode_overlap_prefill_rate": round(len(decode_set & prefill_set) / len(decode_set), 6) if decode_set else None,
                "reuse_gap_calls_median": percentile(reuse_gaps, 0.5),
                "reuse_gap_calls_p95": percentile(reuse_gaps, 0.95),
                "reuse_gap_calls_max": max(reuse_gaps) if reuse_gaps else None,
                "top_experts": [{"expert": expert, "count": count} for expert, count in top],
                "unique_payload_gib": round(sum(span_bytes.get((layer, expert), default_span_bytes) for expert in counts) / (1024 ** 3), 6),
            }
        )
    return rows


def trace_summary(metadata: dict[str, Any], rows: list[TraceRow], calls: list[RouteCall], trace_path: Path) -> dict[str, Any]:
    phases = Counter(row.phase for row in rows)
    calls_by_phase = Counter(call.phase for call in calls)
    passes: list[dict[str, Any]] = []
    for pass_id in sorted({row.pass_id for row in rows}):
        pr = [row for row in rows if row.pass_id == pass_id]
        passes.append(
            {
                "pass_id": pass_id,
                "phase": pr[0].phase if pr else None,
                "events": len(pr),
                "layers": len({row.layer for row in pr}),
                "tokens": sorted({row.token for row in pr}),
                "selected_experts": sum(len(row.experts) for row in pr),
            }
        )
    return {
        "trace": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "metadata": metadata,
        "events": len(rows),
        "calls": len(calls),
        "layers": len({row.layer for row in rows}),
        "passes": len({row.pass_id for row in rows}),
        "phase_events": dict(sorted(phases.items())),
        "phase_calls": dict(sorted(calls_by_phase.items())),
        "selected_experts": sum(len(row.experts) for row in rows),
        "avg_k": round(sum(len(row.experts) for row in rows) / len(rows), 3) if rows else 0.0,
        "pass_details": passes,
    }


def write_markdown(path: Path, result: dict[str, Any], focus_slot: int, focus_policy: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sims = result["simulations"]
    focus = next((row for row in sims if row["slot_bank"] == focus_slot and row["policy"] == focus_policy), None)
    best_by_slot = []
    for slot in sorted({row["slot_bank"] for row in sims}):
        rows = [row for row in sims if row["slot_bank"] == slot]
        best = min(rows, key=lambda row: (row["misses"], row["evictions"], row["policy"]))
        best_by_slot.append(best)
    lines = [
        "# Hy3 route locality analysis",
        "",
        f"Trace: `{result['trace_summary']['trace']}`",
        "",
        "## Trace shape",
        "",
        f"- Events: **{result['trace_summary']['events']}**",
        f"- Layer calls: **{result['trace_summary']['calls']}**",
        f"- Passes: **{result['trace_summary']['passes']}**",
        f"- Selected experts: **{result['trace_summary']['selected_experts']}**",
        f"- Avg top-k: **{result['trace_summary']['avg_k']}**",
        "",
        "## Best policy per slot bank",
        "",
        "| Slot | Best policy | Misses | Hits | Hit rate | Read GiB | Evictions | Oversized calls | Final cache GiB |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_by_slot:
        lines.append(
            f"| {row['slot_bank']} | `{row['policy']}` | {row['misses']} | {row['hits']} | {row['hit_rate']:.3f} | "
            f"{row['gib_read']:.3f} | {row['evictions']} | {row['oversized_calls']} | {row['final_cache_gib']:.3f} |"
        )
    if focus:
        lines += [
            "",
            f"## Focus: `{focus_policy}` at slot {focus_slot}",
            "",
            f"- Misses: **{focus['misses']}**",
            f"- Hits: **{focus['hits']}**",
            f"- Read: **{focus['gib_read']:.3f}GiB**",
            f"- Evictions: **{focus['evictions']}**",
            f"- Final cache: **{focus['final_cache_gib']:.3f}GiB**",
            "",
            "Top miss layers:",
            "",
            "| Layer | Misses | Hits | Hit rate | Evictions | Unique requests | Max unique/call | Read GiB |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in focus["top_miss_layers"][:12]:
            lines.append(
                f"| {row['layer']} | {row['misses']} | {row['hits']} | {row['hit_rate']:.3f} | {row['evictions']} | "
                f"{row['unique_requests']} | {row['max_unique_per_call']} | {row['gib_read']:.3f} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        result["verdict"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_verdict(simulations: list[dict[str, Any]], focus_slot: int, focus_policy: str) -> str:
    by_slot: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in simulations:
        by_slot[row["slot_bank"]].append(row)
    focus_rows = by_slot.get(focus_slot, [])
    if not focus_rows:
        return "No focus-slot simulation was produced."
    focus = next((row for row in focus_rows if row["policy"] == focus_policy), None)
    best = min(focus_rows, key=lambda row: (row["misses"], row["evictions"], row["policy"]))
    parts = []
    if focus:
        if best["policy"] == focus_policy:
            parts.append(f"At slot {focus_slot}, `{focus_policy}` is tied/best by miss count among tested policies.")
        else:
            delta = focus["misses"] - best["misses"]
            parts.append(f"At slot {focus_slot}, `{best['policy']}` beats `{focus_policy}` by {delta} misses in this trace.")
    larger = sorted(slot for slot in by_slot if slot > focus_slot)
    if larger:
        next_slot = larger[0]
        best_next = min(by_slot[next_slot], key=lambda row: (row["misses"], row["evictions"], row["policy"]))
        best_focus = best
        saved = best_focus["misses"] - best_next["misses"]
        parts.append(f"Moving from slot {focus_slot} to slot {next_slot} saves {saved} misses but increases final cache to {best_next['final_cache_gib']:.2f}GiB.")
    if best.get("oversized_calls"):
        parts.append(f"Slot {focus_slot} has {best['oversized_calls']} oversized calls; post-pack trimming is mandatory for memory safety.")
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--slot-banks", default="8,12,16,20,24,32")
    parser.add_argument("--policies", default="trace,id,freq,last,freq_last,last_freq")
    parser.add_argument("--focus-slot", type=int, default=16)
    parser.add_argument("--focus-policy", default="freq")
    parser.add_argument("--top-layers", type=int, default=12)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()

    slot_banks = parse_int_list(args.slot_banks)
    policies = parse_str_list(args.policies, POLICIES)
    if args.focus_policy.replace("-", "_") not in POLICIES:
        raise ValueError(f"unsupported focus policy {args.focus_policy!r}")
    args.focus_policy = args.focus_policy.replace("-", "_")

    metadata, rows, calls = parse_trace(args.trace)
    span_bytes, default_span_bytes = load_span_bytes(args.manifest if args.manifest else None)
    locality_rows = locality_by_layer(calls, span_bytes, default_span_bytes)
    simulations: list[dict[str, Any]] = []
    for slot in slot_banks:
        for policy in policies:
            simulations.append(
                simulate_policy(
                    calls,
                    slot_bank=slot,
                    policy=policy,
                    span_bytes=span_bytes,
                    default_span_bytes=default_span_bytes,
                )
            )
    result = {
        "schema": "hy3-route-locality-analysis-v1",
        "trace_summary": trace_summary(metadata, rows, calls, args.trace),
        "manifest": str(args.manifest) if args.manifest else None,
        "expert_span_default_bytes": default_span_bytes,
        "expert_span_default_mib": round(default_span_bytes / (1024 ** 2), 6) if default_span_bytes else 0.0,
        "slot_banks": slot_banks,
        "policies": policies,
        "focus_slot": args.focus_slot,
        "focus_policy": args.focus_policy,
        "simulations": simulations,
        "locality_by_layer": locality_rows,
        "top_churn_layers": sorted(locality_rows, key=lambda row: (row["unique_experts"], row["unique_per_call_max"], row["selections"]), reverse=True)[: args.top_layers],
    }
    result["verdict"] = build_verdict(simulations, args.focus_slot, args.focus_policy)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.md_out:
        write_markdown(args.md_out, result, args.focus_slot, args.focus_policy)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
