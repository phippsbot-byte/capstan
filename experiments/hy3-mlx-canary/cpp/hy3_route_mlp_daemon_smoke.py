#!/usr/bin/env python3
"""Verify packed ExpertBank reuse against a real routed-layer fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path
from typing import Any

REQ = struct.Struct("<4sIIII")
RESP = struct.Struct("<4sIIIIdQ")


def read_response(proc: subprocess.Popen[bytes]) -> tuple[bytes, dict[str, int | float]]:
    assert proc.stdout is not None
    header = proc.stdout.read(RESP.size)
    if len(header) != RESP.size:
        raise RuntimeError(f"short response header: {len(header)}")
    magic, status, payload_floats, read_calls, cache_hits, compute_s, bytes_read = RESP.unpack(header)
    if magic == b"HY3E" or status != 0:
        message = proc.stdout.read(payload_floats).decode("utf-8", "replace")
        raise RuntimeError(f"daemon error: {message}")
    if magic != b"HY3O":
        raise RuntimeError(f"bad response magic: {magic!r}")
    payload = proc.stdout.read(payload_floats * 4)
    if len(payload) != payload_floats * 4:
        raise RuntimeError("short response payload")
    return payload, {
        "read_calls": read_calls,
        "cache_hits": cache_hits,
        "compute_s": compute_s,
        "bytes_read": bytes_read,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daemon", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--packed-cache-gib", type=float, default=0.1)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text())
    hidden = fixture.get("hidden_tokens", fixture.get("hidden"))
    experts = fixture.get("experts_flat", fixture.get("experts"))
    weights = fixture.get("route_weights_flat", fixture.get("route_weights"))
    if hidden is None or experts is None or weights is None:
        raise SystemExit("fixture is missing hidden, experts, or route weights")
    seq_len = int(fixture.get("seq_len", 1))
    topk = int(fixture["topk"])
    layer = int(fixture["layer"])

    payload = (
        REQ.pack(b"HY3R", layer, seq_len, topk, 0)
        + struct.pack(f"<{len(hidden)}f", *hidden)
        + struct.pack(f"<{len(experts)}i", *experts)
        + struct.pack(f"<{len(weights)}f", *weights)
    )

    command = [
        str(args.daemon),
        "--index", str(args.index),
        "--root", str(args.root),
        "--q4-mode", "direct",
        "--packed-cache-gib", str(args.packed_cache_gib),
    ]
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write(payload)
        proc.stdin.flush()
        first, first_stats = read_response(proc)
        proc.stdin.write(payload)
        proc.stdin.flush()
        second, second_stats = read_response(proc)
        proc.stdin.close()
        return_code = proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    unique_experts = len(set(experts))
    failures: list[str] = []
    if first != second:
        failures.append("cached output differs from uncached output")
    if first_stats["read_calls"] != unique_experts:
        failures.append(f"first read_calls={first_stats['read_calls']} expected={unique_experts}")
    if second_stats["read_calls"] != 0:
        failures.append(f"second read_calls={second_stats['read_calls']} expected=0")
    if second_stats["cache_hits"] != unique_experts:
        failures.append(f"second cache_hits={second_stats['cache_hits']} expected={unique_experts}")
    if second_stats["bytes_read"] != 0:
        failures.append(f"second bytes_read={second_stats['bytes_read']} expected=0")
    if return_code != 0:
        failures.append(f"daemon exited with status {return_code}")

    result: dict[str, Any] = {
        "schema": "hy3-packed-expert-cache-repeat-smoke-v1",
        "status": "FAIL" if failures else "PASS",
        "daemon": str(args.daemon),
        "index": str(args.index),
        "root": str(args.root),
        "fixture": str(args.fixture),
        "layer": layer,
        "seq_len": seq_len,
        "topk": topk,
        "unique_experts": unique_experts,
        "packed_cache_gib": args.packed_cache_gib,
        "output_sha256": hashlib.sha256(first).hexdigest(),
        "byte_identical_outputs": first == second,
        "first": first_stats,
        "second": second_stats,
        "failures": failures,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
