#!/usr/bin/env python3
"""Emit a compact TSV index for the packed Hy3 sidecar.

The packed manifest is correct but enormous (~50MiB JSON, >2M pretty-printed
lines). The C++ hot-path should not drag in a JSON dependency or parse that on
startup. This script turns it into a small line-oriented index that C++ can mmap
or stream cheaply.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path("/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/manifest.json")
FAMILY_ORDER = {"up_proj": 0, "gate_proj": 1, "down_proj": 2}
KIND_ORDER = {"weight": 0, "scales": 1, "biases": 2}


def emit_index(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    entries = manifest.get("packed_entries") or []
    if manifest.get("schema") != "hy3-packed-sidecar-v1":
        raise SystemExit(f"unsupported manifest schema: {manifest.get('schema')!r}")
    if not entries:
        raise SystemExit("manifest has no packed_entries")

    entries = sorted(
        entries,
        key=lambda e: (
            int(e["layer"]),
            int(e["expert"]),
            FAMILY_ORDER[str(e["expert_family"])],
            KIND_ORDER[str(e["tensor_kind"])],
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    total_bytes = 0
    with tmp_path.open("w", encoding="utf-8") as out:
        out.write("# schema=hy3-packed-sidecar-compact-v1\n")
        out.write(f"# source_manifest={manifest_path}\n")
        out.write(f"# source_model_dir={manifest.get('source_model_dir', '')}\n")
        out.write(f"# layout={manifest.get('layout', '')}\n")
        out.write("layer\texpert\tfamily\tkind\tfile_offset\tnbytes\tfile\n")
        for entry in entries:
            nbytes = int(entry["nbytes"])
            total_bytes += nbytes
            out.write(
                "\t".join(
                    [
                        str(int(entry["layer"])),
                        str(int(entry["expert"])),
                        str(entry["expert_family"]),
                        str(entry["tensor_kind"]),
                        str(int(entry["file_offset"])),
                        str(nbytes),
                        str(entry["file"]),
                    ]
                )
                + "\n"
            )
    tmp_path.replace(output_path)
    return {
        "output": str(output_path),
        "entries": len(entries),
        "bytes": total_bytes,
        "gib": round(total_bytes / (1024**3), 6),
        "layers": len({int(e["layer"]) for e in entries}),
        "experts": len({int(e["expert"]) for e in entries}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (args.manifest.parent / "compact-index.tsv")
    result = emit_index(args.manifest, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
