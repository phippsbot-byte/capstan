# Hy3 C++ sidecar IO substrate

First Capstan/C++ cut for the Hy3 packed sidecar. This is deliberately not a
model server yet. It proves the C++ runtime can consume the packed layer-major
sidecar index and issue one contiguous `pread` per selected expert.

Why this exists: the Python canary proved the algorithm, but Python/MLX cache
lifetime and request glue are not the product path. Capstan needs a small native
substrate before we touch decode/prefill kernels.

## Build

```bash
cd /Users/nb/LLM/hy3-mlx-canary
/opt/homebrew/bin/python3.11 hy3_emit_compact_index.py \
  --manifest /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/manifest.json
cmake -S cpp -B build/hy3-sidecar-io
cmake --build build/hy3-sidecar-io --config Release
```

## Smoke

Read the same layer-1 top-8 expert payload shape as the Python smoke:

```bash
./build/hy3-sidecar-io/hy3_sidecar_io \
  --index /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/compact-index.tsv \
  --root /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar \
  --layer 1 --experts 0,1,2,3,4,5,6,7
```

Emulate a cold full-token expert read with the first 8 experts on all 79 MoE
layers:

```bash
./build/hy3-sidecar-io/hy3_sidecar_io \
  --index /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/compact-index.tsv \
  --root /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar \
  --layers 1-79 --topk 8
```

Expected output is JSON with `ok=true`, bytes/read counts, elapsed seconds,
throughput, and an FNV checksum so the compiler cannot optimize away the reads.

## Cache scheduler simulation

Model a native slot-bank cache before wiring real kernels. This exercises the same
sidecar spans while tracking cache hits, misses, evictions, final cache footprint,
and bytes that would cross the sidecar boundary.

Repeated full-top8 trace — should load once, then hit cache:

```bash
./build/hy3-sidecar-io/hy3_sidecar_io \
  --index /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/compact-index.tsv \
  --root /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar \
  --layers 1-79 --topk 8 \
  --simulate-tokens 4 --slot-bank 16 --policy freq --route-pattern fixed
```

Approximate top5 hot trace — two-token expert set repeated over eight tokens:

```bash
./build/hy3-sidecar-io/hy3_sidecar_io \
  --index /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/compact-index.tsv \
  --root /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar \
  --layers 1-79 --topk 5 \
  --simulate-tokens 8 --slot-bank 16 --policy freq --route-pattern hot
```

Adversarial rolling trace — churns through more experts than slot-bank 16 can retain:

```bash
./build/hy3-sidecar-io/hy3_sidecar_io \
  --index /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/compact-index.tsv \
  --root /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar \
  --layers 1-79 --topk 5 \
  --simulate-tokens 8 --slot-bank 16 --policy freq --route-pattern rolling
```

Current benchmark artifact: `cpp/results/20260707-sidecar-io.json`.
