# Hy3 C++ sidecar IO substrate

First Capstan/C++ cut for the Hy3 packed sidecar. This is deliberately not a
model server yet. It proves the C++ runtime can consume the packed layer-major
sidecar index, issue one contiguous `pread` per selected expert, and replay routed
MLP parity fixtures through reusable native modules:

- `hy3_expert_bank.*` — compact-index loading, expert-span materialization, fd reuse.
- `hy3_q4_affine.*` — MLX q4 affine dequantization and dense qlinear.
- `hy3_routed_mlp.*` — routed `up/gate/down + swiglu + route weighting` parity replay.
- `hy3_sidecar_io.cpp` — thin CLI for IO/cache simulation and fixture validation.

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

## Real router trace replay

`hy3_lazy_smoke.py` can now capture real router-selected experts from the Python
prototype and write a TSV trace that C++ can replay:

```bash
HY3_SIDECAR_LAYOUT=/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/manifest.json \
/opt/homebrew/bin/python3.11 hy3_lazy_smoke.py generate-cache \
  --slot-bank 16 --retain-policy freq --topk-cap 5 \
  --prompt 'Reply with exactly pong.' --max-new-tokens 1 \
  --route-trace-out /Volumes/ModelSSD/logs/hy3-mlx-canary/route-traces/<run>/top5-slot16-pong-trace.tsv

./build/hy3-sidecar-io/hy3_sidecar_io \
  --index /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/compact-index.tsv \
  --root /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar \
  --trace /Volumes/ModelSSD/logs/hy3-mlx-canary/route-traces/<run>/top5-slot16-pong-trace.tsv \
  --slot-bank 16 --policy freq
```

First real trace artifact:
`/Volumes/ModelSSD/logs/hy3-mlx-canary/route-traces/20260707-102115/`.

## Routed parity fixtures

Python can now export a compact routed-MoE parity fixture from a real hidden
state/router decision:

```bash
/opt/homebrew/bin/python3.11 hy3_export_layer_fixture.py \
  --layer 1 --token 0 --slot-bank 16 --topk-cap 5 \
  --out /Volumes/ModelSSD/logs/hy3-mlx-canary/parity-fixtures/<run>/hy3-layer1-top5-bos.json
```

The committed fixture is `cpp/fixtures/hy3-layer1-top5-bos.json`.
Native replay materializes the selected expert banks from the packed sidecar,
dequantizes MLX q4 affine weights, runs `up/gate/down + swiglu + route weighting`,
and compares against Python/MLX:

```bash
./build/hy3-sidecar-io/hy3_sidecar_io \
  --index /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/compact-index.tsv \
  --root /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar \
  --fixture cpp/fixtures/hy3-layer1-top5-bos.json
```

Current single-layer result: `parity_pass=true`, max abs error `4.69808e-05`, mean abs error
`3.62875e-06`, RMSE `4.94604e-06`, 5 expert spans / `0.049438GiB` read.

Multi-layer export captures one parity fixture per MoE layer in a single guarded
forward pass:

```bash
/opt/homebrew/bin/python3.11 hy3_export_layer_fixture.py \
  --layers 1-79 --token 0 --slot-bank 16 --topk-cap 5 \
  --out-dir /Volumes/ModelSSD/logs/hy3-mlx-canary/parity-fixtures/<run>

./build/hy3-sidecar-io/hy3_sidecar_io \
  --index /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/compact-index.tsv \
  --root /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar \
  --fixture-list /Volumes/ModelSSD/logs/hy3-mlx-canary/parity-fixtures/<run>/fixtures.txt
```

All-layer artifact: `/Volumes/ModelSSD/logs/hy3-mlx-canary/parity-fixtures/20260707-115831-all-layers/`.
The C++ sweep covered **79/79 MoE layers**, read **3.90564GiB** across **395**
expert spans, and completed in **34.87s**. Absolute error grows with late-layer
activation magnitude, so all-layer pass uses `max_abs <= max(1e-4, 2% of expected
max abs)`: worst absolute error `16.9663` on layer 79, worst relative-to-expected
error `0.0177684` on layer 75, `parity_pass=true`.

Multi-token prefill fixtures use `--all-tokens` plus a short token-id sequence.
This captures `[seq_len, hidden]`, `[seq_len, topk]`, route weights, and expected
routed outputs for each MoE layer:

```bash
/opt/homebrew/bin/python3.11 hy3_export_layer_fixture.py \
  --layers 1-79 --all-tokens --token-ids 120000,79,792,120025 \
  --slot-bank 16 --topk-cap 5 \
  --out-dir /Volumes/ModelSSD/logs/hy3-mlx-canary/parity-fixtures/<run>

./build/hy3-sidecar-io/hy3_sidecar_io \
  --index /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/compact-index.tsv \
  --root /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar \
  --fixture-list /Volumes/ModelSSD/logs/hy3-mlx-canary/parity-fixtures/<run>/fixtures.txt \
  --layer-major
```

Prefill4 artifact: `/Volumes/ModelSSD/logs/hy3-mlx-canary/parity-fixtures/20260707-140203-prefill4-all-layers/`.
The split C++ substrate replayed **79** fixtures with `seq_len=4`, read
**15.6226GiB** across **1,580** naïve expert spans, completed in **225.78s** after
instrumentation, and passed with worst relative-to-expected error `0.0141305`.
Layer-major replay of the same fixtures read each unique expert once per layer,
reducing to **1,408** reads / **13.9219GiB** and saving **172** reads /
**1.70068GiB**, with the same parity verdict. A duplicate-token regression
(`/tmp/hy3-layer1-top5-bos-dup2.json`) confirms the gate catches true prompt reuse:
`10` naïve reads collapse to `5` unique reads.

A 16-token all-layer fixture export also succeeds:
`/Volumes/ModelSSD/logs/hy3-mlx-canary/parity-fixtures/20260707-152258-prefill16-all-layers/`.
Python/MLX exported **79** fixtures with `seq_len=16`, sidecar read **29.752075GiB**,
forward-to-layer wall **97.636s**, and swap delta **0.0GiB**. Full C++ layer-major
replay passed all 79 layers with **3,009** unique reads vs **6,320** naïve route
reads, saving **3,311** reads / **32.7382GiB**; worst relative-to-expected error
was `0.0157926`. Scalar qlinear wall was **899.48s**, so the next compute cut is
SIMD/Metal, not more replay plumbing.
