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
  --route-exec
```

`--route-exec` is the routed-MoE fixture execution path: it requires the rich
parity fixture list because the older route TSV only has expert IDs, not the
hidden activations, route weights, or expected routed outputs needed for math.
It implies layer-major dense q4 execution, preserves token/top-k accumulation
order, rejects incompatible cache/IO flags, validates uniform shape plus strictly
increasing unique layers, and reports prompt-slice totals (`token_layer_events`,
`selected_routes`, unique reads/bytes, tolerance-parity). Use the TSV `--trace`
path only for cache/IO replay.

Prefill4 artifact: `/Volumes/ModelSSD/logs/hy3-mlx-canary/parity-fixtures/20260707-140203-prefill4-all-layers/`.
The split C++ substrate replayed **79** fixtures with `seq_len=4`, read
**15.6226GiB** across **1,580** naïve expert spans, completed in **225.78s** before
layer-major dedup, and passed with worst relative-to-expected error `0.0141305`.
Layer-major replay of the same fixtures read each unique expert once per layer,
reducing to **1,408** reads / **13.9219GiB** and saving **172** reads /
**1.70068GiB**, with the same parity verdict. With the Apple Accelerate-backed
qlinear path and expert-major dense reuse, the order-preserving layer-major replay
wall is **113.84s**. A duplicate-token regression (`/tmp/hy3-layer1-top5-bos-dup2.json`) confirms the
gate catches true prompt reuse: `10` naïve reads collapse to `5` unique reads.

A 16-token all-layer fixture export also succeeds:
`/Volumes/ModelSSD/logs/hy3-mlx-canary/parity-fixtures/20260707-152258-prefill16-all-layers/`.
Python/MLX exported **79** fixtures with `seq_len=16`, sidecar read **29.752075GiB**,
forward-to-layer wall **97.636s**, and swap delta **0.0GiB**. Full C++ layer-major
replay passed all 79 layers with **3,009** unique reads vs **6,320** naïve route
reads, saving **3,311** reads / **32.7382GiB**; worst relative-to-expected error
was `0.0157937`. Apple Accelerate qlinear plus expert-major dense reuse cuts the
order-preserving layer-major wall from **899.48s** to **239.45s**. `--route-exec`
reports the same routed-MoE math as a fixture-backed prompt-slice execution path;
next cuts are real prompt integration beyond captured fixtures and lower-level
Metal/SIMD kernels.

## Persistent routed-MoE daemon

`hy3_route_mlp_daemon` is the first online IPC substrate for calling the native
routed-MoE math from Python. It keeps the compact index and sidecar file handles
alive, speaks a local little-endian binary stdin/stdout protocol, and is wired
behind `HY3_CPP_ROUTE_MLP=1` in `hy_v3_mlx_lazy.py`. Optional dense expert reuse
is enabled with `HY3_CPP_ROUTE_DENSE_CACHE_GIB=<N>` / `--dense-cache-gib N`,
validated as finite `0 <= N <= 16`; one dense expert bank is ~0.070GiB, so the
documented `8` GiB setting can hold roughly 110 expert banks before LRU eviction.

Smoke commands:

```bash
HY3_CPP_ROUTE_MLP=1 /opt/homebrew/bin/python3.11 hy3_lazy_smoke.py forward-one \
  --slot-bank 8 --topk-cap 1 --profile-layers --sync-timers

HY3_CPP_ROUTE_MLP=1 HY3_CPP_ROUTE_DENSE_CACHE_GIB=8 \
  /opt/homebrew/bin/python3.11 hy3_lazy_smoke.py generate-cache \
  --slot-bank 8 --topk-cap 1 --profile-layers --sync-timers \
  --prompt 'Reply with exactly pong.' --max-new-tokens 2
```

Verdict from `/Volumes/ModelSSD/logs/hy3-mlx-canary/cpp-route-daemon/20260707/`
and `cpp/results/20260707-route-daemon-online.json`: the daemon path is correct
and useful as an IPC/correctness substrate, but **not yet a speed win**. Top-k1
`forward-one` matched the MLX next token (`1655`) but took **9.53s** vs **5.87s**
for MLX. Top-k1 two-token `generate-cache` with an 8GiB dense cache took
**[23.17s, 10.49s]** vs MLX **[11.03s, 5.13s]**. Dense cache works — repeated
fixture calls drop from 5 reads to 0 reads and ~176ms to ~12ms — but online CPU
dense SGEMV still loses to MLX `gather_qmm`. A follow-up spike at
`spikes/001-metal-q4-matvec/` found that a simple custom MLX Metal q4 matvec is
feasible but only parity/noisy against `gather_qmm`. A later route-batched SGEMM
spike is recorded at `cpp/results/20260707-route-batched-sgemm-spike.json`: it
kept parity but regressed the seq16 prefill fixture from **118.617s** to
**126.959s** (+7.0%), so do not replace repeated SGEMV with naïve tiny-batch
SGEMM. Next serious speed work needs fused/on-the-fly q4 Metal or a kernel that
avoids dense FP32 materialization and small-M SGEMM overhead.

The next Metal spike at `../spikes/002-metal-fused-up-gate/` partially validates
that direction, but still does not produce a runtime cut. Fused q4
`up_proj + gate_proj + swiglu` is directionally real but modest/noisy:
**1.0658x** median speedup vs `gather_qmm(up/gate)+swiglu`, with one stable
sample regressing to **0.9138x**. The best full expert proxy was fused up+gate
plus normal `gather_qmm(down)`: **1.1245x** median / **1.1726x** mean. Replacing
down with a custom Metal row-reduce projection did **not** help; all-Metal full
expert reached only **1.0505x** median and was slower than keeping MLX down.
Treat this as a direction proof, not a runtime cut; the next kernel needs to
fuse down/route weighting into the same launch or batch many routes without
dense FP32 materialization.

A follow-up route-batched Metal spike at
`../spikes/003-metal-route-batched-up-gate/` tested the real routed shape for
six fixtures (seq4/seq16 × layers 1/40/79): one Metal launch emits weighted
`swiglu(gate_q4(x), up_q4(x))` for all selected routes, then MLX handles down.
Preweighting routes before down was basically neutral under MLX (**0.9949x**
median vs standard), but the custom row-reduce route kernel was much slower:
weighted hidden **0.4708x** median vs `gather_qmm`, full route proxy **0.5227x**
median vs standard. Do not build one-threadgroup-per-route/output-row kernels;
next serious path needs tiled/simdgroup q4 dot, MPS-backed batch matmul, or a
deeper fused route kernel.

`../spikes/004-cpp-q4-direct-dot/` tested a CPU cold-route alternative using the
q4 affine identity `dot(x, q * scale + bias) = scale * sum(q*x) + bias * sum(x)`
so the runtime can skip dense FP32 materialization. On six real seq4/seq16
fixtures (layers 1/40/79), the NEON direct-dot path is **2.6212x** median /
**2.9841x** mean faster than dense-dequant + Accelerate when dequantization is
included, while matching dense output with `max_rel_to_dense < 5e-7`. It is
still only **0.4062x** median vs dense compute once dense weights are already
hot. Verdict: validated for cold/prefill-shaped routes, not a universal
replacement.

Runtime q4 modes now exist in the C++ routed-MoE path via
`--q4-mode dense|direct|hybrid`, and the Python daemon hook accepts
`HY3_CPP_ROUTE_Q4_MODE`. Default remains `dense` for compatibility. `direct`
uses the q4 affine dot path without dense FP32 materialization; `hybrid` uses
`direct` for cold/low-reuse experts and dense+Accelerate once an expert appears
in at least 8 routes in a fixture/request, and the daemon now also routes
already-cached dense experts through the dense path. Artifact:
`cpp/results/20260708-hybrid-q4-summary.json`.

| Shape | Dense | Direct | Hybrid | Verdict |
|---|---:|---:|---:|---|
| six-fixture spread | 2.085s | 1.182s | **1.131s** | all pass parity |
| prefill4 route-exec, 79 layers | 18.573s | 10.178s | **9.405s** / **1.98x** | all pass parity |
| prefill16 route-exec, 79 layers | 42.078s | 30.760s | **29.559s** / **1.42x** | all pass parity |
| online `forward-one`, top-k1 C++ route hook | 1.450s | 0.902s | **0.895s** | next token `1655` |
| online `forward-one`, top-k5 C++ route hook | 6.678s | 4.115s | **4.082s** / **1.64x** | next token `2` |
| online `forward-one`, top-k8 C++ route hook | 9.517s | **5.727s** | 5.736s / **1.66x** | next token `1655` |

This is the first C++/Capstan Hy3 compute cut that is both runtime-shaped and a
real speed win. Online top-k5/top-k8 one-token canaries confirm the cold-route
benefit at less-toy fanout; artifact:
`results/20260708-cpp-route-q4-online-topk-summary.json`.

Multi-token `generate-cache` with a 30-token prompt, 4 generated tokens,
`HY3_CPP_ROUTE_DENSE_CACHE_GIB=16`, and the cache-aware daemon hybrid confirmed
that the speed win survives decode-shaped execution, but token drift blocks
promotion to default. Artifact:
`results/20260708-cpp-route-q4-generate-cache-4tok-cache16-combined-summary.json`.

| Generate-cache shape | Dense | Direct | Hybrid | Token verdict |
|---|---:|---:|---:|---|
| top-k5, 4 tokens | 84.915s | 54.382s / **1.56x** | **52.600s / 1.61x** | direct + hybrid drift |
| top-k8, 4 tokens | 120.941s | 91.766s / **1.32x** | **82.670s / 1.46x** | direct matched, hybrid drift |

Swap delta stayed **0.0GiB**. A forced-token logits harness now exists as
`hy3_lazy_smoke.py forced-logits`; it writes per-step logits to `.npz` so q4
modes can be compared on an identical dense-token sequence instead of drifting
through different prompts. Artifact:
`results/20260708-cpp-route-q4-forced-logits-delta-summary.json`.

| Forced-logits shape | Direct verdict | Hybrid verdict |
|---|---|---|
| top-k5, dense sequence | step1 top-2 swap; dense top1 ranked #2 under direct, margin 0.625 | step0 top-2 swap; dense top1 ranked #2 under hybrid, margin 0.375 |
| top-k8, dense sequence | **all 4 steps same top1**; max RMSE 0.388 | step0 top-2 swap; dense top1 ranked #2 under hybrid, margin 0.125 |

Interpretation: the observed token drift is mostly knife-edge top-2 flips, not
semantic collapse, but the broader logit deltas are nontrivial. Dense remains
the default. `direct` top-k8 is the cleanest speed candidate; `hybrid` stays an
experimental cold/hot speed lane until mixed dense/direct numerics are tightened
or an explicit logit-tolerance policy is accepted.

The persistent daemon now also exposes an opt-in byte-bounded packed
`ExpertBank` LRU through `--packed-cache-gib` and Python
`HY3_CPP_ROUTE_PACKED_CACHE_GIB`. Cache entries own stable moved buffers and
rebind all tensor slices after moves; a focused CTest covers ownership, LRU,
byte ceilings, disabled mode, and oversized entries. Dense plus packed cache
budgets are jointly capped at 16GiB, and packed cache remains off by default.
`hy3_route_mlp_daemon_smoke.py` provides the local sidecar-backed repeated-request
gate and records output hashes plus read/hit telemetry.
The Python singleton serializes each full request/response exchange, protects
singleton lifecycle with a reentrant lock, and resets that lock after `fork()`;
children discard inherited clients by PID and spawn their own daemon. Cache
budgets are forwarded with round-trip-safe precision, while the C++ parser
rejects trailing junk instead of accepting prefixes such as `1junk`. Stress
artifact: `results/20260709-cpp-route-client-hardening-smoke.json`.

Top-k8 direct, 30-token prompt, four generated tokens, 16GiB packed cache:

| Shape | Step sum | Decode median | C++ reads | Packed hits | Tokens | Swap delta |
|---|---:|---:|---:|---:|---|---:|
| direct, cache off | 91.766s | 3.944s | 82.048GiB | 0 | `[185, 120029, 72520, 423]` | 0.0GiB |
| direct, global packed LRU | **82.298s** | **2.994s** | **75.493GiB** | 663 | same | 0.0GiB |

That is **10.3%** lower total step wall and **24.1%** lower median decode wall.
Aggregate read reduction is only **8.0%**, below the initial 15% estimate,
because prefill routing has limited repeated-bank overlap. A layer-sharded LRU
produced 659 hits / 75.532GiB and no material improvement, so the runtime keeps
the simpler global LRU. Artifact:
`results/20260709-cpp-route-packed-cache-summary.json`; repeated-request proof:
`results/20260709-cpp-route-packed-cache-repeat-smoke.json`.
