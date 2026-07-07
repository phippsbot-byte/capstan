# Hy3 lazy SSD sidecar prototype status

Date: 2026-07-06

## Verdict

Replicated the DeepSeek V4 Flash architecture at prototype level for Hy3 preview:

- stock/flat MLX path: **dead** — monolithic q4 hit ~94.7GiB swap before first token
- lazy sidecar path: **works** — resident core loads without expert tensors, routed experts stream from SSD by selected expert id, and the model generated `pong` + EOS through a KV-cache smoke

This is not a production server yet. It is the first real proof that the Hy3 tensor layout can be operated like the DS4/Capstan lane.

## Artifacts

| Artifact | Path |
|---|---|
| Lazy model module | `/Users/nb/LLM/hy3-mlx-canary/hy_v3_mlx_lazy.py` |
| Guarded smoke runner | `/Users/nb/LLM/hy3-mlx-canary/hy3_lazy_smoke.py` |
| Metadata layout builder | `/Users/nb/LLM/hy3-mlx-canary/hy3_sidecar_layout.py` |
| Sidecar layout JSON | `/Users/nb/LLM/hy3-mlx-canary/hy3-sidecar-layout.json` |
| Layout summary | `/Users/nb/LLM/hy3-mlx-canary/hy3-sidecar-layout-summary.md` |

## Verified split

| Bucket | Size |
|---|---:|
| Total MLX payload | 154.589 GiB |
| Resident non-expert core | 4.612 GiB |
| Routed expert sidecar | 149.977 GiB |
| Expert share | 97.02% |

Derived pressure:

| Metric | Value |
|---|---:|
| Expert layers | 79 |
| Experts/layer | 192 |
| Native top-k | 8 |
| Per-expert payload | 10.125 MiB |
| Cold active read/token | 6.249 GiB |

## Smoke results

### 1. Expert-sidecar qmm read

Command shape:

```bash
/opt/homebrew/bin/python3.11 /Users/nb/LLM/hy3-mlx-canary/hy3_lazy_smoke.py \
  expert-read --slot-bank 8 --layer 1 --experts 8
```

Result artifact:
`/Volumes/ModelSSD/logs/hy3-mlx-canary/lazy-expert-read-v2.json`

Result:

- loaded 8 selected experts from safetensor byte offsets
- executed MLX `gather_qmm` for up/gate/down
- elapsed: 0.556s
- swap delta: 0.0GiB

### 2. Resident-only model load

Result artifact:
`/Volumes/ModelSSD/logs/hy3-mlx-canary/lazy-resident-load.json`

Result:

- resident tensors loaded: 2323
- expert parameter leaks: none
- elapsed: 5.627s
- swap delta: 0.0GiB

### 3. Full one-token forward

Result artifact:
`/Volumes/ModelSSD/logs/hy3-mlx-canary/lazy-forward-one-slot8.json`

Result:

- token input: BOS (`120000`)
- logits shape: `[1, 1, 120832]`
- sidecar loads: 632 experts = 79 layers × top-k 8
- cached experts: 632
- load: 4.270s
- forward: 38.474s
- swap delta: 0.0GiB

### 4. KV-cache generation smoke

Prompt used:

```text
<｜hy_begin▁of▁sentence｜><｜hy_User｜>Reply with exactly pong.<｜hy_Assistant｜><think></think>
```

Result artifact:
`/Volumes/ModelSSD/logs/hy3-mlx-canary/lazy-generate-cache-chatlite-4-slot8.json`

Result:

- generated IDs: `[79, 792, 120025]`
- decoded raw: `pong<｜hy_eos｜>`
- clean visible answer: `pong`
- prompt tokens: 11
- step timings: 236.548s, 75.722s, 51.385s
- sidecar loads: 4815
- cache hits: 156
- evictions: 4183
- cached experts: 632
- slot bank: 8
- swap delta during run: 14.055GiB

Interpretation: the runtime path is functionally correct enough to emit the expected answer, but slow and still causes swap during multi-token prompt prefill. The one-token forward staying at 0.0GiB swap proves the architecture; the prompt-prefill path needs DeepSeek-style layer-major dedup / staged prefetch before it is usable.

## Current known limitations

1. Python prototype, not server-grade.
2. Uses safetensors raw byte reads per selected expert; no packed contiguous sidecar yet.
3. Per-layer prompt prefill is still ubatch/token-major and reload-heavy.
4. Slot-bank is a Python LRU of MLX arrays; no secondary sidecar / split physical IO.
5. No OpenAI-compatible server wrapper yet.
6. MTP is not implemented.
7. Tool parser/template integration is not implemented.

## Packed layer-major sidecar result

Implemented the first real packed sidecar:

| Artifact | Path |
|---|---|
| Packer | `/Users/nb/LLM/hy3-mlx-canary/hy3_pack_sidecar.py` |
| Packed sidecar | `/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar` |
| Manifest | `/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/manifest.json` |
| Pack log | `/Volumes/ModelSSD/logs/hy3-mlx-canary/pack-full-sidecar.log` |

Result:

- schema: `hy3-packed-sidecar-v1`
- layout: layer-major / expert-major
- layers: 79
- entries: 136,512
- bytes: 161,036,107,776 / 149.976562 GiB
- on-disk size: 151G
- pack time: 94.45s
- byte verification: 512 sampled payloads, 0 mismatches

Updated `hy_v3_mlx_lazy.py` so `Hy3SidecarStore` supports both original safetensor-offset manifests and packed sidecar manifests via `HY3_SIDECAR_LAYOUT`.

Packed smoke results:

| Smoke | Result |
|---|---:|
| packed expert-read, layer 1, 8 experts | ok, 0.850s, 0.0GiB swap delta |
| packed full one-token forward, BOS | ok, logits `[1,1,120832]`, 72.515s forward, 0.0GiB swap delta |

Forward artifact: `/Volumes/ModelSSD/logs/hy3-mlx-canary/packed-full-forward-one-slot8.json`.

Follow-up optimization/instrumentation:

- changed packed expert load from 9 tensor reads/expert to 1 contiguous `pread`/expert, with fd reuse
- added sidecar stats: read calls/bytes, unique expert request counts, load/pack/remap/qmm timers
- packed expert-read after instrumentation: ok, 8 read calls, 0.079102 GiB read, ~0.95-1.04s wall, 0.0GiB swap delta
- packed one-token forward after instrumentation: ok, 632 read calls, 6.249023 GiB read, 80.156s forward, 3.268GiB swap delta
- sidecar internal time for that forward: load 3.362s, pack 0.411s, remap 0.545s, qmm 0.101s

Important read: packed sidecar IO is **not** the main one-token bottleneck anymore. The forward is burning time outside sidecar reads — likely MLX lazy scheduling / resident dense/shared modules / forced eval placement. The 1-read packed loader is cleaner for the real Capstan path, but not a Python performance win by itself. Next useful move is component-level layer profiling, then either remove bad forced evals or move the hot path out of Python.

Layer-profile result:

- added gated `--profile-layers` / `HY3_PROFILE_LAYERS=1` mode to force per-layer evals and split attention vs MLP timing
- artifact: `/Volumes/ModelSSD/logs/hy3-mlx-canary/packed-forward-one-profiled-slot8.json`
- profiled packed one-token forward: ok, 81.727s forward, 80/80 layers profiled, 3.520GiB swap delta
- total attention time: 1.399s
- total MLP/MoE block time: 79.466s
- residual/eval glue: 0.019s
- sidecar stats during the same run: 632 packed reads, 6.249023 GiB read, load 3.130s, pack 0.242s, remap 0.123s, qmm timer 0.085s

Verdict: the Python sidecar is no longer the scary part for single-token decode. Hy3-preview is spending essentially all wall time in the routed expert path, not the shared MLP. Next target is a routed-expert kernel diagnosis: `gather_qmm`/routing/weighted-sum fusion and avoiding Python/MLX per-layer per-token overhead.

Shared/routed bypass result:

- added `HY3_DISABLE_SHARED_MLP` / `--disable-shared-mlp` diagnostic flag
- added `HY3_DISABLE_ROUTED_MLP` / `--disable-routed-mlp` diagnostic flag
- added `HY3_SYNC_TIMERS` / `--sync-timers` for synchronized timing probes
- shared disabled, routed enabled artifact: `/Volumes/ModelSSD/logs/hy3-mlx-canary/packed-forward-one-profiled-sync-no-shared-mlp-slot8.json`
  - forward: 70.436s
  - MLP/MoE total: 70.298s
  - sidecar: 632 reads, 6.249023 GiB, load 2.413s, swap delta 0.0GiB
  - verdict: shared MLP is not the main bottleneck
- routed disabled, shared enabled artifact: `/Volumes/ModelSSD/logs/hy3-mlx-canary/packed-forward-one-profiled-sync-no-routed-mlp-slot8.json`
  - forward: 0.181s
  - attention total: 0.091s
  - MLP/shared total: 0.062s
  - sidecar: 0 reads
  - verdict: routed expert path accounts for essentially all of the 70-80s/token Python runtime

Corrected interpretation: previous “shared MLP is likely villain” was wrong. The sidecar I/O itself is also not the wall-clock villain. The slow path is the routed expert compute/integration path around `LazySwitchGLU` + top-k weighting. Next useful probe is to isolate `gather_qmm` and the post-qmm weighting/sum with synthetic in-memory expert banks before spending more time on prefill/server polish.

Routed microbench / GC fix result:

- added microbench: `/Users/nb/LLM/hy3-mlx-canary/hy3_routed_microbench.py`
- real packed layer-1/k=8 microbench artifact: `/Volumes/ModelSSD/logs/hy3-mlx-canary/routed-microbench-real-random-layer1-k8.json`
- synthetic in-memory k=8 microbench artifact: `/Volumes/ModelSSD/logs/hy3-mlx-canary/routed-microbench-synthetic-k8.json`
- isolated `gather_qmm` + swiglu + weighting/sum is **not** slow once the expert bank is resident: real packed separated total ~0.0012s, fused expression ~0.0004s for one layer/k=8
- MoE internal timing exposed the actual problem: `LazySwitchGLU` wall time was dominated by `gc.collect()` called after every expert load in `Hy3SidecarStore.get_experts`
- fixed hot path: removed unconditional per-expert `gc.collect()`; GC is now gated to evictions only via `HY3_GC_ON_EVICT`, with `gc_calls`/`gc_time_s` stats
- post-fix full packed one-token artifact: `/Volumes/ModelSSD/logs/hy3-mlx-canary/packed-forward-one-gcfix-sync-full-slot8.json`
  - forward: **3.473s** (down from ~70-82s)
  - attention: 0.124s
  - MLP/MoE: 3.282s
  - sidecar load: 2.430s
  - qmm timer: 0.069s
  - reads: 632 / 6.249023 GiB
  - swap delta: 0.0GiB
- post-fix packed KV-cache pong artifact: `/Volumes/ModelSSD/logs/hy3-mlx-canary/packed-generate-cache-chatlite-gcfix-slot8.json`
  - generated clean `pong`
  - step timings: **17.509s / 3.478s / 2.097s**
  - exact_pong: true
  - reads: 4815 / 47.609253 GiB
  - evictions: 4183
  - gc_calls: 0
  - swap delta: 0.0GiB

Corrected final read: the Python prototype was being murdered by explicit Python GC, not by `gather_qmm`, not by weighting/sum, not by shared MLP, and not by SSD bandwidth. Next useful optimization is reducing sidecar loads/reads for prefill via layer-major expert reuse/prefetch and larger/ smarter slot policy, because post-fix single-token decode is now in the ~3.5s range instead of ~80s.

Slot-bank sweep result:

| Slot bank | Result |
|---:|---|
| 8 | exact `pong`; timings **17.509s / 3.478s / 2.097s**; reads **47.609GiB**; evictions **4183**; swap delta **0.0GiB** |
| 16 | exact `pong`; timings **15.439s / 1.779s / 1.162s**; reads **45.592GiB**; evictions **3347**; swap delta **0.0GiB** |
| 18 | killed by swap guard; swap climbed **3.135 → 12.261GiB** with 8GiB delta gate |
| 20 | killed by swap guard; swap climbed **3.173 → 15.396GiB** with 12GiB delta gate |
| 24 | killed by swap guard; swap climbed **2.774 → 19.305GiB** with 16GiB delta gate |
| 32 | killed by swap guard; swap climbed **1.962 → 26.069GiB** with 24GiB delta gate |

Operational read: slot-bank **16** is the current safe ceiling on the 96GB Studio. It halves decode-ish steps vs slot 8 and stays flat on swap. Slot 18+ causes memory pressure/swap fast; do not use 24/32 without a smarter cache/offload strategy. Next target is layer-major prefill dedup/reuse at slot 16, not brute-force bigger caches.

Frequency-retention cache policy:

Changed `LazySwitchGLU._remap_indices` to load least-frequent prompt experts first so the bounded per-layer LRU retains the most reused experts after prefill. `HY3_RETAIN_FREQUENT_EXPERTS=0` restores old expert-id ordering.

Slot-bank 16 with frequency retention:
- exact `pong`: true
- timings: **14.757s / 1.072s / 1.141s**
- loads: **4412** (down from 4611)
- hits: **559** (up from 360)
- evictions: **3148** (down from 3347)
- reads: **43.625GiB** (down from 45.592GiB)
- swap delta: **0.0GiB**
- artifact: `/Volumes/ModelSSD/logs/hy3-mlx-canary/packed-generate-cache-chatlite-gcfix-slot16-freqretain.json`

This is a cheap win and keeps slot 16 as the default safe **full top-8** policy. Next optimization should target prefill read volume directly, but cache retention order now stops throwing away the most useful prompt experts.

Experimental top-k cap / fast lane:

Added optional `HY3_TOPK_CAP` / `--topk-cap` for approximate local serving. This is **not** fair full-Hy3 behavior; it is a speed/quality tradeoff for making the 295B route usable on the 96GB Studio.

| Lane | Result |
|---|---|
| top-4, slot 16, `pong` | exact `pong`; **5.512s / 0.684s / 0.486s**; reads **22.781GiB**; swap delta **0.0GiB** |
| top-4, slot 32, `pong` | exact `pong`; **4.287s / 0.387s / 0.405s**; reads **22.050GiB**; swap delta **0.0GiB** |
| top-4, slot 32, exact JSON | exact `{\"ok\":true}`; **7.500s** prefill then **0.25-0.45s/token**; reads **28.991GiB**; swap delta **0.0GiB** |
| top-4, slot 32, tool-shaped JSON | exact `{\"tool\":\"calculator\",\"arguments\":{\"expression\":\"17*23\"}}`; **13.643s** prefill then ~**0.5-0.7s/token**; reads **59.692GiB**; swap delta **0.0GiB** |

Added tiny OpenAI-compatible canary server: `/Users/nb/LLM/hy3-mlx-canary/hy3_openai_server.py`. It serves `/v1/models`, `/health`, and `/v1/chat/completions`, renders the real Hy3 chat template with `reasoning_effort=no_think`, and parses Hy3 `<tool_calls>` into OpenAI-style `tool_calls`. Smoke on port `8133` passed exact `pong`, exact JSON, and a calculator tool-call request. The tool-template path is still expensive because the template expands to ~196 prompt tokens; that is a serving/runtime cost issue, not a model-format blocker.

## Next engineering step

Do **not** rerun flat MLX.

Next useful work:

1. Keep two lanes distinct: **fair full top-8** (`slot-bank 16`) vs **fast approximate top-4** (`slot-bank 32`). Do not mix their scores.
2. For re-eval, start with a tiny local slice through `hy3_openai_server.py` before any full Phipps run; the server path now works but prompt-prefill IO is still expensive.
3. Add prompt/prefix cache reuse for repeated eval scaffolding; tool-template prompts currently explode prefill cost.
4. Profile packed reader vs original reader at the per-expert and per-layer level; the Python hot path is now less scary, but prefill IO remains ugly.
5. Only after tiny slice quality is sane, run a signed-off Phipps eval and label the lane clearly (`top8 fair` vs `top4 approximate`).

## DS4 lane cleanup

DeepSeek V4 Flash SSD lane was stopped before Hy3 runs and restored after each heavy test.
Post-restore health passed on `127.0.0.1:8127` with exact JSON smoke and swap delta OK.
