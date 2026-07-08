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
2. Per-layer prompt prefill is still ubatch/token-major and reload-heavy.
3. Slot-bank is a Python LRU of MLX arrays; no secondary sidecar / split physical IO.
4. MTP is not implemented.
5. Current top-4 approximation is quality-damaged on Phipps prompts; use it as plumbing only.
6. Server-side cancellation is still cooperative: request clamps/timeouts stop decode and report failures, but a long MLX prefill can only be observed/rejected after it returns.
7. The local artifact is Hy3-preview 4bit, not the current cloud-scored Hy3 release.

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

Server hardening follow-up:
- `hy3_openai_server.py` now uses a threaded HTTP server plus a single-generation lock so health/model probes can respond while a generation is active and concurrent generations get a clear `409 busy` instead of corrupting the runtime.
- request clamps added: `--hard-max-tokens`, `--max-prompt-tokens`, and `--request-timeout-sec`.
- timeout is cooperative: decode can stop cleanly, but a long MLX prefill can only be rejected after it returns.
- added optimizer sweep runner: `/Users/nb/LLM/hy3-mlx-canary/hy3_local_optimizer.py` for top-k/slot-bank smoke matrices with optional DS4 stop/restore.
- code/metadata were prepared for GitHub as `phippsbot-byte/hy3-mlx-canary`; weights, packed sidecars, logs, and credentials stay out of git.

## Optimizer sweep result — 2026-07-07

Run artifact committed under `optimizer-runs/20260707-063805/`.

| Lane | Pong | JSON | Tool | PE-short | Swap verdict |
|---|---:|---:|---:|---:|---|
| top-4 / slot32 | 13.88s | 8.60s | 144.90s | 87.69s | passed guard but max swap **17.739GiB** / 18GiB delta |
| top-5 / slot24 | 15.41s | 11.51s | 141.97s | failed disconnect | tripped guard: **21.110GiB** used |
| top-6 / slot24 | 17.11s | 20.86s | 147.61s | failed disconnect | tripped guard: **20.094GiB** used |
| top-8 / slot16 | 21.70s | 28.71s | failed disconnect | skipped after guard | tripped guard: **20.232GiB** used |

Interpretation:
- top-4/slot32 is the only lane that completed all four probes under the configured guard, but tool/PE prompts are still painfully slow and it runs right up to the swap cliff.
- top-5/top-6 at slot24 are not safe on the current cache policy; bigger top-k with large slot-bank explodes pressure before the PE probe.
- top-8/slot16 remains too expensive for tool-template prompts under this runner; it is a fairness reference only, not a usable serving lane yet.
- DS4 restore passed after the run with exact JSON smoke.

## Next engineering step

Do **not** rerun flat MLX.

Next useful work:

1. Implement real prefix/KV cache reuse in `hy3_openai_server.py` for repeated system/harness/tool-template scaffolding. Tool-template prefill is the current obvious tax.
2. Add prompt-length/token accounting to optimizer output and split tool-template vs no-tool prompts so we stop mixing model speed with template bloat.
3. Run a safer second sweep: `top4/24`, `top4/28`, `top5/16`, `top5/20`, `top6/16` with a tighter max-token cap and lower swap delta.
4. Only if top5/top6 at smaller slots survive, run a 2-4 test Phipps slice. Otherwise, keep Hy3 local as R&D and put serious work into C++/Capstan-style layer-major prefill.

## Prefix-cache experiment — 2026-07-07

Added tool-scaffold prefix/KV cache reuse in `hy3_openai_server.py` and prompt-token/prefix-cache telemetry in `hy3_local_optimizer.py`.

Validation run: `optimizer-runs/20260707-071352/`, lane `top4/slot32`, prompts `tool,tool-alt`.

| Probe | Prompt tokens | Prefill tokens after split | Prefix | Wall |
|---|---:|---:|---|---:|
| tool `17*23` | 197 | 12 | build, 185-token prefix cached | 90.262s |
| tool-alt `19*29` | 197 | 12 | cache hit | 30.005s |

Both returned correct OpenAI-style calculator tool calls. Max swap during this run was **13.167GiB** and DS4 restore/smoke passed afterward. This proves the repeated-tool-schema tax is real and cacheable; first request still pays the 185-token prefix build, but repeated tool calls are ~3x faster on this top4/slot32 lane.

Next useful work now:

1. Pre-warm common tool/system scaffolds before eval runs so the first measured request does not eat prefix-build cost.
2. Run the smaller-slot sweep with prefix cache active: `top4/24`, `top4/28`, `top5/16`, `top5/20`, `top6/16` over `pong,json,tool,tool-alt,pe-short`.
3. If top5/top6 survives with flat swap, run a tiny Phipps slice. If not, stop optimizing Python UX and move hot prefill to Capstan/C++.

## Expert-cache clear result — 2026-07-07

Smaller-slot prefix sweep without clearing proved the cumulative sidecar expert cache was the memory-pressure villain: every tested lane tripped the 16GiB swap-delta guard even with prefix cache active.

Added `Hy3SidecarStore.clear_cached_experts()`, server flag `--clear-expert-cache-after-request`, and optimizer flag of the same name. This keeps the reusable KV prefix cache but drops sidecar expert tensors after every warm/generation request.

Validation run: `optimizer-runs/20260707-073907/`, lane `top4/slot24`, with prefix prewarm and expert-cache clear.

| Probe | Wall | Prompt toks | Prefill toks | Prefix hit | Result |
|---|---:|---:|---:|---|---|
| pong | 13.245s | 21 | 21 | - | exact `pong` |
| JSON | 10.092s | 25 | 25 | - | exact `{\"ok\":true}` |
| tool `17*23` | 15.164s | 197 | 12 | yes | correct tool call |
| tool-alt `19*29` | 12.718s | 197 | 12 | yes | correct tool call |
| PE-short | 42.252s | 74 | 74 | - | completed, length-capped |

Swap stayed flat: start **2.620GiB**, max **4.530GiB**, end **2.799GiB**. DS4 restore/smoke passed.

Interpretation: for Python Hy3 serving on the 96GB Studio, persistent expert reuse is less valuable than avoiding cumulative memory pressure. Current best operator mode is **top4/slot24 + prefix prewarm + clear expert cache after request**. It is slower than real serving should be, but it is stable enough for more probing.

Next useful work now:

1. Run `top5/16` and `top6/16` with prefix prewarm + expert-cache clear.
2. If either survives, compare quality on the tiny Phipps slice.
3. If neither survives, use top4/slot24 clear mode as the stable canary and move performance work into Capstan/C++.

## Top5/Top6 clear-mode sweep — 2026-07-07

Validation run: `optimizer-runs/20260707-074244/`, with prefix prewarm + expert-cache clear.

| Lane | Pong | JSON | Tool | Tool-alt | PE-short | Swap verdict |
|---|---:|---:|---:|---:|---:|---|
| top5 / slot16 | 14.878s | 12.058s | 18.593s | 14.969s | 54.505s | flat: **2.799 → 2.843GiB**, max **2.843GiB** |
| top6 / slot16 | 18.994s | 15.618s | 23.256s | 17.802s | 73.176s | safe: **2.843 → 2.955GiB**, max **10.527GiB** |

Both lanes returned exact `pong`, exact JSON, correct calculator tool calls, and completed the PE-short probe under the 48-token cap. DS4 restore/smoke passed afterward.

New best candidates:
- **top5/slot16 + prefix prewarm + clear expert cache**: best speed/quality tradeoff candidate.
- **top6/slot16 + prefix prewarm + clear expert cache**: slower but closer to full Hy3 routing; still safe.

Next useful work now:

1. Run a tiny Phipps slice on top5/slot16 clear mode first.
2. If top5 quality is meaningfully better than top4 and latency is tolerable, run the same tiny slice on top6.
3. If neither improves quality enough, freeze Python lane as a stable canary and move the speed work to Capstan/C++.

## Phipps tiny slice — top5/slot16 clear — 2026-07-07

Run: `/Volumes/ModelSSD/logs/hy3-mlx-canary/phipps-slice-results/phipps-eval-v3-20260707-080711.json`
Report: `/Volumes/ModelSSD/logs/hy3-mlx-canary/hy3-phipps-top5-slot16-clear-20260707-top5clear.md`

Config: `top5/slot16`, prefix cache enabled, expert cache cleared after request, 48-token cap.

| Metric | Result |
|---|---:|
| composite | **2.32** |
| avg latency | **94,598 ms** |
| avg output tok/s | **0.6** |

| Test | Score | Gate |
|---|---:|---|
| PE math | 1.00 | fail |
| Conv brevity | 2.22 | fail |
| Tool restraint | 4.08 | pass |
| Tool calculator | 1.12 | fail |
| Voice pushback | 4.02 | pass |
| Edge no hallucinate | 3.62 | pass |
| Edge structured JSON | 1.00 | fail |
| IFEval format | 2.00 | pass-ish |

Compared with the earlier top4/slot32 approximate slice: composite improved only **2.18 → 2.32**, but latency improved about **194.6s → 94.6s** average because clear-mode stops cumulative pressure. Quality is still not acceptable for Phipps; this is a stable runtime canary, not a competitive lane.

Next useful work now:

1. Run the same tiny slice on top6/slot16 clear mode only if we want to check whether one more routed expert fixes quality.
2. Otherwise freeze top5/slot16 clear as the Python operator lane and move performance/quality work into Capstan/C++.

## Phipps tiny slice — top6/slot16 clear — 2026-07-07

Run: `/Volumes/ModelSSD/logs/hy3-mlx-canary/phipps-slice-results/phipps-eval-v3-20260707-082649.json`
Report: `/Volumes/ModelSSD/logs/hy3-mlx-canary/hy3-phipps-top6-slot16-clear-20260707-top6clear.md`

Config: `top6/slot16`, prefix cache enabled, expert cache cleared after request, 48-token cap.

| Metric | Top5 clear | Top6 clear | Verdict |
|---|---:|---:|---|
| composite | **2.32** | **2.23** | top5 wins |
| avg latency | **94,598 ms** | **113,359 ms** | top5 wins |
| avg output tok/s | **0.6** | **0.5** | top5 wins |

Top6 improved only `tool_calculator` and `ifeval_format`, but regressed hallucination, brevity, voice, and overall composite. It is also slower. One more routed expert does **not** fix the local approximate quality problem.

Decision: freeze **top5/slot16 + prefix prewarm + clear expert cache** as the stable Python canary/operator lane. Do not spend more time tuning Python slot/top-k unless we need a very specific smoke. The real path is Capstan/C++ layer-major prefill/decode with the same guardrails.

## Capstan/C++ sidecar IO substrate — 2026-07-07

First native substrate landed under `cpp/` plus compact index emitter `hy3_emit_compact_index.py`. This is not a model server yet; it proves Capstan/C++ can consume the packed Hy3 sidecar with a tiny TSV index and issue one contiguous `pread` per `(layer, expert)` span.

Files:
- `hy3_emit_compact_index.py` → emits `/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/compact-index.tsv` from the 50MiB JSON manifest.
- `cpp/hy3_sidecar_io.cpp` → C++20 IO smoke/benchmark.
- `cpp/results/20260707-sidecar-io.json` → benchmark artifact.

Verified on the Studio with Apple clang 21:

| Probe | Planned spans | Read calls | Payload | Read wall | Throughput | Checksum |
|---|---:|---:|---:|---:|---:|---|
| plan full top8, no read | 632 | 0 | 0 GiB actual / 6.249 GiB planned | 0s | - | `0x14650fb0739d0383` |
| layer1 top8 read | 8 | 8 | 0.079 GiB | 0.028s | 2.78 GiB/s | `0x71ee6d2d6ebd0576` |
| all MoE layers top8 read | 632 | 632 | 6.249 GiB | 3.118s | 2.00 GiB/s | `0x1e173ad573437fa4` |

Notes: filesystem was warm from prior Hy3 work, so this is not a cold-drive worst-case number. Still, the substrate cost is now clearly below Python/MLX generation wall time.

Native slot-bank/cache scheduling is now implemented in the same C++ substrate via `--simulate-tokens`, `--slot-bank`, `--policy`, and deterministic route patterns (`fixed`, `hot`, `rolling`). Updated artifact: `cpp/results/20260707-sidecar-io.json`.

| Cache sim | Hits | Misses | Evictions | Payload read | Read wall | Final cache |
|---|---:|---:|---:|---:|---:|---:|
| top8 fixed, 4 tokens, slot16 | 1,896 | 632 | 0 | 6.249GiB | 2.406s | 6.249GiB |
| top5 hot, 8 tokens, slot16 | 2,370 | 790 | 0 | 7.811GiB | 3.683s | 7.811GiB |
| top5 rolling, 8 tokens, slot16 | 0 | 3,160 | 1,896 | 31.245GiB | 13.029s | 12.498GiB |

Interpretation: native scheduling matches the Python lesson. If routing has locality, slot-bank 16 is enough to keep repeated expert sets hot; if routing churns adversarially, IO explodes even before compute.

Real router trace capture/replay is now implemented:

- Python trace source: `hy3_lazy_smoke.py --route-trace-out`, backed by `HY3_TRACE_ROUTES=1` in `hy_v3_mlx_lazy.py`.
- C++ replay: `hy3_sidecar_io --trace <route.tsv> --slot-bank 16 --policy freq`.
- Artifact: `/Volumes/ModelSSD/logs/hy3-mlx-canary/route-traces/20260707-102115/`.

| Real trace | Events | Selected experts | Hits | Misses | Evictions | Payload read | Read wall | Final cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top5/slot16 `generate-cache`, prompt `Reply with exactly pong.`, 3 generated-token attempts | 632 | 3,160 | 1,051 | 2,109 | 845 | 20.853GiB | 6.648s | 12.498GiB |

Python live run for the same trace read **20.873GiB** and took **50.639s / 2.665s / 2.658s** across prefill/decode steps. C++ replay is not doing compute, but it now gives an apples-ish native IO/cache pressure model for real router decisions.

Routed parity is now implemented:

- Python fixture exporter: `hy3_export_layer_fixture.py`.
- Committed smoke fixture: `cpp/fixtures/hy3-layer1-top5-bos.json`.
- Single-layer artifact: `/Volumes/ModelSSD/logs/hy3-mlx-canary/parity-fixtures/20260707-112829/`.
- All-layer artifact: `/Volumes/ModelSSD/logs/hy3-mlx-canary/parity-fixtures/20260707-115831-all-layers/`.
- C++ replay paths: `hy3_sidecar_io --fixture ...` and `hy3_sidecar_io --fixture-list .../fixtures.txt`.
- C++ implementation is now split into reusable `hy3_expert_bank.*`, `hy3_q4_affine.*`, and `hy3_routed_mlp.*`; the CLI is no longer the math substrate.
- `--layer-major` fixture replay now dedups repeated prompt experts per routed layer while preserving parity metrics.
- Prefill4 artifact: `/Volumes/ModelSSD/logs/hy3-mlx-canary/parity-fixtures/20260707-140203-prefill4-all-layers/`.
- Prefill16 export artifact: `/Volumes/ModelSSD/logs/hy3-mlx-canary/parity-fixtures/20260707-152258-prefill16-all-layers/`.

| Parity fixture | Fixtures/layers | Seq len | Expert spans/read calls | Payload read | Compute wall | Max abs error | Max rel-to-expected | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| layer1 top5 BOS routed MoE, before shared MLP | 1 | 1 | 5 | 0.049438GiB | 0.426s | `4.69808e-05` | ~0.0058 | pass `<1e-4` |
| all MoE layers top5 BOS routed MoE, before shared MLP | 79 | 1 | 395 | 3.90564GiB | 34.87s | `16.9663` on layer 79 | `0.0177684` on layer 75 | pass `max(1e-4, 2% expected max)` |
| all MoE layers top5 4-token prefill fixture, naïve replay | 79 | 4 | 1,580 | 15.6226GiB | 225.78s | `16.9663` on layer 79 | `0.0141305` on layer 74 | pass `max(1e-4, 2% expected max)` |
| all MoE layers top5 4-token prefill fixture, layer-major dedup + Accelerate qlinear + dense reuse | 79 | 4 | 1,408 unique / 1,580 naïve | 13.9219GiB | 51.529s | `16.9663` on layer 79 | `0.0141303` on layer 74 | pass; saves 172 reads / 1.70068GiB |
| all MoE layers top5 16-token prefill fixture, layer-major dedup + Accelerate qlinear + dense reuse | 79 | 16 | 3,009 unique / 6,320 naïve | 29.7521GiB | 118.617s | see artifact | `0.0157937` | pass; saves 3,311 reads / 32.7382GiB |

Prefill16 export and full layer-major replay scaled the fixture shape to **79** all-layer fixtures with `seq_len=16`: Python/MLX sidecar read **29.752075GiB**, forward-to-layer wall **97.636s**, swap delta **0.0GiB**. C++ layer-major replay passed all layers with **3,009** unique reads vs **6,320** naïve route reads, saving **3,311** reads / **32.7382GiB**; current Release HEAD remeasure is **118.617s** compute / **119.02s** process wall with worst relative-to-expected error `0.0157937`. `hy3_sidecar_io --route-exec` now exposes the same routed-MoE math as fixture-backed prompt-slice execution over rich captured prompt fixtures. It explicitly reports `execution_scope=routed_moe_only`, rejects incompatible cache/IO flags, validates uniform shape plus strictly increasing unique layers, and keeps the older `--trace` TSV as cache/IO-only because TSV lacks hidden activations and route weights.

Route-batched SGEMM spike: artifact `cpp/results/20260707-route-batched-sgemm-spike.json`. Hypothesis was that grouping multiple token routes for the same expert into small-batch SGEMM would beat repeated SGEMV. It did not. Candidate code was not committed: prefill4 was only **0.843s** faster (**-1.6%**, basically noise), while prefill16 regressed from **118.617s** to **126.959s** (**+8.342s / +7.0%**) with parity still passing. Verdict: do **not** replace route SGEMV with naïve tiny-batch SGEMM. The next serious compute path needs fused/on-the-fly q4 Metal or another kernel that avoids dense FP32 materialization plus small-M SGEMM overhead.

Interpretation: native code can now materialize selected expert banks from the packed sidecar, dequantize MLX q4 affine weights, run `up/gate/down + swiglu + route weighting`, and match Python/MLX across every routed layer for both single-token and short prefill-shaped fixtures. The executor now exposes the real layer-major accounting: selected routes vs unique expert loads, bytes saved, and per-layer parity. Late-layer absolute errors are large because the expected activation magnitudes are huge; the relative gate is the right ABI/math check here. Apple Accelerate and expert-major dense reuse remove ~73% of the scalar qlinear wall. The persistent C++ routed-MoE daemon plus `HY3_CPP_ROUTE_MLP=1` Python hook is functional and has a working dense expert cache, but online top-k1 canaries are slower than the MLX path (`forward-one` 9.53s vs 5.87s; two-token generate-cache [23.17s, 10.49s] vs [11.03s, 5.13s]). Review hardening now validates request shape before wire writes, caps dense cache env values to finite `0..16` GiB, closes poisoned daemon clients on protocol errors, captures daemon stderr on startup/death, and forces parity fixture export back to Python/MLX even if `HY3_CPP_ROUTE_MLP` leaks from the shell.

Metal compute spikes: the simple one-projection MLX custom Metal q4 matvec is feasible but only parity/noisy against `gather_qmm`. The follow-up fused up+gate/down spike lives at `spikes/002-metal-fused-up-gate/` with summary `spikes/002-metal-fused-up-gate/summary.json`. Stable six-sample pass: fused up+gate+swiglu is directionally real but modest/noisy (**1.0658x** median vs `gather_qmm(up/gate)+swiglu`, one sample at **0.9138x**). The best full-expert proxy is fused up+gate plus normal `gather_qmm(down)`: **1.1245x** median / **1.1726x** mean. Replacing down with a custom Metal row-reduce projection does **not** help: all-Metal full expert is **1.0505x** median and slower than keeping MLX down. Verdict: do not wire this into runtime yet. Next serious kernel needs to fuse down/route weighting into the same launch, or batch many routes without dense FP32 materialization; one-projection clones keep losing to launch overhead and MLX's already-good `gather_qmm`.

Route-batched Metal spike: artifact `spikes/003-metal-route-batched-up-gate/summary.json`. This tested the real routed shape for six layer fixtures (seq4/seq16 × layers 1/40/79): one Metal launch emits `route_weight * swiglu(gate_q4(x), up_q4(x))` for all selected routes, then MLX `gather_qmm(down)` sums the outputs. Algebraically, applying route weights before down is basically neutral under MLX (`0.9949x` median vs standard, tiny early/mid-layer diffs; late-layer absolute diffs reflect huge activation scale). But the row-reduce Metal route kernel is much slower: weighted-hidden primitive **0.4708x** median vs `gather_qmm`; full route proxy **0.5227x** median vs standard. Verdict: invalidated for runtime integration. Do not build one-threadgroup-per-route/output-row kernels into Capstan; the next plausible path is a real tiled/simdgroup q4 dot kernel, MPS-backed batch matmul path, or deeper fused route kernel.

## Default packed Python lane

`hy_v3_mlx_lazy.DEFAULT_LAYOUT` and `hy3_lazy_smoke.LAYOUT_PATH` now default to `/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/manifest.json` instead of the older v0 sidecar layout. The env override `HY3_SIDECAR_LAYOUT` still works. Top-k1 two-token `generate-cache` smoke with no layout env now reports `schema=hy3-packed-sidecar-v1`, **426** span reads / **4.212158GiB**, **[9.859s, 4.485s]** step timings, and **0.0GiB** swap delta. The old v0-layout comparison was **3,834** tensor reads and **[11.027s, 5.128s]**.

## Bounded packed-read coalescing

Python packed expert loading now batches missing experts per layer into bounded contiguous reads. Defaults: `HY3_PACKED_COALESCE_MAX_GIB=0.032` and `HY3_PACKED_COALESCE_MAX_OVERREAD_RATIO=2.0`; set max GiB to `0` to disable. The layout makes this safe to bound: each expert bank is a fixed contiguous **10,616,832 bytes** (~9.89MiB), adjacent experts are byte-adjacent, and a whole layer is ~**1.898GiB**, so never read min→max without a cap.

Measured top5/slot16 `generate-cache`, 8-token decode:

| Coalesce cap | Step sum | Read calls | Payload | Extra payload | Load time | Switch time | Swap delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| disabled | **76.558s** | 2,863 | 28.308GiB | 0.000GiB | 27.074s | 41.247s | 0.0GiB |
| 32MiB | **59.655s** | 2,540 | 29.683GiB | 1.374GiB | 14.790s | 28.643s | 0.0GiB |
| 64MiB | **61.010s** | 2,322 | 33.598GiB | 5.290GiB | 17.482s | 30.436s | 0.0GiB |

Verdict: 32MiB is the best tested default. Fewer syscalls/bigger reads beat byte purity, but 64MiB overreads too much and loses. A 16-token top5/slot16 run with the 32MiB default completed in **110.934s**, **3,921** reads, **43.783GiB** payload, **1.552GiB** extra payload, and **0.0GiB** swap delta. The lane is more efficient, but still not remotely interactive.

Review hardening after this cut preserves `slot_bank` as a hard post-pack cache cap even when a current request needs more experts than the configured bank; requested tensors are protected only until `mx.eval(packed)` finishes, then the layer cache trims back down. Stats now distinguish total packed read groups/experts from true multi-expert groups, and disable mode (`HY3_PACKED_COALESCE_MAX_GIB=0`) skips overread-ratio validation.

Artifacts: `results/20260707-packed-coalesced-loader-summary.json` plus per-run decode JSONs.

## Route locality analyzer

Added `hy3_route_locality.py`, a call-level analyzer for `hy3-route-trace-v1` TSV traces. It infers prefill/decode passes when layer order resets, groups rows into actual Python routed-layer calls, and simulates `Hy3SidecarStore` cache behavior after the post-pack hard-cap fix. This intentionally models unique expert loads per layer call, not the older C++ per-selection replay.

Real trace analyzed: `/Volumes/ModelSSD/logs/hy3-mlx-canary/route-traces/20260707-102115/top5-slot16-pong-3tok-trace.tsv` (**632** events, **237** layer calls, **3** passes, **3,160** selected experts). Artifacts: `results/20260707-route-locality-top5-slot16.json` and `.md`.

Best tested policy by simulated miss count, now reporting actual coalesced read bytes separately from useful expert payload bytes:

| Slot | Best policy | Misses | Hit rate | Actual read GiB | Payload GiB | Extra GiB | Evictions | Oversized calls | Final cache GiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | `freq_last` | 2,146 | 0.151 | 22.564 | 21.219 | 1.345 | 1,514 | 79 | 6.249 |
| 12 | `freq_last` | 2,110 | 0.166 | 22.208 | 20.863 | 1.345 | 1,162 | 79 | 9.374 |
| 16 | `freq_last` | 2,091 | 0.173 | 22.010 | 20.675 | 1.335 | 827 | 76 | 12.498 |
| 20 | `freq` | 2,073 | 0.180 | 21.832 | 20.497 | 1.335 | 498 | 51 | 15.573 |
| 24 | `freq` | 2,060 | 0.185 | 21.703 | 20.369 | 1.335 | 230 | 22 | 18.094 |
| 32 | `freq` | 2,053 | 0.188 | 21.634 | 20.299 | 1.335 | 6 | 0 | 20.240 |

Verdict: cache policy is only a small lever on this trace. At slot16, `freq_last` beats current `freq` by **10** misses in simulation: current `freq` is **22.129GiB** actual / **20.774GiB** payload, while `freq_last` is **22.010GiB** actual / **20.675GiB** payload. Moving slot16→20 saves only **18** misses while adding ~**3.1GiB** final cache. Most churn is baked into route distribution, not LRU ordering.

Live confirmation artifact: `results/20260707-decode8-top5-slot16-coalesce-032-freqlast.json`. Same prompt/config as the 32MiB decode8 baseline (`generate-cache`, top5, slot16, 8 tokens, profile+sync timers) but `HY3_RETAIN_POLICY=freq_last`. Result: **59.925s** step sum vs **59.655s** baseline `freq`; **2,888** loads vs **2,863**; **1,616** hits vs **1,641**; **29.920GiB** read vs **29.683GiB**. Swap delta stayed **0.0GiB**. So `freq_last` is not a runtime win; keep `freq` as default and stop spending time on cache ordering. The next real lever is the kernel/compute path.

## DS4 lane cleanup

DeepSeek V4 Flash SSD lane was stopped before Hy3 runs and restored after each heavy test.
Post-restore health passed on `127.0.0.1:8127` with exact JSON smoke and swap delta OK.
