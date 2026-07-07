# Hy3 MLX Canary

Local Hy3 / `hy_v3` MLX lazy-expert runtime experiments for the Mac Studio.

This repo is **code + metadata only**. It intentionally does not contain model weights, packed sidecars, logs, or credentials.

## Current verdict

- Flat/stock MLX load is not viable on the 96GB Studio; it hit swap before useful generation.
- Lazy SSD sidecar runtime works: resident core stays small and routed experts stream from SSD.
- Packed layer-major sidecar works and removed the worst Python-GC bottleneck.
- Local top-4/top-5/top-6 Python lanes are plumbing canaries, not quality lanes; top5/slot16 + prefix prewarm + expert-cache clear is the stable operator lane.
- First Capstan/C++ substrate exists under `cpp/`: split reusable expert-bank/q4/routed-MLP modules + compact index + contiguous expert-span `pread` benchmark + native slot-bank cache simulation + real router trace replay + routed q4 parity. Full top-8 all-layer expert payload (6.249GiB / 632 spans) reads in ~3.2s warm-cache on the Studio; a real top5/slot16 prefill+decode trace replays 20.853GiB with 1,051 hits / 2,109 misses / 845 evictions. All 79 MoE layers pass native routed parity. Layer-major fixture replay now dedups repeated prompt experts: the 4-token all-layer fixture drops from 1,580 naïve expert reads / 15.6226GiB to 1,408 unique reads / 13.9219GiB with parity intact. Apple Accelerate qlinear plus expert-major dense reuse cuts prefill16 layer-major wall from 899.48s to 239.45s; `--route-exec` now exposes the same substrate as prompt/routed-MoE execution over rich captured prompt fixtures.

## Important files

| File | Purpose |
|---|---|
| `hy_v3_mlx_lazy.py` | Lazy Hy3 model/runtime with SSD-backed expert cache |
| `hy3_lazy_smoke.py` | Guarded substrate smokes: expert read, resident load, one-token, generate |
| `hy3_openai_server.py` | OpenAI-compatible canary server with request clamps/busy state |
| `hy3_local_optimizer.py` | Lane sweep runner for top-k/slot-bank smoke tests |
| `hy3_pack_sidecar.py` | Builds packed layer-major sidecar from MLX safetensors |
| `hy3_sidecar_layout.py` | Metadata-only sidecar layout planner |
| `run_hy3_phipps_slice.sh` | Tiny Phipps slice runner with DS4 restore trap |
| `hy3_emit_compact_index.py` | Emits compact TSV sidecar index for native/C++ experiments |
| `hy3_export_layer_fixture.py` | Exports compact Python/MLX routed-layer parity fixtures for native replay |
| `cpp/` | Split C++20 sidecar substrate, real trace replay, all-layer parity, and layer-major prefill dedup replay |
| `HY3-LAZY-SIDECAR-STATUS.md` | Running status and measured artifacts |

## Local artifact expectations

Default paths used by the canary:

- MLX model: `/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX`
- Packed sidecar manifest: `/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/manifest.json`
- Logs/results: `/Volumes/ModelSSD/logs/hy3-mlx-canary`

## Quick checks

Compile/static sanity:

```bash
/opt/homebrew/bin/python3.11 -m py_compile *.py
```

Metadata-only optimizer dry run:

```bash
/opt/homebrew/bin/python3.11 hy3_local_optimizer.py --dry-run
```

Small sidecar smoke:

```bash
/opt/homebrew/bin/python3.11 hy3_lazy_smoke.py expert-read \
  --slot-bank 8 --layer 1 --experts 1 \
  --layout /Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar/manifest.json
```

## Operational guardrail

When running Hy3 server/slices on the Studio, stop the DS4 SSD lane first and restore it afterward. `run_hy3_phipps_slice.sh` and `hy3_local_optimizer.py --stop-ds4` do that; the optimizer also has a swap guard (`--max-swap-gib`, `--max-swap-delta-gib`). The server has tool-scaffold prefix/KV caching (`--prefix-cache-min-tokens`) so repeated tool calls do not re-prefill the same giant schema block; use `hy3_local_optimizer.py --prewarm-prefixes` to pay that prefix build before measured prompts. If cumulative request pressure is the problem, test `--clear-expert-cache-after-request` before touching slot sizes again. Ad-hoc shells usually forget this stuff and then we get mystery pressure. Computers: still dumb, still fast at hurting themselves.
