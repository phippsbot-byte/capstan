# Hy3 MLX lazy-sidecar canary

Local R&D snapshot for Tencent Hy3 / `hy_v3` expert streaming on the 96GB Mac Studio.

This is **not production Capstan code**. It preserves the working Python/MLX canary so the C++/Capstan port has a verified reference implementation and benchmark artifacts.

## What is tracked here

- `hy_v3_mlx_lazy.py` — lazy routed-expert Hy3 model implementation.
- `hy3_lazy_smoke.py` — guarded resident/load/forward/generate smoke runner.
- `hy3_pack_sidecar.py` — packs original MLX safetensor expert tensors into layer-major expert-major sidecar files.
- `hy3_routed_microbench.py` — isolates routed expert `gather_qmm`, weighting/sum, and sidecar behavior.
- `hy3_openai_server.py` — tiny single-threaded OpenAI-compatible canary server for local eval smokes.
- `hy3_sidecar_layout.py` + `hy3-sidecar-layout.json` — metadata/offset snapshot for the local downloaded preview model.
- `HY3-LAZY-SIDECAR-STATUS.md` — chronological verdicts, measurements, and next steps.

## What is intentionally not tracked

Heavy model artifacts live outside Git:

- `/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX`
- `/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-sidecar`
- `/Volumes/ModelSSD/logs/hy3-mlx-canary`

Do not commit model weights, packed sidecar binaries, or generated logs.

## Current verified result

After removing explicit per-expert `gc.collect()` from the hot path:

- packed one-token forward: ~3.5s, 0.0GiB swap delta
- packed KV-cache `pong`: clean exact `pong`
- slot-bank 16 full top-8 `pong`: ~14.8s / 1.1s / 1.1s with frequency-retention cache policy, 0.0GiB swap delta
- slot-bank 18+ full top-8 currently swap-bombs under guarded runs
- fast approximate lane: top-4 + slot-bank 32 exact `pong` at ~4.3s / 0.4s / 0.4s, 0.0GiB swap delta
- top-4 + slot-bank 32 also passed exact JSON and tool-shaped JSON canaries
- tiny OpenAI-compatible canary server exists at `hy3_openai_server.py`; smoke-passed `/v1/models`, exact `pong`, exact JSON, and OpenAI-style parsed tool call
- top5/slot16 + prefix prewarm + expert-cache clear is the stable Python operator lane; top6 was slower and worse on the tiny Phipps slice
- first C++20 substrate lives under `cpp/`: compact TSV sidecar index + contiguous expert-span `pread`; full top-8 all-layer expert payload reads 6.249GiB / 632 spans in ~3.2s warm-cache on the Studio
- native slot-bank/cache scheduler now models layer-major read pressure before kernel work; top8 fixed 4-token trace at slot16 reads once then hits cache, top5 hot 8-token trace reads 7.811GiB with no evictions, adversarial top5 rolling churn reads 31.245GiB and evicts 1,896 spans
- real router trace capture/replay now works end-to-end; top5/slot16 prefill+decode trace replays 20.853GiB with 1,051 hits / 2,109 misses / 845 evictions through native C++
- one-layer routed parity now passes: native q4 affine `up/gate/down + swiglu + route weighting` matches Python/MLX for layer1 top5 BOS with max abs error `4.69808e-05`

Next useful work: scale the native parity path from one routed layer into layer-major prefill/decode execution. Python canary work should be frozen unless a very specific smoke needs it.
