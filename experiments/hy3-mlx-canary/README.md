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

Next useful work: tiny Phipps slice through the canary server, with lane clearly labeled `top8 fair` vs `top4 approximate`.
