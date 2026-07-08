# Hy3 Metal fused up+gate spike — 2026-07-07

## Question
Can a custom Metal kernel fuse one expert's q4 `up_proj` and `gate_proj` matvecs plus `swiglu`, beating MLX `gather_qmm` enough to justify a larger routed-MoE kernel?

## Method
Script: `metal_fused_up_gate_probe.py`.

For each `(layer, expert)` sample it measures:
- MLX baseline: `gather_qmm(up)`, `gather_qmm(gate)`, then `swiglu`.
- Metal primitive: one fused q4 kernel producing `silu(gate_q4(x)) * up_q4(x)` directly.
- Full expert proxy: fused up+gate hidden followed by normal `gather_qmm(down)`, versus normal `gather_qmm` up/gate/down.

Stable pass: 6 samples, 20 warmups, 200 repeats each.

## Results

| Layer | Expert | Up+gate gather med | Up+gate fused med | Up+gate speedup | Full gather med | Full fused med | Full speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.000939s | 0.000873s | 1.0756x | 0.001046s | 0.000951s | 1.0999x |
| 1 | 17 | 0.000964s | 0.000932s | 1.0343x | 0.001211s | 0.000982s | 1.2332x |
| 40 | 0 | 0.000901s | 0.000828s | 1.0882x | 0.000938s | 0.000902s | 1.0399x |
| 40 | 63 | 0.000911s | 0.000826s | 1.1029x | 0.000956s | 0.000939s | 1.0181x |
| 79 | 0 | 0.000989s | 0.000946s | 1.0455x | 0.001126s | 0.001055s | 1.0673x |
| 79 | 127 | 0.000926s | 0.000833s | 1.1116x | 0.000936s | 0.000952s | 0.9832x |

## Aggregate

- Up+gate fused speedup: median **1.0819x**, mean **1.0763x**, min **1.0343x**.
- Full expert proxy speedup: median **1.0536x**, mean **1.0736x**, min **0.9832x**.
- Mean full-expert median time: fused **0.000963s** vs gather **0.001035s**.

## Verdict

**PARTIAL.** Fused q4 Metal up+gate works and consistently beats `gather_qmm + swiglu` on the primitive. It is not enough to wire into runtime: full expert-route improvement is only about **1.0536x** median and one sample regressed.

Next serious kernel should fuse more of the routed path — down projection and/or route weighting — and avoid dense FP32 materialization plus tiny-M CPU/BLAS overhead. Up+gate-only fusion is a proof of direction, not a product cut.
