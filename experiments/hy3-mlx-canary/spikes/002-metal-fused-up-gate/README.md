# Hy3 Metal fused up+gate/down spike — 2026-07-07

## Question
Can custom Metal q4 kernels beat MLX `gather_qmm` enough to justify a larger routed-MoE kernel?

This spike tests:
- fused `up_proj + gate_proj + swiglu` in one Metal kernel
- full expert proxy with fused up+gate followed by normal `gather_qmm(down)`
- all-Metal full expert proxy with fused up+gate followed by a custom Metal q4 down projection

Stable pass: 6 samples, 20 warmups, 200 repeats each.

## Results

| Layer | Expert | Up+gate speedup | Fused up+gate + gather down speedup | All-Metal full speedup | Full gather med | Fused+gather-down med | All-Metal med |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.9138x | 1.1322x | 1.1254x | 0.001113s | 0.000983s | 0.000989s |
| 1 | 17 | 1.3136x | 1.5784x | 1.6207x | 0.000846s | 0.000536s | 0.000522s |
| 40 | 0 | 1.0717x | 1.0347x | 1.0022x | 0.000925s | 0.000894s | 0.000923s |
| 40 | 63 | 1.1063x | 1.1328x | 1.0859x | 0.001075s | 0.000949s | 0.000990s |
| 79 | 0 | 1.0000x | 1.0410x | 1.0151x | 0.000940s | 0.000903s | 0.000926s |
| 79 | 127 | 1.0599x | 1.1168x | 0.9977x | 0.000880s | 0.000788s | 0.000882s |

## Aggregate

- Fused up+gate primitive: median **1.0658x**, mean **1.0776x**, min **0.9138x**.
- Full expert with fused up+gate + MLX down: median **1.1245x**, mean **1.1726x**.
- All-Metal full expert: median **1.0505x**, mean **1.1412x**, min **0.9977x**.
- Mean full-expert median time: gather **0.000963s**, fused+gather-down **0.000842s**, all-Metal **0.000872s**.

## Verdict

**PARTIAL / NO RUNTIME CUT.** Fused up+gate is a real direction, but still small and noisy. The best full-expert proxy keeps MLX `gather_qmm` for down. Custom Metal down is not better than MLX down here.

Do **not** wire these kernels into the runtime yet. The next useful kernel has to fuse more of the route into one launch — down projection and/or route weighting — or batch many routes without dense FP32 materialization. One-projection clones keep running into launch overhead and MLX's already-good `gather_qmm`.
