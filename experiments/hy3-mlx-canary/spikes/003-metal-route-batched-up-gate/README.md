# Hy3 route-batched Metal up+gate spike — 2026-07-08

## Question
Can we move from toy one-expert Metal probes to a real routed-MoE shape: all selected routes for one layer fixture in one Metal launch?

The probe computes weighted route hidden states:

```text
route_weight * silu(gate_q4(hidden[token])) * up_q4(hidden[token])
```

Then it keeps MLX `gather_qmm(down)` for the down projection and sums routes per token. This tests fewer launches + route-weight integration without trying to solve the entire fused MoE kernel at once.

## Fixture set

Six real fixtures:
- seq4: layers 1, 40, 79
- seq16: layers 1, 40, 79
- top-k: 5

## Results

| Layer | Seq | Routes | Unique experts | Metal hidden speedup | Preweighted gather full vs standard | Metal preweighted full vs standard |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 80 | 38 | 0.3894x | 0.9974x | 0.4423x |
| 1 | 4 | 20 | 17 | 0.5540x | 0.9950x | 0.5998x |
| 40 | 16 | 80 | 36 | 0.3877x | 0.9947x | 0.4316x |
| 40 | 4 | 20 | 16 | 0.5395x | 1.0052x | 0.5895x |
| 79 | 16 | 80 | 40 | 0.4022x | 0.9901x | 0.4559x |
| 79 | 4 | 20 | 19 | 0.5505x | 0.9771x | 0.5940x |

## Aggregate

- Metal weighted-hidden primitive vs MLX gather hidden: median **0.4708x**, mean **0.4706x**.
- Metal preweighted full proxy vs standard gather route: median **0.5227x**, mean **0.5189x**.
- Preweighted gather full proxy vs standard gather route: median **0.9949x**, mean **0.9932x**.

## Verdict

**INVALIDATED for runtime integration.** The route-weight-before-down algebra is acceptable for this probe and basically neutral under MLX. The custom route-batched row-reduce Metal kernel is the loser: it is slower than `gather_qmm` on every sample.

Do **not** build this shape into Capstan. One threadgroup per route/output row does not exploit the GPU well enough. Next credible path needs a real tiled/simdgroup q4 dot kernel, MPS-backed batch matmul path, or a deeper fused kernel that does not clone individual qlinear projections.
