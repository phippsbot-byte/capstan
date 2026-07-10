# Hy3 C++ q4 direct-dot spike — 2026-07-08

## Question

Can a direct C++ q4 affine kernel beat the current dense-dequantize + Apple Accelerate path by avoiding FP32 weight materialization?

The kernel uses the MLX q4 affine identity per 64-wide group:

```text
dot(x, q * scale + bias) = scale * sum(q_i * x_i) + bias * sum(x_i)
```

It fuses `up_proj + gate_proj + swiglu` for the first half of the expert and uses the same direct q4 dot for `down_proj`.

## Fixture set

Six real routed fixtures, same spread as the route-batched Metal spike:

- seq4: layers 1, 40, 79
- seq16: layers 1, 40, 79
- top-k: 5

Result artifact: `results/20260708-q4-direct-dot.json`.

## Results

| Layer | Seq | Routes | Unique experts | Direct vs dense incl. dequant | Direct vs dense compute-only | Direct rel error |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 20 | 17 | 4.4529x | 0.4624x | 0.001495 |
| 40 | 4 | 20 | 16 | 2.4760x | 0.2739x | 0.008619 |
| 79 | 4 | 20 | 19 | 3.4632x | 0.3943x | 0.002897 |
| 1 | 16 | 80 | 38 | 2.7158x | 0.4638x | 0.005121 |
| 40 | 16 | 80 | 36 | 2.2701x | 0.3471x | 0.011102 |
| 79 | 16 | 80 | 40 | 2.5265x | 0.4182x | 0.003224 |

Aggregate:

- Median speedup vs dense path **including dequantization**: **2.6212x**.
- Mean speedup vs dense path **including dequantization**: **2.9841x**.
- Median speedup vs dense **compute-only after dense weights are hot**: **0.4062x**.
- Mean speedup vs dense **compute-only after dense weights are hot**: **0.3933x**.
- Direct output matches dense output tightly (`max_rel_to_dense` below `5e-7` in all sampled fixtures); fixture-relative errors match the existing dense path.

## Verdict

**PARTIAL / VALIDATED for cold and prefill-shaped routes.** Direct q4 dot avoids FP32 materialization and wins decisively when current runtime would otherwise dequantize dense weights for each newly loaded expert.

**Not a universal replacement.** Once a dense expert bank is already hot in cache, Accelerate SGEMV is still about **2.5x** faster than this NEON direct-dot implementation. Wiring direct q4 everywhere would make hot-cache decode worse.

## Recommendation

Add a runtime q4 mode only after a small integration cut:

- `dense`: current path, best when dense expert cache is hot or repeated route count is high.
- `direct`: use for cold/prefill-shaped layer-major routes where dense materialization dominates.
- Future hybrid heuristic: direct for first use / low route reuse, dense for experts reused across enough routes or retained in the daemon dense cache.

This is a more credible next Capstan cut than another one-row-per-threadgroup Metal kernel. The GPU path still needs a tiled/simdgroup q4 dot, but the CPU direct-dot result gives us a real algebraic baseline and a cold-route escape hatch.
