# Hy3 Metal q4 matvec spike — 2026-07-07

## Verdict: PARTIAL

A simple custom MLX Metal q4 affine matvec is feasible, but not a decisive win over `mx.gather_qmm`.

This spike tested one real Hy3 expert tensor loaded from the packed sidecar, using two kernels:

- `naive`: one Metal thread per output row, serial dot over input dim.
- `row-reduce`: one 256-thread threadgroup per output row, parallel reduction over input dim.

## Results

See `summary.json` plus per-run JSON artifacts.

| Projection | Kernel | Metal median | MLX gather median | Speedup |
|---|---:|---:|---:|---:|
| `up_proj` | naive | 0.008544s | 0.006791s | 0.79x |
| `up_proj` | row-reduce | 0.000329s | 0.000322s | 0.98x |
| `gate_proj` | row-reduce | 0.004046s | 0.004739s | 1.17x |
| `down_proj` | row-reduce | 0.000361s | 0.000347s | 0.96x |

Numerical diffs vs `gather_qmm` are nonzero (`~0.027–0.055` max abs), likely from float32 scale/bias conversion plus reduction ordering. This is acceptable for a spike but not enough for runtime integration without fixture-level routed parity.

## Recommendation

Do **not** replace MLX `gather_qmm` with this simple custom matvec. It is roughly parity/noisy, not the kind of win we need.

If we continue Metal work, the target should be a more fused routed-MLP kernel or a route-batched q4 kernel that removes MLX op launch / gather overhead across top-k, not a one-projection matvec clone.
