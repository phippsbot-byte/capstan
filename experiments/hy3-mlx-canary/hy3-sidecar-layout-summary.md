# Hy3 preview MLX sidecar layout summary

Source: `/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX`

## Split verdict

| Bucket | Size |
|---|---:|
| Total MLX safetensor payload | 154.589 GiB |
| Resident non-expert core | 4.612 GiB |
| Routed expert sidecar | 149.977 GiB |
| Expert share | 97.02% |

## Derived runtime pressure

| Metric | Value |
|---|---:|
| Expert MoE layers | 79 |
| Experts/layer | 192 |
| Native top-k | 8 |
| Expert bank/layer | 1.898 GiB |
| Per-expert payload | 10.125 MiB |
| Cold active read/layer | 81.000 MiB |
| Cold active read/token | 6.249 GiB |

## Slot-bank estimates

| Slots/layer | Host/cache footprint |
|---:|---:|
| 16 | 12.498 GiB |
| 24 | 18.747 GiB |
| 32 | 24.996 GiB |
| 48 | 37.494 GiB |
| 64 | 49.992 GiB |
| 80 | 62.490 GiB |
| 96 | 74.988 GiB |
| 128 | 99.984 GiB |

## Categories

| Category | Size | Tensors |
|---|---:|---:|
| `attention_resident` | 3.165 GiB | 1200 |
| `dense_mlp_resident` | 0.086 GiB | 9 |
| `embeddings_lm_head` | 0.519 GiB | 6 |
| `norms_resident` | 0.001 GiB | 81 |
| `routed_experts_sidecar` | 149.977 GiB | 711 |
| `router_gate_resident` | 0.062 GiB | 316 |
| `shared_mlp_resident` | 0.781 GiB | 711 |

## Operator note

This layout is metadata-only. It proves the MLX artifact is almost perfectly split for a Capstan-style runtime: keep ~4.6 GiB resident, stream/cache ~150 GiB of routed experts. It does **not** make stock MLX load the model safely; runtime work is still required.
