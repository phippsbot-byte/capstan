# Hy3 route locality analysis

Trace: `/Volumes/ModelSSD/logs/hy3-mlx-canary/route-traces/20260707-102115/top5-slot16-pong-3tok-trace.tsv`

## Trace shape

- Events: **632**
- Layer calls: **237**
- Passes: **3**
- Selected experts: **3160**
- Avg top-k: **5.0**

## Best policy per slot bank

| Slot | Best policy | Misses | Hits | Hit rate | Read GiB | Evictions | Oversized calls | Final cache GiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | `freq_last` | 2146 | 383 | 0.151 | 21.219 | 1514 | 79 | 6.249 |
| 12 | `freq_last` | 2110 | 419 | 0.166 | 20.863 | 1162 | 79 | 9.374 |
| 16 | `freq_last` | 2091 | 438 | 0.173 | 20.675 | 827 | 76 | 12.498 |
| 20 | `freq` | 2073 | 456 | 0.180 | 20.497 | 498 | 51 | 15.573 |
| 24 | `freq` | 2060 | 469 | 0.185 | 20.369 | 230 | 22 | 18.094 |
| 32 | `freq` | 2053 | 476 | 0.188 | 20.299 | 6 | 0 | 20.240 |

## Focus: `freq` at slot 16

- Misses: **2101**
- Hits: **428**
- Read: **20.774GiB**
- Evictions: **837**
- Final cache: **12.498GiB**

Top miss layers:

| Layer | Misses | Hits | Hit rate | Evictions | Unique requests | Max unique/call | Read GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 35 | 1 | 0.028 | 19 | 36 | 26 | 0.346 |
| 62 | 34 | 5 | 0.128 | 18 | 39 | 29 | 0.336 |
| 4 | 34 | 4 | 0.105 | 18 | 38 | 28 | 0.336 |
| 64 | 34 | 3 | 0.081 | 18 | 37 | 27 | 0.336 |
| 66 | 34 | 3 | 0.081 | 18 | 37 | 27 | 0.336 |
| 73 | 34 | 3 | 0.081 | 18 | 37 | 27 | 0.336 |
| 5 | 33 | 2 | 0.057 | 17 | 35 | 25 | 0.326 |
| 76 | 32 | 6 | 0.158 | 16 | 38 | 28 | 0.316 |
| 29 | 32 | 5 | 0.135 | 16 | 37 | 27 | 0.316 |
| 75 | 32 | 5 | 0.135 | 16 | 37 | 27 | 0.316 |
| 77 | 32 | 5 | 0.135 | 16 | 37 | 27 | 0.316 |
| 3 | 32 | 4 | 0.111 | 16 | 36 | 26 | 0.316 |

## Interpretation

At slot 16, `freq_last` beats `freq` by 10 misses in this trace. Moving from slot 16 to slot 20 saves 18 misses but increases final cache to 15.57GiB. Slot 16 has 76 oversized calls; post-pack trimming is mandatory for memory safety.
