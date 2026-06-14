# Changelog

## v0.3.0

- Add `modelctl ingest` to generate a starter manifest from a running OpenAI-compatible `/v1` endpoint.
- Add `modelctl bench` for synthetic prompt-size benchmarks with timing, usage, server timings, and swap sampling.
- Add `modelctl watchdog` for readiness/swap sampling with optional stop-on-breach.
- Keep `modelctl` dependency-free at runtime and Python 3.11+.

## v0.2.0

- Add `modelctl soak` for repeated smoke tests with latency and swap deltas.
- Add `modelctl doctor` for stale PID/log/readiness/cleanup diagnostics.
- Add `modelctl list` registry scanning via `$MODELCTL_REGISTRY` and `~/.config/modelctl/models`.

## v0.1.0

- Initial manifest-driven local model controller.
- Commands: `validate`, `preflight`, `start`, `wait`, `status`, `smoke`, `cleanup`, `stop`.
- TOML manifests, process group start/stop, OpenAI-compatible smoke, guarded cleanup.
