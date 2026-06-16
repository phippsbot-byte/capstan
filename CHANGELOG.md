# Changelog

## v0.14.0

- Add `modelctl fleet status` for a non-gating operator snapshot across registered manifests.
- Fleet status reports per-lane `ready`/`down`/`invalid` state, PID/log paths, readiness details, current swap, and LaunchAgent plist presence.
- Add `--readiness-timeout` and `--limit` controls for bounded fleet status scans.

## v0.13.0

- Add `modelctl fleet health` to run structured health verdicts across registered manifests.
- Fleet health returns per-model status, issues, warnings, and nested health details, and exits non-zero if any lane is critical/invalid or no registered lanes are found.
- Support fleet-level swap ceilings, swap delta sampling, optional smoke checks, registry overrides, and bounded `--limit` scans.

## v0.12.0

- Add `daemon --health-mode` so long-running supervisors can use the structured health verdict instead of legacy watchdog-only sampling.
- Add daemon/service flags for `--max-swap-delta-gib`, `--sample-sec`, `--smoke`, and `--max-latency-sec`.
- Teach `service install` to persist health-mode daemon arguments into the generated LaunchAgent plist.

## v0.11.0

- Add `modelctl health` for a single PID/readiness/swap health verdict.
- Add swap delta sampling via `--max-swap-delta-gib` and `--sample-sec` so huge local lanes are not judged only by stale absolute swap.
- Add optional `--smoke` and `--max-latency-sec` checks for live endpoint behavior without forcing every health probe to spend tokens.
- Isolate registry tests from the user's real `~/.config/modelctl/models` registry.

## v0.10.0

- Add `modelctl service install` for macOS LaunchAgent plist generation around `modelctl daemon`.
- Add `modelctl service start/stop/restart/status/uninstall` wrappers for `launchctl`, with `--dry-run` previews for safe automation and tests.
- Add guarded daemon service options: explicit `--restart`, swap ceiling override, interval, log path, run-at-load, keepalive, and custom launchd label.
- Keep service install side-effect safe by refusing plist overwrites unless `--overwrite` is passed.

## v0.9.0

- Add `modelctl mlx discover` to find local MLX model directories.
- Add `modelctl mlx inspect` to flag config/template serving hazards, including Qwen/Qwopus `<think>` generation preambles.
- Add `modelctl mlx overlay` to create reversible `-served` overlays that symlink weights and patch only `chat_template.jinja`.
- Add `modelctl mlx manifest` to generate an MLX-focused `modelctl.toml` using `python -m mlx_lm server` with conservative single-user defaults.
- Include docs/examples in source distributions and modernize package license metadata to kill setuptools deprecation noise.

## v0.8.0

- Add `modelctl daemon`, a foreground readiness/swap supervisor loop.
- Support bounded runs with `--iterations` for scripts/tests and continuous operation when omitted.
- Support explicit `--restart` on breach for manifests with `[start]`; restart is opt-in, not a footgun default.

## v0.7.0

- Add `modelctl doctor --fix` for safe local repairs: stale/invalid PID-state removal and state directory creation.
- Add saved report history via `modelctl reports save/list/show` under the modelctl state directory.
- Add global `--pretty` output for human-readable command summaries while keeping JSON as the default.

## v0.6.0

- Add tag-based GitHub Actions release automation that builds wheel/sdist, verifies the wheel, and uploads assets to the GitHub release.
- Add manual PyPI Trusted Publishing workflow for `local-modelctl` once PyPI-side publisher trust is configured.
- Add `modelctl init` for starter `modelctl.toml` manifests.
- Add `modelctl version`.
- Make CI build distributions on every push/PR so packaging breakage is caught before tags.

## v0.5.0

- Add `modelctl registry use NAME` to materialize a registered manifest into a working `modelctl.toml` by copy or symlink.
- Add `modelctl bench --output PATH --format json|md` for shareable benchmark artifacts.
- Fix `registry add` exit codes so duplicate entries fail with a non-zero status unless `--overwrite` is passed.
- Build release distributions for upload with the GitHub release.

## v0.4.0

- Add registry management: `modelctl registry add/list/show/remove`.
- Add `modelctl report` for JSON/Markdown model state artifacts.
- Add `modelctl bench --preset tiny|small|standard` while keeping explicit `--prompt-chars` overrides.
- Keep top-level `modelctl list` as a registry-list convenience alias.

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
