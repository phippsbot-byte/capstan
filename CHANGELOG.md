# Changelog

## v0.24.4

- Validate manifests and action results strictly so malformed lifecycle requests and ambiguous failures stop with non-zero exit codes.
- Add canonical multi-resource lifecycle locks and kernel-backed PID/PGID/birth identity checks before signaling or deleting process state.
- Publish a private durable launch transaction before `Popen`, then promote it to authenticated v2 PID state only after process identity capture.
- Make rotation and promotion single-lock transactions with readiness plus endpoint-ownership proof, authenticated state handoff, post-health gating, and rollback.
- Preserve malformed, untrusted, pending, mismatched, and unproven state instead of signaling, overwriting, or deleting it.

## v0.24.3

- Make `fleet intake` treat existing manifest `[preflight].exclusive_ports` as reserved local ports, skipping matching candidates before `/v1/models` probes or manifest drafts.

## v0.24.2

- Fix `fleet intake` endpoint-key canonicalization to preserve IPv6 brackets for non-local literals, preventing distinct remote IPv6 endpoints from being deduped or skipped as the same lane.

## v0.24.1

- Fix `fleet intake` registered-endpoint skipping for local host aliases such as `localhost`, `0.0.0.0`, and `[::1]` so already-registered local lanes are not probed or duplicated.

## v0.24.0

- Add `fleet intake` to discover live OpenAI-compatible endpoints, skip already-registered lanes, and draft dormant registry manifests from `/v1/models` without start/health/smoke side effects.

## v0.23.1

- Fix `fleet doctor` duplicate-port detection to include manifest-owned `[preflight].exclusive_ports`, not just endpoint URL ports.

## v0.23.0

- Add `fleet doctor` for non-probing fleet inventory audits: duplicate endpoints/ports, missing required paths, stale PID state, and orphaned Capstan LaunchAgents.

## v0.22.0

- Add optional manifest `[fleet] enabled = false` dormant-lane metadata so registered manual/offline lanes stay visible in fleet snapshots while `fleet status`, `fleet health`, and `fleet recover` skip readiness, health, and start side effects.

## v0.21.1

- Fix `fleet health` aggregate status so warning-only latency rows remain top-level `warn` with `warnings`, not `critical` with `issues`.

## v0.21.0

- Add smoke latency telemetry with client wall time plus server prompt/prefill and completion/decode timings when an OpenAI-compatible backend exposes timing fields.
- Add health/fleet/daemon/service prompt and completion latency thresholds (`max_prompt_latency_sec`, `max_completion_latency_sec`) so slow prefill shows up as an operator warning instead of hiding behind a green smoke result.

## v0.20.1

- Fix remaining CLI help branding so `capstan --help` describes `version` as printing the Capstan version.

## v0.20.0

- Rename the public product/CLI to Capstan while preserving `modelctl` as a compatibility console script.
- Add a `capstan` Python package wrapper so `python -m capstan` and `python -m capstan.cli` work without moving the implementation package yet.
- Keep the distribution name `local-modelctl` for this release to avoid two packages owning the same compatibility modules; the future PyPI rename can ship with an explicit migration/deprecation path.
- Keep legacy manifest filenames, state paths, service labels, and `MODELCTL_*` environment variables stable to avoid live LaunchAgent drift.

## v0.19.0

- Add `modelctl promote --candidate TARGET.toml` as a plan-first promotion workflow around preflight, readiness-gated rotate, candidate post-promotion health/smoke gates, and rollback on failed post-health gates.
- Keep promotion side-effect safe by default; `--execute` is required to stop/start processes, and non-finite health-gate values are rejected.
- Add promotion regression coverage for plan-only behavior, shared stable-port preflight tolerance, candidate health defaults, and post-health rollback.

## v0.18.0

- Add manifest-owned `[health]` defaults for `health`, `daemon --health-mode`, and `service install` so launchd supervisors can carry the same safety gates as one-shot checks.
- Add `modelctl rotate --to TARGET.toml` for same-endpoint/same-model process rotation with readiness gating, rollback on failed target startup, atomic PID handoff, and PID ownership guards.
- Harden MLX manifest generation, ingest failure JSON, runner log descriptor cleanup, smoke expectation overrides, and IO latency health checks.

## v0.17.0

- Add `--jobs` to `modelctl fleet status`, `fleet health`, and dry-run `fleet recover` for bounded concurrent fleet operations.
- Parallelize `fleet health` scans while preserving registry output order and structured gate semantics.
- Add per-model and total `elapsed_sec` timing metadata to fleet status, health, and recover outputs.

## v0.16.0

- Add `modelctl fleet recover` to plan or execute safe starts for registered manifests that are down and have a `[start]` section.
- Fleet recovery is dry-run by default; `--execute --wait` is required for real recovery with readiness verification.
- Make bare `python -m unittest` run the real test suite instead of silently reporting zero tests.

## v0.15.0

- Add `modelctl service diff` to compare the installed LaunchAgent plist with desired manifest/service options.
- Service diff returns structured drift rows for plist keys and exits non-zero on drift, missing plist, or invalid plist.
- Support install-like desired flags plus `--content` for full desired/installed plist inspection; default diff preserves the installed Python executable unless `--python` is supplied.
- Treat zombie PIDs as dead so stale PID files do not block restart/doctor recovery.

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
