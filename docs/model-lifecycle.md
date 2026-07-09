# Capstan local model lifecycle

Capstan is still using the legacy `modelctl.toml`, `MODELCTL_*`, and `~/.config/modelctl` names in v0.20 so existing manifests and LaunchAgents stay put.

A model is not "installed" until all of this is true:

1. Artifacts are identified and classified.
2. Runtime is known.
3. Start command is captured in a manifest.
4. Preflight passes.
5. Server reaches readiness.
6. Smoke test passes.
7. Cleanup candidates are documented.
8. Stop/restart works.
9. Optional service wrapper is installed and controllable.

## Artifact classes

- **Active model artifact**: do not delete unless retiring the model.
- **Secondary sidecar / cache lane**: active only for specific runtimes; document it.
- **Runtime cache**: usually safe to delete after stopping the server.
- **Download cache**: often safe if the real model dir exists elsewhere.
- **Old experiment**: unsafe until a human signs off.

## Promotion gates

Minimum useful gate:

```bash
# Create modelctl.toml one of three ways:
capstan init --template llama-cpp --model-id local-model --output modelctl.toml --overwrite

# Or, for MLX artifacts, inspect and promote the served overlay:
capstan mlx discover --root ~/.cache/mlx-models
capstan mlx inspect ~/.cache/mlx-models/my-qwen-model
capstan mlx overlay ~/.cache/mlx-models/my-qwen-model
capstan mlx manifest ~/.cache/mlx-models/my-qwen-model-served --id my-qwen-model-served --port 8123 --output modelctl.toml --overwrite

# Or, if the endpoint is already running:
capstan ingest --endpoint http://127.0.0.1:8080/v1 --output modelctl.toml --overwrite

capstan registry add --source modelctl.toml --name local-test
capstan registry use local-test --output modelctl.toml --overwrite
capstan -m modelctl.toml preflight
capstan -m modelctl.toml start --wait
capstan -m modelctl.toml smoke
capstan -m modelctl.toml soak --count 3
capstan -m modelctl.toml bench --preset tiny --output bench.md --format md
capstan -m modelctl.toml report --format md --output report.md
capstan -m modelctl.toml reports save --format json
capstan reports list
capstan fleet status
capstan fleet health
capstan fleet doctor
capstan fleet intake
capstan fleet recover             # dry-run recovery plan
capstan fleet recover --execute --wait
capstan -m modelctl.toml doctor --fix
capstan -m modelctl.toml health
capstan -m modelctl.toml daemon --iterations 1
capstan -m modelctl.toml service install --restart --interval 120 --dry-run
capstan -m modelctl.toml service install --restart --interval 120 --overwrite
capstan -m modelctl.toml service diff --restart --interval 120
capstan -m modelctl.toml service start
capstan -m modelctl.toml service status
capstan -m modelctl.toml rotate --to candidate.toml --readiness-timeout 300
capstan -m modelctl.toml watchdog --max-swap-gib 4 --duration 0
capstan -m modelctl.toml status
```

## Health checks

`capstan health` is the cheap green/red operator check. By default it checks PID state, readiness, and manifest-owned `[health]` defaults such as swap ceiling/delta, sample window, smoke, and latency gates.

For huge local lanes where macOS may retain stale swap, set delta sampling in `[health]` instead of repeating flags everywhere:

```toml
[health]
max_swap_gib = 128
max_swap_delta_gib = 1
sample_sec = 5
smoke = true
max_prompt_latency_sec = 60
max_completion_latency_sec = 10
```

CLI flags still override the manifest for one-off probes:

```bash
capstan -m modelctl.toml health --max-swap-delta-gib 1 --sample-sec 5
```

Add `--smoke` when you want endpoint behavior included, and `--max-latency-sec` when slow exact-output responses should fail the gate. For llama.cpp-style timing payloads, Capstan also records `latency.server_prompt_s` and `latency.server_completion_s`; use prompt/completion thresholds to catch slow prefill separately from decode:

```bash
capstan -m modelctl.toml health --smoke --max-latency-sec 30 --max-swap-delta-gib 1 --sample-sec 5
capstan -m modelctl.toml health --smoke --max-prompt-latency-sec 60 --max-completion-latency-sec 10
```

## Fleet status and health

Use `fleet status` first when you need to know what is actually alive:

```bash
capstan fleet status
```

It scans `$MODELCTL_REGISTRY` plus `~/.config/modelctl/models`, returns each lane as `ready`, `down`, `dormant`, or `invalid`, and includes PID/log paths, readiness detail, current swap, and whether the expected LaunchAgent plist exists. It is an operator snapshot, not a gate, so down/invalid/dormant rows still return machine-readable JSON with exit code 0.

For registered lanes you want visible but not probed or recovered — parked sidecars, manual bring-up experiments, old candidates — mark them dormant in the manifest:

```toml
[fleet]
enabled = false
reason = "manual bring-up only"
```

Dormant entries appear as `state = dormant` in `fleet status`, `status = skipped` in `fleet health`, and `planned_action = skip` in `fleet recover`. Capstan does not hit readiness/health endpoints or start commands for them.

Once manifests are registered, use `fleet health` as the cheap operator gate across the whole local lane set:

```bash
capstan fleet health
```

It scans `$MODELCTL_REGISTRY` plus `~/.config/modelctl/models`, runs the same structured `health` verdict for each enabled manifest, and exits non-zero if any active lane is critical/invalid/warn or if no registered lanes are found. Add `--smoke` only when you want to spend real endpoint calls across the fleet; prompt/completion latency thresholds work there too.

Use `fleet doctor` when you want inventory truth without hitting model endpoints:

```bash
capstan fleet doctor
```

It catches duplicate endpoints, duplicate endpoint/reserved ports, missing required paths, stale PID state, and orphaned Capstan LaunchAgents. It is intentionally not a liveness probe; use `fleet status` / `fleet health` for runtime state.

Use `fleet intake` when the box has live OpenAI-compatible endpoints that are not registered yet:

```bash
capstan fleet intake                         # dry-run: discover listeners and probe /v1/models
capstan fleet intake --port 8125             # explicit local port
capstan fleet intake --execute --output-dir ~/.config/modelctl/models
```

Intake skips endpoints already present in the registry, probes only `/v1/models`, and writes disabled manifests by default: `[fleet] enabled = false` with the endpoint port reserved in `[preflight].exclusive_ports`. It does not smoke, health-check, start, stop, or recover anything. Review the generated manifest before enabling a lane.

When the fleet is down and you want a controlled recovery path, dry-run first:

```bash
capstan fleet recover
capstan fleet recover --execute --wait
```

`fleet recover` only starts enabled registered manifests that are down and have a `[start]` section. It skips already-ready, dormant, invalid, and inspect-only manifests. No side effects happen unless `--execute` is passed, and executed recovery requires `--wait` so startup readiness is verified before returning green.

## macOS service wrapper

`capstan service install` writes a LaunchAgent plist for the manifest. In v0.20 the plist still invokes the compatibility `modelctl.cli daemon` module so existing service diffs stay stable; the model server itself is not launched directly.

Use `--dry-run` first. It prints the plist path and daemon arguments without touching `~/Library/LaunchAgents`:

```bash
capstan -m modelctl.toml service install --restart --interval 120 --dry-run
```

Then install and control it:

```bash
capstan -m modelctl.toml service install --restart --interval 120 --overwrite
capstan -m modelctl.toml service start
capstan -m modelctl.toml service status
capstan -m modelctl.toml service restart
capstan -m modelctl.toml service stop
capstan -m modelctl.toml service uninstall
```

Use `service diff` whenever you change a manifest or desired daemon flags. It renders the desired LaunchAgent exactly like `service install`, reads the installed plist, preserves the installed Python executable unless `--python` is supplied, and exits non-zero if ProgramArguments, logs, environment, KeepAlive, RunAtLoad, or other plist keys drifted:

```bash
capstan -m modelctl.toml service diff --restart --interval 120
```

`--restart` is explicit because it lets the daemon stop/start the model on readiness or swap breach. No sneaky self-healing time bombs.

For huge macOS model lanes, prefer manifest `[health]` delta checks so stale absolute swap does not trigger a pointless restart loop. Keep `max_swap_gib` as the emergency ceiling.

## Readiness-gated rotation

Use `rotate` when replacing the process behind a stable lane without manual stop/start roulette:

```bash
capstan -m active.toml rotate --to candidate.toml --readiness-timeout 300
```

The target manifest must preserve the current manifest's `[model].endpoint` and `model_id`; it is for rotating the runtime behind a stable lane, not swapping client-facing identities. The sequence is deliberately boring: stop the current process, start the target, wait for the target readiness gate, then atomically move the target PID state into the current manifest's PID path. If target readiness fails, `rotate` stops the target and restarts the current manifest unless `--no-rollback` is set. This is the SSD-lane rotation path; no claiming victory until readiness is green.

For tests or custom service roots, set `MODELCTL_LAUNCHD_DIR`.

For bigger models, add a soak outside this CLI for now:

- exact JSON x5
- normal chat x3
- long prompt x1
- repeated prefix x2
- swap sampling before/after
