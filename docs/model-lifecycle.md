# Local model lifecycle

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
modelctl init --template llama-cpp --model-id local-model --output modelctl.toml --overwrite

# Or, for MLX artifacts, inspect and promote the served overlay:
modelctl mlx discover --root ~/.cache/mlx-models
modelctl mlx inspect ~/.cache/mlx-models/my-qwen-model
modelctl mlx overlay ~/.cache/mlx-models/my-qwen-model
modelctl mlx manifest ~/.cache/mlx-models/my-qwen-model-served --id my-qwen-model-served --port 8123 --output modelctl.toml --overwrite

# Or, if the endpoint is already running:
modelctl ingest --endpoint http://127.0.0.1:8080/v1 --output modelctl.toml --overwrite

modelctl registry add --source modelctl.toml --name local-test
modelctl registry use local-test --output modelctl.toml --overwrite
modelctl -m modelctl.toml preflight
modelctl -m modelctl.toml start --wait
modelctl -m modelctl.toml smoke
modelctl -m modelctl.toml soak --count 3
modelctl -m modelctl.toml bench --preset tiny --output bench.md --format md
modelctl -m modelctl.toml report --format md --output report.md
modelctl -m modelctl.toml reports save --format json
modelctl reports list
modelctl fleet status
modelctl fleet health --max-swap-delta-gib 1 --sample-sec 5
modelctl -m modelctl.toml doctor --fix
modelctl -m modelctl.toml health --max-swap-delta-gib 1 --sample-sec 5
modelctl -m modelctl.toml daemon --health-mode --iterations 1 --max-swap-gib 48 --max-swap-delta-gib 1 --sample-sec 5
modelctl -m modelctl.toml service install --restart --health-mode --max-swap-gib 48 --max-swap-delta-gib 1 --sample-sec 5 --interval 120 --dry-run
modelctl -m modelctl.toml service install --restart --health-mode --max-swap-gib 48 --max-swap-delta-gib 1 --sample-sec 5 --interval 120 --overwrite
modelctl -m modelctl.toml service start
modelctl -m modelctl.toml service status
modelctl -m modelctl.toml watchdog --max-swap-gib 4 --duration 0
modelctl -m modelctl.toml status
```

## Health checks

`modelctl health` is the cheap green/red operator check. By default it checks PID state, readiness, and the manifest's swap ceiling. For huge local lanes where macOS may retain stale swap, use delta sampling instead of pretending absolute swap tells the whole story:

```bash
modelctl -m modelctl.toml health --max-swap-delta-gib 1 --sample-sec 5
```

Add `--smoke` when you want endpoint behavior included, and `--max-latency-sec` when slow exact-output responses should fail the gate:

```bash
modelctl -m modelctl.toml health --smoke --max-latency-sec 30 --max-swap-delta-gib 1 --sample-sec 5
```

## Fleet status and health

Use `fleet status` first when you need to know what is actually alive:

```bash
modelctl fleet status
```

It scans `$MODELCTL_REGISTRY` plus `~/.config/modelctl/models`, returns each lane as `ready`, `down`, or `invalid`, and includes PID/log paths, readiness detail, current swap, and whether the expected LaunchAgent plist exists. It is an operator snapshot, not a gate, so down/invalid rows still return machine-readable JSON with exit code 0.

Once manifests are registered, use `fleet health` as the cheap operator gate across the whole local lane set:

```bash
modelctl fleet health --max-swap-delta-gib 1 --sample-sec 5
```

It scans `$MODELCTL_REGISTRY` plus `~/.config/modelctl/models`, runs the same structured `health` verdict for each manifest, and exits non-zero if any lane is critical/invalid or if no registered lanes are found. Add `--smoke` only when you want to spend real endpoint calls across the fleet.

## macOS service wrapper

`modelctl service install` writes a LaunchAgent plist for the manifest. The plist runs `modelctl daemon`, not the model server directly; that keeps the control plane in one place.

Use `--dry-run` first. It prints the plist path and daemon arguments without touching `~/Library/LaunchAgents`:

```bash
modelctl -m modelctl.toml service install --restart --health-mode --max-swap-gib 48 --max-swap-delta-gib 1 --sample-sec 5 --interval 120 --dry-run
```

Then install and control it:

```bash
modelctl -m modelctl.toml service install --restart --health-mode --max-swap-gib 48 --max-swap-delta-gib 1 --sample-sec 5 --interval 120 --overwrite
modelctl -m modelctl.toml service start
modelctl -m modelctl.toml service status
modelctl -m modelctl.toml service restart
modelctl -m modelctl.toml service stop
modelctl -m modelctl.toml service uninstall
```

`--restart` is explicit because it lets the daemon stop/start the model on readiness or swap breach. No sneaky self-healing time bombs.

For huge macOS model lanes, prefer `--health-mode --max-swap-delta-gib ...` so stale absolute swap does not trigger a pointless restart loop. Keep `--max-swap-gib` as the emergency ceiling.

For tests or custom service roots, set `MODELCTL_LAUNCHD_DIR`.

For bigger models, add a soak outside this CLI for now:

- exact JSON x5
- normal chat x3
- long prompt x1
- repeated prefix x2
- swap sampling before/after
