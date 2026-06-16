# modelctl

`modelctl` is a small manifest-driven CLI for running local LLM servers without turning your workstation into a haunted swap machine.

It is built for messy real local inference work: `llama.cpp`, MLX/oMLX, custom model forks, external SSD model warehouses, sidecars, launch scripts, and "wait, which 140GB directory is live?" cleanup passes.

## Install

```bash
python3.11 -m pip install \
  https://github.com/phippsbot-byte/modelctl/releases/download/v0.14.0/local_modelctl-0.14.0-py3-none-any.whl
```

For local development:

```bash
python3.11 -m pip install -e .
```

Or run module-style:

```bash
python3.11 -m modelctl.cli --help
```

## Quickstart

```bash
modelctl version

# Option A: generate a starter manifest.
modelctl init --template llama-cpp --model-id local-model --output modelctl.toml
$EDITOR modelctl.toml

# Option B: build from an MLX artifact.
modelctl mlx discover --root ~/.cache/mlx-models
modelctl mlx inspect ~/.cache/mlx-models/my-qwen-model
modelctl mlx overlay ~/.cache/mlx-models/my-qwen-model
modelctl mlx manifest ~/.cache/mlx-models/my-qwen-model-served --id my-qwen-model-served --port 8123 --output modelctl.toml --overwrite

# Option C: start from an example.
cp examples/mlx-lm.example.toml modelctl.toml       # MLX
# cp examples/llama-cpp.example.toml modelctl.toml  # llama.cpp
$EDITOR modelctl.toml

# Option D: ingest a running OpenAI-compatible endpoint.
modelctl ingest --endpoint http://127.0.0.1:8080/v1 --output modelctl.toml --overwrite

modelctl --pretty validate
modelctl registry add --source modelctl.toml --name my-model
modelctl registry use my-model --output modelctl.toml --overwrite
modelctl registry list
modelctl preflight
modelctl start --wait
modelctl smoke
modelctl soak --count 3
modelctl bench --preset tiny --output bench.md --format md
modelctl report --format md --output report.md
modelctl reports save --format json
modelctl reports list
modelctl fleet status
modelctl fleet health --max-swap-delta-gib 1 --sample-sec 5
modelctl doctor --fix
modelctl health --max-swap-delta-gib 1 --sample-sec 5
modelctl daemon --health-mode --iterations 1 --max-swap-gib 48 --max-swap-delta-gib 1 --sample-sec 5
modelctl service install --restart --health-mode --max-swap-gib 48 --max-swap-delta-gib 1 --sample-sec 5 --interval 120 --dry-run
modelctl service install --restart --health-mode --max-swap-gib 48 --max-swap-delta-gib 1 --sample-sec 5 --interval 120 --overwrite
modelctl service start
modelctl service status
modelctl watchdog --max-swap-gib 4 --duration 0
modelctl status
modelctl cleanup          # dry-run
modelctl stop
```

## Manifest shape

```toml
[model]
id = "deepseek-v4-flash-ssd"
model_id = "deepseek-v4-flash-ssd-4096"
endpoint = "http://127.0.0.1:8127/v1"
description = "Custom SSD-streaming DeepSeek V4 Flash lane"

[start]
command = ["bash", "-lc", "cd $HOME/LLM/ssd-streaming && exec ./run-dsv4-flash-ssd-server-candidate.sh"]
cwd = "$HOME/LLM/ssd-streaming"
log_path = "$HOME/.local/state/modelctl/deepseek-v4-flash-ssd.log"
pid_path = "$HOME/.local/state/modelctl/deepseek-v4-flash-ssd.pid.json"
startup_timeout_sec = 300
readiness_url = "http://127.0.0.1:8127/v1/models"
readiness_contains = "deepseek-v4-flash-ssd-4096"

[start.env]
DSV4_FLASH_ALIAS = "deepseek-v4-flash-ssd-4096"
DSV4_FLASH_PORT = "8127"
DSV4_FLASH_THREADS = "12"
DSV4_FLASH_THREADS_BATCH = "12"
DSV4_FLASH_THREADS_HTTP = "4"

[preflight]
required_paths = ["$HOME/LLM/ssd-streaming/run-dsv4-flash-ssd-server-candidate.sh"]
exclusive_ports = [8127]
max_swap_gib = 4

[[preflight.disk]]
path = "$HOME"
min_free_gib = 50

[smoke]
prompt = "Return exactly this JSON and nothing else: {\"ok\":true}"
expect = "{\"ok\":true}"
max_tokens = 96
temperature = 0

[[cleanup]]
path = "$HOME/Library/Caches/some-model-kv"
description = "Runtime KV cache; safe to recreate."
safe = true
```

## Commands

- `version` — print installed modelctl version.
- `init --template minimal|llama-cpp --output modelctl.toml` — generate a starter manifest.
- `validate` — parse manifest and print resolved summary; use global `--pretty` for human output.
- `ingest --endpoint URL --output modelctl.toml` — generate a starter manifest from a running `/v1/models` endpoint.
- `mlx discover --root ~/.cache/mlx-models` — find local MLX model directories.
- `mlx inspect PATH` — inspect config/chat template and flag serving hazards like Qwen/Qwopus `<think>` preambles.
- `mlx overlay PATH` — create a reversible sibling `-served` overlay that symlinks weights and patches only `chat_template.jinja`.
- `mlx manifest PATH --id NAME --port N --output modelctl.toml` — generate an MLX-focused manifest using `python -m mlx_lm server`; stock MLX request model defaults to `default_model`.
- `list` — convenience alias for `registry list`; scans `$MODELCTL_REGISTRY` plus `~/.config/modelctl/models`.
- `registry add/list/show/remove/use` — manage durable manifest registry entries and materialize a registered manifest into a workspace.
- `fleet status [--registry DIR]` — show the operator snapshot across registered manifests: ready/down/invalid state, PID/log paths, readiness, swap, and LaunchAgent plist presence.
- `fleet health [--registry DIR] [--smoke]` — run the structured health verdict across all registered manifests and fail if any lane is critical/invalid.
- `preflight` — check paths, exclusive ports, disk floor, and swap ceiling.
- `start --wait` — start server in its own process group, write PID state, optionally wait for readiness.
- `wait` — wait for readiness URL/model string.
- `status` — print PID/readiness/log/swap state.
- `health [--max-swap-delta-gib N] [--smoke]` — one high-signal health verdict for PID, readiness, swap ceiling/delta, and optional smoke latency.
- `doctor --fix` — run diagnostics and apply safe local repairs like stale PID-state removal and state-dir creation.
- `report --format md --output report.md` — write JSON/Markdown model state reports.
- `reports save/list/show` — keep/query saved report history under the modelctl state directory.
- `smoke` — run OpenAI-compatible `/chat/completions` exact-output smoke.
- `soak --count N` — run repeated smoke tests with timing and swap sampling.
- `bench --preset tiny|small|standard --output bench.md --format md` — run synthetic prompt-size benchmarks and write artifacts.
- `watchdog --max-swap-gib N` — sample readiness/swap and optionally stop the manifest process on breach.
- `daemon --health-mode --max-swap-delta-gib N [--restart]` — run a foreground supervisor loop using structured health verdicts; restart is explicit only.
- `daemon --max-swap-gib N [--restart]` — legacy watchdog-style supervisor loop.
- `service install [--restart] [--health-mode] [--dry-run]` — write a macOS LaunchAgent plist that runs `modelctl daemon` for this manifest.
- `service start/stop/restart/status/uninstall [--dry-run]` — control the LaunchAgent with `launchctl`; dry-run prints the exact commands.
- `cleanup` — dry-run cleanup candidates.
- `cleanup --execute` — delete only candidates marked `safe = true`.
- `cleanup --execute --force` — delete unsafe candidates too. Sharp knife; don't juggle it.
- `stop` — terminate the process group from the PID state file.

## Design rules

- Generic lifecycle stays manifest-driven; substrate-specific helpers can generate better manifests.
- MLX/Qwen chat-template fixes must be reversible overlays, not source artifact mutation.
- Manifests are the source of truth.
- Cleanup is dry-run first.
- Start/stop must be reproducible.
- launchd service install should be previewable with `--dry-run`; no invisible plist surgery.
- Every promoted model needs a smoke test.
- If a service manager wedges, make that visible instead of pretending the model is bad.

## Current limitations

- Alpha CLI.
- OpenAI-compatible smoke only for now.
- Process supervision is simple PID/process-group management plus optional macOS `launchd` wrapper, not a cross-platform service manager.
- TOML only; intentionally zero runtime dependencies.

## License

MIT
