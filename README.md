# modelctl

`modelctl` is a small manifest-driven CLI for running local LLM servers without turning your workstation into a haunted swap machine.

It is built for messy real local inference work: `llama.cpp`, MLX/oMLX, custom model forks, external SSD model warehouses, sidecars, launch scripts, and "wait, which 140GB directory is live?" cleanup passes.

## Install

```bash
python3.11 -m pip install \
  https://github.com/phippsbot-byte/modelctl/releases/download/v0.7.0/local_modelctl-0.7.0-py3-none-any.whl
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

# Option B: start from an example.
cp examples/llama-cpp.example.toml modelctl.toml
$EDITOR modelctl.toml

# Option C: ingest a running OpenAI-compatible endpoint.
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
modelctl doctor --fix
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
- `list` — convenience alias for `registry list`; scans `$MODELCTL_REGISTRY` plus `~/.config/modelctl/models`.
- `registry add/list/show/remove/use` — manage durable manifest registry entries and materialize a registered manifest into a workspace.
- `preflight` — check paths, exclusive ports, disk floor, and swap ceiling.
- `start --wait` — start server in its own process group, write PID state, optionally wait for readiness.
- `wait` — wait for readiness URL/model string.
- `status` — print PID/readiness/log/swap state.
- `doctor --fix` — run diagnostics and apply safe local repairs like stale PID-state removal and state-dir creation.
- `report --format md --output report.md` — write JSON/Markdown model state reports.
- `reports save/list/show` — keep/query saved report history under the modelctl state directory.
- `smoke` — run OpenAI-compatible `/chat/completions` exact-output smoke.
- `soak --count N` — run repeated smoke tests with timing and swap sampling.
- `bench --preset tiny|small|standard --output bench.md --format md` — run synthetic prompt-size benchmarks and write artifacts.
- `watchdog --max-swap-gib N` — sample readiness/swap and optionally stop the manifest process on breach.
- `cleanup` — dry-run cleanup candidates.
- `cleanup --execute` — delete only candidates marked `safe = true`.
- `cleanup --execute --force` — delete unsafe candidates too. Sharp knife; don't juggle it.
- `stop` — terminate the process group from the PID state file.

## Design rules

- No model-specific code in the CLI.
- Manifests are the source of truth.
- Cleanup is dry-run first.
- Start/stop must be reproducible.
- Every promoted model needs a smoke test.
- If a service manager wedges, make that visible instead of pretending the model is bad.

## Current limitations

- Alpha CLI.
- OpenAI-compatible smoke only for now.
- Process supervision is simple PID/process-group management, not a full daemon supervisor.
- TOML only; intentionally zero runtime dependencies.

## License

MIT
