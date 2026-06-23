# Capstan

`capstan` is a small manifest-driven CLI for hauling giant local LLM servers under control without turning your workstation into a haunted swap machine.

The legacy `modelctl` console script remains available during the rename. State paths, manifest filenames, and `MODELCTL_*` environment variables intentionally stay stable in v0.20 so existing services do not drift.

It is built for messy real local inference work: `llama.cpp`, MLX/oMLX, custom model forks, external SSD model warehouses, sidecars, launch scripts, and "wait, which 140GB directory is live?" cleanup passes.

## Install

```bash
python3.11 -m pip install \
  https://github.com/phippsbot-byte/capstan/releases/download/v0.24.2/local_modelctl-0.24.2-py3-none-any.whl
```

For local development:

```bash
python3.11 -m pip install -e .
```

Or run module-style:

```bash
python3.11 -m capstan --help
# or
python3.11 -m capstan.cli --help
```

## Quickstart

```bash
capstan version

# Option A: generate a starter manifest.
capstan init --template llama-cpp --model-id local-model --output modelctl.toml
$EDITOR modelctl.toml

# Option B: build from an MLX artifact.
capstan mlx discover --root ~/.cache/mlx-models
capstan mlx inspect ~/.cache/mlx-models/my-qwen-model
capstan mlx overlay ~/.cache/mlx-models/my-qwen-model
capstan mlx manifest ~/.cache/mlx-models/my-qwen-model-served --id my-qwen-model-served --port 8123 --output modelctl.toml --overwrite

# Option C: start from an example.
cp examples/mlx-lm.example.toml modelctl.toml       # MLX
# cp examples/llama-cpp.example.toml modelctl.toml  # llama.cpp
$EDITOR modelctl.toml

# Option D: ingest a running OpenAI-compatible endpoint.
capstan ingest --endpoint http://127.0.0.1:8080/v1 --output modelctl.toml --overwrite

capstan --pretty validate
capstan registry add --source modelctl.toml --name my-model
capstan registry use my-model --output modelctl.toml --overwrite
capstan registry list
capstan preflight
capstan start --wait
capstan smoke
capstan soak --count 3
capstan bench --preset tiny --output bench.md --format md
capstan report --format md --output report.md
capstan reports save --format json
capstan reports list
capstan fleet status
capstan fleet health
capstan fleet doctor              # inventory drift, no endpoint probes
capstan fleet intake              # discover live endpoints, draft dormant manifests
capstan fleet recover             # dry-run recovery plan
capstan fleet recover --execute --wait
capstan doctor --fix
capstan health
capstan daemon --iterations 1
capstan service install --restart --interval 120 --dry-run
capstan service install --restart --interval 120 --overwrite
capstan service diff --restart --interval 120
capstan service start
capstan service status
capstan rotate --to candidate.toml --readiness-timeout 300
capstan promote --candidate candidate.toml          # plan only
capstan promote --candidate candidate.toml --execute --smoke --readiness-timeout 300
capstan watchdog --max-swap-gib 4 --duration 0
capstan status
capstan cleanup          # dry-run
capstan stop
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
max_swap_gib = 128

[health]
max_swap_gib = 128
max_swap_delta_gib = 1
sample_sec = 5
smoke = true
max_latency_sec = 180
max_prompt_latency_sec = 60
max_completion_latency_sec = 10
max_io_latency_sec = 25

[fleet]
enabled = true
# Set enabled=false for registered dormant/manual lanes. Fleet commands keep
# the entry visible but skip readiness, health, recover, and start side effects.
reason = ""

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

- `version` — print installed capstan version.
- `init --template minimal|llama-cpp --output modelctl.toml` — generate a starter manifest.
- `validate` — parse manifest and print resolved summary; use global `--pretty` for human output.
- `ingest --endpoint URL --output modelctl.toml` — generate a starter manifest from a running `/v1/models` endpoint.
- `mlx discover --root ~/.cache/mlx-models` — find local MLX model directories.
- `mlx inspect PATH` — inspect config/chat template and flag serving hazards like Qwen/Qwopus `<think>` preambles.
- `mlx overlay PATH` — create a reversible sibling `-served` overlay that symlinks weights and patches only `chat_template.jinja`.
- `mlx manifest PATH --id NAME --port N --output modelctl.toml` — generate an MLX-focused manifest using `python -m mlx_lm server`; stock MLX request model defaults to `default_model`.
- `list` — convenience alias for `registry list`; scans `$MODELCTL_REGISTRY` plus `~/.config/modelctl/models`.
- `registry add/list/show/remove/use` — manage durable manifest registry entries and materialize a registered manifest into a workspace.
- `fleet status [--registry DIR] [--jobs N]` — show the operator snapshot across registered manifests: ready/down/dormant/invalid state, PID/log paths, readiness, swap, and LaunchAgent plist presence.
- `fleet health [--registry DIR] [--jobs N] [--smoke]` — run the structured health verdict across enabled registered manifests and fail if any active lane is critical/invalid/warn; `[fleet] enabled=false` entries are reported as skipped.
- `fleet doctor [--registry DIR]` — audit registry inventory without endpoint probes: duplicate endpoints, duplicate endpoint/reserved ports, missing required paths, stale PID state, and orphaned Capstan LaunchAgents.
- `fleet intake [--port N|--endpoint URL] [--execute]` — discover live OpenAI-compatible endpoints via `/v1/models`, skip already-registered endpoints, and draft disabled `[fleet] enabled=false` manifests with reserved ports. Dry-run unless `--execute` is passed.
- `fleet recover [--registry DIR] [--jobs N] [--execute] [--wait]` — plan safe starts for down enabled registered manifests with `[start]`; dry-run is parallel-capable, dormant entries are skipped, and real `--execute --wait` recovery stays serial.
- `preflight` — check paths, exclusive ports, disk floor, and swap ceiling.
- `start --wait` — start server in its own process group, write PID state, optionally wait for readiness.
- `rotate --to TARGET.toml` — stop the current manifest process, start a same-endpoint/same-model target, verify readiness, then atomically move target PID ownership to the current manifest PID path; failed target readiness rolls back unless `--no-rollback` is set.
- `promote --candidate TARGET.toml [--execute]` — plan or execute a full promotion: current/candidate preflight, rotate dry-run, readiness-gated rotate, post-promotion health, and rollback if the post-health gate fails.
- `wait` — wait for readiness URL/model string.
- `status` — print PID/readiness/log/swap state.
- `health [--max-swap-delta-gib N] [--smoke]` — one high-signal health verdict for PID, readiness, swap ceiling/delta, optional smoke latency, and manifest `[health]` defaults. OpenAI-style server timings are surfaced as `latency.server_prompt_s` / `latency.server_completion_s`; use `max_prompt_latency_sec` and `max_completion_latency_sec` to warn on slow prefill/decode without failing basic liveness.
- `doctor --fix` — run diagnostics and apply safe local repairs like stale PID-state removal and state-dir creation.
- `report --format md --output report.md` — write JSON/Markdown model state reports.
- `reports save/list/show` — keep/query saved report history under the compatibility `modelctl` state directory.
- `smoke` — run OpenAI-compatible `/chat/completions` exact-output smoke.
- `soak --count N` — run repeated smoke tests with timing and swap sampling.
- `bench --preset tiny|small|standard --output bench.md --format md` — run synthetic prompt-size benchmarks and write artifacts.
- `watchdog --max-swap-gib N` — sample readiness/swap and optionally stop the manifest process on breach.
- `daemon --health-mode --max-swap-delta-gib N [--restart]` — run a foreground supervisor loop using structured health verdicts; restart is explicit only.
- `daemon --max-swap-gib N [--restart]` — legacy watchdog-style supervisor loop.
- `service install [--restart] [--health-mode] [--dry-run]` — write a macOS LaunchAgent plist that runs the Capstan daemon via the compatibility module for this manifest.
- `service diff [install-like flags]` — compare the installed LaunchAgent plist to the desired manifest/service options and fail on drift or missing plist.
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
- Start/stop/rotate must be reproducible and readiness-gated.
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
