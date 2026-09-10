# Qwen4 selected-expert cache profiles

These paired manifests are experimental, guarded opt-ins for the versioned
oMLX Qwen4 cache runtime. They are not a default, a fleet activation, a model
promotion, or launch authority. Both bind only `127.0.0.1:18144`, use one
concurrent request, and invoke `-m omlx.cli serve` (not `mlx_lm`).

`omlx-qwen4-ssd-expert-cache-on` sets
`OMLX_QWEN4_SSD_EXPERT_CACHE_BYTES=1073741824`. Its matching `-off` manifest
sets the same inherited environment explicitly to `0`; it is a fresh-process
rollback because the runtime reads this budget during initialization. Both
retain layers `0,1,46,47`, dedupe `1`, and staged reclaim `1`.

## Required preparation and limits

Set reusable variables before parsing a template:

- `CAPSTAN_OMLX_RUNTIME_ROOT` and `CAPSTAN_OMLX_PYTHON` select the new runtime;
  `PYTHONPATH` is set to that runtime root.
- `CAPSTAN_QWEN4_BASE_PATH` is an isolated oMLX base directory, with
  `CAPSTAN_QWEN4_MODEL_DIR` as its existing `base/models` symlink and
  `CAPSTAN_QWEN4_MODEL_SETTINGS_PATH` as its existing `base/model_settings.json`.
- The per-model settings must explicitly set
  `model_alias="qwen4-ssd-expert-cache-experimental"`, matching the manifests'
  `model_id`. The parent must verify that API alias resolves to the bound
  artifact; it is not a substitute for artifact identity.
  `CAPSTAN_QWEN4_STATE_DIR` and the two evidence-output paths are
  controller-allocated.

Read the selected runtime's `omlx/settings.py` and `omlx/model_settings.py`
before preparation: `--base-path` resolves `base/models`, while per-model
records live in `base/model_settings.json`. The required existing per-model
record must show Qwen4 PLE SSD offload, MTP and VLM MTP disabled, and the exact
`max_context_window=8192` / `max_tokens=1024` limits. Templates only require
these paths; they do not create the symlink, stage settings, auto-start, or
prove artifact/source admission.

Keep `preserve_thinking=true` in the prepared runtime settings. The retained
8K/1024 systems workload disabled thinking per request using
`chat_template_kwargs={"enable_thinking": false}`; it did not qualify a
native-max ability lane. Do not silently change model settings to reproduce it.

`OMLX_QWEN4_SSD_PROFILE_PATH` is a fresh absolute profile JSONL output: the
runtime creates it exclusively. `OMLX_QWEN4_LOAD_TELEMETRY` is likewise a
controller-provisioned writable telemetry sink. Do not reuse those paths across
attempts.

## Admission and rollback

The parent controller must first perform its zero-model, source/artifact
binding, startup, and live gates. The reproducible runtime patch is at
`integrations/qwen4-ssd-expert-cache-v3/runtime.patch`; the adjacent
`receipt.json` records its base, patch, selected-source hashes and admission
status. A packaged patch is not launch authority. Do not manufacture a signed
receipt, and do not treat systems-only evidence as model-promotion approval.

`capstan -m examples/omlx-qwen4-ssd-expert-cache-on.example.toml validate`
is only a manifest syntax/resolution check, not proof of launch eligibility.
There is intentionally no blanket `capstan start` instruction here. A rollback
is the controller-owned stop/restore transaction followed by one fresh off
process. It must restore the original service under the existing lifecycle and
receipt APIs, never run the on and off manifests as overlapping servers.

No cold-SSD or consumer-device speed claim is made by this diagnostic slice.
