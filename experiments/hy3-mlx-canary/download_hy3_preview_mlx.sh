#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/Volumes/ModelSSD/.cache/huggingface
export HF_HUB_DISABLE_TELEMETRY=1

REPO="mlx-community/Hy3-preview-4bit"
DEST="/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX"
LOG_DIR="/Volumes/ModelSSD/logs/hy3-mlx-canary"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/download-$STAMP.log"

mkdir -p "$DEST" "$LOG_DIR" "$HF_HOME"

{
  echo "== Hy3 preview MLX download =="
  date
  echo "repo=$REPO"
  echo "dest=$DEST"
  echo "hf_home=$HF_HOME"
  df -h /Volumes/ModelSSD
  echo
  /opt/homebrew/bin/hf download "$REPO" \
    --repo-type model \
    --local-dir "$DEST" \
    --max-workers 8
  echo
  echo "== verify =="
  date
  /opt/homebrew/bin/python3.11 - <<'PY'
from pathlib import Path
import json, os
p=Path('/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX')
shards=sorted(p.glob('model-*.safetensors'))
size=sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
print('path', p)
print('safetensors', len(shards))
print('total_bytes', size)
print('total_gib', round(size/2**30, 2))
assert (p/'config.json').exists(), 'missing config.json'
assert (p/'tokenizer_config.json').exists(), 'missing tokenizer_config.json'
assert (p/'chat_template.jinja').exists(), 'missing chat_template.jinja'
assert len(shards) == 34, f'expected 34 shards, got {len(shards)}'
cfg=json.loads((p/'config.json').read_text())
print('model_type', cfg.get('model_type'))
print('layers', cfg.get('num_hidden_layers'), 'experts', cfg.get('num_experts'), 'topk', cfg.get('num_experts_per_tok'))
PY
  df -h /Volumes/ModelSSD
  echo "DONE"
} 2>&1 | tee "$LOG"
