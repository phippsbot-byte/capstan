#!/usr/bin/env bash
set -euo pipefail

MODEL="/Volumes/ModelSSD/Models/Hy3-preview-4bit-MLX-served"
PORT="${HY3_PORT:-8133}"
LOG_DIR="/Volumes/ModelSSD/logs/hy3-mlx-canary"
LOG="$LOG_DIR/server-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$LOG_DIR"

if [ ! -f "$MODEL/config.json" ]; then
  echo "missing overlay: $MODEL/config.json" >&2
  exit 2
fi

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port $PORT already in use" >&2
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2 || true
  exit 3
fi

echo "== Hy3 MLX canary server ==" | tee "$LOG"
date | tee -a "$LOG"
echo "model=$MODEL" | tee -a "$LOG"
echo "port=$PORT" | tee -a "$LOG"
sysctl vm.swapusage | tee -a "$LOG"
df -h /Volumes/ModelSSD | tee -a "$LOG"

exec /opt/homebrew/bin/python3.11 -m mlx_lm server \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-tokens 96 \
  --temp 0 \
  --top-p 1 \
  --decode-concurrency 1 \
  --prompt-concurrency 1 \
  --prefill-step-size 128 \
  --prompt-cache-size 1 \
  --prompt-cache-bytes 1073741824 \
  --chat-template-args '{"reasoning_effort":"no_think"}' \
  --log-level INFO \
  2>&1 | tee -a "$LOG"
