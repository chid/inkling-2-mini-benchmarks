#!/usr/bin/env bash
# ==============================================================================
# server-inkling-rocm.sh
# High-performance OpenAI-compatible REST server for Inkling 2 Mini (Native ROCm)
# ==============================================================================
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/inkling-2-mini-uncensored}"
MODEL="${MODEL:-$MODEL_DIR/Inkling-2-Mini-Uncensored-AD-IQ3_S.gguf}"
MMPROJ="${MMPROJ:-$MODEL_DIR/mmproj-Inkling-2-Mini-Uncensored-F16.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/bin/llama-b12626-rocm/llama-server}"

export LD_LIBRARY_PATH="/opt/rocm/core-10.1/lib:${LD_LIBRARY_PATH:-}"

PORT="${PORT:-8093}"
HOST="${HOST:-0.0.0.0}"
THREADS="${THREADS:-8}"
CTX_SIZE="${CTX_SIZE:-8192}"

if [ ! -f "$MODEL" ]; then
    echo "Error: Model file not found at $MODEL" >&2
    exit 1
fi

echo "Starting OpenAI-compatible server for Inkling 2 Mini on http://$HOST:$PORT ..."
exec "$LLAMA_SERVER" \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --device ROCm0 \
    --mmproj-device ROCm0 \
    --host "$HOST" \
    --port "$PORT" \
    -ngl 99 \
    -c "$CTX_SIZE" \
    -t "$THREADS" \
    -fa on \
    "$@"
