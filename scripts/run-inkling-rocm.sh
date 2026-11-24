#!/usr/bin/env bash
# ==============================================================================
# run-inkling-rocm.sh
# Interactive CLI runner for Inkling 2 Mini on AMD Radeon GPUs via Native ROCm/HIP
# ==============================================================================
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/inkling-2-mini-uncensored}"
MODEL="${MODEL:-$MODEL_DIR/Inkling-2-Mini-Uncensored-AD-IQ3_S.gguf}"
MMPROJ="${MMPROJ:-$MODEL_DIR/mmproj-Inkling-2-Mini-Uncensored-F16.gguf}"
LLAMA_CLI="${LLAMA_CLI:-$HOME/bin/llama-b12626-rocm/llama-cli}"

export LD_LIBRARY_PATH="/opt/rocm/core-10.1/lib:${LD_LIBRARY_PATH:-}"

# Recommended defaults based on benchmark sweeps
THREADS="${THREADS:-8}"
CTX_SIZE="${CTX_SIZE:-8192}"

if [ ! -f "$MODEL" ]; then
    echo "Error: Model file not found at $MODEL" >&2
    exit 1
fi

echo "Launching Inkling 2 Mini on ROCm0 (Threads=$THREADS, FA=on, CTX=$CTX_SIZE)..."

exec "$LLAMA_CLI" \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --device ROCm0 \
    --mmproj-device ROCm0 \
    -ngl 99 \
    -c "$CTX_SIZE" \
    -t "$THREADS" \
    -fa on \
    "$@"
