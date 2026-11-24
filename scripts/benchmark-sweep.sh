#!/usr/bin/env bash
# ==============================================================================
# benchmark-sweep.sh
# Automated parameter sweep across CPU threads and Flash Attention configurations
# ==============================================================================
set -euo pipefail

OUTPUT_MD="${OUTPUT_MD:-$HOME/BENCHMARKS_SWEEP.md}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/inkling-2-mini-uncensored}"
MODEL="${MODEL:-$MODEL_DIR/Inkling-2-Mini-Uncensored-AD-IQ3_S.gguf}"
LLAMA_BENCH="${LLAMA_BENCH:-$HOME/bin/llama-b12626-rocm/llama-bench}"

export LD_LIBRARY_PATH="/opt/rocm/core-10.1/lib:${LD_LIBRARY_PATH:-}"

echo "=== PRE-FLIGHT VRAM AUDIT ==="
VRAM_USED=$(cat /sys/class/drm/card*/device/mem_info_vram_used | head -n 1)
VRAM_TOTAL=$(cat /sys/class/drm/card*/device/mem_info_vram_total | head -n 1)
echo "Current VRAM Usage: $((VRAM_USED / 1024 / 1024)) MiB / $((VRAM_TOTAL / 1024 / 1024)) MiB"

echo "Executing benchmark sweep (Threads=4,6,8,12 | FA=on,off)..."
"$LLAMA_BENCH" \
    -m "$MODEL" \
    --device ROCm0 \
    -ngl 99 \
    -r 1 \
    -p 128 \
    -n 64 \
    -t 4,6,8,12 \
    -fa on,off \
    -o md > "$OUTPUT_MD"

echo "Sweep complete. Results written to $OUTPUT_MD"
cat "$OUTPUT_MD"
