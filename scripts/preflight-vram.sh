#!/usr/bin/env bash
# ==============================================================================
# preflight-vram.sh
# Validates dedicated VRAM availability and handles process isolation safely
# ==============================================================================
set -euo pipefail

CARD="${1:-card1}"
SYSFS_DIR="/sys/class/drm/$CARD/device"

if [ ! -d "$SYSFS_DIR" ]; then
    echo "Error: Device $CARD not found in /sys/class/drm/" >&2
    exit 1
fi

VRAM_USED=$(cat "$SYSFS_DIR/mem_info_vram_used")
VRAM_TOTAL=$(cat "$SYSFS_DIR/mem_info_vram_total")
USED_MIB=$((VRAM_USED / 1024 / 1024))
TOTAL_MIB=$((VRAM_TOTAL / 1024 / 1024))
FREE_MIB=$((TOTAL_MIB - USED_MIB))

echo "=== GPU Memory Audit for $CARD ==="
echo "Total VRAM : $TOTAL_MIB MiB"
echo "Used VRAM  : $USED_MIB MiB"
echo "Free VRAM  : $FREE_MIB MiB"

# Check active GPU consumers
echo ""
echo "=== Active Kernel & Compute Consumers ==="
fuser -v /dev/kfd /dev/dri/render* 2>/dev/null || echo "No active compute clients."

if [ "$FREE_MIB" -lt 14000 ]; then
    echo ""
    echo "Warning: Free VRAM ($FREE_MIB MiB) is less than required for unquantized IQ3_S (13,340 MiB)."
    echo "Inference may silently spill to host DDR memory over PCIe GTT."
    exit 2
else
    echo ""
    echo "Status: Sufficient dedicated VRAM available for zero-spill execution."
fi
