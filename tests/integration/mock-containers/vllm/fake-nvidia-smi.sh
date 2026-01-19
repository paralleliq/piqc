#!/bin/bash
# Fake nvidia-smi script for integration testing
#
# Returns realistic GPU metrics for testing the GPU collector.
# Supports the query format used by ModelSpec Introspector.

# Parse arguments
QUERY=""
FORMAT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --query-gpu=*)
            QUERY="${1#*=}"
            shift
            ;;
        --format=*)
            FORMAT="${1#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Get GPU configuration from environment (allows customization per pod)
GPU_COUNT="${FAKE_GPU_COUNT:-1}"
GPU_MODEL="${FAKE_GPU_MODEL:-A100-SXM4-80GB}"
GPU_MEMORY_TOTAL="${FAKE_GPU_MEMORY_TOTAL:-81920}"
GPU_MEMORY_USED="${FAKE_GPU_MEMORY_USED:-45120}"
GPU_UTILIZATION="${FAKE_GPU_UTILIZATION:-87}"
GPU_MEM_UTIL="${FAKE_GPU_MEM_UTIL:-55}"
GPU_TEMPERATURE="${FAKE_GPU_TEMPERATURE:-72}"
GPU_POWER_DRAW="${FAKE_GPU_POWER_DRAW:-320}"
GPU_POWER_LIMIT="${FAKE_GPU_POWER_LIMIT:-400}"

# Calculate free memory
GPU_MEMORY_FREE=$((GPU_MEMORY_TOTAL - GPU_MEMORY_USED))

# Output based on query type
if [[ "$QUERY" == *"index,name,memory.total,memory.used,memory.free"* ]]; then
    # Full query format
    for i in $(seq 0 $((GPU_COUNT - 1))); do
        # Add some variation per GPU
        USED_VAR=$((GPU_MEMORY_USED + (i * 500) - 250))
        FREE_VAR=$((GPU_MEMORY_TOTAL - USED_VAR))
        UTIL_VAR=$((GPU_UTILIZATION + (i * 2) - 1))
        MEM_UTIL_VAR=$((GPU_MEM_UTIL + (i * 3) - 2))
        TEMP_VAR=$((GPU_TEMPERATURE + i))
        POWER_VAR=$((GPU_POWER_DRAW + (i * 10) - 5))
        
        echo "$i, NVIDIA $GPU_MODEL, $GPU_MEMORY_TOTAL, $USED_VAR, $FREE_VAR, $UTIL_VAR, $MEM_UTIL_VAR, $TEMP_VAR, $POWER_VAR, $GPU_POWER_LIMIT"
    done
elif [[ "$QUERY" == *"index,name,memory.total"* ]]; then
    # Simple query format
    for i in $(seq 0 $((GPU_COUNT - 1))); do
        echo "$i, NVIDIA $GPU_MODEL, $GPU_MEMORY_TOTAL"
    done
else
    # Default nvidia-smi output header
    echo "NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2"
    echo ""
    echo "+-----------------------------------------------------------------------------+"
    echo "| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |"
    echo "| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |"
    echo "|=============================================================================|"
    for i in $(seq 0 $((GPU_COUNT - 1))); do
        TEMP_VAR=$((GPU_TEMPERATURE + i))
        POWER_VAR=$((GPU_POWER_DRAW + (i * 10)))
        echo "|   $i  $GPU_MODEL      On | 00000000:$((i+1))7:00.0 Off |                    0 |"
        echo "| 30%   ${TEMP_VAR}C    P0   ${POWER_VAR}W / ${GPU_POWER_LIMIT}W |  ${GPU_MEMORY_USED}MiB / ${GPU_MEMORY_TOTAL}MiB |    ${GPU_UTILIZATION}%      Default |"
    done
    echo "+-----------------------------------------------------------------------------+"
fi

exit 0
