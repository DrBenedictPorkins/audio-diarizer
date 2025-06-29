#!/bin/bash

# Audio Diarization Worker Startup Script with cuDNN Fix
# Sets proper environment variables for cuDNN 9.x on Ubuntu RTX 4090

set -e

# Set cuDNN library paths
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CUDNN_PATH="/usr/lib/x86_64-linux-gnu"

# Optional: Enable TF32 for better performance (uncomment if needed)
# export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

echo "Starting Audio Diarization Worker with cuDNN environment..."
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDNN_PATH: $CUDNN_PATH"
echo "CUDA device: $(nvidia-smi -L 2>/dev/null || echo 'CUDA not available')"
echo ""

# Start the worker
cd "$(dirname "$0")/.."
exec uv run scripts/start_worker.py "$@"