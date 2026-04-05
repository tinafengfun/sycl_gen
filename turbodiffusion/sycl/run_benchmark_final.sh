#!/bin/bash
# Run benchmark in lsv-container with proper oneAPI environment

set -e

LOCAL_DIR="/home/intel/tianfeng/opencode_bench/turbodiffusion-sycl"
CONTAINER_DIR="/workspace/turbodiffusion-sycl"
CONTAINER="lsv-container"

echo "=== Syncing turbodiffusion-sycl to container ==="

# Create directory in container
docker exec $CONTAINER mkdir -p $CONTAINER_DIR

# Sync all files
docker cp $LOCAL_DIR/. $CONTAINER:$CONTAINER_DIR/

echo "=== Running benchmark with proper oneAPI environment ==="
docker exec -w $CONTAINER_DIR $CONTAINER bash -c '
    # Source oneAPI environment properly
    source /opt/intel/oneapi/setvars.sh --force
    
    echo "Environment check:"
    python3 -c "import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"XPU available: {torch.xpu.is_available()}\")"
    
    echo ""
    echo "Checking turbodiffusion_sycl module:"
    python3 -c "import turbodiffusion_sycl; print(\"Module imported successfully\")" 2>&1 || {
        echo "Module import failed"
        exit 1
    }
    
    echo ""
    echo "=== Running TurboDiffusion SYCL Optimization Benchmark ==="
    python3 optimization_benchmark.py --phase baseline --device xpu 2>&1
' 2>&1

echo ""
echo "=== Benchmark complete ==="
