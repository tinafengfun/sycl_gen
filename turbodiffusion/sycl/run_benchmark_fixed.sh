#!/bin/bash
# Fix library paths and run benchmark in lsv-container

set -e
set -x

LOCAL_DIR="/home/intel/tianfeng/opencode_bench/turbodiffusion-sycl"
CONTAINER_DIR="/workspace/turbodiffusion-sycl"
CONTAINER="lsv-container"

echo "=== Syncing turbodiffusion-sycl to container ==="

# Create directory in container
docker exec $CONTAINER mkdir -p $CONTAINER_DIR

# Sync all files
docker cp $LOCAL_DIR/. $CONTAINER:$CONTAINER_DIR/

echo "=== Setting up environment and running benchmark ==="
docker exec -w $CONTAINER_DIR $CONTAINER bash -c '
    # Fix library path - use newer oneAPI version
    export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
    
    echo "Checking PyTorch..."
    python3 -c "import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA/XPU available: {torch.xpu.is_available() if hasattr(torch, \"xpu\") else \"N/A\"}\")" 2>&1 || echo "PyTorch check failed"
    
    echo ""
    echo "Checking turbodiffusion_sycl module..."
    python3 -c "import turbodiffusion_sycl; print(\"Module imported successfully\")" 2>&1 || {
        echo "Module import failed, checking module structure..."
        ls -la turbodiffusion_sycl/
        cat turbodiffusion_sycl/__init__.py | head -50
    }
    
    echo ""
    echo "=== Running benchmark ==="
    python3 optimization_benchmark.py --phase baseline --device xpu 2>&1
'

echo "=== Benchmark complete ==="
