#!/bin/bash
# Final benchmark run with modified iterations

set -e

LOCAL_DIR="/home/intel/tianfeng/opencode_bench/turbodiffusion-sycl"
CONTAINER_DIR="/workspace/turbodiffusion-sycl"
CONTAINER="lsv-container"

echo "=== Syncing updated benchmark to container ==="

# Sync updated benchmark
docker cp $LOCAL_DIR/optimization_benchmark.py $CONTAINER:$CONTAINER_DIR/

echo "=== Running TurboDiffusion SYCL Optimization Benchmark ==="
docker exec -w $CONTAINER_DIR $CONTAINER bash -c '
    source /opt/intel/oneapi/setvars.sh --force
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH
    
    echo "Environment:"
    python3 -c "import torch; import turbodiffusion_sycl_ops; print(f\"PyTorch: {torch.__version__}, XPU: {torch.xpu.is_available()}\")"
    
    echo ""
    echo "Running benchmark (10 iterations per test)..."
    python3 optimization_benchmark.py --phase baseline --device xpu 2>&1
' 2>&1

echo ""
echo "=== Benchmark Complete ==="
