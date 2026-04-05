#!/bin/bash
# Complete build and benchmark script with proper environment

set -e

LOCAL_DIR="/home/intel/tianfeng/opencode_bench/turbodiffusion-sycl"
CONTAINER_DIR="/workspace/turbodiffusion-sycl"
CONTAINER="lsv-container"

echo "=== Syncing turbodiffusion-sycl to container ==="

# Create directory in container
docker exec $CONTAINER mkdir -p $CONTAINER_DIR

# Sync all files
docker cp $LOCAL_DIR/. $CONTAINER:$CONTAINER_DIR/

echo "=== Building SYCL operations module ==="
docker exec -w $CONTAINER_DIR/operators $CONTAINER bash -c '
    source /opt/intel/oneapi/setvars.sh --force
    
    # Clean previous build
    rm -rf build *.so
    
    # Build the extension
    CC=icpx CXX=icpx python3 setup.py build_ext --inplace 2>&1
    
    # Move to parent directory
    mv *.so ../
    
    echo "Build complete. Module files:"
    ls -la ../*.so
' 2>&1

echo ""
echo "=== Running TurboDiffusion SYCL Optimization Benchmark ==="
docker exec -w $CONTAINER_DIR $CONTAINER bash -c '
    source /opt/intel/oneapi/setvars.sh --force
    
    # Set library paths for runtime
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH
    
    echo "Environment verified:"
    python3 -c "import torch; import turbodiffusion_sycl_ops; print(f\"PyTorch: {torch.__version__}\"); print(f\"XPU: {torch.xpu.is_available()}\")"
    
    echo ""
    echo "Starting benchmark..."
    python3 optimization_benchmark.py --phase baseline --device xpu 2>&1
' 2>&1

echo ""
echo "=== Benchmark Complete ==="
