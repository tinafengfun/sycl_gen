#!/bin/bash
# Build turbodiffusion_sycl_ops module and run benchmark

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
    
    echo "Building with:"
    which icpx
    icpx --version | head -1
    
    # Clean previous build
    rm -rf build *.so
    
    # Build the extension
    CC=icpx CXX=icpx python3 setup.py build_ext --inplace 2>&1
    
    echo ""
    echo "Build complete. Moving module to parent directory:"
    mv *.so ../
    ls -la ../*.so
' 2>&1

echo ""
echo "=== Running benchmark ==="
docker exec -w $CONTAINER_DIR $CONTAINER bash -c '
    source /opt/intel/oneapi/setvars.sh --force
    
    echo "Verifying module:"
    python3 -c "import turbodiffusion_sycl_ops; print(\"Module loaded:\", turbodiffusion_sycl_ops)"
    
    echo ""
    python3 optimization_benchmark.py --phase baseline --device xpu 2>&1
' 2>&1

echo ""
echo "=== Complete ==="
