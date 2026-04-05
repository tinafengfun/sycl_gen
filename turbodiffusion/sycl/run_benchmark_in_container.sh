#!/bin/bash
# Sync turbodiffusion-sycl to lsv-container and run benchmark

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

echo "=== Checking turbodiffusion_sycl module ==="
docker exec -w $CONTAINER_DIR $CONTAINER bash -c '
    python3 -c "import turbodiffusion_sycl; print(\"Module found: \", turbodiffusion_sycl)" 2>&1 || {
        echo "Module not found, building..."
        cd operators && python3 setup.py build_ext --inplace 2>&1
    }
'

echo "=== Running benchmark ==="
docker exec -w $CONTAINER_DIR $CONTAINER python3 optimization_benchmark.py --phase baseline --device xpu 2>&1

echo "=== Benchmark complete ==="
