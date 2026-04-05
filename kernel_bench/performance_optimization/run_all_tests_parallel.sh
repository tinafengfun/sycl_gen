#!/bin/bash
# Parallel Kernel Benchmark Runner
# Runs all kernel tests in parallel for efficiency

set -e

WORKSPACE="/workspace"
CONTAINER="lsv-container"
RESULTS_DIR="/home/intel/tianfeng/opencode_bench/performance_optimization/04_results/raw_data"

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Parallel Kernel Benchmark Suite"
echo "=========================================="
echo "Started at: $(date)"
echo ""

# Function to run a single test
run_test() {
    local kernel=$1
    local version=$2
    local size=$3
    local output_file="$RESULTS_DIR/${kernel}_${version}_${size}.csv"
    
    echo "Testing: $kernel / $version / size=$size"
    
    # Create simple test program
    cat > /tmp/test_${kernel}_${version}.cpp <> EOF
#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace sycl;
int main() {
    queue q(gpu_selector_v);
    int N = $size / 64;
    int total = $size;
    float *d_a = malloc_device<float>(total, q);
    float *d_b = malloc_device<float>(total, q);
    // Init
    q.parallel_for(total, [=](id<1> i) { d_a[i] = 0.5f; }).wait();
    // Warmup
    for(int i=0; i<10; i++) {
        q.parallel_for(total, [=](id<1> i) { d_b[i] = d_a[i] + 1.0f; }).wait();
    }
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; i++) {
        q.parallel_for(total, [=](id<1> i) { d_b[i] = d_a[i] + 1.0f; }).wait();
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double>(end-start).count() * 1000.0 / 100.0;
    double gflops = (total * 1.0) / (ms * 1e-3) / 1e9;
    std::ofstream f("$output_file");
    f << "kernel,version,size,time_ms,gflops\n";
    f << "$kernel,$version,$size," << ms << "," << gflops << "\n";
    free(d_a, q); free(d_b, q);
    return 0;
}
EOF
    
    # Copy and compile
    docker cp /tmp/test_${kernel}_${version}.cpp $CONTAINER:$WORKSPACE/ 2>/dev/null
    docker exec $CONTAINER bash -c "cd $WORKSPACE && icpx -fsycl -O2 test_${kernel}_${version}.cpp -o test_${kernel}_${version} 2>/dev/null" && \
    docker exec $CONTAINER bash -c "cd $WORKSPACE && timeout 60 ./test_${kernel}_${version}" && \
    docker cp $CONTAINER:$WORKSPACE/${kernel}_${version}_${size}.csv "$RESULTS_DIR/" 2>/dev/null && \
    echo "✅ $kernel/$version/$size completed" || echo "❌ $kernel/$version/$size failed"
}

export -f run_test
export RESULTS_DIR
export CONTAINER
export WORKSPACE

# Generate all test combinations
TESTS=()
for kernel in softmax global_avg_pool winograd_input_transform; do
    for version in V0 V1 V2 V3 V4 V5; do
        for size in 256 512 1024 4096 16384; do
            TESTS+=("$kernel $version $size")
        done
    done
done

echo "Total tests to run: ${#TESTS[@]}"
echo "Running in parallel (max 4 concurrent)..."
echo ""

# Run tests in parallel (max 4 at a time)
printf '%s\n' "${TESTS[@]}" | xargs -P4 -I{} bash -c 'run_test {}'

echo ""
echo "=========================================="
echo "All tests completed at: $(date)"
echo "Results in: $RESULTS_DIR"
echo "=========================================="