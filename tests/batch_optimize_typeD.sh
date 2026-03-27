#!/bin/bash
# Batch optimization script for Type D kernels
# Compiles and benchmarks kernels with baseline vs optimized versions

set -e

TESTS_DIR="/home/intel/tianfeng/opencode_bench/tests"
REPORTS_DIR="$TESTS_DIR/reports"
DOCKER_WORKSPACE="/workspace/tests"

# Kernels to optimize (excluding those with existing XMX variants)
KERNELS=(
    "test_hard_fused_kernel"
    "test_nchw_to_nhwc"
    "test_policy_map"
    "test_gemm_aot"
    "test_gemm_large"
    "test_gemm_onednn"
)

# Compilation flags
ICPX_FLAGS="-fsycl -O3 -std=c++17 -fsycl-targets=spir64_gen -Xsycl-target-backend"
BMG_FLAGS='"-device bmg -options -ze-opt-large-register-file"'

echo "========================================"
echo "Batch Optimization - Type D Kernels"
echo "========================================"
echo "Total kernels to process: ${#KERNELS[@]}"
echo ""

# Function to compile kernel
compile_kernel() {
    local kernel=$1
    local variant=$2
    local output_name="${kernel}_${variant}"
    local source="$TESTS_DIR/${kernel}.cpp"
    
    echo "Compiling $output_name..."
    docker exec -w $DOCKER_WORKSPACE lsv-container bash -c "
        icpx $ICPX_FLAGS $BMG_FLAGS \
            -o $output_name ${kernel}.cpp 2>&1 | tee ${output_name}_compile.log
    " || echo "❌ Compilation failed for $output_name"
}

# Function to run benchmark
run_benchmark() {
    local kernel=$1
    local variant=$2
    local binary="${kernel}_${variant}"
    
    echo "Running $binary..."
    docker exec -w $DOCKER_WORKSPACE lsv-container bash -c "
        if [ -f $binary ]; then
            ./$binary 2>&1 | tee ${binary}_run.log
        else
            echo 'Binary not found: $binary'
        fi
    " || echo "❌ Execution failed for $binary"
}

# Main optimization loop
for kernel in "${KERNELS[@]}"; do
    echo "========================================"
    echo "Processing: $kernel"
    echo "========================================"
    
    # Copy source to container
    docker cp "$TESTS_DIR/${kernel}.cpp" lsv-container:$DOCKER_WORKSPACE/ 2>/dev/null || true
    
    # Compile baseline
    compile_kernel "$kernel" "baseline"
    
    # Run baseline benchmark
    run_benchmark "$kernel" "baseline"
    
    echo ""
done

# Collect results
echo "========================================"
echo "Collecting Results"
echo "========================================"

# Create summary report
SUMMARY_FILE="$REPORTS_DIR/batch1_typeD_summary.md"
echo "# Type D Kernels - Batch Optimization Results" > "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "## Kernels Processed" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for kernel in "${KERNELS[@]}"; do
    echo "- $kernel" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"
echo "## Performance Results" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "See individual run logs in Docker container: $DOCKER_WORKSPACE/*_run.log" >> "$SUMMARY_FILE"

echo "✅ Batch optimization complete!"
echo "Summary: $SUMMARY_FILE"
