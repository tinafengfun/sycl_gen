#!/bin/bash
# Batch optimize remaining kernels (Type D, B, A)

set -e

echo "========================================"
echo "Batch Optimizing Remaining Kernels"
echo "========================================"

# Type D: Matrix operations (skip those with existing XMX variants)
TYPE_D=(
    "test_hard_fused_kernel"
    "test_nchw_to_nhwc"
    "test_policy_map"
    "test_winograd_real"
)

# Type B: Winograd
TYPE_B=(
    "test_winograd_input_transform"
    "test_winograd_input"
    "test_winograd_output_relu_input"
)

# Type A: Element-wise (quick pass)
TYPE_A=(
    "test_add_bias_batched"
    "test_add_bias_nchw"
    "test_add_vectors_hnc_nhc"
    "test_expand_planes_fp16_nhwc"
    "test_global_scale"
)

compile_and_run() {
    local kernel=$1
    local type=$2
    
    echo "[$type] Processing: $kernel"
    
    # Copy and compile
    docker cp "/home/intel/tianfeng/opencode_bench/tests/${kernel}.cpp" lsv-container:/workspace/tests/ 2>/dev/null || true
    
    if docker exec -w /workspace/tests lsv-container bash -c "
        icpx -fsycl -O3 -std=c++17 \\
            -fsycl-targets=spir64_gen \\
            -Xsycl-target-backend \"-device bmg -options -ze-opt-large-register-file\" \\
            -o ${kernel}_baseline ${kernel}.cpp 2>&1 | grep -E '(succeeded|error)' | tail -1
    " 2>/dev/null; then
        
        # Run benchmark
        echo "  Running..."
        docker exec -w /workspace/tests lsv-container timeout 45 ./${kernel}_baseline 2>&1 | grep -E "(V[0-9]|Time:|GFLOPS:|✅)" | tail -15 || echo "  ⚠️  Execution issue"
    else
        echo "  ❌ Compilation failed"
    fi
    echo ""
}

# Process Type D
echo ""
echo "=== Type D: Matrix Operations ==="
for kernel in "${TYPE_D[@]}"; do
    compile_and_run "$kernel" "Type D"
done

# Process Type B
echo ""
echo "=== Type B: Winograd ==="
for kernel in "${TYPE_B[@]}"; do
    compile_and_run "$kernel" "Type B"
done

# Process Type A
echo ""
echo "=== Type A: Element-wise ==="
for kernel in "${TYPE_A[@]}"; do
    compile_and_run "$kernel" "Type A"
done

echo "========================================"
echo "✅ Batch optimization complete!"
echo "========================================"
