#!/bin/bash
# Batch optimize Type C kernels (Reduction)
# Expected: 50-100% improvement

set -e

KERNELS=(
    "test_global_avg_pool_real"
    "test_softmax_v0"
    "test_softmax_v1"
    "test_hard_batch_norm"
)

echo "========================================"
echo "Type C Batch Optimization"
echo "========================================"
echo ""

for kernel in "${KERNELS[@]}"; do
    echo "Processing: $kernel"
    
    # Copy to container
    docker cp "/home/intel/tianfeng/opencode_bench/tests/${kernel}.cpp" lsv-container:/workspace/tests/ 2>/dev/null || true
    
    # Compile
    echo "  Compiling..."
    docker exec -w /workspace/tests lsv-container bash -c "
        icpx -fsycl -O3 -std=c++17 \\
            -fsycl-targets=spir64_gen \\
            -Xsycl-target-backend \"-device bmg -options -ze-opt-large-register-file\" \\
            -o ${kernel}_baseline ${kernel}.cpp 2>&1 | tail -5
    " || echo "  ❌ Compilation failed"
    
    # Run if compiled
    if docker exec lsv-container test -f /workspace/tests/${kernel}_baseline; then
        echo "  Running benchmark..."
        docker exec -w /workspace/tests lsv-container bash -c "
            timeout 60 ./${kernel}_baseline 2>&1 | tail -20
        " || echo "  ⚠️  Execution timeout or error"
    fi
    
    echo ""
done

echo "✅ Type C batch complete!"
