#!/bin/bash
# 小规模测试脚本 - 测试3个代表性kernel
# 输出所有执行日志到控制台

set -e
set -x

WORKSPACE="/home/intel/tianfeng/opencode_bench"
TESTS_DIR="$WORKSPACE/tests"
REPORTS_DIR="$TESTS_DIR/reports"

# 创建报告目录
mkdir -p "$REPORTS_DIR"

echo "=========================================="
echo "小规模测试开始"
echo "测试kernel: winograd_filter_transform, add_vectors, global_avg_pool"
echo "=========================================="
echo ""

# 函数：编译kernel
compile_kernel() {
    local kernel_name=$1
    local round=$2
    local source_file=$3
    local output_name="${kernel_name}_xmx_r${round}"
    
    echo "[编译] $output_name"
    
    cd "$TESTS_DIR"
    
    # AOT编译命令
    icpx -fsycl -O3 -std=c++17 \
      -fsycl-targets=spir64_gen \
      -Xsycl-target-backend "-device bmg -options -ze-opt-large-register-file" \
      -o "$output_name" "$source_file" 2>&1 | tee "$REPORTS_DIR/${kernel_name}_compile_r${round}.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ 编译成功: $output_name"
        return 0
    else
        echo "❌ 编译失败: $output_name"
        return 1
    fi
}

# 函数：运行性能测试
run_benchmark() {
    local kernel_name=$1
    local round=$2
    
    echo "[测试] ${kernel_name}_xmx_r${round}"
    
    cd "$TESTS_DIR"
    
    if [ -f "${kernel_name}_xmx_r${round}" ]; then
        timeout 300 ./"${kernel_name}_xmx_r${round}" 2>&1 | tee "$REPORTS_DIR/${kernel_name}_run_r${round}.log"
        echo "✅ 测试完成"
    else
        echo "❌ 可执行文件不存在"
    fi
}

# ============================================
# 测试1: test_winograd_filter_transform
# ============================================
echo ""
echo "=========================================="
echo "测试 1/3: test_winograd_filter_transform"
echo "=========================================="

# Round 1
echo ""
echo "[Round 1] 基础XMX优化"
compile_kernel "winograd_filter_transform" "1" "test_winograd_filter_transform_xmx.cpp" || echo "跳过（文件不存在）"
run_benchmark "winograd_filter_transform" "1"

# Round 2  
echo ""
echo "[Round 2] 内存优化"
compile_kernel "winograd_filter_transform" "2" "test_winograd_filter_transform_xmx.cpp" || echo "跳过"
run_benchmark "winograd_filter_transform" "2"

# Round 3
echo ""
echo "[Round 3] 高级优化"
compile_kernel "winograd_filter_transform" "3" "test_winograd_filter_transform_xmx.cpp" || echo "跳过"
run_benchmark "winograd_filter_transform" "3"

# ============================================
# 测试2: test_add_vectors
# ============================================
echo ""
echo "=========================================="
echo "测试 2/3: test_add_vectors"
echo "=========================================="

# Round 1
echo ""
echo "[Round 1] 基础XMX优化"
compile_kernel "add_vectors" "1" "test_add_vectors_xmx.cpp" || echo "跳过（文件不存在）"
run_benchmark "add_vectors" "1"

# Round 2
echo ""
echo "[Round 2] 内存优化"
compile_kernel "add_vectors" "2" "test_add_vectors_xmx.cpp" || echo "跳过"
run_benchmark "add_vectors" "2"

# Round 3
echo ""
echo "[Round 3] 高级优化"
compile_kernel "add_vectors" "3" "test_add_vectors_xmx.cpp" || echo "跳过"
run_benchmark "add_vectors" "3"

# ============================================
# 测试3: test_global_avg_pool
# ============================================
echo ""
echo "=========================================="
echo "测试 3/3: test_global_avg_pool"
echo "=========================================="

# Round 1
echo ""
echo "[Round 1] 基础XMX优化"
compile_kernel "global_avg_pool_nhwc_fp16" "1" "test_global_avg_pool_nhwc_fp16_xmx.cpp" || echo "跳过（文件不存在）"
run_benchmark "global_avg_pool_nhwc_fp16" "1"

# Round 2
echo ""
echo "[Round 2] 内存优化"
compile_kernel "global_avg_pool_nhwc_fp16" "2" "test_global_avg_pool_nhwc_fp16_xmx.cpp" || echo "跳过"
run_benchmark "global_avg_pool_nhwc_fp16" "2"

# Round 3
echo ""
echo "[Round 3] 高级优化"
compile_kernel "global_avg_pool_nhwc_fp16" "3" "test_global_avg_pool_nhwc_fp16_xmx.cpp" || echo "跳过"
run_benchmark "global_avg_pool_nhwc_fp16" "3"

# ============================================
# 生成测试总结
# ============================================
echo ""
echo "=========================================="
echo "小规模测试完成"
echo "=========================================="
echo ""
echo "报告位置: $REPORTS_DIR"
echo ""
ls -lh "$REPORTS_DIR"/

echo ""
echo "所有日志文件已生成，请检查:"
echo "- $REPORTS_DIR/*_compile_r*.log (编译日志)"
echo "- $REPORTS_DIR/*_run_r*.log (运行日志)"
