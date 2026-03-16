#!/bin/bash
# SYCL Build Script - Generated 20260304_062405
# Kernel: winograd_input_transform_kernel.dp
# Source: /workspace/kernel_dataset/sycl/winograd_input_transform_kernel.dp.cpp

set -e
set -x

echo "=== SYCL Compilation Start ==="
echo "Timestamp: 20260304_062405"
echo "Kernel: winograd_input_transform_kernel.dp"
echo "Source: /workspace/kernel_dataset/sycl/winograd_input_transform_kernel.dp.cpp"
echo "Compiler: icpx"
echo "Flags: -fsycl -O2 -std=c++17"
echo ""

# 验证源文件存在
if [ ! -f "/workspace/kernel_dataset/sycl/winograd_input_transform_kernel.dp.cpp" ]; then
    echo "[ERROR] Source file not found: /workspace/kernel_dataset/sycl/winograd_input_transform_kernel.dp.cpp"
    exit 1
fi

# 创建输出目录
mkdir -p /workspace/build_sycl
mkdir -p /workspace/results/b60

# 编译命令
echo "Starting compilation..."
icpx -fsycl -O2 -std=c++17 \
  -c "/workspace/kernel_dataset/sycl/winograd_input_transform_kernel.dp.cpp" \
  -o "/workspace/build_sycl/winograd_input_transform_kernel.dp.o" \
  2>&1 | tee "/workspace/results/b60/compile_winograd_input_transform_kernel.dp_20260304_062405.log"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "=== Compilation Exit Code: $EXIT_CODE ==="

if [ $EXIT_CODE -eq 0 ]; then
    echo "Compilation successful!"
    if [ -f "/workspace/build_sycl/winograd_input_transform_kernel.dp.o" ]; then
        echo "Output file:"
        ls -lh "/workspace/build_sycl/winograd_input_transform_kernel.dp.o"
    else
        echo "[WARNING] Output file not found!"
        EXIT_CODE=1
    fi
else
    echo "Compilation failed!"
fi

exit $EXIT_CODE
