#!/bin/bash
set -e
set -x

echo "=== SYCL Compilation Start ==="
echo "Timestamp: 20260303_092517"
echo "Kernel: vector_add_test.dp"
echo "Source: /workspace/kernel_dataset/sycl/vector_add_test.dp.cpp"
echo "Compiler: icpx"
echo "Flags: -fsycl -O2 -std=c++17"
echo ""

# 创建输出目录
mkdir -p /workspace/build_sycl
mkdir -p /workspace/results/b60

# 编译命令
icpx -fsycl -O2 -std=c++17 \
  -c /workspace/kernel_dataset/sycl/vector_add_test.dp.cpp \
  -o /workspace/build_sycl/vector_add_test.dp.o \
  2>&1 | tee /workspace/results/b60/compile_vector_add_test.dp_20260303_092517.log

EXIT_CODE=$?
echo ""
echo "=== Compilation Exit Code: $EXIT_CODE ==="

if [ $EXIT_CODE -eq 0 ]; then
    echo "Compilation successful!"
    if [ -f /workspace/build_sycl/vector_add_test.dp.o ]; then
        ls -lh /workspace/build_sycl/vector_add_test.dp.o
    fi
else
    echo "Compilation failed!"
fi

exit $EXIT_CODE
