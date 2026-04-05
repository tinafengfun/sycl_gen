#!/bin/bash
set -e
set -x

echo "=== CUDA Compilation in Remote Container ==="
echo "Timestamp: 20260303_092651"
echo "Container: cuda12.9-test"
echo "Kernel: vector_add_test"
echo "Source: /workspace/kernel_dataset/cuda/vector_add_test.cu"
echo "Compiler: /usr/local/cuda/bin/nvcc"
echo "Flags: -O2 -arch=sm_70"
echo ""

# 创建输出目录
mkdir -p /workspace/build_cuda
mkdir -p /workspace/results/cuda

# 检查CUDA环境
echo "Checking CUDA environment..."
which nvcc
nvcc --version

# 编译
/usr/local/cuda/bin/nvcc -O2 -arch=sm_70 \
  -c /workspace/kernel_dataset/cuda/vector_add_test.cu \
  -o /workspace/build_cuda/vector_add_test.o \
  2>&1 | tee /workspace/results/cuda/compile_vector_add_test_20260303_092651.log

EXIT_CODE=$?
echo ""
echo "=== NVCC Exit Code: $EXIT_CODE ==="

if [ $EXIT_CODE -eq 0 ]; then
    echo "CUDA compilation successful!"
    if [ -f /workspace/build_cuda/vector_add_test.o ]; then
        ls -lh /workspace/build_cuda/vector_add_test.o
    fi
else
    echo "CUDA compilation failed!"
fi

exit $EXIT_CODE
