#!/bin/bash
cd /workspace

# Compile kernel + test together
icpx -fsycl -O2 -std=c++17   -I/workspace/kernel_dataset/sycl   -c /home/intel/tianfeng/opencode_bench/kernel_dataset/sycl/winograd_input_transform_kernel.dp.cpp -o /tmp/kernel.o 2>&1

if [ $? -ne 0 ]; then
  echo "Kernel compilation failed"
  exit 1
fi

# Compile test harness
icpx -fsycl -O2 -std=c++17   -I/workspace/kernel_dataset/sycl   /home/intel/tianfeng/opencode_bench/test/accuracy/winograd_sycl_test.cpp /tmp/kernel.o   -o /home/intel/tianfeng/opencode_bench/results/accuracy/winograd_sycl_test 2>&1

if [ $? -ne 0 ]; then
  echo "Test compilation failed"
  exit 1
fi

echo "Compilation successful"
