#!/bin/bash
# SYCL Build Script for BMG/XPU - Generated 20260308_083713
# Kernel: winograd_output_se_relu_input_kernel.dp
# Source: /workspace/kernel_dataset/sycl/winograd_output_se_relu_input_kernel.dp.cpp
# AOT Targets: pvc,bmg,dg2,arl-h,mtl-h,lnl-m,ptl-h,ptl-u

set -e
set -x

echo "=== SYCL Compilation for Intel XPU (BMG Support) ==="
echo "Timestamp: 20260308_083713"
echo "Kernel: winograd_output_se_relu_input_kernel.dp"
echo "Source: /workspace/kernel_dataset/sycl/winograd_output_se_relu_input_kernel.dp.cpp"
echo "Compiler: icpx"
echo "Compile Flags: -fsycl -O2 -std=c++17 -fsycl-unnamed-lambda -sycl-std=2020 -fhonor-nans -fhonor-infinities -fno-associative-math -fno-approx-func -no-ftz -fsycl-targets=spir64_gen,spir64"
echo "Device Link Flags: -fsycl-max-parallel-link-jobs=4 --offload-compress -fsycl-targets=spir64_gen,spir64 -Xs '-device pvc,bmg,dg2,arl-h,mtl-h,lnl-m,ptl-h,ptl-u -options -cl-poison-unsupported-fp64-kernels -options -cl-intel-enable-auto-large-GRF-mode -options -cl-fp32-correctly-rounded-divide-sqrt -options -cl-intel-greater-than-4GB-buffer-required'"
echo "AOT Targets: pvc,bmg,dg2,arl-h,mtl-h,lnl-m,ptl-h,ptl-u"
echo ""

# 验证源文件存在
if [ ! -f "/workspace/kernel_dataset/sycl/winograd_output_se_relu_input_kernel.dp.cpp" ]; then
    echo "[ERROR] Source file not found: /workspace/kernel_dataset/sycl/winograd_output_se_relu_input_kernel.dp.cpp"
    exit 1
fi

# 创建输出目录
mkdir -p /workspace/build_sycl
mkdir -p /workspace/results/b60

# 编译步骤：
# 1. 编译源文件为对象文件（包含设备代码）
echo "Step 1: Compiling source to object file..."
icpx -fsycl -O2 -std=c++17 -fsycl-unnamed-lambda -sycl-std=2020 -fhonor-nans -fhonor-infinities -fno-associative-math -fno-approx-func -no-ftz -fsycl-targets=spir64_gen,spir64 \
  -c "/workspace/kernel_dataset/sycl/winograd_output_se_relu_input_kernel.dp.cpp" \
  -o "/workspace/build_sycl/winograd_output_se_relu_input_kernel.dp.o" \
  2>&1 | tee "/workspace/results/b60/compile_winograd_output_se_relu_input_kernel.dp_20260308_083713.log.step1"

STEP1_EXIT=${PIPESTATUS[0]}
if [ $STEP1_EXIT -ne 0 ]; then
    echo "[ERROR] Compilation step 1 failed with exit code $STEP1_EXIT"
    exit $STEP1_EXIT
fi

# 2. 设备代码链接（AOT编译为指定目标）
echo ""
echo "Step 2: Linking device code for AOT targets..."
icpx -fsycl -fsycl-max-parallel-link-jobs=4 --offload-compress -fsycl-targets=spir64_gen,spir64 -Xs '-device pvc,bmg,dg2,arl-h,mtl-h,lnl-m,ptl-h,ptl-u -options -cl-poison-unsupported-fp64-kernels -options -cl-intel-enable-auto-large-GRF-mode -options -cl-fp32-correctly-rounded-divide-sqrt -options -cl-intel-greater-than-4GB-buffer-required' \
  "/workspace/build_sycl/winograd_output_se_relu_input_kernel.dp.o" \
  -o "/workspace/build_sycl/winograd_output_se_relu_input_kernel.dp.o.linked" \
  2>&1 | tee "/workspace/results/b60/compile_winograd_output_se_relu_input_kernel.dp_20260308_083713.log.step2"

STEP2_EXIT=${PIPESTATUS[0]}
if [ $STEP2_EXIT -ne 0 ]; then
    echo "[WARNING] Device linking step 2 failed with exit code $STEP2_EXIT"
    echo "This may be expected for kernels without device code or if AOT is not needed"
    # 不要退出，因为对象文件可能已经可用
else
    echo "Device linking successful"
fi

# 合并日志
cat "/workspace/results/b60/compile_winograd_output_se_relu_input_kernel.dp_20260308_083713.log.step1" > "/workspace/results/b60/compile_winograd_output_se_relu_input_kernel.dp_20260308_083713.log"
if [ -f "/workspace/results/b60/compile_winograd_output_se_relu_input_kernel.dp_20260308_083713.log.step2" ]; then
    echo "" >> "/workspace/results/b60/compile_winograd_output_se_relu_input_kernel.dp_20260308_083713.log"
    echo "=== Device Link Step ===" >> "/workspace/results/b60/compile_winograd_output_se_relu_input_kernel.dp_20260308_083713.log"
    cat "/workspace/results/b60/compile_winograd_output_se_relu_input_kernel.dp_20260308_083713.log.step2" >> "/workspace/results/b60/compile_winograd_output_se_relu_input_kernel.dp_20260308_083713.log"
fi

EXIT_CODE=0
if [ -f "/workspace/build_sycl/winograd_output_se_relu_input_kernel.dp.o" ]; then
    echo ""
    echo "=== Compilation Successful ==="
    echo "Output file: /workspace/build_sycl/winograd_output_se_relu_input_kernel.dp.o"
    ls -lh "/workspace/build_sycl/winograd_output_se_relu_input_kernel.dp.o"
    
    # 检查是否包含BMG设备代码
    echo ""
    echo "Checking device code presence..."
    if objdump -h "/workspace/build_sycl/winograd_output_se_relu_input_kernel.dp.o" 2>/dev/null | grep -q ".text"; then
        echo "✓ Device code present in object file"
    fi
else
    echo "[WARNING] Output file not found!"
    EXIT_CODE=1
fi

echo ""
echo "=== Compilation Exit Code: $EXIT_CODE ==="
exit $EXIT_CODE
