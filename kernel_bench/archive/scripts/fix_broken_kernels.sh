#!/bin/bash
# 修复损坏的SYCL内核 - 重新转换
# Fix broken SYCL kernels - re-convert

cd /home/intel/tianfeng/opencode_bench

echo "========================================"
echo "修复损坏的SYCL内核"
echo "========================================"
echo "开始时间: $(date)"
echo ""

# 损坏的内核列表（排除generated_v1.dp，它是测试产物）
BROKEN_KERNELS="layer_norm,winograd_filter_transform,winograd_output_se_relu_input,output_input_transform_fp16_shmem"

echo "修复内核:"
echo "  - layer_norm（缺少头文件）"
echo "  - winograd_filter_transform（编译错误）"
echo "  - winograd_output_se_relu_input（编译错误）"
echo "  - output_input_transform_fp16_shmem（编译错误）"
echo ""

# 先备份原文件
mkdir -p kernel_dataset/sycl/broken_backup
cp kernel_dataset/sycl/layer_norm_kernel.dp.cpp kernel_dataset/sycl/broken_backup/ 2>/dev/null || true
cp kernel_dataset/sycl/winograd_filter_transform_kernel.dp.cpp kernel_dataset/sycl/broken_backup/ 2>/dev/null || true
cp kernel_dataset/sycl/winograd_output_se_relu_input_kernel.dp.cpp kernel_dataset/sycl/broken_backup/ 2>/dev/null || true
cp kernel_dataset/sycl/output_input_transform_fp16_shmem_kernel.dp.cpp kernel_dataset/sycl/broken_backup/ 2>/dev/null || true

echo "已备份原文件到 kernel_dataset/sycl/broken_backup/"
echo ""

# 删除损坏的文件重新转换
rm -f kernel_dataset/sycl/layer_norm_kernel.dp.cpp
rm -f kernel_dataset/sycl/winograd_filter_transform_kernel.dp.cpp
rm -f kernel_dataset/sycl/winograd_output_se_relu_input_kernel.dp.cpp
rm -f kernel_dataset/sycl/output_input_transform_fp16_shmem_kernel.dp.cpp

echo "已删除损坏的文件，准备重新转换..."
echo ""

# 运行批量转换
python3 tools/batch_convert.py --kernels "$BROKEN_KERNELS" --output results/batch_conversion/fix_broken --workers 2

echo ""
echo "========================================"
echo "损坏内核修复完成"
echo "结束时间: $(date)"
echo "结果: results/batch_conversion/fix_broken/"
echo "========================================"
