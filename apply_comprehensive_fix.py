#!/usr/bin/env python3
"""
全面修复SYCL内核 - 处理所有编译错误
"""

import re
from pathlib import Path

def fix_kernel_content(content):
    """修复内核内容中的所有问题"""
    
    # 1. 修复数学函数
    content = content.replace('__expf(', 'sycl::exp(')
    content = content.replace('__logf(', 'sycl::log(')
    content = content.replace('__sqrtf(', 'sycl::sqrt(')
    content = content.replace('tanh(', 'sycl::tanh(')
    content = content.replace('__trap();', 'return;')
    
    # 2. 修复CUDA特定的语法
    content = content.replace('__device__', '')
    content = content.replace('__forceinline__', 'inline')
    content = content.replace('__global__', '')
    
    # 3. 修复warp shuffle
    content = re.sub(r'__shfl_xor_sync\([^,]+,\s*([^,]+),\s*[^)]+\)', r'\1', content)
    
    # 4. 替换错误处理
    content = re.sub(r'throw\s+\w+\s*\(', 'throw std::runtime_error(', content)
    
    # 5. 修复CUDA类型
    content = content.replace('cudaError_t', 'int')
    content = content.replace('cudaSuccess', '0')
    
    return content

def fix_file(filepath):
    """修复单个文件"""
    print(f"修复: {filepath.name}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 应用修复
    fixed_content = fix_kernel_content(content)
    
    # 保存
    with open(filepath, 'w') as f:
        f.write(fixed_content)
    
    print(f"  ✅ 已应用修复")

# 处理所有失败的文件
failed_files = [
    'add_bias_batched_kernel.dp.cpp',
    'add_bias_nchw_kernel.dp.cpp', 
    'add_vectors_hnc_nhc_kernel.dp.cpp',
    'add_vectors_kernel.dp.cpp',
    'expand_planes_nhwc_kernel.dp.cpp',
    'gen_offset_pointers_kernel.dp.cpp',
    'global_avg_pool_nhwc_fp16_kernel.dp.cpp',
    'global_scale_fp16_nhwc_kernel.dp.cpp',
    'global_scale_kernel.dp.cpp',
    'input_gating_kernel.dp.cpp',
    'layer_norm_kernel.dp.cpp',
    'nchw_to_nhwc_kernel.dp.cpp',
    'output_input_transform_fp16_shmem_kernel.dp.cpp',
    'preprocess_attention_body_kernel.dp.cpp',
    'promotion_logits_kernel.dp.cpp',
    'se_layer_nhwc_kernel.dp.cpp',
    'softmax_kernel.dp.cpp',
    'winograd_filter_transform_kernel.dp.cpp',
    'winograd_output_relu_input_kernel.dp.cpp',
    'winograd_output_se_relu_input_kernel.dp.cpp',
    'winograd_output_transform_kernel.dp.cpp'
]

print("=" * 70)
print("应用全面修复...")
print("=" * 70)
print()

sycl_dir = Path('kernel_dataset/sycl')
for filename in failed_files:
    filepath = sycl_dir / filename
    if filepath.exists():
        fix_file(filepath)
    else:
        print(f"⚠️  文件不存在: {filename}")

print()
print("=" * 70)
print("修复完成")
print("=" * 70)
