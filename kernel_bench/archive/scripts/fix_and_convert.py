#!/usr/bin/env python3
"""
批量重新转换有问题的内核
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from improved_agent_v3 import ImprovedConversionAgent, ConversionStatus

# 需要重新转换的内核（已有 SYCL 文件但编译失败）
KERNELS_TO_RECONVERT = [
    'add_vectors_hnc_nhc',
    'add_bias_nchw',
    'nchw_to_nhwc',
    'global_scale_fp16_nhwc',
]

# 需要从头转换的内核（没有 SYCL 文件或文件为空）
KERNELS_TO_CONVERT_NEW = [
    'promotion_logits',
    'preprocess_attention_body',
    'input_gating',
    'gen_offset_pointers',
    'winograd_output_transform',
    'winograd_output_se_relu_input',
    'winograd_output_relu_input',
    'output_input_transform_fp16_shmem',
]

async def main():
    print("="*80)
    print("🚀 Agent v3.0 修复 & 新转换")
    print("="*80)
    
    agent = ImprovedConversionAgent(output_dir="results/agent_v3_reconvert")
    
    results = {'reconverted': 0, 'new_converted': 0, 'failed': 0}
    
    # 1. 重新转换有问题的内核
    print("\n" + "="*80)
    print("📌 第一阶段：重新转换有问题的内核")
    print("="*80)
    
    for kernel_id in KERNELS_TO_RECONVERT:
        print(f"\n🔄 重新转换: {kernel_id}")
        
        if kernel_id not in agent.kernels:
            print(f"  ⚠️  内核不存在")
            continue
        
        kernel = agent.kernels[kernel_id]
        
        # 备份并删除现有 SYCL 文件
        if kernel.sycl_file.exists():
            backup_path = kernel.sycl_file.with_suffix('.dp.cpp.bak')
            kernel.sycl_file.rename(backup_path)
            print(f"  💾 已备份原文件")
        
        kernel.has_sycl_mapping = False
        kernel.status = ConversionStatus.PENDING
        
        # 重新转换
        success = await agent.process_kernel(kernel, with_fix_loop=True)
        
        if success:
            results['reconverted'] += 1
            print(f"  ✅ 重新转换成功")
        else:
            results['failed'] += 1
            print(f"  ❌ 重新转换失败")
    
    # 2. 转换新内核
    print("\n" + "="*80)
    print("📌 第二阶段：转换新内核")
    print("="*80)
    
    for kernel_id in KERNELS_TO_CONVERT_NEW:
        print(f"\n🆕 新转换: {kernel_id}")
        
        if kernel_id not in agent.kernels:
            print(f"  ⚠️  内核不存在于索引中")
            results['failed'] += 1
            continue
        
        kernel = agent.kernels[kernel_id]
        
        # 检查是否已有 SYCL 文件
        if kernel.sycl_file.exists() and kernel.sycl_file.stat().st_size > 100:
            print(f"  ℹ️  已有 SYCL 文件，验证中...")
            success = await agent.verify_kernel(kernel)
        else:
            # 全新转换
            success = await agent.process_kernel(kernel, with_fix_loop=True)
        
        if success:
            results['new_converted'] += 1
            print(f"  ✅ 转换成功")
        else:
            results['failed'] += 1
            print(f"  ❌ 转换失败")
    
    # 打印结果
    print("\n" + "="*80)
    print("📊 修复 & 转换完成")
    print("="*80)
    print(f"重新转换成功: {results['reconverted']}")
    print(f"新转换成功: {results['new_converted']}")
    print(f"失败: {results['failed']}")
    print(f"总计成功: {results['reconverted'] + results['new_converted']}")
    
    # 保存结果
    agent._save_session_results()

if __name__ == '__main__':
    asyncio.run(main())
