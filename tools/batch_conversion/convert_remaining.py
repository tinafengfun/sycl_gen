#!/usr/bin/env python3
"""
继续批量转换剩余内核
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from improved_agent_v3 import ImprovedConversionAgent

# 剩余待处理内核 (排除已成功的)
REMAINING_KERNELS = [
    'layer_norm',
    'global_scale_fp16_nhwc',
    'expand_planes_nhwc',
    'promotion_logits',
    'preprocess_attention_body',
    'input_gating',
    'gen_offset_pointers',
    'winograd_output_transform',
    'winograd_output_se_relu_input',
    'winograd_output_relu_input',
    'output_input_transform_fp16_shmem',
]

async def convert_remaining():
    print(f"🚀 继续转换剩余 {len(REMAINING_KERNELS)} 个内核...")
    print()
    
    agent = ImprovedConversionAgent()
    
    success_count = 0
    for kernel_id in REMAINING_KERNELS:
        print(f"\n🔄 {kernel_id}")
        
        if kernel_id not in agent.kernels:
            print(f"  ⚠️ 不存在")
            continue
        
        kernel = agent.kernels[kernel_id]
        
        # 如果已有文件，先删除重新转换
        if kernel.sycl_file.exists():
            kernel.sycl_file.unlink()
            kernel.has_sycl_mapping = False
        
        success = await agent.process_kernel(kernel, with_fix_loop=True)
        
        if success:
            success_count += 1
            print(f"  ✅ 成功")
        else:
            print(f"  ❌ 失败 - {kernel.error_type}")
    
    print(f"\n✅ 本次成功: {success_count}/{len(REMAINING_KERNELS)}")

if __name__ == '__main__':
    asyncio.run(convert_remaining())
