#!/usr/bin/env python3
"""
Agent v3.0 批量转换脚本
处理未转换的 16 个内核
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from improved_agent_v3 import ImprovedConversionAgent, ConversionStatus

# 待处理的 16 个内核
KERNELS_TO_CONVERT = [
    'add_vectors_hnc_nhc',
    'add_bias_nchw',
    'nchw_to_nhwc',
    'global_scale_fp16_nhwc',
    'global_avg_pool_nhwc_fp16',
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

# 排除没有 SYCL 映射的内核
EXCLUDE_NO_SYCL = ['expand_planes_fp32_nchw', 'fused_mha_cutlass']

async def main():
    print("="*80)
    print("🚀 Agent v3.0 批量转换启动")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"待处理内核: {len(KERNELS_TO_CONVERT)}")
    print()
    
    agent = ImprovedConversionAgent(output_dir="results/agent_v3_batch")
    
    # 统计
    results = {
        'start_time': datetime.now().isoformat(),
        'total': len(KERNELS_TO_CONVERT),
        'converted': 0,
        'verified': 0,
        'failed': 0,
        'details': []
    }
    
    # 逐个处理
    for i, kernel_id in enumerate(KERNELS_TO_CONVERT, 1):
        print(f"\n{'='*80}")
        print(f"📌 [{i}/{len(KERNELS_TO_CONVERT)}] 处理: {kernel_id}")
        print(f"{'='*80}")
        
        if kernel_id not in agent.kernels:
            print(f"⚠️  内核 {kernel_id} 不存在于索引中")
            results['details'].append({
                'kernel_id': kernel_id,
                'status': 'not_found',
                'error': 'Not in kernel index'
            })
            continue
        
        kernel_info = agent.kernels[kernel_id]
        
        # 检查是否已有 SYCL 文件
        if kernel_info.sycl_file.exists() and kernel_info.sycl_file.stat().st_size > 0:
            print(f"ℹ️  已有 SYCL 文件，直接验证...")
            success = await agent.verify_kernel(kernel_info)
        else:
            # 执行完整转换流程
            success = await agent.process_kernel(kernel_info, with_fix_loop=True)
        
        # 记录结果
        detail = {
            'kernel_id': kernel_id,
            'status': kernel_info.status.value,
            'cuda_compiles': kernel_info.cuda_compiles,
            'sycl_compiles': kernel_info.sycl_compiles,
            'complexity_score': kernel_info.complexity_score,
            'conversion_attempts': kernel_info.conversion_attempts,
            'fix_attempts': kernel_info.fix_attempts,
            'error_type': kernel_info.error_type
        }
        results['details'].append(detail)
        
        if success:
            results['verified'] += 1
            print(f"✅ 成功!")
        elif kernel_info.conversion_attempts > 0:
            results['converted'] += 1
            print(f"⚠️  转换但未通过验证")
        else:
            results['failed'] += 1
            print(f"❌ 失败")
        
        # 每处理 3 个保存一次结果
        if i % 3 == 0:
            save_results(results, agent)
    
    # 最终保存
    save_results(results, agent)
    
    # 打印摘要
    print("\n" + "="*80)
    print("📊 批量转换完成")
    print("="*80)
    print(f"总处理: {results['total']}")
    print(f"验证通过: {results['verified']}")
    print(f"转换但未验证: {results['converted']}")
    print(f"失败: {results['failed']}")
    print(f"成功率: {results['verified']/results['total']*100:.1f}%")
    print()
    print(f"💾 结果保存: results/agent_v3_batch/")


def save_results(results, agent):
    """保存结果"""
    output_dir = Path("results/agent_v3_batch")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results['end_time'] = datetime.now().isoformat()
    results['llm_stats'] = agent.llm_client.get_stats()
    results['compilation_stats'] = agent.compiler.get_stats()
    
    result_file = output_dir / "batch_conversion_results.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    asyncio.run(main())
