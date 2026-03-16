#!/usr/bin/env python3
"""
批量转换剩余 16 个内核
使用 Agent v3.0 进行自动转换和修复
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from improved_agent_v3 import ImprovedConversionAgent, ConversionStatus

# 16 个待处理内核
KERNELS_TO_CONVERT = [
    'add_vectors_hnc_nhc',
    'add_bias_nchw', 
    'nchw_to_nhwc',
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
    # 以下两个没有SYCL映射，需要特殊处理
    # 'expand_planes_fp32_nchw',
    # 'fused_mha_cutlass',
]

async def batch_convert():
    print("="*80)
    print("🚀 Agent v3.0 批量转换 - 16 个内核")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目标: {len(KERNELS_TO_CONVERT)} 个内核")
    print()
    
    agent = ImprovedConversionAgent(output_dir="results/batch_convert_16")
    
    results = {
        'start_time': datetime.now().isoformat(),
        'total': len(KERNELS_TO_CONVERT),
        'converted': 0,
        'verified': 0,
        'failed': 0,
        'skipped': 0,
        'details': []
    }
    
    # 逐个处理
    for i, kernel_id in enumerate(KERNELS_TO_CONVERT, 1):
        print(f"\n{'='*80}")
        print(f"📌 [{i}/{len(KERNELS_TO_CONVERT)}] {kernel_id}")
        print(f"{'='*80}")
        
        if kernel_id not in agent.kernels:
            print(f"⚠️  内核 {kernel_id} 不存在")
            results['skipped'] += 1
            results['details'].append({
                'kernel_id': kernel_id,
                'status': 'not_found'
            })
            continue
        
        kernel = agent.kernels[kernel_id]
        
        # 检查当前状态
        print(f"当前状态:")
        print(f"  - 有SYCL映射: {kernel.has_sycl_mapping}")
        print(f"  - SYCL文件存在: {kernel.sycl_file.exists()}")
        if kernel.sycl_file.exists():
            print(f"  - SYCL文件大小: {kernel.sycl_file.stat().st_size} bytes")
        
        # 如果已有SYCL文件，先验证
        if kernel.sycl_file.exists() and kernel.sycl_file.stat().st_size > 100:
            print(f"\n🔍 已有SYCL文件，直接验证...")
            
            # 测试CUDA编译
            cuda_ok, cuda_err_type, cuda_err = agent.compiler.test_cuda_compilation(kernel_id)
            print(f"  CUDA: {'✅ 通过' if cuda_ok else '❌ 失败'}")
            
            # 测试SYCL编译
            sycl_ok, sycl_err_type, sycl_err = agent.compiler.test_sycl_compilation(kernel_id)
            print(f"  SYCL: {'✅ 通过' if sycl_ok else '❌ 失败'}")
            
            if cuda_ok and sycl_ok:
                print(f"\n✅ {kernel_id} 双平台编译通过！")
                results['verified'] += 1
                results['details'].append({
                    'kernel_id': kernel_id,
                    'status': 'verified',
                    'cuda_ok': True,
                    'sycl_ok': True
                })
                continue
            else:
                print(f"\n🔄 编译失败，需要重新转换...")
                # 备份并删除现有SYCL文件
                if kernel.sycl_file.exists():
                    backup = kernel.sycl_file.with_suffix('.dp.cpp.bak')
                    kernel.sycl_file.rename(backup)
                    print(f"  💾 已备份原文件")
                kernel.has_sycl_mapping = False
        
        # 执行转换
        print(f"\n🔄 开始转换...")
        success = await agent.process_kernel(kernel, with_fix_loop=True)
        
        detail = {
            'kernel_id': kernel_id,
            'status': kernel.status.value,
            'cuda_compiles': kernel.cuda_compiles,
            'sycl_compiles': kernel.sycl_compiles,
            'complexity_score': kernel.complexity_score,
            'conversion_attempts': kernel.conversion_attempts,
            'fix_attempts': kernel.fix_attempts,
            'error_type': kernel.error_type
        }
        results['details'].append(detail)
        
        if success:
            results['verified'] += 1
            print(f"\n✅ {kernel_id} 转换成功并通过验证！")
        elif kernel.conversion_attempts > 0:
            results['converted'] += 1
            print(f"\n⚠️  {kernel_id} 转换完成但未通过验证")
        else:
            results['failed'] += 1
            print(f"\n❌ {kernel_id} 转换失败")
        
        # 每3个保存一次
        if i % 3 == 0:
            save_results(results, agent)
    
    # 最终保存
    save_results(results, agent)
    
    # 打印摘要
    print("\n" + "="*80)
    print("📊 批量转换完成")
    print("="*80)
    print(f"总处理: {results['total']}")
    print(f"✅ 验证通过: {results['verified']}")
    print(f"⚠️  转换但未验证: {results['converted']}")
    print(f"❌ 失败: {results['failed']}")
    print(f"⏭️  跳过: {results['skipped']}")
    print(f"📈 成功率: {results['verified']/results['total']*100:.1f}%")
    print()
    print(f"💾 结果: results/batch_convert_16/")
    print("="*80)
    
    return results


def save_results(results, agent):
    """保存结果"""
    output_dir = Path("results/batch_convert_16")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results['end_time'] = datetime.now().isoformat()
    results['llm_stats'] = agent.llm_client.get_stats()
    results['compilation_stats'] = agent.compiler.get_stats()
    
    result_file = output_dir / "batch_results.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    asyncio.run(batch_convert())
