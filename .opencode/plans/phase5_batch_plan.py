#!/usr/bin/env python3
"""
Phase 5: Batch Convert Remaining Kernels
Phase 5: 批量转换剩余内核以达到25+目标

目标：
- 当前：7个内核有完整 harness
- 目标：25+ 个内核
- 需要：批量转换 ~18 个内核

策略：
1. 优先处理简单内核（vector ops, data conversion）
2. 使用已验证的 harness 模板
3. 并行验证（使用 ParallelRealAccuracyTester）
4. 批量生成和测试
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/.opencode/plans')
from phase1_fixed_harnesses import FIXED_HARNESSES as PHASE1_HARNESSES
from phase2_improved_harnesses import PHASE2_IMPROVED_HARNESSES

# 当前已完成的内核
COMPLETED_KERNELS = {
    # Phase 1
    'add_vectors', 'winograd_input_transform', 'add_vectors_hnc_nhc',
    'add_bias_nchw', 'nchw_to_nhwc',
    # Phase 2
    'add_bias_batched', 'global_scale'
}

# 需要转换的内核（按优先级排序）
# 优先级：简单 > 复杂，独立 > 依赖
PRIORITY_KERNELS = {
    # P0 - 最简单，data conversion
    'P0': [
        'copy_type_converted',
        'expand_planes_nhwc',
        'expand_planes_nchw',
    ],
    # P1 - 归一化操作
    'P1': [
        'batch_norm',
        'layer_norm',
        'global_scale_fp16_nhwc',
    ],
    # P2 - Pooling
    'P2': [
        'global_avg_pool',
        'global_avg_pool_nhwc_fp16',
    ],
    # P3 - Policy & Softmax
    'P3': [
        'policy_map',
        'softmax',
        'softmax_opt_64',
    ],
    # P4 - Attention operations (更复杂)
    'P4': [
        'promotion_logits',
        'preprocess_attention_body',
        'input_gating',
        'gen_offset_pointers',
        'se_layer_nhwc',
    ],
    # P5 - Winograd (最复杂)
    'P5': [
        'winograd_filter_transform',
        'winograd_output_transform',
        'winograd_output_se_relu_input',
        'winograd_output_relu_input',
        'output_input_transform_fp16_shmem',
    ],
    # P6 - CUDA only (CUTLASS)
    'P6': [
        'fused_mha_cutlass',  # NVIDIA-specific, needs custom SYCL impl
    ]
}

# 需要修复 placeholder 的内核（run_extended_accuracy_test.py 中的）
PLACEHOLDER_KERNELS = [
    'copy_type_converted',
    'batch_norm',
    'layer_norm',
    'expand_planes_nhwc',
    'expand_planes_nchw',
    'global_avg_pool',
    'global_avg_pool_nhwc_fp16',
    'policy_map',
    'softmax',
    'softmax_opt_64',
]


def calculate_progress():
    """计算当前进度"""
    total_needed = 25
    completed = len(COMPLETED_KERNELS)
    remaining = total_needed - completed
    
    all_kernels = []
    for priority, kernels in PRIORITY_KERNELS.items():
        all_kernels.extend(kernels)
    
    print("=" * 80)
    print("📊 Phase 5: Batch Convert Progress")
    print("=" * 80)
    print(f"✅ 已完成: {completed} 个内核")
    print(f"🎯 目标: {total_needed} 个内核")
    print(f"⏳ 剩余: {remaining} 个内核需要转换")
    print(f"📁 可用内核池: {len(all_kernels)} 个")
    print()
    
    for priority, kernels in PRIORITY_KERNELS.items():
        print(f"  {priority}: {len(kernels)} 个内核")
        if priority != 'P6':  # Skip CUDA-only
            for k in kernels[:3]:  # Show first 3
                status = "✅" if k in COMPLETED_KERNELS else "⏳"
                print(f"    {status} {k}")
            if len(kernels) > 3:
                print(f"    ... and {len(kernels) - 3} more")
    
    print()
    print(f"优先级策略: P0 > P1 > P2 > P3 > P4 > P5")
    print("=" * 80)
    
    return {
        'completed': completed,
        'target': total_needed,
        'remaining': remaining,
        'available': len(all_kernels)
    }


def select_kernels_for_batch(batch_size: int = 5) -> List[str]:
    """选择下一批要转换的内核"""
    selected = []
    
    # 按优先级选择
    for priority in ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']:
        for kernel in PRIORITY_KERNELS[priority]:
            if kernel not in COMPLETED_KERNELS and kernel not in selected:
                selected.append(kernel)
                if len(selected) >= batch_size:
                    return selected
    
    return selected


def create_batch_plan(batch_size: int = 5):
    """创建批量转换计划"""
    selected = select_kernels_for_batch(batch_size)
    
    print("\n" + "=" * 80)
    print(f"📋 Phase 5 Batch Plan (Batch Size: {batch_size})")
    print("=" * 80)
    
    for i, kernel in enumerate(selected, 1):
        # Find priority
        priority = "Unknown"
        for p, kernels in PRIORITY_KERNELS.items():
            if kernel in kernels:
                priority = p
                break
        
        print(f"{i}. {kernel} ({priority})")
    
    print()
    print(f"预计时间: ~{batch_size * 2} 分钟（使用并行测试）")
    print("=" * 80)
    
    return selected


if __name__ == '__main__':
    # 显示当前进度
    progress = calculate_progress()
    
    # 创建下一批计划
    if progress['remaining'] > 0:
        batch = create_batch_plan(batch_size=5)
        
        print("\n下一步操作选项:")
        print("1. 执行本批次转换 (5个内核)")
        print("2. 调整批次大小")
        print("3. 查看特定内核详情")
        print("4. 运行并行测试验证当前7个内核")
        print("\n建议选择: 执行本批次转换")
