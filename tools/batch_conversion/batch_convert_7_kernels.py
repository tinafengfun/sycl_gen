#!/usr/bin/env python3
"""
批量转换7个测试内核
Batch conversion of 7 test kernels
"""

import asyncio
import json
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_agent_v2 import EnhancedConversionAgent

async def main():
    """主函数 - 转换7个测试内核"""
    
    # 7个测试内核（从21个失败内核中选择）
    test_kernels = [
        'add_vectors',              # 基础操作
        'add_bias_batched',         # 批量操作
        'layer_norm',               # 归一化
        'softmax',                  # 激活函数
        'global_scale',             # 全局操作
        'winograd_filter_transform', # Winograd
        'se_layer_nhwc',            # SE层
    ]
    
    print("="*70)
    print("🚀 增强版Agent v2.0 - 批量转换测试")
    print("="*70)
    print(f"📊 转换内核数: {len(test_kernels)}")
    print(f"🤖 使用模型: minimax-m2.5")
    print(f"⏱️  预计时间: 30-45分钟")
    print("="*70)
    print()
    
    print("📝 测试内核列表:")
    for i, kernel in enumerate(test_kernels, 1):
        print(f"  {i}. {kernel}")
    print()
    
    # 创建Agent并运行
    agent = EnhancedConversionAgent()
    results = await agent.run_batch_conversion(test_kernels)
    
    # 保存结果
    output_dir = Path('results/batch_conversion_7')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / 'conversion_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印汇总
    print()
    print("="*70)
    print("📊 转换完成汇总")
    print("="*70)
    
    total = len(test_kernels)
    success = sum(1 for r in results if r.get('success'))
    failed = total - success
    
    print(f"  总内核数: {total}")
    print(f"  ✅ 成功: {success} ({success/total*100:.1f}%)")
    print(f"  ❌ 失败: {failed} ({failed/total*100:.1f}%)")
    print()
    print(f"📁 详细结果: {result_file}")
    print("="*70)
    
    # 验证编译
    print()
    print("🔍 验证编译状态...")
    print()
    
    import subprocess
    pass_count = 0
    for kernel in test_kernels:
        sycl_file = f"kernel_dataset/sycl/{kernel}_kernel.dp.cpp"
        if Path(sycl_file).exists():
            # 复制到docker测试编译
            try:
                subprocess.run(['docker', 'cp', sycl_file, 'lsv-container:/workspace/test.cpp'],
                              capture_output=True, timeout=10, check=True)
                result = subprocess.run(['docker', 'exec', 'lsv-container', 'bash', '-c',
                                        'cd /workspace && icpx -fsycl -c test.cpp -o test.o'],
                                       capture_output=True, timeout=30)
                if result.returncode == 0:
                    print(f"  ✅ {kernel:30s} - 编译通过")
                    pass_count += 1
                else:
                    print(f"  ❌ {kernel:30s} - 编译失败")
            except Exception as e:
                print(f"  ⚠️  {kernel:30s} - 测试异常: {e}")
    
    print()
    print(f"📊 编译验证: {pass_count}/{total} 通过 ({pass_count/total*100:.1f}%)")
    print()
    print("✅ 批量转换测试完成!")
    print("="*70)

if __name__ == '__main__':
    asyncio.run(main())
