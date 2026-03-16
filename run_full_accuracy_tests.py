#!/usr/bin/env python3
"""
运行完整的CUDA vs SYCL准确度测试套件
Run full CUDA vs SYCL accuracy test suite
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# 添加tools目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'tools'))

from llm_accuracy_test_agent import LLMAccuracyTestAgent, AccuracyTestResult

async def run_full_test_suite():
    """运行完整的测试套件"""
    
    print("=" * 80)
    print("🧪 CUDA vs SYCL 准确度完整测试套件")
    print("🧪 Full CUDA vs SYCL Accuracy Test Suite")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 加载内核索引
    with open('kernel_dataset/index.json', 'r') as f:
        index = json.load(f)
    
    # 找出所有有CUDA和SYCL版本的内核
    testable_kernels = []
    for kernel in index['kernels']:
        if kernel.get('cuda') and kernel.get('sycl'):
            testable_kernels.append(kernel)
    
    print(f"📊 发现 {len(testable_kernels)} 个可测试内核")
    print(f"📊 Found {len(testable_kernels)} testable kernels")
    print()
    
    # 创建输出目录
    output_dir = Path('results') / 'accuracy_tests' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试结果
    results = {
        'test_date': datetime.now().isoformat(),
        'total_kernels': len(testable_kernels),
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'kernels': []
    }
    
    # 逐个测试内核
    for i, kernel in enumerate(testable_kernels, 1):
        kernel_id = kernel['id']
        cuda_file = kernel['cuda']['file']
        sycl_file = kernel['sycl']['file']
        
        print(f"\n{'='*80}")
        print(f"[{i}/{len(testable_kernels)}] 测试内核: {kernel_id}")
        print(f"[{i}/{len(testable_kernels)}] Testing kernel: {kernel_id}")
        print(f"{'='*80}")
        
        # 检查文件是否存在
        cuda_path = Path('kernel_dataset') / cuda_file
        sycl_path = Path('kernel_dataset') / sycl_file
        
        if not cuda_path.exists():
            print(f"❌ CUDA文件不存在: {cuda_path}")
            results['errors'] += 1
            continue
            
        if not sycl_path.exists():
            print(f"❌ SYCL文件不存在: {sycl_path}")
            results['errors'] += 1
            continue
        
        # 创建Agent并运行测试
        agent = LLMAccuracyTestAgent(kernel_id=kernel_id)
        
        kernel_output_dir = output_dir / kernel_id
        kernel_output_dir.mkdir(exist_ok=True)
        
        try:
            result = await agent.run_full_accuracy_test(
                cuda_file=str(cuda_path),
                sycl_file=str(sycl_path),
                output_dir=str(kernel_output_dir)
            )
            
            kernel_result = {
                'kernel_id': kernel_id,
                'name': kernel['name'],
                'success': result.success,
                'duration_seconds': result.duration_seconds,
                'error': result.error
            }
            
            if result.success and result.report:
                kernel_result['report'] = result.report
                # 计算通过率
                if 'summary' in result.report:
                    summary = result.report['summary']
                    if summary.get('pass_rate', 0) >= 0.95:
                        print(f"✅ 测试通过! 通过率: {summary['pass_rate']*100:.1f}%")
                        results['passed'] += 1
                    else:
                        print(f"⚠️  测试未达标. 通过率: {summary['pass_rate']*100:.1f}%")
                        results['failed'] += 1
                else:
                    results['passed'] += 1
            else:
                print(f"❌ 测试失败: {result.error}")
                results['errors'] += 1
                
            results['kernels'].append(kernel_result)
            
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results['errors'] += 1
            results['kernels'].append({
                'kernel_id': kernel_id,
                'name': kernel['name'],
                'success': False,
                'error': str(e)
            })
    
    # 保存汇总报告
    summary_file = output_dir / 'test_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("📊 测试汇总 / Test Summary")
    print("=" * 80)
    print(f"总内核数: {results['total_kernels']}")
    print(f"通过: {results['passed']} ({results['passed']/results['total_kernels']*100:.1f}%)")
    print(f"失败: {results['failed']} ({results['failed']/results['total_kernels']*100:.1f}%)")
    print(f"错误: {results['errors']} ({results['errors']/results['total_kernels']*100:.1f}%)")
    print(f"\n详细报告: {summary_file}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return results

if __name__ == '__main__':
    asyncio.run(run_full_test_suite())
