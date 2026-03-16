#!/usr/bin/env python3
"""
Phase 5 Batch 1 Test Runner
使用 ParallelRealAccuracyTester 测试第一批5个内核
"""

import sys
sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/.opencode/plans')

from phase1_fixed_harnesses import FIXED_HARNESSES as PHASE1_HARNESSES
from phase2_improved_harnesses import PHASE2_IMPROVED_HARNESSES
from phase5_batch1_harnesses import PHASE5_BATCH1_HARNESSES
from improvement1_parallel_tester import ParallelRealAccuracyTester

# 合并所有 harnesses
ALL_HARNESSES = {}
ALL_HARNESSES.update(PHASE1_HARNESSES)
ALL_HARNESSES.update(PHASE2_IMPROVED_HARNESSES)
ALL_HARNESSES.update(PHASE5_BATCH1_HARNESSES)

# 更新 tester 的 harness 数据库
class Phase5Tester(ParallelRealAccuracyTester):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.harness_db = ALL_HARNESSES

def main():
    print("=" * 80)
    print("🚀 Phase 5 Batch 1 - Parallel Testing")
    print("=" * 80)
    print()
    
    # 显示当前状态
    completed_kernels = set(PHASE1_HARNESSES.keys()) | set(PHASE2_IMPROVED_HARNESSES.keys())
    new_kernels = list(PHASE5_BATCH1_HARNESSES.keys())
    
    print(f"📊 当前状态:")
    print(f"  ✅ 已有内核: {len(completed_kernels)} 个")
    print(f"  🆕 新测试内核: {len(new_kernels)} 个")
    print(f"  🎯 目标: 25+ 个内核")
    print()
    
    print("📝 新测试内核列表:")
    for i, kernel in enumerate(new_kernels, 1):
        print(f"  {i}. {kernel}")
    print()
    
    # 创建 tester
    tester = Phase5Tester(max_workers=3)
    
    print("⚙️  配置:")
    print(f"  并行度: {tester.max_workers}")
    print(f"  CUDA Host: {tester.cuda_host}")
    print(f"  CUDA Container: {tester.cuda_container}")
    print(f"  SYCL Container: {tester.sycl_container}")
    print()
    
    # 询问是否开始测试
    print("准备开始并行测试这5个内核...")
    print("预计时间: ~2-3分钟")
    print()
    
    # 执行测试
    try:
        results = tester.batch_test_parallel(new_kernels)
        
        # 显示详细结果
        print("\n" + "=" * 80)
        print("📋 详细测试结果")
        print("=" * 80)
        
        for result in results['results']:
            kid = result['kernel_id']
            if result.get('passed'):
                print(f"✅ {kid:30s} - MAE: {result.get('mae', 0):.2e}, Max Err: {result.get('max_error', 0):.2e}")
            else:
                print(f"❌ {kid:30s} - {result.get('details', {}).get('status', 'failed')}")
        
        print("=" * 80)
        
        # 计算新进度
        passed_new = sum(1 for r in results['results'] if r.get('passed'))
        total_now = len(completed_kernels) + passed_new
        remaining = max(0, 25 - total_now)
        
        print(f"\n📊 进度更新:")
        print(f"  ✅ 本次通过: {passed_new}/{len(new_kernels)} 个")
        print(f"  📈 总计完成: {total_now} 个内核")
        print(f"  ⏳ 距离25目标: {remaining} 个")
        print()
        
        if passed_new == len(new_kernels):
            print("🎉 恭喜！全部5个内核测试通过！")
            print("\n下一步建议:")
            print("  1. 继续 Phase 5 Batch 2 (再转换5个内核)")
            print("  2. 修复任何失败的 harnesses")
            print("  3. 运行完整测试验证所有内核")
        else:
            print("⚠️  部分内核测试失败，需要修复")
            print("\n建议:")
            print("  1. 检查失败内核的 CUDA/SYCL 代码")
            print("  2. 对比输出差异")
            print("  3. 调整 tolerance 阈值")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main()
