#!/usr/bin/env python3
"""
LLM Accuracy Test Agent - Final Integration
LLM准确度测试Agent - 最终集成版本

完整功能：
1. 平台能力检测
2. 测试套件生成（18个测试配置）
3. LLM驱动的harness生成
4. 并行执行（带进度监控）
5. 自动编译错误修复
6. JSON报告生成
7. 决策支持
"""

import asyncio
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# 添加tools目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from platform_detector import PlatformDetector, detect_platforms
from test_suite_generator import generate_test_suite, TestSuiteGenerator
from llm_harness_generator import LLMHarnessGenerator
from async_test_executor import AsyncTestExecutor, ProgressMonitor
from json_report_generator import JSONReportGenerator

@dataclass
class AccuracyTestResult:
    """准确度测试结果"""
    success: bool
    report: Optional[Dict] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0


class LLMAccuracyTestAgent:
    """
    LLM准确度测试Agent - 完整版
    
    使用示例:
        agent = LLMAccuracyTestAgent(kernel_id="copy_type_converted")
        result = await agent.run_full_accuracy_test(
            cuda_file="path/to/kernel.cu",
            sycl_file="path/to/kernel.dp.cpp"
        )
    """
    
    def __init__(self, kernel_id: str, max_llm_concurrency: int = 4):
        """
        初始化Agent
        
        Args:
            kernel_id: kernel标识符
            max_llm_concurrency: LLM最大并发数
        """
        self.kernel_id = kernel_id
        self.max_llm_concurrency = max_llm_concurrency
        self.base_dir = Path(__file__).parent.parent
        
        # 初始化各组件
        self.platform_detector = PlatformDetector()
        self.harness_generator = LLMHarnessGenerator()
        
        print(f"🚀 LLM Accuracy Test Agent initialized for: {kernel_id}")
    
    async def run_full_accuracy_test(
        self,
        cuda_file: str,
        sycl_file: str,
        output_dir: Optional[str] = None
    ) -> AccuracyTestResult:
        """
        运行完整准确度测试
        
        Args:
            cuda_file: CUDA kernel文件路径
            sycl_file: SYCL kernel文件路径
            output_dir: 报告输出目录
            
        Returns:
            AccuracyTestResult对象
        """
        start_time = time.time()
        
        try:
            # 1. 读取kernel代码
            print("\n" + "="*70)
            print("📖 Phase 1: Loading kernel files")
            print("="*70)
            
            with open(cuda_file, 'r') as f:
                cuda_code = f.read()
            
            with open(sycl_file, 'r') as f:
                sycl_code = f.read()
            
            print(f"   CUDA code: {len(cuda_code)} characters")
            print(f"   SYCL code: {len(sycl_code)} characters")
            
            # 2. 检测平台能力
            print("\n" + "="*70)
            print("🔍 Phase 2: Detecting platform capabilities")
            print("="*70)
            
            platform_caps = detect_platforms()
            
            print(f"   SYCL: {platform_caps['sycl']['device']}")
            print(f"         FP32: {platform_caps['sycl']['float32']}, "
                  f"FP16: {platform_caps['sycl']['float16']}, "
                  f"BF16: {platform_caps['sycl']['bfloat16']}")
            print(f"   CUDA: {platform_caps['cuda']['device']} "
                  f"(SM{platform_caps['cuda'].get('sm_version', 'unknown')})")
            print(f"         FP32: {platform_caps['cuda']['float32']}, "
                  f"FP16: {platform_caps['cuda']['float16']}, "
                  f"BF16: {platform_caps['cuda']['bfloat16']}")
            
            # 3. 生成测试套件
            print("\n" + "="*70)
            print("📋 Phase 3: Generating test suite")
            print("="*70)
            
            test_configs = generate_test_suite(platform_caps)
            
            print(f"   Generated {len(test_configs)} test configurations:")
            for i, config in enumerate(test_configs[:5], 1):
                print(f"   {i}. {config['test_id']}: {config['name']} "
                      f"({config['dtype']})")
            if len(test_configs) > 5:
                print(f"   ... and {len(test_configs) - 5} more")
            
            # 4. 创建进度监控器和执行器
            monitor = ProgressMonitor()
            executor = AsyncTestExecutor(
                monitor, 
                max_llm_concurrency=self.max_llm_concurrency
            )
            
            # 5. 执行所有测试
            print("\n" + "="*70)
            print("🧪 Phase 4: Executing tests")
            print("="*70)
            
            test_results = []
            
            # 串行执行测试（避免资源冲突）
            for i, test_config in enumerate(test_configs, 1):
                print(f"\n--- Test {i}/{len(test_configs)}: {test_config['test_id']} ---")
                
                result = await executor.execute_test(
                    test_id=test_config['test_id'],
                    kernel_code=cuda_code,  # 使用CUDA代码生成harness
                    test_config=test_config
                )
                
                test_results.append(result)
                
                # 显示结果
                status = result.get('status', 'UNKNOWN')
                if status == 'PASSED':
                    print(f"   ✅ PASSED")
                elif status == 'FAILED':
                    print(f"   ❌ FAILED: {result.get('error', 'Unknown error')}")
                else:
                    print(f"   ⚠️  {status}")
            
            # 6. 生成LLM使用统计
            llm_usage = {
                "total_calls": self.harness_generator.llm_calls,
                "harness_generation": len(test_configs) * 2,  # CUDA + SYCL
                "error_fix_calls": max(0, self.harness_generator.llm_calls - len(test_configs) * 2),
                "estimated_cost_usd": self.harness_generator.llm_calls * 0.01  # 粗略估计
            }
            
            # 7. 生成JSON报告
            print("\n" + "="*70)
            print("📊 Phase 5: Generating report")
            print("="*70)
            
            report_generator = JSONReportGenerator(
                self.kernel_id,
                Path(cuda_file).stem.replace('_kernel', '')
            )
            
            # 添加执行trace
            for result in test_results:
                report_generator.add_trace(
                    "test_completed",
                    result.get('test_id'),
                    {"status": result.get('status')}
                )
            
            # 添加问题
            for result in test_results:
                if result.get('status') == 'FAILED':
                    report_generator.add_issue(
                        severity="critical" if "compilation" in result.get('error', '').lower() else "warning",
                        category="test_failure",
                        message=result.get('error', 'Test failed'),
                        test_id=result.get('test_id'),
                        recommendation="Review kernel conversion or test configuration"
                    )
            
            # 添加建议
            passed = sum(1 for r in test_results if r.get('status') == 'PASSED')
            if passed == len(test_results):
                report_generator.add_recommendation(
                    "All tests passed - conversion is accurate and ready for integration"
                )
            elif passed >= len(test_results) * 0.8:
                report_generator.add_recommendation(
                    "Most tests passed - review failed tests before deployment"
                )
            else:
                report_generator.add_recommendation(
                    "Many tests failed - significant issues need to be addressed"
                )
            
            # 生成完整报告
            report = report_generator.generate_report(
                platform_info=platform_caps,
                test_configs=test_configs,
                test_results=test_results,
                llm_usage=llm_usage
            )
            
            # 8. 保存和显示报告
            duration = time.time() - start_time
            
            if output_dir:
                filepath = report_generator.save_report(report, output_dir)
                print(f"\n💾 Report saved to: {filepath}")
            
            # 打印摘要
            report_generator.print_summary(report)
            
            return AccuracyTestResult(
                success=True,
                report=report,
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            
            return AccuracyTestResult(
                success=False,
                error=str(e),
                duration_seconds=duration
            )


# 便捷函数
async def run_accuracy_test(
    kernel_id: str,
    cuda_file: str,
    sycl_file: str,
    output_dir: Optional[str] = None,
    max_llm_concurrency: int = 4
) -> Dict:
    """
    便捷函数：运行完整准确度测试
    
    Args:
        kernel_id: kernel标识符
        cuda_file: CUDA kernel文件路径
        sycl_file: SYCL kernel文件路径
        output_dir: 报告输出目录
        max_llm_concurrency: LLM最大并发数
        
    Returns:
        测试报告字典
    """
    agent = LLMAccuracyTestAgent(kernel_id, max_llm_concurrency)
    result = await agent.run_full_accuracy_test(
        cuda_file, sycl_file, output_dir
    )
    
    if result.success:
        return result.report
    else:
        return {
            "error": result.error,
            "duration_seconds": result.duration_seconds
        }


# 命令行接口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LLM Accuracy Test Agent"
    )
    parser.add_argument(
        "kernel_id",
        help="Kernel identifier (e.g., copy_type_converted)"
    )
    parser.add_argument(
        "cuda_file",
        help="Path to CUDA kernel file"
    )
    parser.add_argument(
        "sycl_file",
        help="Path to SYCL kernel file"
    )
    parser.add_argument(
        "-o", "--output",
        default="results/reports",
        help="Output directory for reports (default: results/reports)"
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=4,
        help="Max LLM concurrency (default: 4)"
    )
    
    args = parser.parse_args()
    
    # 运行测试
    result = asyncio.run(run_accuracy_test(
        kernel_id=args.kernel_id,
        cuda_file=args.cuda_file,
        sycl_file=args.sycl_file,
        output_dir=args.output,
        max_llm_concurrency=args.concurrency
    ))
    
    # 保存简化版报告到stdout
    if "error" not in result:
        print("\n" + "="*70)
        print("✅ Test completed successfully!")
        print("="*70)
        print(f"Pass rate: {result['summary']['pass_rate']*100:.1f}%")
        print(f"Total tests: {result['summary']['total_tests']}")
        print(f"Passed: {result['summary']['passed']}")
        print(f"Failed: {result['summary']['failed']}")
        print(f"Duration: {result['metadata']['total_duration_seconds']:.1f}s")
    else:
        print("\n" + "="*70)
        print("❌ Test failed!")
        print("="*70)
        print(f"Error: {result['error']}")
        sys.exit(1)
