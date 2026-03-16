#!/usr/bin/env python3
"""
Integrated Conversion Agent with Accuracy Verification
集成准确度验证的完整转换Agent

这是最终集成模块，将以下组件整合：
1. Enhanced Agent v2.0 - CUDA到SYCL转换
2. Accuracy Verifier - 准确度验证
3. Conversion Pipeline - 流水线管理
4. Smart Fix Agent - 自动修复

主要特性:
- 转换完成后自动验证准确度
- 验证失败时自动重试和修复
- 支持批量处理
- 提供详细的进度和报告

使用示例:
    agent = IntegratedConversionAgent()
    
    # 启用准确度验证
    agent.enable_accuracy_verification(
        auto_fix=True,
        max_attempts=3
    )
    
    # 转换并验证
    result = await agent.convert_kernel("softmax")
    
    # 批量处理
    results = await agent.convert_batch([
        'copy_type_converted',
        'global_avg_pool',
        'softmax'
    ])
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import sys

# 添加tools目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'tools'))

from accuracy_verifier import (
    AccuracyVerifier, VerificationResult, VerificationStatus,
    ToleranceConfig
)
from conversion_pipeline import (
    ConversionPipeline, PipelineConfig,
    AccuracyVerificationHook, AutoFixHook, CompilationCheckHook,
    PipelineEvent, ConversionContext
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ConversionReport:
    """转换报告"""
    kernel_id: str
    conversion_success: bool = False
    compilation_success: bool = False
    verification_success: bool = False
    verification_result: Optional[VerificationResult] = None
    attempts: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def overall_success(self) -> bool:
        """整体成功（转换+编译+验证）"""
        return (self.conversion_success and 
                self.compilation_success and 
                self.verification_success)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'kernel_id': self.kernel_id,
            'conversion_success': self.conversion_success,
            'compilation_success': self.compilation_success,
            'verification_success': self.verification_success,
            'verification_result': self.verification_result.to_dict() if self.verification_result else None,
            'attempts': self.attempts,
            'errors': self.errors,
            'duration_seconds': self.duration_seconds,
            'timestamp': self.timestamp,
            'overall_success': self.overall_success
        }


class IntegratedConversionAgent:
    """
    集成转换Agent - 完整解决方案
    
    功能:
    1. CUDA到SYCL代码转换
    2. 编译验证
    3. 准确度验证
    4. 自动修复
    5. 批量处理
    6. 详细报告
    """
    
    def __init__(
        self,
        cuda_host: str = "10.112.229.160",
        sycl_container: str = "lsv-container",
        output_dir: str = "results/integrated_conversion"
    ):
        """
        初始化Agent
        
        Args:
            cuda_host: CUDA远程主机
            sycl_container: SYCL容器名称
            output_dir: 输出目录
        """
        self.cuda_host = cuda_host
        self.sycl_container = sycl_container
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 核心组件
        self.pipeline = ConversionPipeline()
        self.verifier = AccuracyVerifier()
        
        # 配置
        self._accuracy_verification_enabled = False
        self._auto_fix_enabled = False
        self._max_attempts = 1
        
        # 统计
        self._stats = {
            'total': 0,
            'converted': 0,
            'compiled': 0,
            'verified': 0,
            'failed': 0
        }
        
        self._reports: List[ConversionReport] = []
        
        self.logger = logging.getLogger(__name__)
    
    def enable_accuracy_verification(
        self,
        auto_fix: bool = True,
        max_attempts: int = 3,
        skip_on_failure: bool = False,
        custom_tolerance: Optional[ToleranceConfig] = None
    ):
        """
        启用准确度验证
        
        Args:
            auto_fix: 验证失败时是否自动修复
            max_attempts: 最大尝试次数
            skip_on_failure: 验证失败时是否继续
            custom_tolerance: 自定义容忍度配置
        """
        self._accuracy_verification_enabled = True
        self._auto_fix_enabled = auto_fix
        self._max_attempts = max_attempts
        
        # 配置流水线
        self.pipeline = ConversionPipeline()
        
        # 添加编译检查钩子
        self.pipeline.add_hook(CompilationCheckHook())
        
        # 添加准确度验证钩子
        verification_hook = AccuracyVerificationHook(
            skip_on_failure=skip_on_failure,
            max_retries=max_attempts if auto_fix else 0
        )
        if custom_tolerance:
            verification_hook.verifier.tolerance = custom_tolerance
        self.pipeline.add_hook(verification_hook)
        
        # 添加自动修复钩子
        if auto_fix:
            self.pipeline.add_hook(AutoFixHook(max_attempts=max_attempts))
        
        self.logger.info(
            f"Accuracy verification enabled: auto_fix={auto_fix}, "
            f"max_attempts={max_attempts}"
        )
    
    def disable_accuracy_verification(self):
        """禁用准确度验证"""
        self._accuracy_verification_enabled = False
        self.pipeline = ConversionPipeline()  # 重置为空的流水线
        self.logger.info("Accuracy verification disabled")
    
    async def convert_kernel(
        self, 
        kernel_id: str,
        skip_if_exists: bool = True
    ) -> ConversionReport:
        """
        转换单个kernel
        
        Args:
            kernel_id: 内核标识符
            skip_if_exists: 如果结果已存在则跳过
            
        Returns:
            ConversionReport
        """
        import time
        start_time = time.time()
        
        report = ConversionReport(kernel_id=kernel_id)
        
        self.logger.info(f"=" * 70)
        self.logger.info(f"🚀 Converting kernel: {kernel_id}")
        self.logger.info(f"=" * 70)
        
        try:
            # 使用流水线执行转换
            context = await self.pipeline.convert(
                kernel_id=kernel_id,
                skip_if_cached=skip_if_exists
            )
            
            # 从context提取结果
            report.attempts = context.attempt_count
            report.conversion_success = True  # 转换成功
            report.compilation_success = context.metadata.get('compilation_success', False)
            
            if context.verification_result:
                report.verification_result = context.verification_result
                report.verification_success = context.verification_result.passed
            
            report.errors = context.errors
            
        except Exception as e:
            self.logger.error(f"Conversion failed: {e}")
            report.errors.append(str(e))
        
        report.duration_seconds = time.time() - start_time
        
        # 更新统计
        self._stats['total'] += 1
        if report.conversion_success:
            self._stats['converted'] += 1
        if report.compilation_success:
            self._stats['compiled'] += 1
        if report.verification_success:
            self._stats['verified'] += 1
        if not report.overall_success:
            self._stats['failed'] += 1
        
        # 保存报告
        self._reports.append(report)
        self._log_report(report)
        
        return report
    
    def _log_report(self, report: ConversionReport):
        """记录报告摘要"""
        status = "✅" if report.overall_success else "❌"
        self.logger.info(f"{status} Kernel: {report.kernel_id}")
        self.logger.info(f"   Conversion: {'✅' if report.conversion_success else '❌'}")
        self.logger.info(f"   Compilation: {'✅' if report.compilation_success else '❌'}")
        self.logger.info(f"   Verification: {'✅' if report.verification_success else '❌'}")
        
        if report.verification_result:
            self.logger.info(
                f"   MAE: {report.verification_result.mae:.2e}, "
                f"MaxErr: {report.verification_result.max_error:.2e}"
            )
        
        if report.errors:
            self.logger.warning(f"   Errors: {len(report.errors)}")
    
    async def convert_batch(
        self,
        kernel_ids: List[str],
        max_concurrency: int = 3,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, ConversionReport]:
        """
        批量转换kernel
        
        Args:
            kernel_ids: 内核ID列表
            max_concurrency: 最大并发数
            progress_callback: 进度回调函数
            
        Returns:
            {kernel_id: ConversionReport}
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        results = {}
        
        async def convert_single(kernel_id: str):
            async with semaphore:
                report = await self.convert_kernel(kernel_id)
                results[kernel_id] = report
                
                if progress_callback:
                    progress_callback(kernel_id, report)
                
                return report
        
        # 执行所有任务
        tasks = [convert_single(kid) for kid in kernel_ids]
        await asyncio.gather(*tasks)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self._stats['total']
        if total == 0:
            return self._stats
        
        return {
            **self._stats,
            'conversion_rate': self._stats['converted'] / total,
            'compilation_rate': self._stats['compiled'] / total,
            'verification_rate': self._stats['verified'] / total,
            'success_rate': self._stats['verified'] / total
        }
    
    def generate_summary_report(self) -> str:
        """生成摘要报告"""
        stats = self.get_statistics()
        total = stats['total']
        
        lines = [
            "=" * 70,
            "📊 Conversion Summary Report",
            "=" * 70,
            f"Total kernels: {total}",
            f"Converted: {stats['converted']} ({stats.get('conversion_rate', 0)*100:.1f}%)",
            f"Compiled: {stats['compiled']} ({stats.get('compilation_rate', 0)*100:.1f}%)",
            f"Verified: {stats['verified']} ({stats.get('verification_rate', 0)*100:.1f}%)",
            f"Failed: {stats['failed']}",
            "=" * 70,
            "✅ Successful kernels:",
        ]
        
        for report in self._reports:
            if report.overall_success:
                lines.append(f"  • {report.kernel_id}")
        
        lines.append("=" * 70)
        lines.append("❌ Failed kernels:")
        
        for report in self._reports:
            if not report.overall_success:
                lines.append(f"  • {report.kernel_id}")
                if report.errors:
                    lines.append(f"    Error: {report.errors[0][:80]}")
        
        lines.append("=" * 70)
        
        return '\n'.join(lines)
    
    def save_reports(self, filename: Optional[str] = None):
        """保存所有报告到文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversion_reports_{timestamp}.json"
        
        output_file = self.output_dir / filename
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'reports': [r.to_dict() for r in self._reports]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Reports saved to: {output_file}")
        
        # 同时保存文本摘要
        summary_file = output_file.with_suffix('.txt')
        with open(summary_file, 'w') as f:
            f.write(self.generate_summary_report())
        
        return output_file


# 便捷函数
async def quick_convert(
    kernel_id: str,
    enable_verification: bool = True
) -> ConversionReport:
    """
    快速转换单个kernel
    
    Example:
        >>> report = await quick_convert("softmax")
        >>> print(f"Success: {report.overall_success}")
    """
    agent = IntegratedConversionAgent()
    
    if enable_verification:
        agent.enable_accuracy_verification(auto_fix=True, max_attempts=3)
    
    return await agent.convert_kernel(kernel_id)


async def batch_convert_with_verification(
    kernel_ids: List[str],
    max_concurrency: int = 3
) -> Dict[str, ConversionReport]:
    """
    批量转换并验证
    
    Example:
        >>> results = await batch_convert_with_verification([
        ...     'copy_type_converted',
        ...     'global_avg_pool',
        ...     'softmax'
        ... ])
    """
    agent = IntegratedConversionAgent()
    agent.enable_accuracy_verification(auto_fix=True, max_attempts=2)
    
    results = await agent.convert_batch(kernel_ids, max_concurrency)
    
    # 保存报告
    agent.save_reports()
    
    # 打印摘要
    print(agent.generate_summary_report())
    
    return results


# 演示
async def demo():
    """演示集成Agent的使用"""
    print("🚀 Integrated Conversion Agent Demo")
    print("=" * 70)
    
    # 创建Agent
    agent = IntegratedConversionAgent()
    
    # 启用准确度验证
    agent.enable_accuracy_verification(
        auto_fix=True,
        max_attempts=2,
        skip_on_failure=True  # 验证失败也继续
    )
    
    # 测试kernel列表
    test_kernels = [
        'copy_type_converted',
        'global_avg_pool',
        'softmax'
    ]
    
    print(f"Converting {len(test_kernels)} kernels...\n")
    
    # 批量转换
    results = await agent.convert_batch(test_kernels, max_concurrency=2)
    
    # 打印摘要
    print("\n" + agent.generate_summary_report())
    
    # 保存报告
    agent.save_reports("demo_report.json")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(demo())
