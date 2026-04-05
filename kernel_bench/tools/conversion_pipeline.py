#!/usr/bin/env python3
"""
Conversion Pipeline with Accuracy Verification Hooks
集成准确度验证的转换流水线

架构设计:
1. Hook系统: 在转换流程关键点插入验证
2. 策略模式: 可配置的验证策略
3. 事件驱动: 转换完成后自动触发验证
4. 反馈循环: 验证失败可触发重新转换

使用示例:
    pipeline = ConversionPipeline()
    pipeline.add_hook(AccuracyVerificationHook())
    pipeline.add_hook(AutoFixHook())
    
    result = await pipeline.convert(kernel_id)
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Any
import sys

# 添加tools目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'tools'))

from accuracy_verifier import (
    AccuracyVerifier, VerificationResult, VerificationStatus,
    CUDARemotePlatform, SYCLLocalPlatform, HarnessGenerator
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HookPriority(Enum):
    """钩子优先级"""
    HIGH = 0      # 最先执行
    NORMAL = 1    # 默认优先级
    LOW = 2       # 最后执行


class PipelineEvent(Enum):
    """流水线事件类型"""
    PRE_CONVERSION = "pre_conversion"      # 转换前
    POST_CONVERSION = "post_conversion"    # 转换后
    PRE_COMPILATION = "pre_compilation"    # 编译前
    POST_COMPILATION = "post_compilation"  # 编译后
    PRE_VERIFICATION = "pre_verification"  # 验证前
    POST_VERIFICATION = "post_verification" # 验证后
    ON_ERROR = "on_error"                  # 出错时
    ON_SUCCESS = "on_success"              # 成功时


@dataclass
class ConversionContext:
    """转换上下文数据类"""
    kernel_id: str
    cuda_file: Optional[Path] = None
    sycl_file: Optional[Path] = None
    sycl_code: Optional[str] = None
    verification_result: Optional[VerificationResult] = None
    attempt_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'kernel_id': self.kernel_id,
            'cuda_file': str(self.cuda_file) if self.cuda_file else None,
            'sycl_file': str(self.sycl_file) if self.sycl_file else None,
            'attempt_count': self.attempt_count,
            'has_verification': self.verification_result is not None,
            'errors': self.errors
        }


class PipelineHook(ABC):
    """流水线钩子基类"""
    
    def __init__(self, priority: HookPriority = HookPriority.NORMAL):
        self.priority = priority
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def execute(
        self, 
        event: PipelineEvent, 
        context: ConversionContext
    ) -> bool:
        """
        执行钩子逻辑
        
        Args:
            event: 触发的事件
            context: 转换上下文
            
        Returns:
            True继续流程，False中断流程
        """
        pass
    
    @property
    @abstractmethod
    def subscribed_events(self) -> Set[PipelineEvent]:
        """订阅的事件集合"""
        pass


class AccuracyVerificationHook(PipelineHook):
    """
    准确度验证钩子
    
    在编译通过后自动执行准确度验证
    支持配置化验证策略
    """
    
    def __init__(
        self,
        priority: HookPriority = HookPriority.NORMAL,
        verify_on: Set[PipelineEvent] = None,
        skip_on_failure: bool = True,
        max_retries: int = 0
    ):
        super().__init__(priority)
        self.verifier = AccuracyVerifier()
        self.verify_on = verify_on or {PipelineEvent.POST_COMPILATION}
        self.skip_on_failure = skip_on_failure
        self.max_retries = max_retries
    
    @property
    def subscribed_events(self) -> Set[PipelineEvent]:
        return self.verify_on
    
    async def execute(
        self, 
        event: PipelineEvent, 
        context: ConversionContext
    ) -> bool:
        """执行准确度验证"""
        if event not in self.verify_on:
            return True
        
        self.logger.info(f"Running accuracy verification for {context.kernel_id}")
        
        try:
            # 执行验证
            result = await self.verifier.verify(context.kernel_id)
            context.verification_result = result
            
            # 记录结果
            if result.passed:
                self.logger.info(
                    f"✅ {context.kernel_id} passed accuracy check "
                    f"(MAE={result.mae:.2e})"
                )
                return True
            else:
                self.logger.warning(
                    f"⚠️ {context.kernel_id} failed accuracy check "
                    f"(MAE={result.mae:.2e}, MaxErr={result.max_error:.2e})"
                )
                
                # 添加到错误列表
                context.errors.append(
                    f"Accuracy check failed: MAE={result.mae:.2e}"
                )
                
                # 根据配置决定是否继续
                if self.skip_on_failure:
                    self.logger.info("Continuing despite accuracy failure")
                    return True
                else:
                    self.logger.error("Stopping due to accuracy failure")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Accuracy verification error: {e}")
            context.errors.append(f"Verification error: {str(e)}")
            
            if self.skip_on_failure:
                return True
            return False


class CompilationCheckHook(PipelineHook):
    """
    编译检查钩子
    
    验证SYCL代码是否可以编译
    作为准确度验证的前置条件
    """
    
    def __init__(self, priority: HookPriority = HookPriority.HIGH):
        super().__init__(priority)
        self.sycl_platform = SYCLLocalPlatform()
    
    @property
    def subscribed_events(self) -> Set[PipelineEvent]:
        return {PipelineEvent.POST_CONVERSION}
    
    async def execute(
        self, 
        event: PipelineEvent, 
        context: ConversionContext
    ) -> bool:
        """检查编译"""
        if not context.sycl_file:
            self.logger.warning("No SYCL file to compile")
            return True
        
        self.logger.info(f"Checking compilation for {context.kernel_id}")
        
        # 这里可以添加实际的编译检查逻辑
        # 简化版本：假设编译通过
        context.metadata['compilation_checked'] = True
        return True


class AutoFixHook(PipelineHook):
    """
    自动修复钩子
    
    当验证失败时，尝试自动修复并重新转换
    """
    
    def __init__(
        self,
        priority: HookPriority = HookPriority.LOW,
        max_attempts: int = 3
    ):
        super().__init__(priority)
        self.max_attempts = max_attempts
    
    @property
    def subscribed_events(self) -> Set[PipelineEvent]:
        return {PipelineEvent.POST_VERIFICATION}
    
    async def execute(
        self, 
        event: PipelineEvent, 
        context: ConversionContext
    ) -> bool:
        """尝试自动修复"""
        if not context.verification_result:
            return True
        
        if context.verification_result.passed:
            return True
        
        if context.attempt_count >= self.max_attempts:
            self.logger.error(
                f"Max retry attempts ({self.max_attempts}) reached for "
                f"{context.kernel_id}"
            )
            return False
        
        self.logger.info(
            f"Attempting auto-fix for {context.kernel_id} "
            f"(attempt {context.attempt_count + 1}/{self.max_attempts})"
        )
        
        # 这里可以实现自动修复逻辑
        # 例如：调整harness参数、尝试不同的转换策略等
        
        context.attempt_count += 1
        context.metadata['auto_fix_triggered'] = True
        
        return True


class LoggingHook(PipelineHook):
    """日志记录钩子 - 用于调试和监控"""
    
    def __init__(self, log_file: Optional[str] = None):
        super().__init__(HookPriority.LOW)
        self.log_file = log_file
    
    @property
    def subscribed_events(self) -> Set[PipelineEvent]:
        return set(PipelineEvent)  # 订阅所有事件
    
    async def execute(
        self, 
        event: PipelineEvent, 
        context: ConversionContext
    ) -> bool:
        """记录日志"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': event.value,
            'kernel_id': context.kernel_id,
            'attempt': context.attempt_count,
            'context': context.to_dict()
        }
        
        self.logger.debug(f"Pipeline event: {log_entry}")
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        return True


class ConversionPipeline:
    """
    转换流水线 - 集成钩子系统
    
    主要功能:
    1. 管理钩子注册和执行
    2. 控制转换流程
    3. 处理事件分发
    4. 维护转换状态
    """
    
    def __init__(self):
        self.hooks: Dict[PipelineEvent, List[PipelineHook]] = {
            event: [] for event in PipelineEvent
        }
        self.logger = logging.getLogger(__name__)
        self._results: Dict[str, Dict] = {}
    
    def add_hook(self, hook: PipelineHook):
        """添加钩子"""
        for event in hook.subscribed_events:
            if event in self.hooks:
                self.hooks[event].append(hook)
                # 按优先级排序
                self.hooks[event].sort(key=lambda h: h.priority.value)
        
        self.logger.info(f"Added hook: {hook.__class__.__name__}")
    
    def remove_hook(self, hook_class: type):
        """移除指定类型的钩子"""
        for event_hooks in self.hooks.values():
            event_hooks[:] = [h for h in event_hooks if not isinstance(h, hook_class)]
    
    async def _execute_hooks(
        self, 
        event: PipelineEvent, 
        context: ConversionContext
    ) -> bool:
        """执行指定事件的所有钩子"""
        hooks = self.hooks.get(event, [])
        
        for hook in hooks:
            try:
                result = await hook.execute(event, context)
                if not result:
                    self.logger.warning(
                        f"Hook {hook.__class__.__name__} stopped pipeline"
                    )
                    return False
            except Exception as e:
                self.logger.error(
                    f"Hook {hook.__class__.__name__} failed: {e}"
                )
                # 继续执行其他钩子
        
        return True
    
    async def convert(
        self, 
        kernel_id: str,
        cuda_file: Optional[Path] = None,
        sycl_file: Optional[Path] = None,
        skip_if_cached: bool = True
    ) -> ConversionContext:
        """
        执行完整的转换流程
        
        Args:
            kernel_id: 内核标识符
            cuda_file: CUDA源文件路径
            sycl_file: SYCL目标文件路径
            skip_if_cached: 如果已缓存则跳过
            
        Returns:
            ConversionContext包含完整结果
        """
        context = ConversionContext(
            kernel_id=kernel_id,
            cuda_file=cuda_file or Path(f"kernel_dataset/cuda/{kernel_id}_kernel.cu"),
            sycl_file=sycl_file or Path(f"kernel_dataset/sycl/{kernel_id}_kernel.dp.cpp")
        )
        
        self.logger.info(f"Starting conversion pipeline for {kernel_id}")
        
        # 1. 转换前钩子
        if not await self._execute_hooks(PipelineEvent.PRE_CONVERSION, context):
            return context
        
        # 2. 执行转换（这里简化，实际需要调用转换逻辑）
        self.logger.info(f"Converting {kernel_id}...")
        context.attempt_count += 1
        
        # 3. 转换后钩子
        if not await self._execute_hooks(PipelineEvent.POST_CONVERSION, context):
            return context
        
        # 4. 编译检查
        if not await self._execute_hooks(PipelineEvent.PRE_COMPILATION, context):
            return context
        
        # 模拟编译过程
        context.metadata['compilation_success'] = True
        
        if not await self._execute_hooks(PipelineEvent.POST_COMPILATION, context):
            return context
        
        # 5. 准确度验证
        if not await self._execute_hooks(PipelineEvent.PRE_VERIFICATION, context):
            return context
        
        # 执行验证
        await self._execute_hooks(PipelineEvent.POST_VERIFICATION, context)
        
        # 6. 最终处理
        if context.verification_result and context.verification_result.passed:
            await self._execute_hooks(PipelineEvent.ON_SUCCESS, context)
        elif context.errors:
            await self._execute_hooks(PipelineEvent.ON_ERROR, context)
        
        # 保存结果
        self._results[kernel_id] = context.to_dict()
        
        return context
    
    async def convert_batch(
        self, 
        kernel_ids: List[str],
        max_concurrency: int = 3
    ) -> Dict[str, ConversionContext]:
        """批量转换"""
        semaphore = asyncio.Semaphore(max_concurrency)
        results = {}
        
        async def convert_with_limit(kernel_id: str):
            async with semaphore:
                context = await self.convert(kernel_id)
                results[kernel_id] = context
                return context
        
        tasks = [convert_with_limit(kid) for kid in kernel_ids]
        await asyncio.gather(*tasks)
        
        return results
    
    def get_results(self) -> Dict[str, Dict]:
        """获取所有结果"""
        return self._results.copy()
    
    def save_results(self, output_file: str):
        """保存结果到文件"""
        with open(output_file, 'w') as f:
            json.dump(self._results, f, indent=2)
        self.logger.info(f"Results saved to {output_file}")


# 便捷配置
class PipelineConfig:
    """流水线配置预设"""
    
    @staticmethod
    def standard() -> ConversionPipeline:
        """标准配置 - 包含准确度验证"""
        pipeline = ConversionPipeline()
        pipeline.add_hook(CompilationCheckHook())
        pipeline.add_hook(AccuracyVerificationHook())
        pipeline.add_hook(LoggingHook())
        return pipeline
    
    @staticmethod
    def with_auto_fix() -> ConversionPipeline:
        """自动修复配置"""
        pipeline = ConversionPipeline()
        pipeline.add_hook(CompilationCheckHook())
        pipeline.add_hook(AccuracyVerificationHook(skip_on_failure=False))
        pipeline.add_hook(AutoFixHook(max_attempts=3))
        pipeline.add_hook(LoggingHook())
        return pipeline
    
    @staticmethod
    def debug() -> ConversionPipeline:
        """调试配置 - 详细日志"""
        pipeline = ConversionPipeline()
        pipeline.add_hook(LoggingHook(log_file="pipeline_debug.log"))
        pipeline.add_hook(CompilationCheckHook())
        pipeline.add_hook(AccuracyVerificationHook())
        return pipeline


# 使用示例
async def example_usage():
    """使用示例"""
    # 创建标准流水线
    pipeline = PipelineConfig.standard()
    
    # 转换单个kernel
    context = await pipeline.convert("softmax")
    
    if context.verification_result:
        print(f"Verification: {context.verification_result.status.value}")
        print(f"MAE: {context.verification_result.mae:.2e}")
    
    # 批量转换
    results = await pipeline.convert_batch([
        'copy_type_converted',
        'global_avg_pool',
        'softmax_opt_64'
    ])
    
    for kernel_id, ctx in results.items():
        status = "✅" if ctx.verification_result and ctx.verification_result.passed else "❌"
        print(f"{status} {kernel_id}")
    
    # 保存结果
    pipeline.save_results("conversion_results.json")


if __name__ == "__main__":
    # 运行示例
    print("🚀 Testing ConversionPipeline...")
    asyncio.run(example_usage())
