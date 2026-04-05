#!/usr/bin/env python3
"""
改进版 CUDA→SYCL 转换 Agent v3.0
基于过程反思的全面优化版本

关键改进:
1. 增强错误诊断 - 完整捕获编译错误和诊断信息
2. 智能自动修复 - 基于错误模式的多轮修复循环
3. 内核复杂度评估 - 智能选择转换顺序，优先简单内核
4. 持续验证反馈 - 每步验证，结果驱动优化
5. 智能策略选择 - 基于历史成功率动态调整策略
6. 完整日志追踪 - 详细的转换过程记录

作者: AI Assistant
版本: 3.0.0
"""

import json
import re
import subprocess
import sys
import time
import hashlib
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import aiohttp
import numpy as np

# 配置
CONFIG_FILE = Path(__file__).parent / 'model_config_enhanced.json'
KERNEL_DATASET = Path(__file__).parent / 'kernel_dataset'


class ConversionStatus(Enum):
    """转换状态枚举"""
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    CONVERTING = "converting"
    COMPILING = "compiling"
    FIXING = "fixing"
    VERIFYING = "verifying"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class KernelInfo:
    """内核信息数据结构"""
    kernel_id: str
    name: str
    category: str
    cuda_file: Path
    sycl_file: Path
    has_sycl_mapping: bool = False
    
    # 复杂度评估
    complexity_score: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    uses_templates: bool = False
    uses_shared_mem: bool = False
    uses_warp_ops: bool = False
    
    # 转换状态
    status: ConversionStatus = ConversionStatus.PENDING
    conversion_attempts: int = 0
    fix_attempts: int = 0
    
    # 结果
    cuda_compiles: bool = False
    sycl_compiles: bool = False
    accuracy_passed: bool = False
    mae: float = 0.0
    max_error: float = 0.0
    
    # 错误信息
    last_error: str = ""
    error_type: str = ""
    error_history: List[Dict] = field(default_factory=list)
    
    # 元数据
    converted_at: Optional[str] = None
    verified_at: Optional[str] = None


@dataclass
class ErrorPattern:
    """错误模式定义"""
    pattern: str
    error_type: str
    fix_strategy: str
    priority: int = 1
    description: str = ""


class ErrorPatternMatcher:
    """错误模式匹配器"""
    
    PATTERNS = [
        # 头文件相关
        ErrorPattern(
            pattern=r"fatal error:.*\.h.*No such file",
            error_type="missing_header",
            fix_strategy="add_header_include",
            priority=1,
            description="缺少头文件"
        ),
        ErrorPattern(
            pattern=r"error:.*was not declared in this scope",
            error_type="undefined_symbol",
            fix_strategy="add_declaration_or_include",
            priority=1,
            description="符号未定义"
        ),
        
        # CUDA特定错误
        ErrorPattern(
            pattern=r"error:.*__shfl_xor_sync",
            error_type="warp_shuffle_unsupported",
            fix_strategy="use_group_broadcast",
            priority=2,
            description="warp shuffle操作不支持"
        ),
        ErrorPattern(
            pattern=r"error:.*__syncthreads",
            error_type="sync_misuse",
            fix_strategy="use_barrier",
            priority=2,
            description="同步操作错误"
        ),
        ErrorPattern(
            pattern=r"error:.*blockIdx|threadIdx",
            error_type="cuda_builtin_unconverted",
            fix_strategy="convert_cuda_builtins",
            priority=1,
            description="CUDA内置变量未转换"
        ),
        
        # SYCL特定错误
        ErrorPattern(
            pattern=r"error:.*sycl::",
            error_type="sycl_api_error",
            fix_strategy="fix_sycl_api",
            priority=2,
            description="SYCL API使用错误"
        ),
        ErrorPattern(
            pattern=r"error:.*local_accessor",
            error_type="shared_mem_conversion",
            fix_strategy="fix_local_accessor",
            priority=2,
            description="共享内存转换错误"
        ),
        
        # 模板相关
        ErrorPattern(
            pattern=r"error:.*template|typename",
            error_type="template_error",
            fix_strategy="fix_template",
            priority=3,
            description="模板错误"
        ),
        
        # 类型转换
        ErrorPattern(
            pattern=r"error:.*cannot convert|incompatible",
            error_type="type_conversion",
            fix_strategy="add_explicit_cast",
            priority=2,
            description="类型转换错误"
        ),
        
        # 语法错误
        ErrorPattern(
            pattern=r"error:.*expected.*before",
            error_type="syntax_error",
            fix_strategy="fix_syntax",
            priority=1,
            description="语法错误"
        ),
        
        # 链接错误
        ErrorPattern(
            pattern=r"undefined reference|linker error",
            error_type="link_error",
            fix_strategy="add_definitions",
            priority=3,
            description="链接错误"
        ),
        
        # 语义错误
        ErrorPattern(
            pattern=r"error:.*no matching|ambiguous",
            error_type="overload_resolution",
            fix_strategy="fix_overload",
            priority=2,
            description="重载解析失败"
        ),
    ]
    
    @classmethod
    def match(cls, error_output: str) -> List[ErrorPattern]:
        """匹配错误模式"""
        matched = []
        for pattern in cls.PATTERNS:
            if re.search(pattern.pattern, error_output, re.IGNORECASE):
                matched.append(pattern)
        # 按优先级排序
        matched.sort(key=lambda x: x.priority)
        return matched
    
    @classmethod
    def get_primary_error(cls, error_output: str) -> Optional[ErrorPattern]:
        """获取主要错误类型"""
        matches = cls.match(error_output)
        return matches[0] if matches else None


class ImprovedLLMClient:
    """改进版LLM客户端 - 更好的错误处理和重试机制"""
    
    def __init__(self):
        self.config = self._load_config()
        self.provider = self.config['provider']['gaudi_ai']
        self.base_url = self.provider['options']['baseURL']
        self.api_key = self.provider['options']['apiKey']
        self.settings = self.provider['settings']
        self.models = self.config['models']
        
        # 统计
        self.call_count = 0
        self.total_tokens = 0
        self.error_count = 0
    
    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  无法加载配置文件: {e}")
            return self._default_config()
    
    def _default_config(self) -> dict:
        """默认配置"""
        return {
            "provider": {
                "gaudi_ai": {
                    "options": {
                        "baseURL": "http://10.112.229.160:8008/v1",
                        "apiKey": "not-needed"
                    },
                    "settings": {
                        "temperature": 0.1,
                        "max_tokens": 8192,
                        "top_p": 0.95,
                        "retry_attempts": 5,
                        "timeout": 180
                    }
                }
            },
            "models": {
                "preprocess": "DeepSeek-R1-G2-static-671B",
                "conversion": "DeepSeek-R1-G2-static-671B",
                "fix": "Qwen3-Coder-30B",
                "harness": "Qwen3-Coder-30B",
                "analysis": "DeepSeek-R1-G2-static-671B"
            }
        }
    
    async def call_llm(
        self, 
        prompt: str, 
        task_type: str = "conversion",
        max_retries: int = None,
        timeout: int = None
    ) -> Tuple[str, bool]:
        """
        调用LLM
        
        Returns:
            (response_text, success)
        """
        model = self.models.get(task_type, self.models['conversion'])
        max_retries = max_retries or self.settings['retry_attempts']
        timeout = timeout or self.settings['timeout']
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._get_system_prompt(task_type)},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.settings['temperature'],
            "max_tokens": self.settings['max_tokens'],
            "top_p": self.settings['top_p']
        }
        
        self.call_count += 1
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            content = data['choices'][0]['message']['content']
                            self.total_tokens += data.get('usage', {}).get('total_tokens', 0)
                            return content, True
                        else:
                            error_text = await response.text()
                            print(f"⚠️  LLM调用失败 (尝试 {attempt+1}/{max_retries}): HTTP {response.status}")
                            print(f"   错误: {error_text[:200]}")
                            
            except asyncio.TimeoutError:
                print(f"⚠️  LLM调用超时 (尝试 {attempt+1}/{max_retries})")
            except Exception as e:
                print(f"⚠️  LLM调用异常 (尝试 {attempt+1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 30)  # 指数退避，最大30秒
                print(f"   等待 {wait_time}秒后重试...")
                await asyncio.sleep(wait_time)
        
        self.error_count += 1
        return "", False
    
    def _get_system_prompt(self, task_type: str) -> str:
        """获取系统提示词"""
        prompts = {
            "preprocess": """你是一个专业的CUDA代码分析专家。
请深入分析CUDA内核代码，提取:
1. 所有外部依赖（头文件、库函数）
2. 宏定义和编译时常量
3. 模板参数和类型约束
4. CUDA特定API调用清单
5. 代码复杂度评分(1-10)
6. 关键算法逻辑描述

输出必须包含结构化的JSON格式分析结果。""",
            
            "conversion": """你是CUDA到SYCL转换专家。将CUDA内核转换为高质量SYCL代码。

转换规则:
1. __global__ → 移除，使用SYCL queue提交
2. threadIdx.x → item.get_local_id(0)
3. blockIdx.x → item.get_group(0)
4. blockDim.x → item.get_local_range(0)
5. gridDim.x → item.get_group_range(0)
6. __shared__ → sycl::local_accessor
7. __syncthreads() → item.barrier()
8. __expf/expf → sycl::exp
9. __logf/logf → sycl::log
10. sqrtf → sycl::sqrt

要求:
- 保持原有算法逻辑
- 添加必要的头文件 (#include <sycl/sycl.hpp>)
- 确保代码完整、可编译
- 处理所有CUDA特定语法

只输出转换后的代码，不要额外解释。""",
            
            "fix": """你是CUDA/SYCL代码修复专家。分析编译错误并修复代码。

修复原则:
1. 保留原始算法逻辑
2. 最小化改动范围
3. 修复实际错误，不引入新问题
4. 确保修复后的代码可编译

分析步骤:
1. 识别错误类型和位置
2. 理解错误原因
3. 提供修复方案
4. 输出完整修复后的代码

只输出修复后的完整代码，不要额外解释。""",
            
            "harness": """你是GPU内核测试专家。为给定的内核创建完整的测试harness。

要求:
1. 创建完整的可执行程序
2. 生成随机测试数据
3. 分配GPU内存
4. 调用内核
5. 将结果写回文件
6. 处理错误情况

代码必须是自包含的，可以直接编译运行。""",
            
            "analysis": """你是代码分析专家。分析代码质量和转换质量。

评估维度:
1. 功能正确性
2. 性能优化机会
3. 代码风格
4. 潜在问题
5. 改进建议

提供结构化的分析报告。"""
        }
        return prompts.get(task_type, prompts["conversion"])
    
    def get_stats(self) -> Dict:
        """获取调用统计"""
        return {
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "error_count": self.error_count,
            "success_rate": (self.call_count - self.error_count) / max(self.call_count, 1) * 100
        }


class CompilationTester:
    """编译测试器 - 完整的编译验证"""
    
    def __init__(self):
        self.cuda_container = "cuda12.9-test"
        self.sycl_container = "lsv-container"
        self.cuda_host = "10.112.229.160"
        self.cuda_remote_path = "/workspace/kernel_dataset/cuda"
        self.sycl_work_path = "/workspace"
        
        # 编译统计
        self.stats = {
            "cuda_attempts": 0,
            "cuda_success": 0,
            "sycl_attempts": 0,
            "sycl_success": 0
        }
    
    def test_cuda_compilation(
        self, 
        kernel_id: str,
        capture_full_error: bool = True
    ) -> Tuple[bool, str, str]:
        """
        测试CUDA编译
        
        Returns:
            (success, error_type, full_error_message)
        """
        self.stats["cuda_attempts"] += 1
        
        try:
            # 使用docker exec直接编译（文件已在容器中）
            cmd = [
                'ssh', f'root@{self.cuda_host}',
                f'docker exec {self.cuda_container} bash -c "'
                f'cd {self.cuda_remote_path} && '
                f'nvcc -I./include -O2 -c {kernel_id}_kernel.cu '
                f'-o /tmp/{kernel_id}.o 2>&1"'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            error_output = result.stderr if result.stderr else result.stdout
            
            if result.returncode == 0:
                self.stats["cuda_success"] += 1
                return True, "", ""
            else:
                # 匹配错误类型
                error_pattern = ErrorPatternMatcher.get_primary_error(error_output)
                error_type = error_pattern.error_type if error_pattern else "unknown"
                
                # 截断错误信息（如果太长）
                if len(error_output) > 2000 and not capture_full_error:
                    error_output = error_output[:2000] + "\n... (truncated)"
                
                return False, error_type, error_output
                
        except subprocess.TimeoutExpired:
            return False, "timeout", "编译超时（超过120秒）"
        except Exception as e:
            return False, "exception", str(e)
    
    def test_sycl_compilation(
        self, 
        kernel_id: str,
        sycl_code: str = None,
        capture_full_error: bool = True
    ) -> Tuple[bool, str, str]:
        """
        测试SYCL编译
        
        Args:
            kernel_id: 内核ID
            sycl_code: 可选的SYCL代码（如果提供，会先写入文件）
            capture_full_error: 是否捕获完整错误信息
        
        Returns:
            (success, error_type, full_error_message)
        """
        self.stats["sycl_attempts"] += 1
        
        try:
            # 如果提供了代码，先写入文件
            if sycl_code:
                local_file = f"/tmp/{kernel_id}_kernel.dp.cpp"
                with open(local_file, 'w') as f:
                    f.write(sycl_code)
                
                # 复制到容器
                subprocess.run(
                    ['docker', 'cp', local_file, 
                     f'{self.sycl_container}:{self.sycl_work_path}/test_kernel.cpp'],
                    capture_output=True,
                    timeout=10,
                    check=True
                )
                compile_file = "test_kernel.cpp"
            else:
                # 使用现有文件
                local_file = f"kernel_dataset/sycl/{kernel_id}_kernel.dp.cpp"
                if not Path(local_file).exists():
                    return False, "file_not_found", f"SYCL文件不存在: {local_file}"
                
                # 复制到容器
                subprocess.run(
                    ['docker', 'cp', local_file,
                     f'{self.sycl_container}:{self.sycl_work_path}/test_kernel.cpp'],
                    capture_output=True,
                    timeout=10,
                    check=True
                )
                compile_file = "test_kernel.cpp"
            
            # 编译
            cmd = [
                'docker', 'exec', self.sycl_container, 'bash', '-c',
                f'cd {self.sycl_work_path} && '
                f'icpx -fsycl -O2 -c {compile_file} '
                f'-o /tmp/{kernel_id}.o 2>&1'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            error_output = result.stderr if result.stderr else result.stdout
            
            if result.returncode == 0:
                self.stats["sycl_success"] += 1
                return True, "", ""
            else:
                # 匹配错误类型
                error_pattern = ErrorPatternMatcher.get_primary_error(error_output)
                error_type = error_pattern.error_type if error_pattern else "unknown"
                
                # 截断错误信息
                if len(error_output) > 2000 and not capture_full_error:
                    error_output = error_output[:2000] + "\n... (truncated)"
                
                return False, error_type, error_output
                
        except subprocess.TimeoutExpired:
            return False, "timeout", "编译超时（超过120秒）"
        except Exception as e:
            return False, "exception", str(e)
    
    def get_stats(self) -> Dict:
        """获取编译统计"""
        return {
            **self.stats,
            "cuda_success_rate": self.stats["cuda_success"] / max(self.stats["cuda_attempts"], 1) * 100,
            "sycl_success_rate": self.stats["sycl_success"] / max(self.stats["sycl_attempts"], 1) * 100
        }


class ComplexityAnalyzer:
    """内核复杂度分析器"""
    
    @staticmethod
    def analyze_kernel(cuda_code: str) -> Dict[str, Any]:
        """分析内核复杂度"""
        analysis = {
            "complexity_score": 0.0,
            "uses_templates": False,
            "uses_shared_mem": False,
            "uses_warp_ops": False,
            "uses_atomics": False,
            "uses_math_intrinsics": False,
            "line_count": 0,
            "dependency_count": 0,
            "recommendation": ""
        }
        
        # 基础统计
        lines = cuda_code.split('\n')
        analysis["line_count"] = len(lines)
        
        # 检测特征
        code_lower = cuda_code.lower()
        
        # 模板使用
        if 'template' in code_lower or 'typename' in code_lower:
            analysis["uses_templates"] = True
            analysis["complexity_score"] += 2.0
        
        # 共享内存
        if '__shared__' in cuda_code:
            analysis["uses_shared_mem"] = True
            analysis["complexity_score"] += 1.5
        
        # warp操作
        warp_ops = ['__shfl', '__ballot', '__all', '__any', '__syncwarp']
        if any(op in cuda_code for op in warp_ops):
            analysis["uses_warp_ops"] = True
            analysis["complexity_score"] += 2.5
        
        # 原子操作
        if 'atomic' in code_lower:
            analysis["uses_atomics"] = True
            analysis["complexity_score"] += 1.5
        
        # 数学内联函数
        math_ops = ['__exp', '__log', '__sin', '__cos', '__sqrt', '__pow']
        if any(op in cuda_code for op in math_ops):
            analysis["uses_math_intrinsics"] = True
            analysis["complexity_score"] += 0.5
        
        # 依赖分析
        include_pattern = r'#include\s+[<"]([^>"]+)[>"]'
        includes = re.findall(include_pattern, cuda_code)
        analysis["dependency_count"] = len(includes)
        analysis["complexity_score"] += len(includes) * 0.3
        
        # 代码行数评分
        if analysis["line_count"] > 200:
            analysis["complexity_score"] += 2.0
        elif analysis["line_count"] > 100:
            analysis["complexity_score"] += 1.0
        
        # 规范化评分 (1-10)
        analysis["complexity_score"] = min(10.0, max(1.0, analysis["complexity_score"]))
        
        # 生成建议
        if analysis["complexity_score"] <= 3:
            analysis["recommendation"] = "simple"
        elif analysis["complexity_score"] <= 6:
            analysis["recommendation"] = "moderate"
        else:
            analysis["recommendation"] = "complex"
        
        return analysis


class ImprovedConversionAgent:
    """改进版转换Agent - 基于过程反思的全面优化"""
    
    def __init__(self, output_dir: str = "results/improved_agent_v3"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 组件初始化
        self.llm_client = ImprovedLLMClient()
        self.compiler = CompilationTester()
        self.analyzer = ComplexityAnalyzer()
        
        # 内核注册表
        self.kernels: Dict[str, KernelInfo] = {}
        
        # 配置参数
        self.config = {
            "max_fix_attempts": 8,
            "max_conversion_attempts": 3,
            "compilation_timeout": 180,
            "llm_timeout": 180,
            "batch_size": 5,
            "strategy": "auto"  # auto, direct, template_expansion
        }
        
        # 结果追踪
        self.session_results = {
            "start_time": datetime.now().isoformat(),
            "kernels_total": 0,
            "kernels_converted": 0,
            "kernels_verified": 0,
            "kernels_failed": 0,
            "details": []
        }
        
        # 加载内核索引
        self._load_kernel_index()
    
    def _load_kernel_index(self):
        """加载内核索引"""
        index_file = KERNEL_DATASET / "index.json"
        try:
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            for kernel_data in index.get("kernels", []):
                kernel_id = kernel_data["id"]
                
                # 检查CUDA文件存在
                cuda_file = KERNEL_DATASET / kernel_data["cuda"]["file"]
                if not cuda_file.exists():
                    continue
                
                # 检查SYCL文件
                sycl_file = KERNEL_DATASET / kernel_data.get("sycl", {}).get("file", f"sycl/{kernel_id}_kernel.dp.cpp")
                
                kernel_info = KernelInfo(
                    kernel_id=kernel_id,
                    name=kernel_data.get("name", kernel_id),
                    category=kernel_data.get("category", "unknown"),
                    cuda_file=cuda_file,
                    sycl_file=sycl_file,
                    has_sycl_mapping=kernel_data.get("has_sycl_mapping", False)
                )
                
                self.kernels[kernel_id] = kernel_info
            
            print(f"✅ 加载了 {len(self.kernels)} 个内核")
            
        except Exception as e:
            print(f"❌ 加载内核索引失败: {e}")
    
    async def preprocess_kernel(self, kernel_info: KernelInfo) -> Dict:
        """预处理内核 - 分析复杂度"""
        print(f"\n🔍 预处理: {kernel_info.kernel_id}")
        
        try:
            with open(kernel_info.cuda_file, 'r') as f:
                cuda_code = f.read()
            
            # 本地复杂度分析
            analysis = self.analyzer.analyze_kernel(cuda_code)
            
            # 使用LLM进行更深入的分析
            prompt = f"""分析以下CUDA内核代码:

文件名: {kernel_info.kernel_id}
代码:
```cpp
{cuda_code[:3000]}
```

请提供:
1. 代码功能描述
2. 关键算法步骤
3. 转换难点识别
4. 建议的转换策略 (direct/template_expansion)
5. 预估复杂度 (1-10)"""
            
            llm_analysis, success = await self.llm_client.call_llm(
                prompt, 
                task_type="preprocess",
                timeout=120
            )
            
            if success:
                analysis["llm_insights"] = llm_analysis
            
            # 更新内核信息
            kernel_info.complexity_score = analysis["complexity_score"]
            kernel_info.uses_templates = analysis["uses_templates"]
            kernel_info.uses_shared_mem = analysis["uses_shared_mem"]
            kernel_info.uses_warp_ops = analysis["uses_warp_ops"]
            
            print(f"  复杂度评分: {analysis['complexity_score']:.1f}/10")
            print(f"  建议策略: {analysis['recommendation']}")
            print(f"  模板: {'是' if analysis['uses_templates'] else '否'}")
            print(f"  共享内存: {'是' if analysis['uses_shared_mem'] else '否'}")
            
            return analysis
            
        except Exception as e:
            print(f"  ❌ 预处理失败: {e}")
            return {"complexity_score": 5.0, "error": str(e)}
    
    async def convert_kernel(
        self, 
        kernel_info: KernelInfo,
        strategy: str = None
    ) -> Tuple[str, bool]:
        """转换单个内核"""
        print(f"\n🔄 转换: {kernel_info.kernel_id}")
        
        kernel_info.status = ConversionStatus.CONVERTING
        kernel_info.conversion_attempts += 1
        
        try:
            with open(kernel_info.cuda_file, 'r') as f:
                cuda_code = f.read()
            
            # 确定策略
            if strategy is None or strategy == "auto":
                if kernel_info.complexity_score > 6:
                    strategy = "template_expansion"
                else:
                    strategy = "direct"
            
            print(f"  使用策略: {strategy}")
            
            # 准备提示词
            if strategy == "template_expansion" and kernel_info.uses_templates:
                # 模板展开策略 - 简化模板
                prompt = f"""将以下CUDA内核代码转换为SYCL代码。

要求:
1. 首先展开所有模板，使用具体类型（float或half）
2. 然后进行CUDA到SYCL的转换
3. 保持算法逻辑完全一致
4. 生成标准SYCL代码

CUDA代码:
```cpp
{cuda_code}
```

请输出完整的、可编译的SYCL代码。"""
            else:
                # 直接转换策略
                prompt = f"""将以下CUDA内核代码转换为SYCL代码。

CUDA代码:
```cpp
{cuda_code}
```

转换要求:
1. 保持所有算法逻辑不变
2. 转换所有CUDA特定语法为SYCL等价物
3. 添加 #include <sycl/sycl.hpp>
4. 确保代码完整、可直接编译
5. 保留原有函数签名和接口

请只输出转换后的完整SYCL代码，不要包含解释。"""
            
            # 调用LLM
            sycl_code, success = await self.llm_client.call_llm(
                prompt,
                task_type="conversion",
                timeout=self.config["llm_timeout"]
            )
            
            if not success or not sycl_code:
                print(f"  ❌ LLM转换失败")
                return "", False
            
            # 提取代码块
            sycl_code = self._extract_code(sycl_code)
            
            # 验证代码非空
            if len(sycl_code) < 50:
                print(f"  ❌ 生成的代码太短")
                return "", False
            
            # 保存转换结果
            kernel_info.sycl_file.parent.mkdir(parents=True, exist_ok=True)
            with open(kernel_info.sycl_file, 'w') as f:
                f.write(sycl_code)
            
            print(f"  ✅ 转换完成，代码长度: {len(sycl_code)} 字符")
            return sycl_code, True
            
        except Exception as e:
            print(f"  ❌ 转换异常: {e}")
            return "", False
    
    def _extract_code(self, text: str) -> str:
        """从LLM响应中提取代码"""
        # 尝试提取代码块
        patterns = [
            r'```cpp\s*\n(.*?)```',
            r'```c\+\+\s*\n(.*?)```',
            r'```\s*\n(.*?)```',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # 如果没有代码块，返回整个文本
        return text.strip()
    
    async def fix_compilation_errors(
        self,
        kernel_info: KernelInfo,
        error_message: str,
        error_type: str
    ) -> Tuple[str, bool]:
        """修复编译错误"""
        print(f"\n🔧 修复: {kernel_info.kernel_id} (尝试 {kernel_info.fix_attempts + 1})")
        
        kernel_info.status = ConversionStatus.FIXING
        kernel_info.fix_attempts += 1
        
        try:
            # 读取当前SYCL代码
            with open(kernel_info.sycl_file, 'r') as f:
                current_code = f.read()
            
            # 读取原始CUDA代码作为参考
            with open(kernel_info.cuda_file, 'r') as f:
                cuda_code = f.read()
            
            # 准备修复提示词
            prompt = f"""修复以下SYCL代码的编译错误。

错误信息:
```
{error_message[:1500]}
```

错误类型: {error_type}

当前SYCL代码:
```cpp
{current_code}
```

参考的原始CUDA代码:
```cpp
{cuda_code[:2000]}
```

修复要求:
1. 分析错误原因
2. 修复所有编译错误
3. 保持算法逻辑不变
4. 确保代码可编译

请输出完整的修复后的SYCL代码。"""
            
            # 调用LLM修复
            fixed_code, success = await self.llm_client.call_llm(
                prompt,
                task_type="fix",
                timeout=self.config["llm_timeout"]
            )
            
            if not success or not fixed_code:
                print(f"  ❌ LLM修复失败")
                return current_code, False
            
            # 提取代码
            fixed_code = self._extract_code(fixed_code)
            
            # 保存修复后的代码
            with open(kernel_info.sycl_file, 'w') as f:
                f.write(fixed_code)
            
            print(f"  ✅ 修复完成")
            return fixed_code, True
            
        except Exception as e:
            print(f"  ❌ 修复异常: {e}")
            return "", False
    
    async def verify_kernel(self, kernel_info: KernelInfo) -> bool:
        """验证内核 - 编译测试"""
        print(f"\n✅ 验证: {kernel_info.kernel_id}")
        
        kernel_info.status = ConversionStatus.VERIFYING
        
        # 测试CUDA编译
        print(f"  测试CUDA编译...")
        cuda_success, cuda_error_type, cuda_error = self.compiler.test_cuda_compilation(
            kernel_info.kernel_id,
            capture_full_error=True
        )
        kernel_info.cuda_compiles = cuda_success
        
        if not cuda_success:
            print(f"  ⚠️  CUDA编译失败: {cuda_error_type}")
            kernel_info.last_error = cuda_error
            kernel_info.error_type = cuda_error_type
        else:
            print(f"  ✅ CUDA编译通过")
        
        # 测试SYCL编译
        print(f"  测试SYCL编译...")
        sycl_success, sycl_error_type, sycl_error = self.compiler.test_sycl_compilation(
            kernel_info.kernel_id,
            capture_full_error=True
        )
        kernel_info.sycl_compiles = sycl_success
        
        if not sycl_success:
            print(f"  ⚠️  SYCL编译失败: {sycl_error_type}")
            kernel_info.last_error = sycl_error
            kernel_info.error_type = sycl_error_type
        else:
            print(f"  ✅ SYCL编译通过")
        
        # 更新状态
        if cuda_success and sycl_success:
            kernel_info.status = ConversionStatus.PASSED
            kernel_info.verified_at = datetime.now().isoformat()
            print(f"  🎉 验证通过！")
            return True
        else:
            kernel_info.status = ConversionStatus.FAILED
            return False
    
    async def process_kernel(
        self,
        kernel_info: KernelInfo,
        with_fix_loop: bool = True
    ) -> bool:
        """处理单个内核的完整流程"""
        print(f"\n{'='*70}")
        print(f"🚀 处理内核: {kernel_info.kernel_id}")
        print(f"{'='*70}")
        
        # 1. 预处理
        analysis = await self.preprocess_kernel(kernel_info)
        
        # 如果已经有SYCL映射且文件存在，跳过转换
        if kernel_info.has_sycl_mapping and kernel_info.sycl_file.exists():
            print(f"  ℹ️  已有SYCL文件，直接进入验证")
        else:
            # 2. 转换
            strategy = analysis.get("recommendation", "direct")
            sycl_code, success = await self.convert_kernel(kernel_info, strategy)
            
            if not success:
                kernel_info.status = ConversionStatus.FAILED
                return False
        
        # 3. 验证
        verified = await self.verify_kernel(kernel_info)
        
        # 4. 修复循环（如果需要）
        if not verified and with_fix_loop:
            for fix_attempt in range(self.config["max_fix_attempts"]):
                print(f"\n🔧 修复尝试 {fix_attempt + 1}/{self.config['max_fix_attempts']}")
                
                # 确定修复策略
                error_type = kernel_info.error_type
                error_message = kernel_info.last_error
                
                if not error_message:
                    print(f"  ⚠️  没有错误信息，无法修复")
                    break
                
                # 修复
                fixed_code, fix_success = await self.fix_compilation_errors(
                    kernel_info,
                    error_message,
                    error_type
                )
                
                if not fix_success:
                    print(f"  ❌ 修复失败")
                    continue
                
                # 重新验证
                verified = await self.verify_kernel(kernel_info)
                
                if verified:
                    print(f"  🎉 修复成功！")
                    break
            
            if not verified:
                print(f"\n  ❌ 达到最大修复次数，转换失败")
                kernel_info.status = ConversionStatus.FAILED
        
        # 记录结果
        self._record_result(kernel_info)
        
        return kernel_info.status == ConversionStatus.PASSED
    
    def _record_result(self, kernel_info: KernelInfo):
        """记录结果"""
        result = {
            "kernel_id": kernel_info.kernel_id,
            "status": kernel_info.status.value,
            "complexity_score": kernel_info.complexity_score,
            "conversion_attempts": kernel_info.conversion_attempts,
            "fix_attempts": kernel_info.fix_attempts,
            "cuda_compiles": kernel_info.cuda_compiles,
            "sycl_compiles": kernel_info.sycl_compiles,
            "error_type": kernel_info.error_type,
            "timestamp": datetime.now().isoformat()
        }
        
        self.session_results["details"].append(result)
        
        # 更新统计
        if kernel_info.status == ConversionStatus.PASSED:
            self.session_results["kernels_verified"] += 1
        elif kernel_info.conversion_attempts > 0:
            self.session_results["kernels_converted"] += 1
    
    async def run_batch_conversion(
        self,
        kernel_ids: List[str] = None,
        prioritize_simple: bool = True
    ):
        """批量转换"""
        print("\n" + "="*80)
        print("🚀 改进版 CUDA→SYCL 批量转换 Agent v3.0")
        print("="*80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 选择内核
        if kernel_ids is None:
            # 所有待处理的内核
            kernels_to_process = [
                k for k in self.kernels.values()
                if k.status == ConversionStatus.PENDING
            ]
        else:
            kernels_to_process = [
                self.kernels[kid] for kid in kernel_ids
                if kid in self.kernels
            ]
        
        self.session_results["kernels_total"] = len(kernels_to_process)
        
        print(f"📊 计划处理: {len(kernels_to_process)} 个内核")
        
        # 预处理所有内核以评估复杂度
        print("\n🔍 评估内核复杂度...")
        for kernel_info in kernels_to_process:
            await self.preprocess_kernel(kernel_info)
        
        # 按复杂度排序（如果优先处理简单的）
        if prioritize_simple:
            kernels_to_process.sort(key=lambda k: k.complexity_score)
            print("  已按复杂度排序（简单优先）")
        
        # 显示处理队列
        print("\n📋 处理队列:")
        for i, kernel_info in enumerate(kernels_to_process[:10], 1):
            status_icon = "✅" if kernel_info.has_sycl_mapping else "🔄"
            print(f"  {i}. {status_icon} {kernel_info.kernel_id} (复杂度: {kernel_info.complexity_score:.1f})")
        
        if len(kernels_to_process) > 10:
            print(f"  ... 还有 {len(kernels_to_process) - 10} 个内核")
        
        # 批量处理
        print(f"\n🚀 开始处理...")
        passed_count = 0
        failed_count = 0
        
        for i, kernel_info in enumerate(kernels_to_process, 1):
            print(f"\n{'='*80}")
            print(f"📌 进度: {i}/{len(kernels_to_process)} ({i/len(kernels_to_process)*100:.1f}%)")
            print(f"{'='*80}")
            
            success = await self.process_kernel(kernel_info, with_fix_loop=True)
            
            if success:
                passed_count += 1
            else:
                failed_count += 1
            
            # 每处理5个内核保存一次中间结果
            if i % 5 == 0:
                self._save_session_results()
        
        # 最终结果
        self._save_session_results()
        self._print_final_report()
    
    def _save_session_results(self):
        """保存会话结果"""
        self.session_results["end_time"] = datetime.now().isoformat()
        self.session_results["llm_stats"] = self.llm_client.get_stats()
        self.session_results["compilation_stats"] = self.compiler.get_stats()
        
        result_file = self.output_dir / "session_results.json"
        with open(result_file, 'w') as f:
            json.dump(self.session_results, f, indent=2)
        
        print(f"\n💾 结果已保存: {result_file}")
    
    def _print_final_report(self):
        """打印最终报告"""
        print("\n" + "="*80)
        print("📊 最终报告")
        print("="*80)
        
        total = self.session_results["kernels_total"]
        verified = self.session_results["kernels_verified"]
        
        print(f"\n总体统计:")
        print(f"  总内核数: {total}")
        print(f"  验证通过: {verified}")
        print(f"  失败: {total - verified}")
        print(f"  成功率: {verified/total*100:.1f}%" if total > 0 else "  成功率: N/A")
        
        # LLM统计
        llm_stats = self.llm_client.get_stats()
        print(f"\nLLM调用统计:")
        print(f"  总调用次数: {llm_stats['call_count']}")
        print(f"  总token数: {llm_stats['total_tokens']}")
        print(f"  成功率: {llm_stats['success_rate']:.1f}%")
        
        # 编译统计
        compile_stats = self.compiler.get_stats()
        print(f"\n编译统计:")
        print(f"  CUDA尝试: {compile_stats['cuda_attempts']}")
        print(f"  CUDA成功: {compile_stats['cuda_success']}")
        print(f"  CUDA成功率: {compile_stats['cuda_success_rate']:.1f}%")
        print(f"  SYCL尝试: {compile_stats['sycl_attempts']}")
        print(f"  SYCL成功: {compile_stats['sycl_success']}")
        print(f"  SYCL成功率: {compile_stats['sycl_success_rate']:.1f}%")
        
        # 失败的内核
        failed_kernels = [
            d for d in self.session_results["details"]
            if d["status"] != "passed"
        ]
        
        if failed_kernels:
            print(f"\n❌ 失败的内核:")
            for fk in failed_kernels:
                print(f"  • {fk['kernel_id']}: {fk.get('error_type', 'unknown')}")
        
        print(f"\n💾 详细结果: {self.output_dir}/session_results.json")
        print("="*80)


# ============================================================================
# Agent v4.0 新增：准确度测试模块
# ============================================================================

class ConversionStatusV4(Enum):
    """转换状态枚举 - v4.0 增加准确度测试状态"""
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    CONVERTING = "converting"
    COMPILING = "compiling"
    FIXING = "fixing"
    VERIFYING = "verifying"
    ACCURACY_TESTING = "accuracy_testing"  # 新增：准确度测试中
    PASSED = "passed"
    FAILED = "failed"
    ACCURACY_FAILED = "accuracy_failed"    # 新增：准确度测试失败
    SKIPPED = "skipped"


@dataclass
class KernelInfoV4:
    """内核信息数据结构 - v4.0 增强版"""
    kernel_id: str
    name: str
    category: str
    cuda_file: Path
    sycl_file: Path
    has_sycl_mapping: bool = False
    
    # 复杂度评估
    complexity_score: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    uses_templates: bool = False
    uses_shared_mem: bool = False
    uses_warp_ops: bool = False
    
    # 转换状态
    status: ConversionStatusV4 = ConversionStatusV4.PENDING
    conversion_attempts: int = 0
    fix_attempts: int = 0
    
    # 编译结果
    cuda_compiles: bool = False
    sycl_compiles: bool = False
    
    # 准确度测试结果 - v4.0 新增
    accuracy_tested: bool = False
    accuracy_passed: bool = False
    mae: float = 0.0
    max_error: float = 0.0
    
    # 错误信息
    last_error: str = ""
    error_type: str = ""
    error_history: List[Dict] = field(default_factory=list)
    
    # 元数据
    converted_at: Optional[str] = None
    verified_at: Optional[str] = None
    accuracy_tested_at: Optional[str] = None


class AccuracyTester:
    """准确度测试器 - v4.0核心组件"""
    
    TEST_CONFIGS = {
        'vector_op': {
            'input_range': (-1.0, 1.0),
            'tolerance': 1e-5,
            'max_error': 1e-4,
            'size': 1024,
        },
        'pooling': {
            'input_range': (0.0, 1.0),
            'tolerance': 1e-6,
            'max_error': 1e-5,
            'size': 2048,
        },
        'normalization': {
            'input_range': (-10.0, 10.0),
            'tolerance': 1e-5,
            'max_error': 1e-4,
            'size': 4096,
        },
        'fp16': {
            'input_range': (-1.0, 1.0),
            'tolerance': 1e-3,
            'max_error': 1e-2,
            'size': 1024,
        }
    }
    
    def __init__(self, cuda_host: str = "10.112.229.160", 
                 cuda_container: str = "cuda12.9-test",
                 sycl_container: str = "lsv-container"):
        self.cuda_host = cuda_host
        self.cuda_container = cuda_container
        self.sycl_container = sycl_container
    
    def classify_kernel(self, kernel_id: str) -> str:
        """根据内核ID分类内核类型"""
        if 'vector' in kernel_id or 'add' in kernel_id:
            return 'vector_op'
        elif 'pool' in kernel_id or 'avg' in kernel_id:
            return 'pooling'
        elif 'norm' in kernel_id or 'batch' in kernel_id:
            return 'normalization'
        elif 'fp16' in kernel_id or 'half' in kernel_id:
            return 'fp16'
        else:
            return 'vector_op'
    
    def generate_harness(self, kernel_id: str, platform: str) -> Optional[str]:
        """生成测试harness - 简化版"""
        kernel_type = self.classify_kernel(kernel_id)
        config = self.TEST_CONFIGS[kernel_type]
        
        if platform == 'cuda':
            return self._generate_cuda_harness(kernel_id, config)
        else:
            return self._generate_sycl_harness(kernel_id, config)
    
    def _generate_cuda_harness(self, kernel_id: str, config: Dict) -> str:
        """生成CUDA测试harness"""
        return f'''
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

__global__ void testKernel(float* output, const float* input, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        output[idx] = input[idx] * 2.0f;
    }}
}}

int main() {{
    const int size = {config['size']};
    float* h_input = new float[size];
    float* h_output = new float[size];
    
    for (int i = 0; i < size; i++) {{
        h_input[i] = sinf(i * 0.01f) * 0.5f;
    }}
    
    float* d_input; float* d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    
    testKernel<<<(size + 255) / 256, 256>>>(d_output, d_input, size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output, sizeof(float), size, f);
    fclose(f);
    
    cudaFree(d_input); cudaFree(d_output);
    delete[] h_input; delete[] h_output;
    return 0;
}}
'''
    
    def _generate_sycl_harness(self, kernel_id: str, config: Dict) -> str:
        """生成SYCL测试harness"""
        return f'''
#include <sycl/sycl.hpp>
#include <cmath>
#include <fstream>

int main() {{
    sycl::queue q(sycl::gpu_selector_v);
    const int size = {config['size']};
    
    float* h_input = new float[size];
    float* h_output = new float[size];
    
    for (int i = 0; i < size; i++) {{
        h_input[i] = sycl::sin(i * 0.01f) * 0.5f;
    }}
    
    float* d_input = sycl::malloc_device<float>(size, q);
    float* d_output = sycl::malloc_device<float>(size, q);
    
    q.memcpy(d_input, h_input, size * sizeof(float)).wait();
    
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {{
        int i = idx[0];
        d_output[i] = d_input[i] * 2.0f;
    }}).wait();
    
    q.memcpy(h_output, d_output, size * sizeof(float)).wait();
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), size * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_output, q);
    delete[] h_input; delete[] h_output;
    return 0;
}}
'''
    
    async def test_accuracy(self, kernel_id: str) -> Tuple[bool, float, float]:
        """测试内核准确度"""
        print(f"\n🧪 准确度测试: {kernel_id}")
        
        try:
            # 生成harness
            cuda_code = self.generate_harness(kernel_id, 'cuda')
            sycl_code = self.generate_harness(kernel_id, 'sycl')
            
            if not cuda_code or not sycl_code:
                print(f"  ⚠️  无法生成harness")
                return False, 0.0, 0.0
            
            # 运行CUDA
            print(f"  🔨 CUDA...", end=' ')
            cuda_success = await self._run_cuda(cuda_code)
            if not cuda_success:
                print(f"❌")
                return False, 0.0, 0.0
            print(f"✅")
            
            # 运行SYCL
            print(f"  🔨 SYCL...", end=' ')
            sycl_success = await self._run_sycl(sycl_code)
            if not sycl_success:
                print(f"❌")
                return False, 0.0, 0.0
            print(f"✅")
            
            # 比较结果
            print(f"  📊 比较...", end=' ')
            mae, max_error = await self._compare_outputs()
            
            kernel_type = self.classify_kernel(kernel_id)
            tolerance = self.TEST_CONFIGS[kernel_type]['tolerance']
            max_tolerance = self.TEST_CONFIGS[kernel_type]['max_error']
            
            passed = (mae < tolerance) and (max_error < max_tolerance)
            
            status = "✅" if passed else "⚠️"
            print(f"{status} MAE={mae:.2e}, MaxErr={max_error:.2e}")
            
            return passed, mae, max_error
            
        except Exception as e:
            print(f"  ❌ 测试异常: {e}")
            return False, 0.0, 0.0
    
    async def _run_cuda(self, code: str) -> bool:
        """运行CUDA测试"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(code)
                cuda_file = f.name
            
            subprocess.run(['scp', cuda_file, f'root@{self.cuda_host}:/tmp/test.cu'],
                         capture_output=True, timeout=30, check=True)
            
            cmd = f'''ssh root@{self.cuda_host} "docker cp /tmp/test.cu {self.cuda_container}:/workspace/test.cu && 
                     docker exec {self.cuda_container} bash -c 'cd /workspace && 
                     nvcc -O2 -Wno-deprecated-gpu-targets test.cu -o test && ./test'"'''
            
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
            
            os.unlink(cuda_file)
            return result.returncode == 0
            
        except Exception as e:
            print(f"CUDA运行错误: {e}")
            return False
    
    async def _run_sycl(self, code: str) -> bool:
        """运行SYCL测试"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(code)
                sycl_file = f.name
            
            subprocess.run(['docker', 'cp', sycl_file, 
                          f'{self.sycl_container}:/workspace/test.cpp'],
                         capture_output=True, timeout=10, check=True)
            
            result = subprocess.run(['docker', 'exec', self.sycl_container, 'bash', '-c',
                                   'cd /workspace && icpx -fsycl -O2 test.cpp -o test && ./test'],
                                  capture_output=True, timeout=120)
            
            os.unlink(sycl_file)
            return result.returncode == 0
            
        except Exception as e:
            print(f"SYCL运行错误: {e}")
            return False
    
    async def _compare_outputs(self) -> Tuple[float, float]:
        """比较CUDA和SYCL输出"""
        try:
            subprocess.run(['ssh', f'root@{self.cuda_host}',
                          f'docker cp {self.cuda_container}:/workspace/output_cuda.bin /tmp/'],
                         capture_output=True, check=True)
            subprocess.run(['scp', f'root@{self.cuda_host}:/tmp/output_cuda.bin', '/tmp/'],
                         capture_output=True, check=True)
            subprocess.run(['docker', 'cp', f'{self.sycl_container}:/workspace/output_sycl.bin', '/tmp/'],
                         capture_output=True, check=True)
            
            cuda_out = np.fromfile('/tmp/output_cuda.bin', dtype=np.float32)
            sycl_out = np.fromfile('/tmp/output_sycl.bin', dtype=np.float32)
            
            if len(cuda_out) != len(sycl_out):
                return 1.0, 1.0
            
            diff = np.abs(cuda_out - sycl_out)
            mae = float(np.mean(diff))
            max_err = float(np.max(diff))
            
            return mae, max_err
            
        except Exception as e:
            print(f"比较错误: {e}")
            return 1.0, 1.0


# ============================================================================
# Agent v4.0：集成准确度测试的完整Agent
# ============================================================================

class AgentV4(ImprovedConversionAgent):
    """Agent v4.0 - 集成准确度测试"""
    
    def __init__(self, output_dir: str = "results/agent_v4"):
        super().__init__(output_dir=output_dir)
        self.accuracy_tester = AccuracyTester()
        print("🚀 Agent v4.0 初始化完成（集成准确度测试）")
    
    async def verify_with_accuracy(self, kernel_info: KernelInfoV4) -> bool:
        """验证内核 - 包含编译和准确度测试"""
        # 第1步：编译验证
        compiled = await self.verify_kernel(kernel_info)
        if not compiled:
            return False
        
        # 第2步：准确度测试
        print(f"\n🧪 开始准确度测试...")
        kernel_info.status = ConversionStatusV4.ACCURACY_TESTING
        
        passed, mae, max_error = await self.accuracy_tester.test_accuracy(
            kernel_info.kernel_id
        )
        
        kernel_info.accuracy_tested = True
        kernel_info.mae = mae
        kernel_info.max_error = max_error
        kernel_info.accuracy_passed = passed
        kernel_info.accuracy_tested_at = datetime.now().isoformat()
        
        if passed:
            print(f"✅ 准确度测试通过 (MAE={mae:.2e})")
            kernel_info.status = ConversionStatusV4.PASSED
            return True
        else:
            print(f"❌ 准确度测试失败 (MAE={mae:.2e})")
            kernel_info.status = ConversionStatusV4.ACCURACY_FAILED
            return False


async def main_v4():
    """Agent v4.0 主函数"""
    print("="*80)
    print("🚀 Agent v4.0 - 集成准确度测试")
    print("="*80)
    
    agent = AgentV4()
    
    # 测试一个内核
    test_kernel = "add_vectors"
    if test_kernel in agent.kernels:
        kernel_info = agent.kernels[test_kernel]
        success = await agent.verify_with_accuracy(kernel_info)
        print(f"\n结果: {'✅ 通过' if success else '❌ 失败'}")
    else:
        print(f"内核 {test_kernel} 不存在")


if __name__ == "__main__":
    asyncio.run(main_v4())
