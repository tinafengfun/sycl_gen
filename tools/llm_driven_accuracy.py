#!/usr/bin/env python3
"""
LLM-Driven Intelligent Accuracy Test System
LLM驱动的智能准确度测试系统

核心特性:
1. LLM智能Harness生成 - 自动从原始kernel代码提取测试逻辑
2. LLM错误分析 - 自动诊断失败原因并提供修复建议
3. LLM测试用例生成 - 智能生成边界条件测试
4. LLM执行优化 - 动态调整测试策略

使用LLM: minimax-m2.5 via Gaudi AI API
"""

import asyncio
import json
import logging
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any, Union
import numpy as np
import sys

# 添加tools目录到路径
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 1. LLM客户端封装
# ============================================================================

class LLMClient:
    """LLM客户端 - 封装Gaudi AI API调用"""
    
    def __init__(self, config_file: str = "model_config_minimax.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.provider = self.config.get('provider', {}).get('gaudi_ai', {})
        self.base_url = self.provider.get('options', {}).get('baseURL', '')
        self.api_key = self.provider.get('options', {}).get('apiKey', '')
        self.settings = self.provider.get('settings', {})
        
    def _load_config(self) -> Dict:
        """加载配置"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return {
                'provider': {
                    'gaudi_ai': {
                        'options': {
                            'baseURL': 'https://api.gaudi.ai/v1',
                            'apiKey': ''
                        },
                        'settings': {
                            'temperature': 0.3,
                            'max_tokens': 4000,
                            'top_p': 0.95,
                            'retry_attempts': 3,
                            'timeout': 180
                        }
                    }
                },
                'models': {
                    'default': 'minimax-m2.5'
                }
            }
    
    async def call(self, 
                   prompt: str, 
                   system_prompt: str = "",
                   model: str = None,
                   max_retries: int = 3) -> str:
        """
        调用LLM
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            model: 模型名称
            max_retries: 最大重试次数
            
        Returns:
            LLM响应文本
        """
        import aiohttp
        
        model = model or self.config.get('models', {}).get('default', 'minimax-m2.5')
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt or "You are an expert CUDA/SYCL kernel testing engineer."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.settings.get('temperature', 0.3),
            "max_tokens": self.settings.get('max_tokens', 4000),
            "top_p": self.settings.get('top_p', 0.95)
        }
        
        timeout = aiohttp.ClientTimeout(total=self.settings.get('timeout', 180))
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=timeout
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data['choices'][0]['message']['content']
                        else:
                            logger.warning(f"LLM call failed (attempt {attempt+1}): HTTP {response.status}")
            except Exception as e:
                logger.warning(f"LLM call error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        return ""


# ============================================================================
# 2. LLM智能Harness生成器
# ============================================================================

@dataclass
class KernelAnalysis:
    """Kernel分析结果"""
    kernel_id: str
    function_name: str
    input_params: List[Dict]
    output_params: List[Dict]
    thread_hierarchy: Dict  # grid/block配置
    data_types: List[str]
    algorithm_summary: str
    dependencies: List[str]
    test_scenarios: List[str]


class LLMHarnessGenerator:
    """
    LLM驱动的智能Harness生成器
    
    功能:
    1. 分析原始CUDA kernel代码
    2. 自动生成测试harness
    3. 生成确定性测试输入
    4. 确保CUDA和SYCL逻辑一致
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.cache: Dict[str, Dict] = {}
    
    async def analyze_kernel(self, kernel_id: str, cuda_code: str) -> KernelAnalysis:
        """
        使用LLM分析kernel代码
        
        Args:
            kernel_id: 内核标识符
            cuda_code: CUDA源代码
            
        Returns:
            KernelAnalysis对象
        """
        system_prompt = """You are an expert CUDA kernel analyzer. Extract the following information from the CUDA kernel code:
1. Kernel function name
2. Input parameters with types
3. Output parameters with types
4. Thread hierarchy (grid size, block size)
5. Data types used
6. Algorithm summary
7. Dependencies
8. Recommended test scenarios

Respond in JSON format only."""

        prompt = f"""Analyze this CUDA kernel and extract key information:

Kernel ID: {kernel_id}

CUDA Code:
```cuda
{cuda_code[:2000]}  # 限制长度
```

Provide your analysis in this exact JSON format:
{{
    "function_name": "kernel_function_name",
    "input_params": [
        {{"name": "param_name", "type": "data_type", "description": "what it does"}}
    ],
    "output_params": [
        {{"name": "output_name", "type": "data_type", "description": "what it stores"}}
    ],
    "thread_hierarchy": {{
        "grid_dims": "e.g., (N+255)/256",
        "block_dims": "e.g., 256",
        "description": "how threads are organized"
    }},
    "data_types": ["float", "half", etc],
    "algorithm_summary": "brief description of what the kernel does",
    "dependencies": ["header files", "helper functions"],
    "test_scenarios": [
        "small input test",
        "large input test", 
        "edge case test"
    ]
}}"""

        response = await self.llm.call(prompt, system_prompt)
        
        # 解析JSON响应
        try:
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return KernelAnalysis(
                    kernel_id=kernel_id,
                    function_name=data.get('function_name', f'{kernel_id}_kernel'),
                    input_params=data.get('input_params', []),
                    output_params=data.get('output_params', []),
                    thread_hierarchy=data.get('thread_hierarchy', {}),
                    data_types=data.get('data_types', ['float']),
                    algorithm_summary=data.get('algorithm_summary', ''),
                    dependencies=data.get('dependencies', []),
                    test_scenarios=data.get('test_scenarios', ['basic test'])
                )
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
        
        # 返回默认值
        return KernelAnalysis(
            kernel_id=kernel_id,
            function_name=f'{kernel_id}_kernel',
            input_params=[],
            output_params=[],
            thread_hierarchy={},
            data_types=['float'],
            algorithm_summary='',
            dependencies=[],
            test_scenarios=['basic test']
        )
    
    async def generate_harness(self, 
                               kernel_id: str,
                               analysis: KernelAnalysis,
                               platform: str) -> str:
        """
        使用LLM生成测试harness
        
        Args:
            kernel_id: 内核标识符
            analysis: kernel分析结果
            platform: 'cuda' 或 'sycl'
            
        Returns:
            harness代码
        """
        system_prompt = f"""You are an expert {platform.upper()} test harness generator. 
Generate a complete, compilable test harness that:
1. Uses deterministic inputs (sin/cos functions, not random)
2. Properly allocates and manages memory
3. Calls the kernel with correct parameters
4. Saves output to binary file
5. Handles errors properly

The code must be self-contained and ready to compile."""

        prompt = f"""Generate a {platform.upper()} test harness for this kernel:

Kernel ID: {kernel_id}
Function: {analysis.function_name}
Algorithm: {analysis.algorithm_summary}
Data Types: {', '.join(analysis.data_types)}
Thread Hierarchy: {json.dumps(analysis.thread_hierarchy)}

Requirements:
1. Use deterministic inputs based on index (e.g., sin(i*0.01f))
2. Allocate appropriate memory sizes
3. Use proper {platform.upper()} APIs
4. Save output to /workspace/output_{platform}.bin
5. Include error checking
6. Make it compilable with standard flags

Generate the complete code:"""

        harness_code = await self.llm.call(prompt, system_prompt)
        
        # 提取代码块
        code_match = re.search(r'```(?:cuda|cpp|c\+\+)?\s*(.*?)```', harness_code, re.DOTALL)
        if code_match:
            harness_code = code_match.group(1).strip()
        
        return harness_code
    
    async def generate_pair(self, 
                           kernel_id: str, 
                           cuda_file: Path) -> Tuple[str, str]:
        """
        生成CUDA和SYCL harness对
        
        Args:
            kernel_id: 内核标识符
            cuda_file: CUDA源文件路径
            
        Returns:
            (cuda_harness, sycl_harness)
        """
        # 读取CUDA代码
        try:
            with open(cuda_file, 'r') as f:
                cuda_code = f.read()
        except Exception as e:
            logger.error(f"Failed to read {cuda_file}: {e}")
            return "", ""
        
        # 分析kernel
        logger.info(f"Analyzing kernel: {kernel_id}")
        analysis = await self.analyze_kernel(kernel_id, cuda_code)
        
        # 并行生成两个harness
        cuda_task = self.generate_harness(kernel_id, analysis, 'cuda')
        sycl_task = self.generate_harness(kernel_id, analysis, 'sycl')
        
        cuda_harness, sycl_harness = await asyncio.gather(cuda_task, sycl_task)
        
        # 缓存结果
        self.cache[kernel_id] = {
            'analysis': analysis,
            'cuda_harness': cuda_harness,
            'sycl_harness': sycl_harness,
            'timestamp': datetime.now().isoformat()
        }
        
        return cuda_harness, sycl_harness


# ============================================================================
# 3. LLM错误分析器
# ============================================================================

@dataclass
class ErrorAnalysis:
    """错误分析结果"""
    error_type: str  # 'compilation', 'runtime', 'accuracy'
    severity: str    # 'low', 'medium', 'high', 'critical'
    root_cause: str
    description: str
    suggestions: List[str]
    code_fixes: List[Dict]  # {'location': '...', 'original': '...', 'fixed': '...'}
    confidence: float  # 0-1


class LLMErrorAnalyzer:
    """
    LLM驱动的错误分析器
    
    功能:
    1. 分析编译错误
    2. 分析运行时错误
    3. 分析准确度失败
    4. 提供修复建议
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
    
    async def analyze_compilation_error(self, 
                                       kernel_id: str,
                                       platform: str,
                                       source_code: str,
                                       error_output: str) -> ErrorAnalysis:
        """分析编译错误"""
        system_prompt = """You are a compilation error expert. Analyze the compilation error and provide:
1. Error type classification
2. Root cause
3. Detailed description
4. Specific fix suggestions
5. Code fixes if applicable

Respond in JSON format."""

        prompt = f"""Analyze this {platform.upper()} compilation error:

Kernel: {kernel_id}

Source Code:
```{platform}
{source_code}
```

Error Output:
```
{error_output[:2000]}
```

Provide analysis in JSON format:
{{
    "error_type": "syntax_error|type_error|link_error|other",
    "severity": "low|medium|high|critical",
    "root_cause": "brief description",
    "description": "detailed explanation",
    "suggestions": ["fix suggestion 1", "fix suggestion 2"],
    "code_fixes": [
        {{
            "location": "line number or function",
            "original": "original code",
            "fixed": "fixed code"
        }}
    ],
    "confidence": 0.95
}}"""

        response = await self.llm.call(prompt, system_prompt)
        return self._parse_error_analysis(response)
    
    async def analyze_accuracy_failure(self,
                                      kernel_id: str,
                                      cuda_output: np.ndarray,
                                      sycl_output: np.ndarray,
                                      mae: float,
                                      max_error: float) -> ErrorAnalysis:
        """分析准确度失败"""
        system_prompt = """You are a numerical accuracy expert. Analyze why CUDA and SYCL outputs differ:
1. Identify likely causes
2. Compare error patterns
3. Suggest algorithm fixes
4. Recommend tolerance adjustments

Respond in JSON format."""

        # 计算错误统计
        diff = np.abs(cuda_output - sycl_output)
        error_stats = {
            'mean_error': float(np.mean(diff)),
            'max_error': float(np.max(diff)),
            'std_error': float(np.std(diff)),
            'error_percentiles': {
                'p50': float(np.percentile(diff, 50)),
                'p90': float(np.percentile(diff, 90)),
                'p99': float(np.percentile(diff, 99))
            }
        }
        
        # 找出错误最大的位置
        max_idx = np.unravel_index(np.argmax(diff), diff.shape) if diff.size > 0 else (0,)
        
        prompt = f"""Analyze this accuracy failure:

Kernel: {kernel_id}
MAE: {mae:.2e}
Max Error: {max_error:.2e}

Error Statistics:
{json.dumps(error_stats, indent=2)}

Max Error Location: {max_idx}
CUDA value at max: {cuda_output[max_idx]:.6e}
SYCL value at max: {sycl_output[max_idx]:.6e}

Sample of differences (first 20):
{diff.flatten()[:20].tolist()}

Provide analysis in JSON format:
{{
    "error_type": "algorithm_mismatch|precision_loss|boundary_error|other",
    "severity": "low|medium|high|critical",
    "root_cause": "brief description",
    "description": "detailed explanation of why results differ",
    "suggestions": [
        "specific fix 1",
        "specific fix 2"
    ],
    "code_fixes": [],
    "recommended_tolerance": {{
        "abs": 1e-5,
        "rel": 1e-4
    }},
    "confidence": 0.85
}}"""

        response = await self.llm.call(prompt, system_prompt)
        return self._parse_error_analysis(response)
    
    def _parse_error_analysis(self, response: str) -> ErrorAnalysis:
        """解析LLM的错误分析响应"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return ErrorAnalysis(
                    error_type=data.get('error_type', 'unknown'),
                    severity=data.get('severity', 'medium'),
                    root_cause=data.get('root_cause', 'Unknown'),
                    description=data.get('description', ''),
                    suggestions=data.get('suggestions', []),
                    code_fixes=data.get('code_fixes', []),
                    confidence=data.get('confidence', 0.5)
                )
        except Exception as e:
            logger.error(f"Failed to parse error analysis: {e}")
        
        return ErrorAnalysis(
            error_type='unknown',
            severity='medium',
            root_cause='Failed to analyze',
            description='',
            suggestions=['Check logs manually'],
            code_fixes=[],
            confidence=0.0
        )


# ============================================================================
# 4. LLM测试用例生成器
# ============================================================================

@dataclass
class TestCase:
    """测试用例"""
    name: str
    description: str
    input_generator: str  # Python代码生成输入
    expected_behavior: str
    tolerance_override: Optional[Dict] = None


class LLMTestCaseGenerator:
    """
    LLM驱动的测试用例生成器
    
    生成各种测试场景:
    1. 边界值测试
    2. 异常情况测试
    3. 性能边界测试
    4. 数值稳定性测试
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
    
    async def generate_test_cases(self, 
                                  kernel_id: str,
                                  analysis: KernelAnalysis) -> List[TestCase]:
        """生成测试用例集"""
        system_prompt = """You are a test case design expert. Generate comprehensive test cases for a GPU kernel.

For each test case, provide:
1. Name
2. Description
3. Python code to generate deterministic inputs
4. Expected behavior
5. Special tolerance if needed

Respond in JSON format."""

        prompt = f"""Generate test cases for kernel: {kernel_id}

Algorithm: {analysis.algorithm_summary}
Input Parameters: {json.dumps(analysis.input_params)}
Data Types: {', '.join(analysis.data_types)}

Generate these types of test cases:
1. Basic functionality test
2. Edge case (zeros, very small values)
3. Large value test
4. Boundary condition test
5. Numerical stability test

Response format:
{{
    "test_cases": [
        {{
            "name": "test_name",
            "description": "what this tests",
            "input_generator": "python code to generate inputs",
            "expected_behavior": "what should happen",
            "tolerance_override": {{"abs": 1e-5, "rel": 1e-4}}
        }}
    ]
}}"""

        response = await self.llm.call(prompt, system_prompt)
        
        test_cases = []
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for tc_data in data.get('test_cases', []):
                    test_cases.append(TestCase(
                        name=tc_data.get('name', 'unnamed'),
                        description=tc_data.get('description', ''),
                        input_generator=tc_data.get('input_generator', ''),
                        expected_behavior=tc_data.get('expected_behavior', ''),
                        tolerance_override=tc_data.get('tolerance_override')
                    ))
        except Exception as e:
            logger.error(f"Failed to parse test cases: {e}")
        
        return test_cases


# ============================================================================
# 5. LLM执行策略优化器
# ============================================================================

class LLMExecutionOptimizer:
    """
    LLM驱动的执行策略优化器
    
    优化测试执行:
    1. 动态调整并行度
    2. 智能重试策略
    3. 测试优先级排序
    4. 资源分配优化
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
    
    async def optimize_execution_plan(self,
                                     kernel_ids: List[str],
                                     historical_results: Dict) -> Dict:
        """优化执行计划"""
        system_prompt = """You are a test execution optimization expert. 
Given a list of kernels and their historical performance, create an optimal execution plan.

Consider:
1. Which kernels are likely to fail (test first)
2. Which kernels are independent (can run in parallel)
3. Resource constraints
4. Execution time estimates

Respond in JSON format."""

        # 构建历史结果摘要
        history_summary = []
        for kid, results in historical_results.items():
            history_summary.append({
                'kernel_id': kid,
                'success_rate': results.get('success_rate', 0),
                'avg_duration': results.get('avg_duration', 30),
                'failure_pattern': results.get('failure_pattern', 'unknown')
            })
        
        prompt = f"""Optimize test execution plan:

Kernels to test: {kernel_ids}

Historical Performance:
{json.dumps(history_summary, indent=2)}

Create execution plan with:
1. Execution order (which first)
2. Batch grouping (which can run together)
3. Parallelism level for each batch
4. Retry strategy

Response format:
{{
    "execution_plan": {{
        "batches": [
            {{
                "kernels": ["kernel1", "kernel2"],
                "parallelism": 2,
                "priority": "high",
                "rationale": "why this grouping"
            }}
        ],
        "estimated_total_time": 180,
        "risk_mitigation": "description"
    }}
}}"""

        response = await self.llm.call(prompt, system_prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"Failed to parse execution plan: {e}")
        
        # 返回默认计划
        return {
            'execution_plan': {
                'batches': [{'kernels': kernel_ids, 'parallelism': 3, 'priority': 'normal'}],
                'estimated_total_time': len(kernel_ids) * 30,
                'risk_mitigation': 'Default sequential execution'
            }
        }


# ============================================================================
# 6. 智能准确度Agent集成
# ============================================================================

@dataclass
class SmartVerificationResult:
    """智能验证结果"""
    kernel_id: str
    passed: bool
    mae: float
    max_error: float
    llm_generated_harness: bool
    error_analysis: Optional[ErrorAnalysis] = None
    suggested_fixes: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class LLMDrivenAccuracyAgent:
    """
    LLM驱动的智能准确度Agent
    
    完整集成所有LLM功能:
    1. 智能Harness生成
    2. 自动错误分析
    3. 智能修复建议
    4. 测试用例生成
    5. 执行优化
    """
    
    def __init__(self, 
                 cuda_host: str = "10.112.229.160",
                 sycl_container: str = "lsv-container"):
        self.cuda_host = cuda_host
        self.sycl_container = sycl_container
        
        # LLM组件
        self.llm = LLMClient()
        self.harness_gen = LLMHarnessGenerator(self.llm)
        self.error_analyzer = LLMErrorAnalyzer(self.llm)
        self.test_case_gen = LLMTestCaseGenerator(self.llm)
        self.optimizer = LLMExecutionOptimizer(self.llm)
        
        # 结果缓存
        self.results: Dict[str, SmartVerificationResult] = {}
        self.history: Dict[str, Dict] = {}
        
        self.logger = logging.getLogger(__name__)
    
    async def verify_with_llm(self, 
                             kernel_id: str,
                             use_llm_harness: bool = True) -> SmartVerificationResult:
        """
        使用LLM智能验证kernel
        
        Args:
            kernel_id: 内核标识符
            use_llm_harness: 是否使用LLM生成harness
            
        Returns:
            SmartVerificationResult
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"🚀 LLM-driven verification: {kernel_id}")
        
        cuda_file = Path(f"kernel_dataset/cuda/{kernel_id}_kernel.cu")
        
        if use_llm_harness and cuda_file.exists():
            # 使用LLM生成harness
            self.logger.info("  🤖 Generating harness with LLM...")
            cuda_harness, sycl_harness = await self.harness_gen.generate_pair(
                kernel_id, cuda_file
            )
        else:
            # 使用内置模板
            self.logger.info("  📋 Using built-in harness templates...")
            from accuracy_verifier import HarnessGenerator
            gen = HarnessGenerator()
            cuda_harness = gen.generate(kernel_id, 'cuda')
            sycl_harness = gen.generate(kernel_id, 'sycl')
        
        if not cuda_harness or not sycl_harness:
            return SmartVerificationResult(
                kernel_id=kernel_id,
                passed=False,
                mae=0.0,
                max_error=0.0,
                llm_generated_harness=use_llm_harness,
                suggested_fixes=["Failed to generate harness"]
            )
        
        # 执行测试
        self.logger.info("  🔨 Compiling and running...")
        cuda_success, cuda_error = await self._run_cuda(cuda_harness)
        sycl_success, sycl_error = await self._run_sycl(sycl_harness)
        
        if not cuda_success or not sycl_success:
            # 编译/运行失败，使用LLM分析
            error_msg = cuda_error or sycl_error
            self.logger.warning(f"  ⚠️ Execution failed: {error_msg[:100]}")
            
            # TODO: 使用LLM分析错误
            return SmartVerificationResult(
                kernel_id=kernel_id,
                passed=False,
                mae=0.0,
                max_error=0.0,
                llm_generated_harness=use_llm_harness,
                suggested_fixes=[error_msg[:200]]
            )
        
        # 比较输出
        self.logger.info("  📊 Comparing outputs...")
        try:
            cuda_output = np.fromfile('/tmp/output_cuda.bin', dtype=np.float32)
            sycl_output = np.fromfile('/tmp/output_sycl.bin', dtype=np.float32)
            
            diff = np.abs(cuda_output - sycl_output)
            mae = float(np.mean(diff))
            max_error = float(np.max(diff))
            
            passed = mae < 1e-4  # 使用宽松标准
            
            result = SmartVerificationResult(
                kernel_id=kernel_id,
                passed=passed,
                mae=mae,
                max_error=max_error,
                llm_generated_harness=use_llm_harness,
                duration_seconds=time.time() - start_time
            )
            
            # 如果失败，使用LLM分析
            if not passed:
                self.logger.info("  🔍 Analyzing failure with LLM...")
                error_analysis = await self.error_analyzer.analyze_accuracy_failure(
                    kernel_id, cuda_output, sycl_output, mae, max_error
                )
                result.error_analysis = error_analysis
                result.suggested_fixes = error_analysis.suggestions
            
            self.results[kernel_id] = result
            return result
            
        except Exception as e:
            self.logger.error(f"  ❌ Comparison failed: {e}")
            return SmartVerificationResult(
                kernel_id=kernel_id,
                passed=False,
                mae=0.0,
                max_error=0.0,
                llm_generated_harness=use_llm_harness,
                suggested_fixes=[str(e)]
            )
    
    async def _run_cuda(self, harness_code: str) -> Tuple[bool, Optional[str]]:
        """运行CUDA测试"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(harness_code)
                local_file = f.name
            
            subprocess.run(
                ['scp', '-o', 'StrictHostKeyChecking=no', local_file, 
                 f'{self.cuda_host}:/tmp/test.cu'],
                capture_output=True, timeout=30, check=True
            )
            
            cmd = f'''
            ssh -o StrictHostKeyChecking=no {self.cuda_host} "
                docker cp /tmp/test.cu cuda12.9-test:/workspace/test.cu &&
                docker exec cuda12.9-test bash -c '
                    cd /workspace && 
                    nvcc -O2 -Wno-deprecated-gpu-targets test.cu -o test &&
                    ./test'
            "
            '''
            
            result = subprocess.run(
                cmd, shell=True, capture_output=True, 
                text=True, timeout=120
            )
            
            Path(local_file).unlink(missing_ok=True)
            
            if result.returncode == 0:
                subprocess.run(
                    ['ssh', '-o', 'StrictHostKeyChecking=no', self.cuda_host,
                     'docker cp cuda12.9-test:/workspace/output_cuda.bin /tmp/'],
                    capture_output=True, timeout=10, check=True
                )
                subprocess.run(
                    ['scp', '-o', 'StrictHostKeyChecking=no',
                     f'{self.cuda_host}:/tmp/output_cuda.bin', '/tmp/output_cuda.bin'],
                    capture_output=True, timeout=10, check=True
                )
                return True, None
            
            return False, result.stderr
            
        except Exception as e:
            return False, str(e)
    
    async def _run_sycl(self, harness_code: str) -> Tuple[bool, Optional[str]]:
        """运行SYCL测试"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(harness_code)
                local_file = f.name
            
            subprocess.run(
                ['docker', 'cp', local_file, f'{self.sycl_container}:/workspace/test.cpp'],
                capture_output=True, timeout=10, check=True
            )
            
            result = subprocess.run(
                ['docker', 'exec', self.sycl_container, 'bash', '-c',
                 'cd /workspace && icpx -fsycl -O2 test.cpp -o test && ./test'],
                capture_output=True, text=True, timeout=120
            )
            
            Path(local_file).unlink(missing_ok=True)
            
            if result.returncode == 0:
                subprocess.run(
                    ['docker', 'cp', f'{self.sycl_container}:/workspace/output_sycl.bin', 
                     '/tmp/output_sycl.bin'],
                    capture_output=True, timeout=10, check=True
                )
                return True, None
            
            return False, result.stderr
            
        except Exception as e:
            return False, str(e)
    
    async def verify_batch(self,
                          kernel_ids: List[str],
                          use_llm_harness: bool = True,
                          optimize_execution: bool = True) -> Dict[str, SmartVerificationResult]:
        """批量智能验证"""
        
        if optimize_execution and self.history:
            # 使用LLM优化执行计划
            self.logger.info("🎯 Optimizing execution plan with LLM...")
            plan = await self.optimizer.optimize_execution_plan(kernel_ids, self.history)
            # TODO: 根据计划执行
        
        results = {}
        for kernel_id in kernel_ids:
            result = await self.verify_with_llm(kernel_id, use_llm_harness)
            results[kernel_id] = result
            
            status = "✅" if result.passed else "❌"
            harness_type = "LLM" if result.llm_generated_harness else "Built-in"
            self.logger.info(f"{status} {kernel_id} ({harness_type} harness, "
                           f"MAE={result.mae:.2e})")
        
        return results
    
    def generate_report(self) -> str:
        """生成智能报告"""
        lines = [
            "=" * 70,
            "🤖 LLM-Driven Accuracy Test Report",
            "=" * 70,
            f"Total kernels: {len(self.results)}",
            f"Passed: {sum(1 for r in self.results.values() if r.passed)}",
            f"LLM-generated harness: {sum(1 for r in self.results.values() if r.llm_generated_harness)}",
            "=" * 70
        ]
        
        for kernel_id, result in self.results.items():
            lines.append(f"\n{kernel_id}:")
            lines.append(f"  Status: {'PASS' if result.passed else 'FAIL'}")
            lines.append(f"  MAE: {result.mae:.2e}, Max Error: {result.max_error:.2e}")
            lines.append(f"  LLM Harness: {'Yes' if result.llm_generated_harness else 'No'}")
            if result.suggested_fixes:
                lines.append(f"  Suggestions: {result.suggested_fixes[0][:80]}")
        
        lines.append("=" * 70)
        return '\n'.join(lines)


# ============================================================================
# 7. 便捷函数
# ============================================================================

async def llm_verify_kernel(kernel_id: str, use_llm_harness: bool = True):
    """便捷函数: 使用LLM验证单个kernel"""
    agent = LLMDrivenAccuracyAgent()
    return await agent.verify_with_llm(kernel_id, use_llm_harness)


async def llm_verify_batch(kernel_ids: List[str]):
    """便捷函数: 使用LLM批量验证"""
    agent = LLMDrivenAccuracyAgent()
    results = await agent.verify_batch(kernel_ids)
    print(agent.generate_report())
    return results


# ============================================================================
# 8. 演示
# ============================================================================

async def demo():
    """演示LLM驱动的准确度测试"""
    print("🤖 LLM-Driven Accuracy Test Demo")
    print("=" * 70)
    
    agent = LLMDrivenAccuracyAgent()
    
    # 测试一个kernel
    test_kernels = ['copy_type_converted', 'softmax']
    
    for kernel_id in test_kernels:
        print(f"\n🧪 Testing: {kernel_id}")
        print("-" * 50)
        
        result = await agent.verify_with_llm(kernel_id, use_llm_harness=True)
        
        print(f"Status: {'✅ PASS' if result.passed else '❌ FAIL'}")
        print(f"MAE: {result.mae:.2e}")
        print(f"LLM Harness: {'Yes' if result.llm_generated_harness else 'No'}")
        
        if result.suggested_fixes:
            print(f"Suggestions: {result.suggested_fixes[0]}")
    
    print("\n" + agent.generate_report())


if __name__ == "__main__":
    asyncio.run(demo())
