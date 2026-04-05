#!/usr/bin/env python3
"""
Unified CUDA-to-SYCL Converter Agent v3.0
统一转换Agent系统 - 集成所有功能

Features:
- 6个Agent统一调度
- 5个Phase优化执行
- 完整Trace日志
- 集成准确度测试
- 智能缓存和并行
"""

import os
import sys
import json
import asyncio
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class Phase(Enum):
    """执行阶段"""
    INITIALIZATION = "initialization"
    ANALYSIS = "analysis"
    CONVERSION = "conversion"
    VALIDATION = "validation"
    ACCURACY = "accuracy"
    REPORTING = "reporting"

class Status(Enum):
    """状态枚举"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

@dataclass
class ConversionResult:
    """转换结果"""
    kernel_id: str
    status: str
    duration_seconds: float
    phases_completed: int
    compilation_success: bool
    accuracy_pass_rate: float
    output_file: str

class UnifiedTracer:
    """统一Tracer"""
    def __init__(self, session_id: str, kernel_id: str):
        self.session_id = session_id
        self.kernel_id = kernel_id
        self.base_dir = Path(__file__).parent.parent
        self.trace_dir = self.base_dir / ".traces" / "sessions" / session_id
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.trace_dir / "unified_trace.jsonl"
        self.metrics = {
            "total_steps": 0,
            "total_tool_calls": 0,
            "errors": 0,
            "fixes": 0
        }
    
    def log(self, agent: str, action: str, details: dict):
        """记录Trace"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session": self.session_id,
            "kernel": self.kernel_id,
            "agent": agent,
            "action": action,
            "details": details
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        self.metrics["total_steps"] += 1
        
    def log_tool_call(self, tool: str, success: bool, duration_ms: float):
        """记录工具调用"""
        self.metrics["total_tool_calls"] += 1
        if not success:
            self.metrics["errors"] += 1

class UnifiedOrchestrator:
    """统一主控Agent"""
    
    def __init__(self, kernel_id: str, cuda_file: str, use_model: bool = False):
        self.kernel_id = kernel_id
        self.cuda_file = cuda_file
        self.use_model = use_model
        self.session_id = f"{kernel_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tracer = UnifiedTracer(self.session_id, kernel_id)
        
        self.state = {
            "current_phase": Phase.INITIALIZATION,
            "phases_completed": [],
            "attempts": 0,
            "fixes_applied": 0
        }
        
        # 初始化子Agent
        self.analyzer = UnifiedAnalyzer(self.tracer)
        self.converter = UnifiedConverter(self.tracer, use_model=use_model)
        self.validator = UnifiedValidator(self.tracer)
        self.accuracy_tester = UnifiedAccuracyTester(self.tracer)
        self.reporter = UnifiedReporter(self.tracer)
        
    async def execute_full_conversion(self) -> ConversionResult:
        """执行完整转换流程"""
        start_time = datetime.now()
        
        print(f"🚀 [UnifiedOrchestrator] 启动统一转换: {self.kernel_id}")
        print(f"   Session ID: {self.session_id}")
        self.tracer.log("UnifiedOrchestrator", "task_start", {
            "kernel_id": self.kernel_id,
            "cuda_file": str(self.cuda_file)
        })
        
        try:
            # Phase 1: 分析
            print("\n📊 Phase 1: CUDA分析...")
            analysis = await self.run_phase1_analysis()
            
            # Phase 2: 转换
            print("\n🔄 Phase 2: SYCL转换...")
            sycl_code = await self.run_phase2_conversion(analysis)
            
            # Phase 3: 编译验证
            print("\n🔨 Phase 3: 编译验证...")
            build_result = await self.run_phase3_validation(sycl_code)
            
            if not build_result["success"]:
                raise Exception(f"编译失败: {build_result['error']}")
            
            # Phase 4: 准确度测试 ⭐
            print("\n🎯 Phase 4: 准确度验证...")
            accuracy_result = await self.run_phase4_accuracy_test()
            
            # Phase 5: 报告
            print("\n📋 Phase 5: 生成报告...")
            await self.run_phase5_reporting(analysis, build_result, accuracy_result)
            
            # 计算总耗时
            duration = (datetime.now() - start_time).total_seconds()
            
            result = ConversionResult(
                kernel_id=self.kernel_id,
                status="success",
                duration_seconds=duration,
                phases_completed=5,
                compilation_success=True,
                accuracy_pass_rate=accuracy_result["pass_rate"],
                output_file=f"kernel_dataset/sycl/{self.kernel_id}_kernel.dp.cpp"
            )
            
            print(f"\n✅ [UnifiedOrchestrator] 转换完成!")
            print(f"   耗时: {duration:.1f}秒")
            print(f"   编译: {'通过' if result.compilation_success else '失败'}")
            print(f"   准确度: {result.accuracy_pass_rate*100:.1f}%")
            
            self.tracer.log("UnifiedOrchestrator", "task_complete", asdict(result))
            
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            result = ConversionResult(
                kernel_id=self.kernel_id,
                status="failed",
                duration_seconds=duration,
                phases_completed=len(self.state["phases_completed"]),
                compilation_success=False,
                accuracy_pass_rate=0.0,
                output_file=""
            )
            
            print(f"\n❌ [UnifiedOrchestrator] 转换失败: {e}")
            self.tracer.log("UnifiedOrchestrator", "task_failed", {
                "error": str(e),
                "phase": self.state["current_phase"].value
            })
            
            return result
    
    async def run_phase1_analysis(self) -> dict:
        """Phase 1: CUDA分析"""
        self.state["current_phase"] = Phase.ANALYSIS
        
        analysis = await self.analyzer.analyze(self.cuda_file)
        
        self.state["phases_completed"].append(Phase.ANALYSIS)
        return analysis
    
    async def run_phase2_conversion(self, analysis: dict) -> str:
        """Phase 2: 代码转换"""
        self.state["current_phase"] = Phase.CONVERSION
        
        sycl_code = await self.converter.convert(self.cuda_file, analysis)
        
        self.state["phases_completed"].append(Phase.CONVERSION)
        return sycl_code
    
    async def run_phase3_validation(self, sycl_code: str) -> dict:
        """Phase 3: 编译验证 (含自动修复)"""
        self.state["current_phase"] = Phase.VALIDATION
        
        max_attempts = 5
        
        for attempt in range(1, max_attempts + 1):
            print(f"   编译尝试 {attempt}/{max_attempts}...")
            
            result = await self.validator.validate(sycl_code, self.kernel_id)
            
            if result["success"]:
                print(f"   ✅ 编译成功!")
                self.state["phases_completed"].append(Phase.VALIDATION)
                return result
            
            # 尝试自动修复
            if attempt < max_attempts:
                print(f"   🔧 检测到错误，尝试自动修复...")
                fixes = await self.validator.auto_fix(result["errors"])
                
                if fixes:
                    sycl_code = self.apply_fixes(sycl_code, fixes)
                    self.state["fixes_applied"] += len(fixes)
                    print(f"   ✅ 应用了 {len(fixes)} 个修复")
                else:
                    print(f"   ⚠️  无法自动修复，继续下一次尝试")
        
        return {"success": False, "error": "Max retry attempts exceeded"}
    
    async def run_phase4_accuracy_test(self) -> dict:
        """Phase 4: 准确度测试 ⭐"""
        self.state["current_phase"] = Phase.ACCURACY
        
        # 检查是否已经有编译好的二进制
        cuda_binary = f"/tmp/{self.kernel_id}_cuda"
        sycl_binary = f"/tmp/{self.kernel_id}_sycl"
        
        result = await self.accuracy_tester.test(
            self.kernel_id,
            self.cuda_file,
            f"kernel_dataset/sycl/{self.kernel_id}_kernel.dp.cpp",
            cuda_binary,
            sycl_binary
        )
        
        self.state["phases_completed"].append(Phase.ACCURACY)
        return result
    
    async def run_phase5_reporting(self, analysis, build, accuracy):
        """Phase 5: 报告生成"""
        self.state["current_phase"] = Phase.REPORTING
        
        report = {
            "session_id": self.session_id,
            "kernel_id": self.kernel_id,
            "timestamp": datetime.now().isoformat(),
            "phases_completed": [p.value for p in self.state["phases_completed"]],
            "fixes_applied": self.state["fixes_applied"],
            "compilation": build,
            "accuracy": accuracy,
            "trace_metrics": self.tracer.metrics
        }
        
        # 保存报告
        report_file = self.tracer.trace_dir / "final_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.state["phases_completed"].append(Phase.REPORTING)
    
    def apply_fixes(self, code: str, fixes: List[dict]) -> str:
        """应用修复"""
        for fix in fixes:
            code = code.replace(fix["old"], fix["new"])
        return code


class UnifiedAnalyzer:
    """统一分析Agent"""
    
    def __init__(self, tracer: UnifiedTracer):
        self.tracer = tracer
    
    async def analyze(self, cuda_file: str) -> dict:
        """分析CUDA代码"""
        self.tracer.log("UnifiedAnalyzer", "start_analysis", {"file": cuda_file})
        
        # 读取文件
        with open(cuda_file, 'r') as f:
            code = f.read()
        
        lines = code.split('\n')
        
        # 分析组件
        analysis = {
            "total_lines": len(lines),
            "device_functions": code.count("__device__"),
            "global_kernels": code.count("__global__"),
            "templates": code.count("template"),
            "constants": code.count("constexpr"),
            "complexity_level": 3 if "winograd" in cuda_file else 2,
            "estimated_conversion_time": "15-20 minutes"
        }
        
        self.tracer.log("UnifiedAnalyzer", "analysis_complete", analysis)
        
        print(f"   📈 分析结果: {analysis['total_lines']}行")
        print(f"      Device函数: {analysis['device_functions']}")
        print(f"      Global kernels: {analysis['global_kernels']}")
        print(f"      复杂度: Level {analysis['complexity_level']}")
        
        return analysis


class ConversionError(Exception):
    """Conversion error exception"""
    pass


class ModelBasedConverter:
    """CUDA-to-SYCL converter using AI model"""
    
    def __init__(self, tracer: UnifiedTracer):
        self.tracer = tracer
        self.base_dir = Path(__file__).parent.parent
        
    def _load_prompt(self, filename: str) -> str:
        """Load prompt from file"""
        prompt_path = self.base_dir / "prompts" / filename
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    def _build_prompt(self, cuda_code: str, analysis: dict) -> str:
        """Build prompt for AI model using enhanced template"""
        from model_config import USER_PROMPT_TEMPLATE
        
        return USER_PROMPT_TEMPLATE.format(
            cuda_code=cuda_code,
            kernel_name=analysis.get('kernel_name', 'unknown'),
            total_lines=analysis.get('total_lines', 0),
            device_functions=analysis.get('device_functions', 0),
            global_kernels=analysis.get('global_kernels', 0),
            templates=analysis.get('templates', 0),
            complexity_level=analysis.get('complexity_level', 1)
        )
    
    async def _call_model(self, prompt: str) -> str:
        """
        Call Gaudi AI model to generate SYCL code
        
        Uses Gaudi AI API (DeepSeek/Qwen models) to generate SYCL code
        from the provided prompt.
        """
        # Save prompt for debugging
        prompt_file = f"/tmp/model_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        self.tracer.log("ModelBasedConverter", "calling_gaudi_ai", {
            "prompt_file": prompt_file,
            "prompt_length": len(prompt)
        })
        
        # Import and use Gaudi AI client
        try:
            from gaudi_ai_client import GaudiAIClient
            from model_config import MODEL_CONFIG, SYSTEM_PROMPT
            
            # Initialize client with API key from config
            api_key = MODEL_CONFIG["options"]["apiKey"]
            base_url = MODEL_CONFIG["options"]["baseURL"]
            model = MODEL_CONFIG["default_model"]
            
            client = GaudiAIClient(api_key=api_key, base_url=base_url)
            
            # Generate code using Gaudi AI
            print(f"   🤖 Calling Gaudi AI ({model})...")
            
            # Use larger max_tokens for complex kernels
            max_tokens = MODEL_CONFIG["models"][model]["max_tokens"] if model in MODEL_CONFIG["models"] else 8192
            temperature = MODEL_CONFIG["models"][model]["temperature"] if model in MODEL_CONFIG["models"] else 0.05
            
            generated_code = await client.generate_code(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=180
            )
            
            # Post-process to clean up generated code
            from model_config import post_process_generated_code
            generated_code = post_process_generated_code(generated_code)
            
            self.tracer.log("ModelBasedConverter", "gaudi_ai_success", {
                "model": model,
                "generated_length": len(generated_code),
                "max_tokens": max_tokens
            })
            
            return generated_code
            
        except ImportError as e:
            raise ConversionError(f"Failed to import Gaudi AI client: {e}")
        except Exception as e:
            raise ConversionError(f"Gaudi AI generation failed: {e}")
    
    async def _validate_syntax(self, code: str) -> bool:
        """Quick syntax validation"""
        required_patterns = [
            '#include <sycl/sycl.hpp>',
            'namespace sycldnn_backend'
        ]
        
        forbidden_patterns = [
            '1. ', '2. ', '3. ', '4. ', '5. ',  # Numbered lists
            'Analysis:', 'Conversion:', 'Step ',   # Explanations
            '```cpp', '```',                      # Markdown
            '*Alternative*', '*Note*',             # Explanatory notes
        ]
        
        for pattern in required_patterns:
            if pattern not in code:
                self.tracer.log("ModelBasedConverter", "validation_failed", {
                    "missing": pattern
                })
                return False
        
        for pattern in forbidden_patterns:
            if pattern in code:
                self.tracer.log("ModelBasedConverter", "validation_failed", {
                    "forbidden_content": pattern
                })
                return False
        
        return True
    
    async def convert(self, cuda_file: str, analysis: dict) -> str:
        """
        Convert CUDA to SYCL using AI model
        
        Args:
            cuda_file: Path to CUDA source file
            analysis: Analysis report from UnifiedAnalyzer
            
        Returns:
            SYCL code string
        """
        self.tracer.log("ModelBasedConverter", "start_conversion", {
            "file": cuda_file,
            "kernel_name": analysis.get('kernel_name', 'unknown')
        })
        
        # Read CUDA code
        with open(cuda_file, 'r', encoding='utf-8') as f:
            cuda_code = f.read()
        
        # Build prompt
        prompt = self._build_prompt(cuda_code, analysis)
        
        # Call AI model
        try:
            sycl_code = await self._call_model(prompt)
        except NotImplementedError:
            # If model not available, raise to trigger fallback
            raise
        
        # Post-process generated code to fix common issues
        try:
            from code_post_processor import post_process_code
            sycl_code, fixes = post_process_code(sycl_code, analysis.get('kernel_name', ''))
            if fixes:
                print(f"   🔧 Applied {len(fixes)} post-processing fixes")
                for fix in fixes[:5]:  # Show first 5 fixes
                    print(f"      - {fix}")
        except Exception as e:
            print(f"   ⚠️  Post-processing warning: {e}")
        
        # Validate generated code
        if not await self._validate_syntax(sycl_code):
            raise ConversionError("Model generated invalid SYCL code")
        
        self.tracer.log("ModelBasedConverter", "conversion_complete", {
            "lines": len(sycl_code.split('\n'))
        })
        
        return sycl_code


class RuleBasedConverter:
    """Rule-based CUDA-to-SYCL converter"""
    
    def __init__(self, tracer: UnifiedTracer):
        self.tracer = tracer
    
    async def convert(self, cuda_file: str, analysis: dict) -> str:
        """Convert CUDA to SYCL using replacement rules"""
        self.tracer.log("RuleBasedConverter", "start_conversion", {})
        
        # Read CUDA code
        with open(cuda_file, 'r', encoding='utf-8') as f:
            cuda_code = f.read()
        
        # Apply replacements
        replacements = [
            ("#include <cuda_runtime.h>", "#include <sycl/sycl.hpp>"),
            ("#include <cuda_fp16.h>", ""),
            ("__device__", ""),
            ("__global__", ""),
            ("__forceinline__", "inline"),
            ("namespace cudnn_backend", "namespace sycldnn_backend"),
            ("half", "sycl::half"),
            ("uint4", "sycl::uint4"),
            ("threadIdx.x", "item.get_local_id(0)"),
            ("threadIdx.y", "item.get_local_id(1)"),
            ("blockIdx.x", "item.get_group(0)"),
            ("blockIdx.y", "item.get_group(1)"),
            ("blockDim.x", "item.get_local_range(0)"),
            ("blockDim.y", "item.get_local_range(1)"),
            ("gridDim.x", "item.get_group_range(0)"),
            ("gridDim.y", "item.get_group_range(1)"),
            ("__syncthreads()", "item.barrier()"),
        ]
        
        sycl_code = cuda_code
        for old, new in replacements:
            sycl_code = sycl_code.replace(old, new)
        
        self.tracer.log("UnifiedConverter", "conversion_complete", {
            "replacements": len(replacements),
            "lines": len(sycl_code.split('\n'))
        })
        
        print("   📝 Applied {} replacement rules".format(len(replacements)))
        
        return sycl_code


class UnifiedConverter:
    """Enhanced converter with model-based and rule-based options"""
    
    def __init__(self, tracer: UnifiedTracer, use_model: bool = True):
        self.tracer = tracer
        self.use_model = use_model
        self.model_converter = ModelBasedConverter(tracer)
        self.rule_converter = RuleBasedConverter(tracer)
    
    async def convert(self, cuda_file: str, analysis: dict) -> str:
        """
        Convert CUDA to SYCL with fallback strategy
        
        Strategy:
        1. Try model-based conversion (if enabled)
        2. If fails, fallback to rule-based
        """
        self.tracer.log("UnifiedConverter", "start_conversion", {
            "file": cuda_file,
            "use_model": self.use_model,
            "complexity": analysis.get('complexity_level', 1)
        })
        
        if self.use_model:
            try:
                print("   🤖 Attempting model-based conversion...")
                sycl_code = await self.model_converter.convert(cuda_file, analysis)
                
                self.tracer.log("UnifiedConverter", "model_conversion_success", {
                    "lines": len(sycl_code.split('\n'))
                })
                print("   ✅ Model-based conversion successful")
                return sycl_code
                
            except (NotImplementedError, ConversionError) as e:
                self.tracer.log("UnifiedConverter", "model_conversion_failed", {
                    "error": str(e)
                })
                print(f"   ⚠️  Model conversion failed: {e}")
                print("   🔄 Falling back to rule-based conversion...")
        
        # Fallback to rule-based
        sycl_code = await self.rule_converter.convert(cuda_file, analysis)
        
        self.tracer.log("UnifiedConverter", "rule_conversion_complete", {
            "lines": len(sycl_code.split('\n'))
        })
        print("   ✅ Rule-based conversion complete")
        
        return sycl_code
        
        # 读取CUDA代码
        with open(cuda_file, 'r') as f:
            cuda_code = f.read()
        
        # 基础替换规则
        replacements = [
            ("#include <cuda_runtime.h>", "#include <sycl/sycl.hpp>"),
            ("#include <cuda_fp16.h>", ""),
            ("__device__", ""),
            ("__global__", ""),
            ("__forceinline__", "inline"),
            ("namespace cudnn_backend", "namespace sycldnn_backend"),
            ("half", "sycl::half"),
            ("uint4", "sycl::uint4"),
        ]
        
        sycl_code = cuda_code
        for old, new in replacements:
            sycl_code = sycl_code.replace(old, new)
        
        self.tracer.log("UnifiedConverter", "conversion_complete", {
            "replacements": len(replacements)
        })
        
        print(f"   📝 应用了 {len(replacements)} 个基础替换规则")
        
        return sycl_code


class UnifiedValidator:
    """统一验证Agent"""
    
    def __init__(self, tracer: UnifiedTracer):
        self.tracer = tracer
    
    async def validate(self, sycl_code: str, kernel_id: str) -> dict:
        """验证编译"""
        self.tracer.log("UnifiedValidator", "start_validation", {})
        
        # 保存代码
        output_file = f"kernel_dataset/sycl/{kernel_id}_kernel.dp.cpp"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(sycl_code)
        
        # 调用编译工具
        result = subprocess.run(
            ["./tools/b60_sycl_builder.sh", "compile", output_file],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        success = result.returncode == 0
        
        self.tracer.log("UnifiedValidator", "validation_result", {
            "success": success,
            "returncode": result.returncode
        })
        
        return {
            "success": success,
            "error": result.stderr if not success else None,
            "errors": [result.stderr] if not success and result.stderr else []
        }
    
    async def auto_fix(self, errors: List[str]) -> List[dict]:
        """自动修复错误"""
        fixes = []
        
        # 常见错误修复规则
        fix_rules = {
            "bfloat16.hpp": {
                "pattern": '#include <sycl/ext/oneapi/experimental/bfloat16.hpp>',
                "replacement": ''
            },
            "template_param_C_IndexNHCW": {
                "pattern": r'IndexNHCW<(\w+)>\(([^)]+)\)',
                "replacement": r'IndexNHCW(\2, \1)'
            },
            "template_param_C_IndexNCHW": {
                "pattern": r'IndexNCHW<(\w+)>\(([^)]+)\)',
                "replacement": r'IndexNCHW(\2, \1)'
            },
            "template_param_NC_TempIndexHWNC": {
                "pattern": r'TempIndexHWNC<(\w+),\s*(\w+)>\(([^)]+)\)',
                "replacement": r'TempIndexHWNC(\3, \1, \2)'
            }
        }
        
        for error in errors:
            for key, rule in fix_rules.items():
                if key in error:
                    fixes.append(rule)
        
        self.tracer.log("UnifiedValidator", "auto_fix", {
            "fixes_found": len(fixes)
        })
        
        return fixes


class UnifiedAccuracyTester:
    """统一准确度Agent ⭐ - 集成LLM驱动真实测试"""
    
    def __init__(self, tracer: UnifiedTracer):
        self.tracer = tracer
        self.base_dir = Path(__file__).parent.parent
        
        # 导入LLM Accuracy Test Agent组件
        sys.path.insert(0, str(self.base_dir / "tools"))
        from platform_detector import detect_platforms
        from test_suite_generator import generate_test_suite
        from llm_accuracy_test_agent import LLMAccuracyTestAgent
        
        self.detect_platforms = detect_platforms
        self.generate_test_suite = generate_test_suite
        self.LLMAccuracyTestAgent = LLMAccuracyTestAgent
        
        # 检测平台能力
        self.platform_caps = None
    
    async def test(self, kernel_id: str, cuda_file: str, sycl_file: str,
                   cuda_binary: str, sycl_binary: str) -> dict:
        """执行LLM驱动的真实准确度测试"""
        import time
        start_time = time.time()
        
        self.tracer.log("UnifiedAccuracyTester", "start_test", {
            "kernel": kernel_id,
            "mode": "llm_driven"
        })
        
        print(f"\n   🚀 启动LLM驱动准确度测试")
        print(f"   📁 CUDA文件: {cuda_file}")
        print(f"   📁 SYCL文件: {sycl_file}")
        
        try:
            # 使用集成的LLM Accuracy Test Agent
            agent = self.LLMAccuracyTestAgent(kernel_id, max_llm_concurrency=2)
            
            # 运行完整测试
            result = await agent.run_full_accuracy_test(
                cuda_file=cuda_file,
                sycl_file=sycl_file,
                output_dir=str(self.base_dir / ".traces" / "sessions" / self.tracer.session_id / "reports")
            )
            
            duration = time.time() - start_time
            
            if result.success and result.report:
                report = result.report
                summary = report.get('summary', {})
                ds = report.get('decision_support', {})
                quality = ds.get('conversion_quality', {})
                readiness = ds.get('deployment_readiness', {})
                
                self.tracer.log("UnifiedAccuracyTester", "test_complete", {
                    "duration": duration,
                    "total_tests": summary.get('total_tests', 0),
                    "passed": summary.get('passed', 0),
                    "failed": summary.get('failed', 0),
                    "pass_rate": summary.get('pass_rate', 0)
                })
                
                print(f"\n   ✅ LLM准确度测试完成!")
                print(f"      总测试数: {summary.get('total_tests', 0)}")
                print(f"      通过: {summary.get('passed', 0)} ✅")
                print(f"      失败: {summary.get('failed', 0)} ❌")
                print(f"      跳过: {summary.get('skipped', 0)} ⏭️")
                pass_rate = summary.get('pass_rate', 0)
                print(f"      通过率: {pass_rate*100:.1f}%")
                print(f"      耗时: {duration:.1f}s")
                
                # 显示决策支持
                print(f"\n   📊 质量评估: {quality.get('score', 'N/A')} - {quality.get('verdict', 'Unknown')}")
                is_ready = readiness.get('ready', False)
                print(f"   🚀 部署准备度: {'✅ 就绪' if is_ready else '❌ 未就绪'}")
                print(f"      置信度: {readiness.get('confidence', 'Unknown')}")
                
                # 返回简化版结果（与旧接口兼容）
                return {
                    "total_tests": summary.get('total_tests', 0),
                    "passed_tests": summary.get('passed', 0),
                    "pass_rate": pass_rate,
                    "status": "PASS" if pass_rate >= 0.99 else "FAIL",
                    "test_results": report.get('test_results', []),
                    "summary": summary,
                    "decision_support": ds,
                    "duration_seconds": duration
                }
            else:
                print(f"\n   ❌ LLM准确度测试失败: {result.error}")
                
                self.tracer.log("UnifiedAccuracyTester", "test_failed", {
                    "error": result.error,
                    "duration": duration
                })
                
                return {
                    "total_tests": 0,
                    "passed_tests": 0,
                    "pass_rate": 0.0,
                    "status": "FAILED",
                    "error": result.error,
                    "duration_seconds": duration
                }
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"\n   ❌ 测试执行异常: {e}")
            import traceback
            traceback.print_exc()
            
            self.tracer.log("UnifiedAccuracyTester", "test_error", {
                "error": str(e),
                "duration": duration
            })
            
            return {
                "total_tests": 0,
                "passed_tests": 0,
                "pass_rate": 0.0,
                "status": "ERROR",
                "error": str(e),
                "duration_seconds": duration
            }
    
class UnifiedReporter:
    """Unified Reporter Agent for generating comprehensive reports"""
    
    def __init__(self, tracer: UnifiedTracer):
        self.tracer = tracer
        self.base_dir = Path(__file__).parent.parent
    
    async def generate_reports(self, session_data: dict) -> dict:
        """Generate all report formats"""
        self.tracer.log("UnifiedReporter", "start_reporting", {
            "kernel_id": session_data.get("kernel_id"),
            "session_id": session_data.get("session_id")
        })
        
        # Create output directory
        output_dir = self.base_dir / ".traces" / "sessions" / session_data["session_id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        reports = {}
        
        # Generate JSON report
        try:
            json_content = self.generate_json_report(session_data)
            json_path = output_dir / "final_report.json"
            with open(json_path, 'w') as f:
                f.write(json_content)
            reports["json"] = str(json_path)
            print("   ✓ JSON report generated")
        except Exception as e:
            self.tracer.log("UnifiedReporter", "json_error", {"error": str(e)})
            print(f"   ⚠ JSON report failed: {e}")
            reports["json"] = None
        
        # Generate HTML report
        try:
            html_content = self.generate_html_report(session_data)
            html_path = output_dir / "final_report.html"
            with open(html_path, 'w') as f:
                f.write(html_content)
            reports["html"] = str(html_path)
            print("   ✓ HTML report generated")
        except Exception as e:
            self.tracer.log("UnifiedReporter", "html_error", {"error": str(e)})
            print(f"   ⚠ HTML report failed: {e}")
            reports["html"] = None
        
        # Generate Markdown report
        try:
            md_content = self.generate_markdown_report(session_data)
            md_path = output_dir / "final_report.md"
            with open(md_path, 'w') as f:
                f.write(md_content)
            reports["markdown"] = str(md_path)
            print("   ✓ Markdown report generated")
        except Exception as e:
            self.tracer.log("UnifiedReporter", "md_error", {"error": str(e)})
            print(f"   ⚠ Markdown report failed: {e}")
            reports["markdown"] = None
        
        self.tracer.log("UnifiedReporter", "reports_complete", reports)
        return reports
    
    def generate_json_report(self, data: dict) -> str:
        """Generate JSON format report"""
        import json
        return json.dumps(data, indent=2, default=str)
    
    def generate_html_report(self, data: dict) -> str:
        """Generate HTML format report with styling"""
        kernel_id = data.get("kernel_id", "unknown")
        session_id = data.get("session_id", "unknown")
        timestamp = data.get("timestamp", "")
        overall_status = data.get("overall_status", "unknown")
        
        # Determine status color
        status_color = "green" if overall_status == "success" else "red"
        overall_status_str = str(overall_status).upper()
        
        # Build phases table
        phases = data.get("phases", {})
        phases_rows = ""
        for phase_name, phase_data in phases.items():
            phase_status = phase_data.get("status", "unknown")
            phase_duration = phase_data.get("duration_seconds", 0)
            row_class = "status-success" if phase_status == "completed" else "status-fail"
            phases_rows += "<tr class='{0}'><td>{1}</td><td>{2}</td><td>{3:.1f}s</td></tr>\n".format(
                row_class, phase_name, phase_status, phase_duration
            )
        
        # Build performance metrics
        perf = data.get("performance", {})
        total_duration = perf.get("total_duration_seconds", 0)
        
        # Build trace metrics
        trace = data.get("trace_summary", {})
        total_steps = trace.get("total_steps", 0)
        total_tool_calls = trace.get("total_tool_calls", 0)
        
        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Conversion Report - {kernel_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .status-success {{ color: green; font-weight: bold; }}
        .status-fail {{ color: red; font-weight: bold; }}
        h1 {{ color: #333; margin-top: 0; }}
        h2 {{ color: #666; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .metric-card {{ display: inline-block; margin: 10px 15px 10px 0; padding: 20px; background: #e3f2fd; border-radius: 8px; min-width: 150px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #1976d2; }}
        .metric-label {{ color: #666; font-size: 14px; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 CUDA-to-SYCL Conversion Report</h1>
            <p><strong>Kernel:</strong> {kernel_id}</p>
            <p><strong>Session:</strong> {session_id}</p>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            <p><strong>Status:</strong> <span class="status-{status_color}">{overall_status_str}</span></p>
        </div>
        
        <h2>📊 Phase Summary</h2>
        <table>
            <tr>
                <th>Phase</th>
                <th>Status</th>
                <th>Duration</th>
            </tr>
            {phases_rows}
        </table>
        
        <h2>⚡ Performance Metrics</h2>
        <div class="metric-card">
            <div class="metric-value">{total_duration:.1f}s</div>
            <div class="metric-label">Total Duration</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_steps}</div>
            <div class="metric-label">Total Steps</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_tool_calls}</div>
            <div class="metric-label">Tool Calls</div>
        </div>
        
        <div class="footer">
            <p>Generated by UnifiedReporter Agent | CUDA-to-SYCL Conversion Tool</p>
        </div>
    </div>
</body>
</html>"""
        
        return html_template.format(
            kernel_id=kernel_id,
            session_id=session_id,
            timestamp=timestamp,
            status_color=status_color,
            overall_status_str=overall_status_str,
            phases_rows=phases_rows,
            total_duration=total_duration,
            total_steps=total_steps,
            total_tool_calls=total_tool_calls
        )
    
    def generate_markdown_report(self, data: dict) -> str:
        """Generate Markdown format report"""
        kernel_id = data.get("kernel_id", "unknown")
        session_id = data.get("session_id", "unknown")
        timestamp = data.get("timestamp", "")
        overall_status = data.get("overall_status", "unknown")
        
        status_emoji = "✅" if overall_status == "success" else "❌"
        overall_status_str = str(overall_status).upper()
        
        # Build phases section
        phases = data.get("phases", {})
        phases_md = ""
        for phase_name, phase_data in phases.items():
            phase_status = phase_data.get("status", "unknown")
            phase_duration = phase_data.get("duration_seconds", 0)
            phases_md += "- **{0}**: {1} ({2:.1f}s)\n".format(
                phase_name.capitalize(), phase_status, phase_duration
            )
        
        # Build performance section
        perf = data.get("performance", {})
        total_duration = perf.get("total_duration_seconds", 0)
        
        # Build trace section
        trace = data.get("trace_summary", {})
        total_steps = trace.get("total_steps", 0)
        total_tool_calls = trace.get("total_tool_calls", 0)
        errors = trace.get("errors", 0)
        fixes = trace.get("fixes", 0)
        
        md_template = """# CUDA-to-SYCL Conversion Report

{status_emoji} **Status**: {overall_status_str}

## Summary

- **Kernel ID**: {kernel_id}
- **Session ID**: {session_id}
- **Timestamp**: {timestamp}
- **Total Duration**: {total_duration:.1f} seconds

## Phase Breakdown

{phases_md}

## Performance Metrics

| Metric | Value |
|--------|-------|
| Total Duration | {total_duration:.1f}s |
| Total Steps | {total_steps} |
| Tool Calls | {total_tool_calls} |
| Errors | {errors} |
| Fixes Applied | {fixes} |

## Details

### Analysis
- Lines analyzed: {analysis_lines}
- Complexity level: {complexity_level}

### Compilation
- Success: {compilation_success}
- Fixes applied: {compilation_fixes}

### Accuracy
- Pass rate: {accuracy_pass_rate:.1%}
- Max absolute diff: {accuracy_max_abs:.2e}
- Max relative diff: {accuracy_max_rel:.2e}

---

*Report generated by UnifiedReporter Agent*
"""
        
        # Extract additional data safely
        analysis = data.get("analysis", {})
        compilation = data.get("compilation", {})
        accuracy = data.get("accuracy", {})
        
        return md_template.format(
            status_emoji=status_emoji,
            overall_status_str=overall_status_str,
            kernel_id=kernel_id,
            session_id=session_id,
            timestamp=timestamp,
            total_duration=total_duration,
            phases_md=phases_md,
            total_steps=total_steps,
            total_tool_calls=total_tool_calls,
            errors=errors,
            fixes=fixes,
            analysis_lines=analysis.get("total_lines", 0),
            complexity_level=analysis.get("complexity_level", 0),
            compilation_success="Yes" if compilation.get("success") else "No",
            compilation_fixes=data.get("fixes_applied", 0),
            accuracy_pass_rate=accuracy.get("pass_rate", 0),
            accuracy_max_abs=accuracy.get("summary", {}).get("overall_max_abs_diff", 0),
            accuracy_max_rel=accuracy.get("summary", {}).get("overall_max_rel_diff", 0)
        )


async def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: unified_converter.py <kernel_id>")
        print("Example: unified_converter.py winograd_input_transform")
        sys.exit(1)
    
    kernel_id = sys.argv[1]
    cuda_file = f"kernel_dataset/cuda/{kernel_id}_kernel.cu"
    
    if not os.path.exists(cuda_file):
        print(f"Error: CUDA file not found: {cuda_file}")
        sys.exit(1)
    
    # 创建并运行统一Agent
    orchestrator = UnifiedOrchestrator(kernel_id, cuda_file)
    result = await orchestrator.execute_full_conversion()
    
    # 输出结果
    print("\n" + "="*60)
    print("📊 转换结果汇总")
    print("="*60)
    print(f"Kernel: {result.kernel_id}")
    print(f"Status: {result.status}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"Phases: {result.phases_completed}/5")
    print(f"Compilation: {'✅ Pass' if result.compilation_success else '❌ Fail'}")
    print(f"Accuracy: {result.accuracy_pass_rate*100:.1f}%")
    print(f"Output: {result.output_file}")
    print("="*60)
    
    sys.exit(0 if result.status == "success" else 1)


if __name__ == "__main__":
    asyncio.run(main())
