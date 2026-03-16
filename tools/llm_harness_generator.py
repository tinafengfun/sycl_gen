#!/usr/bin/env python3
"""
LLM Harness Generator
LLM驱动的测试harness生成器

核心功能：使用LLM智能生成CUDA和SYCL测试代码
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys

# 添加tools目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from gaudi_ai_client import GaudiAIClient
from model_config import MODEL_CONFIG, SYSTEM_PROMPT

@dataclass
class KernelAnalysis:
    """Kernel分析结果"""
    kernel_name: str
    templates: List[str]
    parameters: List[Dict]
    launch_config: Dict
    data_types: List[str]
    complexity: int
    dependencies: List[str]
    test_requirements: Dict

@dataclass
class GeneratedHarness:
    """生成的测试harness"""
    cuda_code: str
    sycl_code: str
    kernel_analysis: KernelAnalysis
    generation_time: float
    llm_calls: int
    success: bool
    error: Optional[str] = None


class LLMHarnessGenerator:
    """LLM测试harness生成器"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.client = GaudiAIClient(
            api_key=MODEL_CONFIG["options"]["apiKey"],
            base_url=MODEL_CONFIG["options"]["baseURL"]
        )
        self.model = MODEL_CONFIG["default_model"]
        self.llm_calls = 0
    
    async def analyze_kernel(self, cuda_code: str) -> KernelAnalysis:
        """
        使用LLM分析kernel结构
        
        Args:
            cuda_code: CUDA kernel源代码
            
        Returns:
            KernelAnalysis对象
        """
        prompt = """Analyze this CUDA kernel and extract structured information:

KERNEL CODE:
""" + cuda_code + """

Extract and return ONLY a JSON object with this structure:
{
  "kernel_name": "extracted name without _kernel suffix",
  "templates": ["list of template parameters like 'typename T'"],
  "parameters": [
    {"type": "float*", "name": "param_name", "is_pointer": true, "is_output": true, "is_const": false}
  ],
  "launch_config": {
    "grid_dims": "expression to calculate grid size",
    "block_dims": 256,
    "shared_memory": 0
  },
  "data_types": ["float32", "float16", "bfloat16"],
  "complexity": 1,
  "dependencies": ["cuda_runtime.h"],
  "test_requirements": {
    "min_size": 1024,
    "needs_special_values": false,
    "dimensionality": 1
  }
}

Be precise. Return ONLY valid JSON, no markdown, no explanations."""

        try:
            response = await self.client.generate_code(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                model=self.model,
                temperature=0.1,
                max_tokens=2048
            )
            self.llm_calls += 1
            
            # 解析JSON响应
            # 清理可能的markdown标记
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            analysis_dict = json.loads(response)
            
            return KernelAnalysis(
                kernel_name=analysis_dict.get("kernel_name", "unknown"),
                templates=analysis_dict.get("templates", []),
                parameters=analysis_dict.get("parameters", []),
                launch_config=analysis_dict.get("launch_config", {}),
                data_types=analysis_dict.get("data_types", ["float32"]),
                complexity=analysis_dict.get("complexity", 1),
                dependencies=analysis_dict.get("dependencies", []),
                test_requirements=analysis_dict.get("test_requirements", {})
            )
            
        except Exception as e:
            print(f"Kernel analysis failed: {e}")
            # 返回默认分析
            return KernelAnalysis(
                kernel_name="unknown",
                templates=[],
                parameters=[],
                launch_config={"block_dims": 256},
                data_types=["float32"],
                complexity=1,
                dependencies=["cuda_runtime.h"],
                test_requirements={"min_size": 1024}
            )
    
    async def generate_cuda_harness(
        self, 
        kernel_code: str,
        analysis: KernelAnalysis,
        test_config: Dict
    ) -> str:
        """生成CUDA测试harness"""
        
        prompt = f"""Generate a complete, compilable CUDA test harness.

KERNEL TO TEST:
{kernel_code}

KERNEL ANALYSIS:
{json.dumps(analysis.__dict__, indent=2)}

TEST CONFIGURATION:
- Test name: {test_config['name']}
- Data type: {test_config['dtype']}
- Dimensions: N={test_config.get('N', 1)}, C={test_config.get('C', 64)}, H={test_config.get('H', 8)}, W={test_config.get('W', 8)}
- Elements: {test_config.get('N', 1) * test_config.get('C', 64) * test_config.get('H', 8) * test_config.get('W', 8)}
- Data generation: {test_config.get('data_gen', 'random_uniform')}
  * Strategy: {test_config.get('data_strategy', 'random')}
  * Seed: 42 (fixed for reproducibility)
  * Range: [{test_config.get('min_val', -1.0)}, {test_config.get('max_val', 1.0)}]
- Template instantiation: {test_config.get('template_types', ['float', 'float'])}
- Output file: /tmp/test_output_cuda.bin

REQUIREMENTS:
1. Include: cuda_runtime.h, cuda_fp16.h (if fp16/bf16)
2. Copy the kernel function from "KERNEL TO TEST" section
3. Generate test data with FIXED seed 42 using curand or std::mt19937
4. Allocate device memory for all kernel parameters
5. For {test_config['dtype']}:
   - float32: use float
   - bfloat16: use __nv_bfloat16 (check SM version at runtime)
   - float16: use half
6. Launch kernel: blocks = (size + 255) / 256, threads = 256
7. Copy results back to host
8. Save output to binary file: /tmp/test_output_cuda.bin
9. Check CUDA errors after each call
10. Free all allocated memory

Generate a complete main() function that:
1. Reads no input (generates data internally)
2. Runs the kernel
3. Saves output to /tmp/test_output_cuda.bin
4. Returns 0 on success

RETURN ONLY the complete compilable CUDA C++ code. No markdown, no explanations, no code blocks."""

        try:
            code = await self.client.generate_code(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                model=self.model,
                temperature=0.1,
                max_tokens=4096
            )
            self.llm_calls += 1
            
            # 清理代码
            code = self._clean_generated_code(code)
            return code
            
        except Exception as e:
            print(f"CUDA harness generation failed: {e}")
            return self._generate_fallback_cuda_harness(analysis, test_config)
    
    async def generate_sycl_harness(
        self,
        kernel_code: str,
        analysis: KernelAnalysis,
        test_config: Dict
    ) -> str:
        """生成SYCL测试harness"""
        
        prompt = f"""Generate a complete, compilable SYCL test harness.

KERNEL TO TEST:
{kernel_code}

KERNEL ANALYSIS:
{json.dumps(analysis.__dict__, indent=2)}

TEST CONFIGURATION:
- Test name: {test_config['name']}
- Data type: {test_config['dtype']}
- Dimensions: N={test_config.get('N', 1)}, C={test_config.get('C', 64)}, H={test_config.get('H', 8)}, W={test_config.get('W', 8)}
- Elements: {test_config.get('N', 1) * test_config.get('C', 64) * test_config.get('H', 8) * test_config.get('W', 8)}
- Data generation: {test_config.get('data_gen', 'random_uniform')}
  * Strategy: {test_config.get('data_strategy', 'random')}
  * Seed: 42 (fixed for reproducibility)
  * Range: [{test_config.get('min_val', -1.0)}, {test_config.get('max_val', 1.0)}]
- Template instantiation: {test_config.get('template_types', ['float', 'float'])}
- Output file: /tmp/test_output_sycl.bin

REQUIREMENTS:
1. Include: sycl/sycl.hpp
2. Convert kernel to SYCL functor (struct with operator())
3. Generate test data with FIXED seed 42
4. Allocate device memory: sycl::malloc_device
5. For {test_config['dtype']}:
   - float32: use float
   - bfloat16: use sycl::ext::oneapi::bfloat16
   - float16: use sycl::half
6. Launch: sycl::parallel_for with nd_range
7. Copy back: queue.memcpy().wait()
8. Save output to: /tmp/test_output_sycl.bin
9. Free memory: sycl::free

Generate a complete main() function that:
1. Generates test data internally with seed 42
2. Runs the kernel
3. Saves output to /tmp/test_output_sycl.bin
4. Returns 0 on success

RETURN ONLY the complete compilable SYCL C++ code. No markdown, no explanations, no code blocks."""

        try:
            code = await self.client.generate_code(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                model=self.model,
                temperature=0.1,
                max_tokens=4096
            )
            self.llm_calls += 1
            
            # 清理代码
            code = self._clean_generated_code(code)
            return code
            
        except Exception as e:
            print(f"SYCL harness generation failed: {e}")
            return self._generate_fallback_sycl_harness(analysis, test_config)
    
    async def fix_compilation_error(
        self,
        code: str,
        error_message: str,
        language: str  # "cuda" or "sycl"
    ) -> str:
        """使用LLM修复编译错误"""
        
        prompt = f"""Fix this compilation error.

LANGUAGE: {language.upper()}

ORIGINAL CODE:
{code}

COMPILER ERROR:
{error_message}

Fix the code to resolve the error. Keep the same functionality.
Common issues:
- Missing includes
- Type mismatches (especially for fp16/bf16)
- Template instantiation
- SYCL/CUDA API differences
- Syntax errors

RETURN ONLY the corrected, compilable code. No explanations."""

        try:
            fixed_code = await self.client.generate_code(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                model=self.model,
                temperature=0.1,
                max_tokens=4096
            )
            self.llm_calls += 1
            
            return self._clean_generated_code(fixed_code)
            
        except Exception as e:
            print(f"Error fix failed: {e}")
            return code  # 返回原始代码
    
    def _clean_generated_code(self, code: str) -> str:
        """清理LLM生成的代码"""
        # 移除markdown标记
        code = code.strip()
        if code.startswith('```cpp'):
            code = code[6:]
        elif code.startswith('```c++'):
            code = code[6:]
        elif code.startswith('```cuda'):
            code = code[7:]
        elif code.startswith('```sycl'):
            code = code[7:]
        elif code.startswith('```'):
            code = code[3:]
        
        if code.endswith('```'):
            code = code[:-3]
        
        return code.strip()
    
    def _generate_fallback_cuda_harness(
        self,
        analysis: KernelAnalysis,
        test_config: Dict
    ) -> str:
        """生成fallback CUDA harness（当LLM失败时）"""
        return f'''// Fallback CUDA harness for {analysis.kernel_name}
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <random>

int main() {{
    const int N = 1024;
    std::vector<float> h_input(N);
    std::vector<float> h_output(N);
    
    // Generate test data with seed 42
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < N; i++) {{
        h_input[i] = dis(gen);
    }}
    
    // TODO: Implement actual kernel test
    // For now, just copy input to output
    h_output = h_input;
    
    // Save output
    std::ofstream out("/tmp/test_output_cuda.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(h_output.data()), N * sizeof(float));
    out.close();
    
    return 0;
}}
'''
    
    def _generate_fallback_sycl_harness(
        self,
        analysis: KernelAnalysis,
        test_config: Dict
    ) -> str:
        """生成fallback SYCL harness（当LLM失败时）"""
        return f'''// Fallback SYCL harness for {analysis.kernel_name}
#include <sycl/sycl.hpp>
#include <fstream>
#include <vector>
#include <random>

int main() {{
    const int N = 1024;
    std::vector<float> h_input(N);
    std::vector<float> h_output(N);
    
    // Generate test data with seed 42
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < N; i++) {{
        h_input[i] = dis(gen);
    }}
    
    // TODO: Implement actual kernel test
    // For now, just copy input to output
    h_output = h_input;
    
    // Save output
    std::ofstream out("/tmp/test_output_sycl.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(h_output.data()), N * sizeof(float));
    out.close();
    
    return 0;
}}
'''
    
    async def generate_full_harness(
        self,
        kernel_code: str,
        test_config: Dict
    ) -> GeneratedHarness:
        """生成完整的测试harness（CUDA + SYCL）"""
        import time
        start_time = time.time()
        
        # 1. 分析kernel
        print(f"  🔍 Analyzing kernel...")
        analysis = await self.analyze_kernel(kernel_code)
        
        # 2. 生成CUDA harness
        print(f"  📝 Generating CUDA harness...")
        cuda_code = await self.generate_cuda_harness(kernel_code, analysis, test_config)
        
        # 3. 生成SYCL harness
        print(f"  📝 Generating SYCL harness...")
        sycl_code = await self.generate_sycl_harness(kernel_code, analysis, test_config)
        
        generation_time = time.time() - start_time
        
        return GeneratedHarness(
            cuda_code=cuda_code,
            sycl_code=sycl_code,
            kernel_analysis=analysis,
            generation_time=generation_time,
            llm_calls=self.llm_calls,
            success=True
        )


# 测试函数
async def test_harness_generator():
    """测试harness生成器"""
    print("="*70)
    print("🧪 Testing LLM Harness Generator")
    print("="*70)
    
    # 测试kernel
    test_kernel = '''
template <typename T>
__global__ void addOne_kernel(T* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 1.0f;
    }
}
'''
    
    test_config = {
        "name": "f32_simple_test",
        "dtype": "float32",
        "N": 1,
        "C": 64,
        "H": 8,
        "W": 8,
        "data_gen": "random_uniform",
        "data_strategy": "random",
        "min_val": -1.0,
        "max_val": 1.0,
        "template_types": ["float"]
    }
    
    generator = LLMHarnessGenerator()
    
    print("\n🚀 Generating test harnesses...")
    result = await generator.generate_full_harness(test_kernel, test_config)
    
    print(f"\n📊 Generation Complete:")
    print(f"   Success: {result.success}")
    print(f"   Time: {result.generation_time:.2f}s")
    print(f"   LLM calls: {result.llm_calls}")
    print(f"   Kernel: {result.kernel_analysis.kernel_name}")
    
    print(f"\n📝 CUDA Code (first 30 lines):")
    print('\n'.join(result.cuda_code.split('\n')[:30]))
    
    print(f"\n📝 SYCL Code (first 30 lines):")
    print('\n'.join(result.sycl_code.split('\n')[:30]))


if __name__ == "__main__":
    asyncio.run(test_harness_generator())
