# Model Configuration for Gaudi AI
# Gaudi AI 模型配置

import json
import os
from pathlib import Path

# 模型配置
MODEL_CONFIG = {
    "provider": "gaudi_ai",
    "name": "Gaudi AI",
    "npm": "@ai-sdk/openai-compatible",
    "models": {
        "DeepSeek-R1-G2-static-671B": {
            "name": "DeepSeek-R1-G2-static-671B",
            "description": "DeepSeek R1 671B - Best for complex code generation",
            "max_tokens": 8192,
            "temperature": 0.1
        },
        "Qwen3-Coder-30B-A3B-Instruct": {
            "name": "Qwen3-Coder-30B-A3B-Instruct",
            "description": "Qwen3 Coder 30B - Optimized for code",
            "max_tokens": 4096,
            "temperature": 0.1
        },
        "glm-4.6-fp8": {
            "name": "glm-4.6-fp8",
            "description": "GLM 4.6 FP8 - Balanced performance",
            "max_tokens": 4096,
            "temperature": 0.1
        },
        "glm-4.7-fp8": {
            "name": "glm-4.7-fp8",
            "description": "GLM 4.7 FP8 - Latest version",
            "max_tokens": 8192,  # Increased for large kernels
            "temperature": 0.05  # Lower for more deterministic output
        }
    },
    "options": {
        "apiKey": "sk-43Y6B7acOvFuOPelLuLw8Q",
        "baseURL": "http://10.112.110.111/v1",
        "headers": {
            "Authorization": "Bearer sk-43Y6B7acOvFuOPelLuLw8Q"
        }
    },
    "default_model": "glm-4.7-fp8"
}

# Enhanced System prompt with strict constraints
SYSTEM_PROMPT = """You are a CUDA to SYCL code converter. 

ABSOLUTE RULES (VIOLATION = FAILURE):
1. OUTPUT ONLY C++ CODE - NO explanations, NO comments about conversion, NO markdown
2. NO ```cpp or ``` markers around code
3. NO numbered lists, NO bullet points, NO analysis steps
4. CODE MUST BE COMPLETE - never truncate, never end mid-function
5. CODE MUST COMPILE with: icpx -fsycl -std=c++17

CONVERSION RULES:
1. HEADERS:
   - Replace: <cuda_runtime.h>, <cuda_fp16.h>
   - With: <sycl/sycl.hpp>

2. NAMESPACES:
   - Use: namespace lczero { namespace sycldnn_backend { } }

3. REMOVE CUDA KEYWORDS:
   - Remove: __device__, __global__, __forceinline__, __launch_bounds__(...)
   - Use: inline for helper functions

4. THREAD INDEXING:
   - threadIdx.x -> item.get_local_id(0)
   - blockIdx.x -> item.get_group(0)
   - blockDim.x -> item.get_local_range(0)
   - gridDim.x -> item.get_group_range(0)

5. SHARED MEMORY:
   - OLD: __shared__ float arr[SIZE];
   - NEW: sycl::local_accessor<float, 1> arr(sycl::range<1>(SIZE), cgh);
   - Use sycl::range<> and handler (cgh)

6. SYNCHRONIZATION:
   - __syncthreads() -> item.barrier(sycl::access::fence_space::local_space)

7. MATH FUNCTIONS:
   - __expf(x) -> sycl::exp(x)
   - __fdividef(a, b) -> a / b
   - __shfl_xor_sync(mask, x, offset) -> sycl::reduce_over_group(item.get_sub_group(), x, sycl::plus<float>())
   - __shfl_down_sync(mask, x, offset) -> sycl::shift_group_left(item.get_sub_group(), x, offset)
   - tanh(x) -> sycl::tanh(x)

8. KERNEL STRUCTURE:
   Convert to functor:
   template<typename T>
   struct KernelName {
     void operator()(sycl::nd_item<1> item) const { ... }
   };

9. KEEP ORIGINAL:
   - Template parameters, constants, macros
   - GPL license header
   - Algorithm logic

START WITH GPL HEADER AND INCLUDE, THEN CODE. NOTHING ELSE."""

# Strict user prompt template
USER_PROMPT_TEMPLATE = """Convert this CUDA code to SYCL. OUTPUT ONLY CODE.

CUDA CODE:
{cuda_code}

Kernel: {kernel_name} | Lines: {total_lines}

REQUIREMENTS:
- Output ONLY compilable C++ code
- NO explanations, NO markdown ``` blocks
- MUST handle __expf, __shfl_xor_sync, __shfl_down_sync correctly
- MUST use sycl::range for local_accessor
- MUST preserve all templates
- Code must end with complete functions

SYCL CODE:"""


def post_process_generated_code(code: str) -> str:
    """Post-process AI generated code to remove unwanted content"""
    lines = code.split('\n')
    result = []
    in_code_block = False
    
    for line in lines:
        # Skip markdown code block markers
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        
        # Skip common non-code patterns
        stripped = line.strip()
        if stripped.startswith(('1. ', '2. ', '3. ', '4. ', '5. ', '6. ', '7. ', '8. ', '9. ')):
            continue
        if stripped.startswith(('* ', '- ', '> ', '# ')):
            continue
        if stripped.lower() in ['analyze', 'analysis', 'conversion', 'converted code:', 
                                'sycl code:', 'here is', 'note:', 'important:']:
            continue
        
        # Keep code lines
        result.append(line)
    
    # Join and clean up
    code = '\n'.join(result)
    
    # Remove leading/trailing whitespace
    code = code.strip()
    
    return code
