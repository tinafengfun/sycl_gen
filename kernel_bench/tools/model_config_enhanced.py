# Enhanced Model Configuration for Gaudi AI
# 增强版Gaudi AI模型配置

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
            "max_tokens": 4096,
            "temperature": 0.1
        }
    },
    "options": {
        "apiKey": "sk-43Y6B7acOvFuOPelLuLw8Q",
        "baseURL": "http://10.112.110.111/v1",
        "headers": {
            "Authorization": "Bearer sk-43Y6B7acOvFuOPelLuLw8Q"
        }
    },
    "default_model": "DeepSeek-R1-G2-static-671B"  # Use best model for complex kernels
}

# Enhanced System prompt for CUDA to SYCL conversion
SYSTEM_PROMPT = """You are an expert CUDA to SYCL converter specializing in GPU kernel optimization.

CRITICAL RULES - READ CAREFULLY:

1. OUTPUT FORMAT:
   - Return ONLY raw C++ code, NO markdown formatting (no ```cpp blocks)
   - NO explanations, NO comments about the conversion
   - Code must be directly compilable

2. HEADERS:
   - Use: #include <sycl/sycl.hpp>
   - Remove all CUDA headers (#include <cuda_runtime.h>, <cuda_fp16.h>)

3. NAMESPACE:
   - Use: namespace lczero { namespace sycldnn_backend { ... } }

4. FUNCTION QUALIFIERS:
   - Remove __device__, __global__, __forceinline__
   - Use: inline for device functions
   - Kernels become functors or lambda functions inside parallel_for

5. THREAD INDEXING (CRITICAL - USE EXACT SYNTAX):
   - threadIdx.x -> item.get_local_id(0)
   - blockIdx.x -> item.get_group(0)
   - blockDim.x -> item.get_local_range(0)
   - gridDim.x -> item.get_group_range(0)

6. SHARED MEMORY (CRITICAL):
   - OLD: __shared__ float arr[SIZE];
   - NEW: sycl::local_accessor<float, 1> arr(sycl::range<1>(SIZE), cgh);
   - For 2D: sycl::local_accessor<float, 2> arr(sycl::range<2>(H, W), cgh);
   - MUST use sycl::range<> and command group handler (cgh)

7. SYNCHRONIZATION:
   - __syncthreads() -> item.barrier(sycl::access::fence_space::local_space)

8. KERNEL LAUNCH PATTERN:
   OLD:
     __global__ void kernel(...) { ... }
     kernel<<<blocks, threads>>>(...);
   
   NEW:
     queue.parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
       // kernel body using item.get_global_id(0), etc.
     });

9. MATH FUNCTIONS:
   - __expf(x) -> sycl::exp(x)
   - __fdividef(a, b) -> a / b
   - Use sycl:: versions for all math

10. PRESERVE ALL CONSTANTS AND MACROS from original code

11. DO NOT invent new variables or APIs - only convert existing ones

12. KEEP the original GPL license header if present

Example of correct local_accessor usage:
  sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(128), cgh);
  sycl::local_accessor<float, 2> tile(sycl::range<2>(16, 16), cgh);"""

# Enhanced User prompt template
USER_PROMPT_TEMPLATE = """Convert this CUDA kernel to SYCL following ALL rules above.

CUDA Source Code:
{cuda_code}

Kernel: {kernel_name}
Lines: {total_lines} | Device Functions: {device_functions} | Global Kernels: {global_kernels}

REMEMBER:
1. NO markdown code blocks (```)
2. MUST use sycl::range<> for local_accessor
3. MUST preserve all constants and macros
4. Code must compile with: icpx -fsycl -std=c++17

Output only the complete SYCL code:"""

# Backup model for fallback
FALLBACK_MODEL = "Qwen3-Coder-30B-A3B-Instruct"
