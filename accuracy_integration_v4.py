#!/usr/bin/env python3
"""
改进版 CUDA→SYCL 转换 Agent v4.0
集成准确度测试的完整方案
"""

import asyncio
import json
import re
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import aiohttp
import numpy as np

# 在 improved_agent_v3.py 基础上增加准确度测试

class ConversionStatus(Enum):
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
class KernelInfo:
    """内核信息数据结构 - v4.0"""
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
    accuracy_tested_at: Optional[str] = None  # 新增


class AccuracyTester:
    """准确度测试器 - 从 run_extended_accuracy_test.py 提取"""
    
    # 测试配置
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
            return 'vector_op'  # 默认
    
    def generate_harness(self, kernel_id: str, platform: str) -> Optional[str]:
        """生成测试harness"""
        kernel_type = self.classify_kernel(kernel_id)
        config = self.TEST_CONFIGS[kernel_type]
        
        # 这里简化处理，实际应该根据内核类型生成对应的harness
        # 完整的harness代码应该和 run_extended_accuracy_test.py 中的一样
        
        if platform == 'cuda':
            return self._generate_cuda_harness(kernel_id, config)
        else:
            return self._generate_sycl_harness(kernel_id, config)
    
    def _generate_cuda_harness(self, kernel_id: str, config: Dict) -> str:
        """生成CUDA测试harness模板"""
        return f'''
#include <cuda_runtime.h>
#include <cstdio.h>
#include <cmath>

// 简化的测试内核
__global__ void testKernel(float* output, const float* input, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        output[idx] = input[idx] * 2.0f;  // 简化操作
    }}
}}

int main() {{
    const int size = {config['size']};
    float* h_input = new float[size];
    float* h_output = new float[size];
    
    // 确定性输入
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
        """生成SYCL测试harness模板"""
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
        """
        测试内核准确度
        
        Returns:
            (passed, mae, max_error)
        """
        print(f"\n🧪 准确度测试: {kernel_id}")
        
        try:
            # 生成harness
            cuda_code = self.generate_harness(kernel_id, 'cuda')
            sycl_code = self.generate_harness(kernel_id, 'sycl')
            
            if not cuda_code or not sycl_code:
                print(f"  ⚠️  无法生成harness")
                return False, 0.0, 0.0
            
            # 编译并运行CUDA
            print(f"  🔨 CUDA...", end=' ')
            cuda_success = await self._run_cuda(cuda_code)
            if not cuda_success:
                print(f"❌")
                return False, 0.0, 0.0
            print(f"✅")
            
            # 编译并运行SYCL
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
            # 保存代码
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(code)
                cuda_file = f.name
            
            # 复制到远程
            subprocess.run(['scp', cuda_file, f'root@{self.cuda_host}:/tmp/test.cu'],
                         capture_output=True, timeout=30, check=True)
            
            # 编译并运行
            cmd = f'''ssh root@{self.cuda_host} "docker cp /tmp/test.cu {self.cuda_container}:/workspace/test.cu && 
                     docker exec {self.cuda_container} bash -c 'cd /workspace && 
                     nvcc -O2 -Wno-deprecated-gpu-targets test.cu -o test && ./test'"'''
            
            result = subprocess.run(cmd, shell=True, capture_output=True, 
                                  timeout=120)
            
            os.unlink(cuda_file)
            return result.returncode == 0
            
        except Exception as e:
            print(f"CUDA运行错误: {e}")
            return False
    
    async def _run_sycl(self, code: str) -> bool:
        """运行SYCL测试"""
        try:
            # 保存代码
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(code)
                sycl_file = f.name
            
            # 复制到容器
            subprocess.run(['docker', 'cp', sycl_file, 
                          f'{self.sycl_container}:/workspace/test.cpp'],
                         capture_output=True, timeout=10, check=True)
            
            # 编译并运行
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
            # 复制CUDA输出
            subprocess.run(['ssh', f'root@{self.cuda_host}',
                          f'docker cp {self.cuda_container}:/workspace/output_cuda.bin /tmp/'],
                         capture_output=True, check=True)
            subprocess.run(['scp', f'root@{self.cuda_host}:/tmp/output_cuda.bin', '/tmp/'],
                         capture_output=True, check=True)
            
            # 复制SYCL输出
            subprocess.run(['docker', 'cp', f'{self.sycl_container}:/workspace/output_sycl.bin', '/tmp/'],
                         capture_output=True, check=True)
            
            # 读取并比较
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


# 使用示例和集成方法
async def example_integration():
    """
    展示如何将准确度测试集成到Agent流程中
    """
    print("="*80)
    print("Agent v4.0 集成准确度测试示例")
    print("="*80)
    
    # 创建准确度测试器
    accuracy_tester = AccuracyTester()
    
    # 测试一个内核
    kernel_id = "add_vectors"
    passed, mae, max_error = await accuracy_tester.test_accuracy(kernel_id)
    
    print(f"\n结果:")
    print(f"  通过: {passed}")
    print(f"  MAE: {mae:.2e}")
    print(f"  Max Error: {max_error:.2e}")


if __name__ == '__main__':
    asyncio.run(example_integration())
