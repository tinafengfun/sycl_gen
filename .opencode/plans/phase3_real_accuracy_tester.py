#!/usr/bin/env python3
"""
RealAccuracyTester - Phase 3 Implementation
整合所有修复后的harness，提供统一的准确度测试接口
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import tempfile
import os

# Import all fixed harnesses from Phase 1
import sys
sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/.opencode/plans')

from phase1_fixed_harnesses import FIXED_HARNESSES as PHASE1_HARNESSES

# Include Phase 2 harnesses directly
PHASE2_IMPROVED_HARNESSES = {
    'add_bias_batched': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cstdio.h>
__global__ void addBiasBatchedKernel(float* output, const float* input, 
                                      const float* bias, int Batch, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Batch * N * C;
    if (idx < total) {
        int nc = idx / C;
        int c = nc % C;
        output[idx] = input[idx] + bias[c];
    }
}
int main() {
    const int Batch = 2, N = 4, C = 32;
    const int size = Batch * N * C;
    float* h_input = new float[size];
    float* h_bias = new float[C];
    float* h_output = new float[size];
    for (int i = 0; i < size; i++) h_input[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < C; i++) h_bias[i] = (float)i / 100.0f;
    float *d_input, *d_bias, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_bias, C * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, C * sizeof(float), cudaMemcpyHostToDevice);
    addBiasBatchedKernel<<<(size + 255) / 256, 256>>>(d_output, d_input, d_bias, Batch, N, C);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output, sizeof(float), size, f);
    fclose(f);
    cudaFree(d_input); cudaFree(d_bias); cudaFree(d_output);
    delete[] h_input; delete[] h_bias; delete[] h_output;
    return 0;
}
''',
        'sycl': '''
#include <sycl/sycl.hpp>
#include <fstream>
int main() {
    sycl::queue q(sycl::gpu_selector_v);
    const int Batch = 2, N = 4, C = 32;
    const int size = Batch * N * C;
    float* h_input = new float[size];
    float* h_bias = new float[C];
    float* h_output = new float[size];
    for (int i = 0; i < size; i++) h_input[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < C; i++) h_bias[i] = (float)i / 100.0f;
    float* d_input = sycl::malloc_device<float>(size, q);
    float* d_bias = sycl::malloc_device<float>(C, q);
    float* d_output = sycl::malloc_device<float>(size, q);
    q.memcpy(d_input, h_input, size * sizeof(float)).wait();
    q.memcpy(d_bias, h_bias, C * sizeof(float)).wait();
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
        int i = idx[0];
        int nc = i / C;
        int c = nc % C;
        d_output[i] = d_input[i] + d_bias[c];
    }).wait();
    q.memcpy(h_output, d_output, size * sizeof(float)).wait();
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), size * sizeof(float));
    f.close();
    sycl::free(d_input, q); sycl::free(d_bias, q); sycl::free(d_output, q);
    delete[] h_input; delete[] h_bias; delete[] h_output;
    return 0;
}
'''
    },
    'global_scale': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio.h>
#include <cmath>
__global__ void globalScaleKernel(float* output, const float* input,
                                   const float* scaleBias, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int planeSize = 64;
    int total = N * C * planeSize;
    if (idx < total) {
        int nc = idx / planeSize;
        int n = nc / C;
        int c = nc % C;
        float val = input[idx];
        float s = scaleBias[n * 2 * C + c];
        s = 1.0f / (1.0f + expf(-s));
        output[idx] = val * s;
    }
}
int main() {
    const int N = 2, C = 32, planeSize = 64;
    const int inputSize = N * C * planeSize;
    const int scaleBiasSize = N * 2 * C;
    float* h_input = new float[inputSize];
    float* h_scaleBias = new float[scaleBiasSize];
    float* h_output = new float[inputSize];
    for (int i = 0; i < inputSize; i++) h_input[i] = sinf(i * 0.01f) * 0.5f;
    for (int i = 0; i < scaleBiasSize; i++) h_scaleBias[i] = cosf(i * 0.03f) * 0.2f;
    float *d_input, *d_scaleBias, *d_output;
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_scaleBias, scaleBiasSize * sizeof(float));
    cudaMalloc(&d_output, inputSize * sizeof(float));
    cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scaleBias, h_scaleBias, scaleBiasSize * sizeof(float), cudaMemcpyHostToDevice);
    globalScaleKernel<<<(inputSize + 255) / 256, 256>>>(d_output, d_input, d_scaleBias, N, C);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, inputSize * sizeof(float), cudaMemcpyDeviceToHost);
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output, sizeof(float), inputSize, f);
    fclose(f);
    cudaFree(d_input); cudaFree(d_scaleBias); cudaFree(d_output);
    delete[] h_input; delete[] h_scaleBias; delete[] h_output;
    return 0;
}
''',
        'sycl': '''
#include <sycl/sycl.hpp>
#include <cmath>
#include <fstream>
int main() {
    sycl::queue q(sycl::gpu_selector_v);
    const int N = 2, C = 32, planeSize = 64;
    const int inputSize = N * C * planeSize;
    const int scaleBiasSize = N * 2 * C;
    float* h_input = new float[inputSize];
    float* h_scaleBias = new float[scaleBiasSize];
    float* h_output = new float[inputSize];
    for (int i = 0; i < inputSize; i++) h_input[i] = sycl::sin(i * 0.01f) * 0.5f;
    for (int i = 0; i < scaleBiasSize; i++) h_scaleBias[i] = sycl::cos(i * 0.03f) * 0.2f;
    float* d_input = sycl::malloc_device<float>(inputSize, q);
    float* d_scaleBias = sycl::malloc_device<float>(scaleBiasSize, q);
    float* d_output = sycl::malloc_device<float>(inputSize, q);
    q.memcpy(d_input, h_input, inputSize * sizeof(float)).wait();
    q.memcpy(d_scaleBias, h_scaleBias, scaleBiasSize * sizeof(float)).wait();
    q.parallel_for(sycl::range<1>(inputSize), [=](sycl::id<1> idx) {
        int i = idx[0];
        int planeSize = 64;
        int nc = i / planeSize;
        int n = nc / C;
        int c = nc % C;
        float val = d_input[i];
        float s = d_scaleBias[n * 2 * C + c];
        s = 1.0f / (1.0f + sycl::exp(-s));
        d_output[i] = val * s;
    }).wait();
    q.memcpy(h_output, d_output, inputSize * sizeof(float)).wait();
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), inputSize * sizeof(float));
    f.close();
    sycl::free(d_input, q); sycl::free(d_scaleBias, q); sycl::free(d_output, q);
    delete[] h_input; delete[] h_scaleBias; delete[] h_output;
    return 0;
}
'''
    }
}

# Merge all harnesses
ALL_FIXED_HARNESSES = {**PHASE1_HARNESSES, **PHASE2_IMPROVED_HARNESSES}


class RealAccuracyTester:
    """
    真实准确度测试器 - 使用修复后的正确harness
    
    改进点：
    1. 使用真实内核逻辑（非placeholder）
    2. 整合Phase 1和Phase 2的所有修复
    3. 提供统一的测试接口
    4. 完整的错误处理和日志
    """
    
    def __init__(self, cuda_host: str = "10.112.229.160",
                 cuda_container: str = "cuda12.9-test",
                 sycl_container: str = "lsv-container"):
        self.cuda_host = cuda_host
        self.cuda_container = cuda_container
        self.sycl_container = sycl_container
        self.harness_db = ALL_FIXED_HARNESSES
        
        # Statistics
        self.stats = {
            'tested': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0
        }
    
    def list_available_harnesses(self) -> List[str]:
        """列出所有可用的harness"""
        return list(self.harness_db.keys())
    
    def has_harness(self, kernel_id: str) -> bool:
        """检查是否有该内核的harness"""
        return kernel_id in self.harness_db
    
    def get_harness(self, kernel_id: str, platform: str) -> Optional[str]:
        """获取指定内核的harness代码"""
        if kernel_id not in self.harness_db:
            return None
        return self.harness_db[kernel_id].get(platform)
    
    def test_accuracy(self, kernel_id: str) -> Tuple[bool, float, float, Dict]:
        """
        测试内核准确度
        
        Args:
            kernel_id: 内核ID
            
        Returns:
            (passed, mae, max_error, details)
        """
        print(f"\n🧪 测试: {kernel_id}")
        print("-" * 50)
        
        self.stats['tested'] += 1
        
        # Check if harness exists
        if not self.has_harness(kernel_id):
            print(f"  ⚠️  无harness，跳过")
            self.stats['skipped'] += 1
            return False, 0.0, 0.0, {'status': 'skipped', 'reason': 'no_harness'}
        
        # Get harness codes
        cuda_code = self.get_harness(kernel_id, 'cuda')
        sycl_code = self.get_harness(kernel_id, 'sycl')
        
        if not cuda_code or not sycl_code:
            print(f"  ⚠️  harness不完整")
            self.stats['skipped'] += 1
            return False, 0.0, 0.0, {'status': 'skipped', 'reason': 'incomplete_harness'}
        
        try:
            # Run CUDA
            print("  🔨 CUDA...", end=' ')
            cuda_success = self._run_cuda(cuda_code)
            if not cuda_success:
                print("❌")
                self.stats['failed'] += 1
                return False, 0.0, 0.0, {'status': 'failed', 'stage': 'cuda'}
            print("✅")
            
            # Run SYCL
            print("  🔨 SYCL...", end=' ')
            sycl_success = self._run_sycl(sycl_code)
            if not sycl_success:
                print("❌")
                self.stats['failed'] += 1
                return False, 0.0, 0.0, {'status': 'failed', 'stage': 'sycl'}
            print("✅")
            
            # Compare results
            print("  📊 比较...", end=' ')
            mae, max_error = self._compare_outputs()
            
            # Determine pass/fail
            passed = (mae < 1e-5) and (max_error < 1e-4)
            
            status_icon = "✅" if passed else "⚠️"
            print(f"{status_icon} MAE={mae:.2e}, MaxErr={max_error:.2e}")
            
            if passed:
                self.stats['passed'] += 1
            else:
                self.stats['failed'] += 1
            
            return passed, mae, max_error, {
                'status': 'passed' if passed else 'failed',
                'mae': mae,
                'max_error': max_error
            }
            
        except Exception as e:
            print(f"❌ 异常: {e}")
            self.stats['failed'] += 1
            return False, 0.0, 0.0, {'status': 'error', 'error': str(e)}
    
    def _run_cuda(self, code: str) -> bool:
        """运行CUDA测试"""
        try:
            # Save code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(code)
                cuda_file = f.name
            
            # Copy to remote
            subprocess.run(['scp', cuda_file, f'root@{self.cuda_host}:/tmp/test.cu'],
                         capture_output=True, timeout=30, check=True)
            
            # Compile and run
            cmd = f'''ssh root@{self.cuda_host} "docker cp /tmp/test.cu {self.cuda_container}:/workspace/test.cu && 
                     docker exec {self.cuda_container} bash -c 'cd /workspace && 
                     nvcc -O2 -Wno-deprecated-gpu-targets test.cu -o test && ./test'"'''
            
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
            
            os.unlink(cuda_file)
            return result.returncode == 0
            
        except Exception as e:
            print(f"[CUDA Error: {e}]")
            return False
    
    def _run_sycl(self, code: str) -> bool:
        """运行SYCL测试"""
        try:
            # Save code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(code)
                sycl_file = f.name
            
            # Copy to container
            subprocess.run(['docker', 'cp', sycl_file, f'{self.sycl_container}:/workspace/test.cpp'],
                         capture_output=True, timeout=10, check=True)
            
            # Compile and run
            result = subprocess.run(['docker', 'exec', self.sycl_container, 'bash', '-c',
                                   'cd /workspace && icpx -fsycl -O2 test.cpp -o test && ./test'],
                                  capture_output=True, timeout=120)
            
            os.unlink(sycl_file)
            return result.returncode == 0
            
        except Exception as e:
            print(f"[SYCL Error: {e}]")
            return False
    
    def _compare_outputs(self) -> Tuple[float, float]:
        """比较CUDA和SYCL输出"""
        try:
            # Copy outputs
            subprocess.run(['ssh', f'root@{self.cuda_host}',
                          f'docker cp {self.cuda_container}:/workspace/output_cuda.bin /tmp/'],
                         capture_output=True, check=True)
            subprocess.run(['scp', f'root@{self.cuda_host}:/tmp/output_cuda.bin', '/tmp/'],
                         capture_output=True, check=True)
            subprocess.run(['docker', 'cp', f'{self.sycl_container}:/workspace/output_sycl.bin', '/tmp/'],
                         capture_output=True, check=True)
            
            # Read and compare
            cuda_out = np.fromfile('/tmp/output_cuda.bin', dtype=np.float32)
            sycl_out = np.fromfile('/tmp/output_sycl.bin', dtype=np.float32)
            
            if len(cuda_out) != len(sycl_out):
                return 1.0, 1.0
            
            diff = np.abs(cuda_out - sycl_out)
            mae = float(np.mean(diff))
            max_err = float(np.max(diff))
            
            return mae, max_err
            
        except Exception as e:
            print(f"[Compare Error: {e}]")
            return 1.0, 1.0
    
    def batch_test(self, kernel_ids: List[str]) -> Dict:
        """批量测试多个内核"""
        print("=" * 80)
        print("🚀 RealAccuracyTester - 批量测试")
        print("=" * 80)
        print(f"可用harness: {len(self.list_available_harnesses())}")
        print(f"测试内核: {len(kernel_ids)}")
        print()
        
        results = []
        for kernel_id in kernel_ids:
            passed, mae, max_err, details = self.test_accuracy(kernel_id)
            results.append({
                'kernel_id': kernel_id,
                'passed': passed,
                'mae': mae,
                'max_error': max_err,
                'details': details
            })
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 测试摘要")
        print("=" * 80)
        print(f"总测试: {self.stats['tested']}")
        print(f"✅ 通过: {self.stats['passed']}")
        print(f"❌ 失败: {self.stats['failed']}")
        print(f"⏭️  跳过: {self.stats['skipped']}")
        print(f"📈 通过率: {self.stats['passed']/max(self.stats['tested'],1)*100:.1f}%")
        print("=" * 80)
        
        return {
            'results': results,
            'stats': self.stats
        }
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()


# Test function
def test_real_accuracy_tester():
    """测试RealAccuracyTester"""
    print("\n" + "=" * 80)
    print("🔍 Testing RealAccuracyTester")
    print("=" * 80)
    
    tester = RealAccuracyTester()
    
    # List available harnesses
    print(f"\n📋 可用harnesses ({len(tester.list_available_harnesses())}):")
    for i, kid in enumerate(tester.list_available_harnesses(), 1):
        print(f"  {i}. {kid}")
    
    # Test a few kernels
    test_kernels = ['add_vectors', 'copy_type_converted', 'global_avg_pool']
    print(f"\n🧪 测试 {len(test_kernels)} 个内核...")
    
    results = tester.batch_test(test_kernels)
    
    return results


if __name__ == '__main__':
    results = test_real_accuracy_tester()
    
    print("\n✅ RealAccuracyTester implementation complete!")
    print(f"   可用harnesses: {len(ALL_FIXED_HARNESSES)}")
    print("   Ready for integration with Agent v4.1")
