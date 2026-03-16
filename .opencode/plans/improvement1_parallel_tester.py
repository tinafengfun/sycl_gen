#!/usr/bin/env python3
"""
ParallelRealAccuracyTester - 改进版：支持并行测试
基于反思：串行测试效率低，需要并行化
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import tempfile
import os
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/.opencode/plans')
from phase1_fixed_harnesses import FIXED_HARNESSES as PHASE1_HARNESSES

# Include Phase 2 harnesses
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

ALL_FIXED_HARNESSES = {**PHASE1_HARNESSES, **PHASE2_IMPROVED_HARNESSES}


class ParallelRealAccuracyTester:
    """
    改进版准确度测试器 - 支持并行测试
    
    改进点（基于反思）：
    1. ✅ 并行化：使用ThreadPoolExecutor并行测试多个内核
    2. ✅ 时间估算：记录每个测试耗时，预测总时间
    3. ✅ 进度跟踪：实时显示测试进度
    4. ✅ 性能对比：串行 vs 并行效率对比
    """
    
    def __init__(self, cuda_host: str = "10.112.229.160",
                 cuda_container: str = "cuda12.9-test",
                 sycl_container: str = "lsv-container",
                 max_workers: int = 3):
        self.cuda_host = cuda_host
        self.cuda_container = cuda_container
        self.sycl_container = sycl_container
        self.harness_db = ALL_FIXED_HARNESSES
        self.max_workers = max_workers
        
        # Statistics with timing
        self.stats = {
            'tested': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'total_time': 0,
            'avg_time_per_kernel': 0
        }
        
        # Timing data
        self.timing_data = {}
    
    def estimate_total_time(self, kernel_ids: List[str]) -> float:
        """估算总测试时间"""
        # Based on empirical data: ~30s per kernel
        estimated_per_kernel = 30
        parallel_factor = min(len(kernel_ids), self.max_workers)
        estimated_total = (len(kernel_ids) / parallel_factor) * estimated_per_kernel
        return estimated_total
    
    def batch_test_parallel(self, kernel_ids: List[str]) -> Dict:
        """
        并行批量测试多个内核
        
        改进：使用ThreadPoolExecutor并行执行
        """
        print("=" * 80)
        print("🚀 ParallelRealAccuracyTester - 并行批量测试")
        print("=" * 80)
        print(f"可用harness: {len(self.list_available_harnesses())}")
        print(f"测试内核: {len(kernel_ids)}")
        print(f"并行度: {self.max_workers}")
        
        # Estimate time
        estimated_time = self.estimate_total_time(kernel_ids)
        print(f"预计时间: ~{estimated_time:.0f}秒")
        print()
        
        start_time = time.time()
        results = []
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_kernel = {
                executor.submit(self.test_accuracy_single, kid): kid 
                for kid in kernel_ids
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_kernel):
                kernel_id = future_to_kernel[future]
                try:
                    passed, mae, max_err, details = future.result()
                    results.append({
                        'kernel_id': kernel_id,
                        'passed': passed,
                        'mae': mae,
                        'max_error': max_err,
                        'details': details
                    })
                    completed += 1
                    
                    # Progress update
                    progress = (completed / len(kernel_ids)) * 100
                    print(f"\r📊 进度: {completed}/{len(kernel_ids)} ({progress:.1f}%)", end='', flush=True)
                    
                except Exception as e:
                    print(f"\n❌ {kernel_id} 测试失败: {e}")
                    results.append({
                        'kernel_id': kernel_id,
                        'passed': False,
                        'error': str(e)
                    })
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n\n" + "=" * 80)
        print("📊 测试摘要")
        print("=" * 80)
        print(f"总测试: {len(kernel_ids)}")
        print(f"✅ 通过: {sum(1 for r in results if r.get('passed'))}")
        print(f"❌ 失败: {sum(1 for r in results if not r.get('passed'))}")
        print(f"⏱️  总耗时: {total_time:.1f}秒")
        print(f"⚡ 并行效率: {len(kernel_ids) * 30 / total_time:.1f}x (vs 串行)")
        print("=" * 80)
        
        return {
            'results': results,
            'stats': self.stats,
            'timing': {
                'total': total_time,
                'estimated': estimated_time,
                'avg_per_kernel': total_time / len(kernel_ids)
            }
        }
    
    def test_accuracy_single(self, kernel_id: str) -> Tuple[bool, float, float, Dict]:
        """测试单个内核（用于并行执行）"""
        start = time.time()
        
        if not self.has_harness(kernel_id):
            self.stats['skipped'] += 1
            return False, 0.0, 0.0, {'status': 'skipped'}
        
        cuda_code = self.get_harness(kernel_id, 'cuda')
        sycl_code = self.get_harness(kernel_id, 'sycl')
        
        if not cuda_code or not sycl_code:
            self.stats['skipped'] += 1
            return False, 0.0, 0.0, {'status': 'incomplete'}
        
        try:
            cuda_success = self._run_cuda(cuda_code)
            if not cuda_success:
                self.stats['failed'] += 1
                return False, 0.0, 0.0, {'status': 'cuda_failed'}
            
            sycl_success = self._run_sycl(sycl_code)
            if not sycl_success:
                self.stats['failed'] += 1
                return False, 0.0, 0.0, {'status': 'sycl_failed'}
            
            mae, max_error = self._compare_outputs()
            passed = (mae < 1e-5) and (max_error < 1e-4)
            
            elapsed = time.time() - start
            self.timing_data[kernel_id] = elapsed
            
            if passed:
                self.stats['passed'] += 1
            else:
                self.stats['failed'] += 1
            
            self.stats['tested'] += 1
            
            return passed, mae, max_error, {
                'status': 'passed' if passed else 'failed',
                'time': elapsed
            }
            
        except Exception as e:
            self.stats['failed'] += 1
            return False, 0.0, 0.0, {'status': 'error', 'error': str(e)}
    
    def has_harness(self, kernel_id: str) -> bool:
        return kernel_id in self.harness_db
    
    def get_harness(self, kernel_id: str, platform: str) -> Optional[str]:
        if kernel_id not in self.harness_db:
            return None
        return self.harness_db[kernel_id].get(platform)
    
    def list_available_harnesses(self) -> List[str]:
        return list(self.harness_db.keys())
    
    def _run_cuda(self, code: str) -> bool:
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
        except:
            return False
    
    def _run_sycl(self, code: str) -> bool:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(code)
                sycl_file = f.name
            
            subprocess.run(['docker', 'cp', sycl_file, f'{self.sycl_container}:/workspace/test.cpp'],
                         capture_output=True, timeout=10, check=True)
            
            result = subprocess.run(['docker', 'exec', self.sycl_container, 'bash', '-c',
                                   'cd /workspace && icpx -fsycl -O2 test.cpp -o test && ./test'],
                                  capture_output=True, timeout=120)
            os.unlink(sycl_file)
            return result.returncode == 0
        except:
            return False
    
    def _compare_outputs(self) -> Tuple[float, float]:
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
            return float(np.mean(diff)), float(np.max(diff))
        except:
            return 1.0, 1.0


if __name__ == '__main__':
    tester = ParallelRealAccuracyTester(max_workers=3)
    
    # Test available kernels
    available = tester.list_available_harnesses()
    print(f"可用harnesses: {len(available)}")
    for kid in available:
        print(f"  - {kid}")
    
    print("\n✅ ParallelRealAccuracyTester 改进完成！")
    print("改进点：")
    print("  1. ✅ 并行测试支持（多线程）")
    print("  2. ✅ 时间估算功能")
    print("  3. ✅ 实时进度显示")
    print("  4. ✅ 性能对比统计")
