#!/usr/bin/env python3
"""
LLM Accuracy Tester Prototype
LLM准确度测试器原型

快速原型验证：真正的kernel调用 + 准确度对比
"""

import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json

class PrototypeAccuracyTester:
    """原型准确度测试器"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.remote_host = "root@10.112.229.160"
    
    def test_copy_type_converted(self) -> Dict:
        """
        测试copy_type_converted kernel
        这是一个简单的类型转换kernel，适合作为原型验证
        """
        print("="*70)
        print("🧪 PROTOTYPE: Testing copy_type_converted kernel")
        print("="*70)
        
        # 测试配置
        test_size = 1024
        dtype = np.float32
        
        # 生成测试数据（固定种子确保可复现）
        np.random.seed(42)
        test_data = np.random.uniform(-1.0, 1.0, test_size).astype(dtype)
        
        print(f"\n📊 Test Configuration:")
        print(f"   Size: {test_size} elements")
        print(f"   Dtype: float32")
        print(f"   Range: [-1.0, 1.0]")
        print(f"   Seed: 42")
        
        # 步骤1: 运行CUDA测试
        print("\n🚀 Step 1: Running CUDA test...")
        cuda_output = self._run_cuda_test(test_data)
        
        if cuda_output is None:
            print("   ❌ CUDA test failed")
            return {"status": "FAILED", "error": "CUDA test failed"}
        
        print(f"   ✓ CUDA output shape: {cuda_output.shape}")
        print(f"   ✓ CUDA output sample: {cuda_output[:5]}")
        
        # 步骤2: 运行SYCL测试
        print("\n🚀 Step 2: Running SYCL test...")
        sycl_output = self._run_sycl_test(test_data)
        
        if sycl_output is None:
            print("   ❌ SYCL test failed")
            return {"status": "FAILED", "error": "SYCL test failed"}
        
        print(f"   ✓ SYCL output shape: {sycl_output.shape}")
        print(f"   ✓ SYCL output sample: {sycl_output[:5]}")
        
        # 步骤3: 对比结果
        print("\n📊 Step 3: Comparing results...")
        comparison = self._compare_results(cuda_output, sycl_output)
        
        print(f"   Max absolute error: {comparison['max_abs_error']:.2e}")
        print(f"   Max relative error: {comparison['max_rel_error']:.2e}")
        print(f"   Mean absolute error: {comparison['mean_abs_error']:.2e}")
        print(f"   Pass: {comparison['pass']}")
        
        if comparison['pass']:
            print("\n✅ TEST PASSED!")
        else:
            print("\n❌ TEST FAILED!")
            print(f"   Reason: {comparison.get('reason', 'Unknown')}")
        
        return {
            "status": "PASSED" if comparison['pass'] else "FAILED",
            "comparison": comparison,
            "cuda_sample": cuda_output[:5].tolist(),
            "sycl_sample": sycl_output[:5].tolist()
        }
    
    def _run_cuda_test(self, test_data: np.ndarray) -> np.ndarray:
        """运行CUDA测试 - 真正的kernel调用"""
        
        # 创建测试代码 - 真正调用kernel
        test_code = '''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <fstream>
#include <vector>
#include <math>

// Kernel definition
template <typename DstType, typename SrcType>
__global__ void copyTypeConverted_kernel(DstType* op, SrcType* ip, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    DstType el = (DstType)ip[tid];
    op[tid] = el;
}

int main() {
    const int N = 1024;
    
    // Allocate host memory
    std::vector<float> h_input(N);
    std::vector<float> h_output(N);
    
    // Read input from file
    std::ifstream in("/tmp/test_input.bin", std::ios::binary);
    in.read(reinterpret_cast<char*>(h_input.data()), N * sizeof(float));
    in.close();
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    copyTypeConverted_kernel<<<blocks, threads>>>(d_output, d_input, N);
    cudaDeviceSynchronize();
    
    // Copy back to host
    cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Write output to file
    std::ofstream out("/tmp/test_output.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(h_output.data()), N * sizeof(float));
    out.close();
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
'''
        
        try:
            # 保存输入数据
            test_data.tofile("/tmp/test_input.bin")
            
            # 写入测试代码
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(test_code)
                local_cu = f.name
            
            # 复制到远程主机，再复制到docker容器
            remote_cu = "/tmp/prototype_test.cu"
            subprocess.run(
                ['scp', local_cu, f'{self.remote_host}:{remote_cu}'],
                capture_output=True, timeout=30
            )
            
            # 复制到docker容器
            docker_cp = f'ssh {self.remote_host} "docker cp {remote_cu} cuda12.9-test:/workspace/test.cu"'
            subprocess.run(docker_cp, shell=True, capture_output=True, timeout=30)
            
            # 复制输入文件到容器
            docker_cp_input = f'ssh {self.remote_host} "docker cp /tmp/test_input.bin cuda12.9-test:/workspace/test_input.bin"'
            subprocess.run(docker_cp_input, shell=True, capture_output=True, timeout=30)
            
            # 编译并运行
            ssh_cmd = f'''
            ssh {self.remote_host} '
            docker exec cuda12.9-test bash -c "
            cd /workspace && 
            nvcc -O2 test.cu -o test && 
            ./test
            "
            '
            '''
            
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True,
                                  text=True, timeout=120)
            
            if result.returncode != 0:
                print(f"   CUDA execution error: {result.stderr[:200]}")
                return None
            
            # 复制输出文件回来
            docker_cp_output = f'ssh {self.remote_host} "docker cp cuda12.9-test:/workspace/test_output.bin /tmp/test_output_cuda.bin"'
            subprocess.run(docker_cp_output, shell=True, capture_output=True, timeout=30)
            
            scp_back = subprocess.run(
                ['scp', f'{self.remote_host}:/tmp/test_output_cuda.bin', '/tmp/test_output_cuda.bin'],
                capture_output=True, timeout=30
            )
            
            # 读取输出
            output = np.fromfile("/tmp/test_output_cuda.bin", dtype=np.float32)
            return output
            
        except Exception as e:
            print(f"   CUDA test error: {e}")
            return None
        finally:
            import os
            try:
                os.unlink(local_cu)
            except:
                pass
    
    def _run_sycl_test(self, test_data: np.ndarray) -> np.ndarray:
        """运行SYCL测试 - 真正的kernel调用"""
        
        # 创建测试代码 - 真正调用kernel
        test_code = '''
#include <sycl/sycl.hpp>
#include <fstream>
#include <vector>
#include <math>

// Kernel definition as functor
template <typename DstType, typename SrcType>
struct copyTypeConverted_kernel {
    DstType* op;
    SrcType* ip;
    int N;
    
    void operator()(sycl::id<1> idx) const {
        int tid = idx[0];
        if (tid >= N) return;
        DstType el = (DstType)ip[tid];
        op[tid] = el;
    }
};

int main() {
    const int N = 1024;
    
    // Read input from file
    std::vector<float> h_input(N);
    std::ifstream in("/workspace/test_input.bin", std::ios::binary);
    in.read(reinterpret_cast<char*>(h_input.data()), N * sizeof(float));
    in.close();
    
    // Create queue
    sycl::queue q;
    
    // Allocate device memory
    float* d_input = sycl::malloc_device<float>(N, q);
    float* d_output = sycl::malloc_device<float>(N, q);
    
    // Copy to device
    q.memcpy(d_input, h_input.data(), N * sizeof(float)).wait();
    
    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    q.parallel_for(
        sycl::range<1>(blocks * threads),
        copyTypeConverted_kernel<float, float>{d_output, d_input, N}
    ).wait();
    
    // Copy back to host
    std::vector<float> h_output(N);
    q.memcpy(h_output.data(), d_output, N * sizeof(float)).wait();
    
    // Write output to file
    std::ofstream out("/workspace/test_output.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(h_output.data()), N * sizeof(float));
    out.close();
    
    // Cleanup
    sycl::free(d_input, q);
    sycl::free(d_output, q);
    
    return 0;
}
'''
        
        try:
            # 保存输入数据到docker容器
            test_data.tofile("/tmp/test_input_sycl.bin")
            subprocess.run(
                ['docker', 'cp', '/tmp/test_input_sycl.bin', 'lsv-container:/workspace/test_input.bin'],
                capture_output=True, timeout=30
            )
            
            # 写入测试代码
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(test_code)
                local_cpp = f.name
            
            # 复制到docker容器
            subprocess.run(
                ['docker', 'cp', local_cpp, 'lsv-container:/workspace/test.cpp'],
                capture_output=True, timeout=30
            )
            
            # 编译
            compile_cmd = [
                'docker', 'exec', 'lsv-container', 'bash', '-c',
                'cd /workspace && icpx -fsycl -O2 test.cpp -o test'
            ]
            result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                print(f"   SYCL compilation error: {result.stderr[:200]}")
                return None
            
            # 运行
            run_cmd = ['docker', 'exec', 'lsv-container', '/workspace/test']
            result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"   SYCL execution error: {result.stderr[:200]}")
                return None
            
            # 复制输出文件回来
            subprocess.run(
                ['docker', 'cp', 'lsv-container:/workspace/test_output.bin', '/tmp/test_output_sycl.bin'],
                capture_output=True, timeout=30
            )
            
            # 读取输出
            output = np.fromfile("/tmp/test_output_sycl.bin", dtype=np.float32)
            return output
            
        except Exception as e:
            print(f"   SYCL test error: {e}")
            return None
        finally:
            import os
            try:
                os.unlink(local_cpp)
            except:
                pass
    
    def _compare_results(self, cuda_output: np.ndarray, sycl_output: np.ndarray) -> Dict:
        """对比CUDA和SYCL输出结果"""
        
        if len(cuda_output) != len(sycl_output):
            return {
                "pass": False,
                "reason": f"Size mismatch: CUDA={len(cuda_output)}, SYCL={len(sycl_output)}"
            }
        
        # 计算误差
        abs_error = np.abs(cuda_output - sycl_output)
        rel_error = abs_error / (np.abs(cuda_output) + 1e-10)
        
        max_abs_error = float(np.max(abs_error))
        max_rel_error = float(np.max(rel_error))
        mean_abs_error = float(np.mean(abs_error))
        
        # 容差（对于float32，期望完全一致或极小误差）
        abs_tolerance = 1e-5
        rel_tolerance = 1e-4
        
        # 判断是否通过
        violations = np.sum((abs_error > abs_tolerance) & (rel_error > rel_tolerance))
        violation_rate = violations / len(cuda_output)
        
        passed = violation_rate < 0.001  # 0.1%容忍
        
        return {
            "pass": passed,
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "mean_abs_error": mean_abs_error,
            "violations": int(violations),
            "violation_rate": float(violation_rate),
            "reason": f"{violations} elements exceed tolerance" if not passed else None
        }


if __name__ == "__main__":
    tester = PrototypeAccuracyTester()
    result = tester.test_copy_type_converted()
    
    print("\n" + "="*70)
    print("FINAL RESULT:")
    print("="*70)
    print(json.dumps(result, indent=2))
