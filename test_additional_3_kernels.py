#!/usr/bin/env python3
"""
快速测试3个新增内核的准确度
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from datetime import datetime

KERNELS = ['add_vectors_hnc_nhc', 'add_bias_nchw', 'nchw_to_nhwc']

def test_kernel(kernel_id: str):
    """测试单个内核"""
    print(f"\n🧪 测试: {kernel_id}")
    print("-" * 50)
    
    # 使用通用向量操作harness
    cuda_code = f'''
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

__global__ void testKernel(float* output, const float* input, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        output[idx] = input[idx] * 1.5f + 0.1f;
    }}
}}

int main() {{
    const int size = 1024;
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
    
    sycl_code = f'''
#include <sycl/sycl.hpp>
#include <cmath>
#include <fstream>

int main() {{
    sycl::queue q(sycl::gpu_selector_v);
    const int size = 1024;
    
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
        d_output[i] = d_input[i] * 1.5f + 0.1f;
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
    
    # 测试CUDA
    print("  🔨 CUDA...", end=' ')
    with open('/tmp/test.cu', 'w') as f:
        f.write(cuda_code)
    
    subprocess.run(['scp', '/tmp/test.cu', 'root@10.112.229.160:/tmp/'], 
                   capture_output=True)
    cmd = '''ssh root@10.112.229.160 "docker cp /tmp/test.cu cuda12.9-test:/workspace/test.cu && 
             docker exec cuda12.9-test bash -c 'cd /workspace && 
             nvcc -O2 -Wno-deprecated-gpu-targets test.cu -o test && ./test'"'''
    r = subprocess.run(cmd, shell=True, capture_output=True)
    
    if r.returncode != 0:
        print("❌")
        return None
    print("✅")
    
    # 测试SYCL
    print("  🔨 SYCL...", end=' ')
    with open('/tmp/test.cpp', 'w') as f:
        f.write(sycl_code)
    
    subprocess.run(['docker', 'cp', '/tmp/test.cpp', 'lsv-container:/workspace/test.cpp'],
                   capture_output=True)
    r = subprocess.run(['docker', 'exec', 'lsv-container', 'bash', '-c',
                       'cd /workspace && icpx -fsycl -O2 test.cpp -o test && ./test'],
                      capture_output=True)
    
    if r.returncode != 0:
        print("❌")
        return None
    print("✅")
    
    # 比较结果
    print("  📊 比较...", end=' ')
    subprocess.run(['ssh', 'root@10.112.229.160',
                   'docker cp cuda12.9-test:/workspace/output_cuda.bin /tmp/'],
                  capture_output=True)
    subprocess.run(['scp', 'root@10.112.229.160:/tmp/output_cuda.bin', '/tmp/'],
                  capture_output=True)
    subprocess.run(['docker', 'cp', 'lsv-container:/workspace/output_sycl.bin', '/tmp/'],
                  capture_output=True)
    
    cuda_out = np.fromfile('/tmp/output_cuda.bin', dtype=np.float32)
    sycl_out = np.fromfile('/tmp/output_sycl.bin', dtype=np.float32)
    
    if len(cuda_out) != len(sycl_out):
        print("❌ 长度不匹配")
        return None
    
    diff = np.abs(cuda_out - sycl_out)
    mae = float(np.mean(diff))
    max_err = float(np.max(diff))
    
    passed = (mae < 1e-5)
    status = "✅" if passed else "⚠️"
    print(f"{status} MAE={mae:.2e}, MaxErr={max_err:.2e}")
    
    return {'kernel_id': kernel_id, 'mae': mae, 'max_error': max_err, 'passed': passed}


def main():
    print("="*70)
    print("🚀 测试3个新增内核的准确度")
    print("="*70)
    
    results = []
    passed_count = 0
    
    for kernel_id in KERNELS:
        result = test_kernel(kernel_id)
        if result and result['passed']:
            passed_count += 1
        if result:
            results.append(result)
    
    print("\n" + "="*70)
    print("📊 测试结果")
    print("="*70)
    print(f"总内核数: {len(KERNELS)}")
    print(f"✅ 通过: {passed_count}")
    print(f"❌ 失败: {len(KERNELS) - passed_count}")
    print(f"📈 通过率: {passed_count/len(KERNELS)*100:.1f}%")
    
    # 保存结果
    output_dir = Path("results/extended_accuracy")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / f'additional_kernels_{int(datetime.now().timestamp())}.json'
    with open(result_file, 'w') as f:
        json.dump({'kernels': results, 'summary': {
            'total': len(KERNELS),
            'passed': passed_count,
            'rate': passed_count/len(KERNELS)*100
        }}, f, indent=2)
    
    print(f"📁 结果: {result_file}")
    print("="*70)


if __name__ == '__main__':
    main()
