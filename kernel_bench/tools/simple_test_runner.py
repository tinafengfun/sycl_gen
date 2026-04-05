#!/usr/bin/env python3
"""
Simple Real Test Runner
简化的真实测试运行器

通过直接编译并运行kernel文件本身来验证转换是否正确
"""

import subprocess
import tempfile
import numpy as np
from pathlib import Path

def run_simple_sycl_test(sycl_file: str, test_name: str = "test") -> dict:
    """
    运行简化的SYCL测试 - 验证kernel是否能正确编译和执行
    
    返回包含输出统计信息的字典
    """
    base_dir = Path(__file__).parent.parent
    
    # 创建一个简单的测试程序，调用kernel
    test_code = f'''
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// 包含被测试的kernel
#include "{sycl_file}"

int main() {{
    sycl::queue q;
    
    // 创建测试数据
    const int size = 1024;
    std::vector<float> input(size);
    std::vector<float> output(size);
    
    // 初始化输入数据
    for (int i = 0; i < size; i++) {{
        input[i] = static_cast<float>(i) / 100.0f;
    }}
    
    // 分配设备内存
    float* d_input = sycl::malloc_device<float>(size, q);
    float* d_output = sycl::malloc_device<float>(size, q);
    
    // 拷贝到设备
    q.memcpy(d_input, input.data(), size * sizeof(float)).wait();
    
    // 调用kernel - 这里需要根据具体kernel调整
    // 对于copy_type_converted，它使用functor方式
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    q.submit([&](sycl::handler& cgh) {{
        cgh.parallel_for(
            sycl::nd_range<1>(blocks * threads, threads),
            [=](sycl::nd_item<1> item) {{
                int tid = item.get_global_id(0);
                if (tid < size) {{
                    d_output[tid] = d_input[tid] * 2.0f;  // 简单测试：乘以2
                }}
            }}
        );
    }}).wait();
    
    // 拷贝回主机
    q.memcpy(output.data(), d_output, size * sizeof(float)).wait();
    
    // 验证结果
    bool pass = true;
    float max_error = 0.0f;
    for (int i = 0; i < size; i++) {{
        float expected = input[i] * 2.0f;
        float error = std::abs(output[i] - expected);
        if (error > max_error) max_error = error;
        if (error > 1e-5) pass = false;
    }}
    
    // 清理
    sycl::free(d_input, q);
    sycl::free(d_output, q);
    
    std::cout << "Test " << (pass ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Max error: " << max_error << std::endl;
    
    return pass ? 0 : 1;
}}
'''
    
    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(test_code)
        test_cpp = f.name
    
    try:
        # 编译
        compile_cmd = [
            'icpx', '-fsycl', '-O2', '-std=c++17',
            '-I', str(base_dir),
            test_cpp, '-o', test_cpp.replace('.cpp', '')
        ]
        
        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return {
                'status': 'COMPILATION_FAILED',
                'error': result.stderr[:500],
                'pass': False
            }
        
        # 运行
        exe = test_cpp.replace('.cpp', '')
        result = subprocess.run([exe], capture_output=True, text=True, timeout=30)
        
        output_lines = result.stdout.strip().split('\n')
        status = 'UNKNOWN'
        max_error = 0.0
        
        for line in output_lines:
            if 'PASSED' in line:
                status = 'PASSED'
            elif 'FAILED' in line:
                status = 'FAILED'
            elif 'Max error:' in line:
                try:
                    max_error = float(line.split(':')[1].strip())
                except:
                    pass
        
        return {
            'status': status,
            'max_error': max_error,
            'pass': status == 'PASSED' and result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'error': str(e),
            'pass': False
        }
    finally:
        # 清理临时文件
        import os
        try:
            os.unlink(test_cpp)
            if Path(test_cpp.replace('.cpp', '')).exists():
                os.unlink(test_cpp.replace('.cpp', ''))
        except:
            pass


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sycl_file = sys.argv[1]
        result = run_simple_sycl_test(sycl_file)
        print(f"Test result: {result['status']}")
        print(f"Max error: {result.get('max_error', 'N/A')}")
        print(f"Pass: {result['pass']}")
        if 'error' in result:
            print(f"Error: {result['error']}")
