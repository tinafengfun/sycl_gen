#!/usr/bin/env python3
"""
Real Test Harness Generator
真实测试harness生成器 - 生成真正调用kernel的测试代码

修复问题：原来的harness只是复制输入到输出，没有实际调用kernel
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class KernelInfo:
    """Kernel信息"""
    name: str
    template_types: List[str]  # 如 ['T']
    params: List[Tuple[str, str]]  # (type, name)
    launch_params: Dict[str, any]  # 启动参数

class RealHarnessGenerator:
    """真实测试harness生成器"""
    
    def __init__(self, kernel_id: str):
        self.kernel_id = kernel_id
        self.base_dir = Path(__file__).parent.parent
    
    def extract_kernel_info(self, cuda_file: str) -> Optional[KernelInfo]:
        """从CUDA文件中提取kernel信息"""
        with open(cuda_file, 'r') as f:
            content = f.read()
        
        # 查找 __global__ kernel
        # 支持模板和非模板
        pattern = r'''
            (?:template\s*<([^>]+)>)?\s*
            __global__\s+void\s+
            (\w+)_kernel\s*
            \(([^)]*)\)
        '''
        
        match = re.search(pattern, content, re.VERBOSE)
        if not match:
            return None
        
        template_part = match.group(1) or ""
        kernel_name = match.group(2)
        params_str = match.group(3)
        
        # 解析模板参数
        template_types = []
        if template_part:
            # 提取类型参数名，如 "typename T" -> "T"
            for t in template_part.split(','):
                t = t.strip()
                if 'typename' in t:
                    template_types.append(t.split()[-1])
        
        # 解析函数参数
        params = []
        for param in params_str.split(','):
            param = param.strip()
            if param and param != 'void':
                # 简单解析：找最后一个空格前的所有内容作为类型
                if ' ' in param:
                    parts = param.rsplit(' ', 1)
                    param_type = parts[0].strip()
                    param_name = parts[1].strip()
                    params.append((param_type, param_name))
        
        return KernelInfo(
            name=kernel_name,
            template_types=template_types,
            params=params,
            launch_params={}
        )
    
    def generate_real_cuda_harness(self, info: KernelInfo, input_size: int = 1024) -> str:
        """生成真正调用CUDA kernel的测试harness"""
        kernel_name = info.name
        
        # 为每个指针参数生成内存分配代码
        ptr_params = [(t, n) for t, n in info.params if '*' in t]
        scalar_params = [(t, n) for t, n in info.params if '*' not in t]
        
        # 生成内存分配
        alloc_code = []
        memcpy_in = []
        memcpy_out = []
        free_code = []
        
        for i, (ptype, pname) in enumerate(ptr_params):
            base_type = ptype.replace('*', '').strip()
            
            # 第一个指针通常是输出，其他是输入
            if i == 0:
                # 输出：从输入文件读取大小，分配内存
                alloc_code.append(f'    // Output: {pname}')
                alloc_code.append(f'    {base_type}* {pname};')
                alloc_code.append(f'    cudaMalloc(&{pname}, size * sizeof({base_type}));')
                memcpy_out.append(f'    cudaMemcpy(output_data.data(), {pname}, size * sizeof({base_type}), cudaMemcpyDeviceToHost);')
            else:
                # 输入：分配并复制数据
                alloc_code.append(f'    // Input: {pname}')
                alloc_code.append(f'    {base_type}* {pname};')
                alloc_code.append(f'    cudaMalloc(&{pname}, size * sizeof({base_type}));')
                memcpy_in.append(f'    cudaMemcpy({pname}, input_data.data(), size * sizeof({base_type}), cudaMemcpyHostToDevice);')
            
            free_code.append(f'    cudaFree({pname});')
        
        # 生成标量参数（如果有）
        scalar_args = []
        for ptype, pname in scalar_params:
            if 'int' in ptype.lower():
                # 假设整数参数从命令行获取或使用默认值
                scalar_args.append(f'size')  # 简化：使用size作为参数
            else:
                scalar_args.append('0')  # 其他类型使用默认值
        
        # 生成kernel调用
        if info.template_types:
            template_inst = '<' + ', '.join(['float' for _ in info.template_types]) + '>'
        else:
            template_inst = ''
        
        ptr_names = [n for _, n in ptr_params]
        all_args = ptr_names + scalar_args
        
        code = f'''// Real CUDA Test Harness for {kernel_name}
// Auto-generated - actually calls the kernel

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

// Include the kernel
#include "{cuda_file}"

int main(int argc, char* argv[]) {{
    if (argc < 3) {{
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }}
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    
    // Read input data
    std::ifstream in(input_file, std::ios::binary);
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    
    size_t size = file_size / sizeof(float);
    std::vector<float> input_data(size);
    std::vector<float> output_data(size);
    
    in.read(reinterpret_cast<char*>(input_data.data()), file_size);
    in.close();
    
    // Allocate device memory
{chr(10).join(alloc_code)}
    
    // Copy input to device
{chr(10).join(memcpy_in)}
    
    // Launch kernel
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    {kernel_name}_kernel{template_inst}<<<blocks, threads>>>({', '.join(all_args)});
    cudaDeviceSynchronize();
    
    // Copy output back
{chr(10).join(memcpy_out)}
    
    // Write output
    std::ofstream out(output_file, std::ios::binary);
    out.write(reinterpret_cast<char*>(output_data.data()), size * sizeof(float));
    out.close();
    
    // Cleanup
{chr(10).join(free_code)}
    
    return 0;
}}
'''
        return code
    
    def generate_real_sycl_harness(self, info: KernelInfo, input_size: int = 1024) -> str:
        """生成真正调用SYCL kernel的测试harness"""
        kernel_name = info.name
        
        ptr_params = [(t, n) for t, n in info.params if '*' in t]
        scalar_params = [(t, n) for t, n in info.params if '*' not in t]
        
        # 生成SYCL内存分配和队列操作
        alloc_code = []
        memcpy_in = []
        memcpy_out = []
        
        for i, (ptype, pname) in enumerate(ptr_params):
            base_type = ptype.replace('*', '').strip()
            
            if i == 0:
                alloc_code.append(f'    // Output: {pname}')
                alloc_code.append(f'    {base_type}* {pname} = sycl::malloc_device<{base_type}>(size, q);')
                memcpy_out.append(f'    q.memcpy(output_data.data(), {pname}, size * sizeof({base_type})).wait();')
            else:
                alloc_code.append(f'    // Input: {pname}')
                alloc_code.append(f'    {base_type}* {pname} = sycl::malloc_device<{base_type}>(size, q);')
                memcpy_in.append(f'    q.memcpy({pname}, input_data.data(), size * sizeof({base_type})).wait();')
        
        # 生成kernel调用（使用functor方式）
        ptr_names = [n for _, n in ptr_params]
        scalar_args = ['size' for _ in scalar_params]  # 简化处理
        all_args = ptr_names + scalar_args
        
        if info.template_types:
            template_inst = '<' + ', '.join(['float' for _ in info.template_types]) + '>'
        else:
            template_inst = ''
        
        code = f'''// Real SYCL Test Harness for {kernel_name}
// Auto-generated - actually calls the kernel

#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

// Include the kernel
#include "{sycl_file}"

int main(int argc, char* argv[]) {{
    if (argc < 3) {{
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }}
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    
    // Read input data
    std::ifstream in(input_file, std::ios::binary);
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    
    size_t size = file_size / sizeof(float);
    std::vector<float> input_data(size);
    std::vector<float> output_data(size);
    
    in.read(reinterpret_cast<char*>(input_data.data()), file_size);
    in.close();
    
    // Create queue
    sycl::queue q;
    
    // Allocate device memory
{chr(10).join(alloc_code)}
    
    // Copy input to device
{chr(10).join(memcpy_in)}
    
    // Launch kernel
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    q.parallel_for(sycl::nd_range<1>(blocks * threads, threads), [=](sycl::nd_item<1> item) {{
        {kernel_name}_kernel{template_inst}({', '.join(all_args)});
    }}).wait();
    
    // Copy output back
{chr(10).join(memcpy_out)}
    
    // Write output
    std::ofstream out(output_file, std::ios::binary);
    out.write(reinterpret_cast<char*>(output_data.data()), size * sizeof(float));
    out.close();
    
    // Cleanup
{chr(10).join([f'    sycl::free({n}, q);' for _, n in ptr_params])}
    
    return 0;
}}
'''
        return code


def generate_real_test_harness(kernel_id: str, cuda_file: str, sycl_file: str) -> Tuple[str, str]:
    """
    生成真实的测试harness（真正调用kernel）
    
    Args:
        kernel_id: kernel ID
        cuda_file: CUDA kernel文件路径
        sycl_file: SYCL kernel文件路径
        
    Returns:
        (cuda_harness_code, sycl_harness_code)
    """
    generator = RealHarnessGenerator(kernel_id)
    info = generator.extract_kernel_info(cuda_file)
    
    if not info:
        raise ValueError(f"Could not extract kernel info from {cuda_file}")
    
    cuda_harness = generator.generate_real_cuda_harness(info)
    sycl_harness = generator.generate_real_sycl_harness(info)
    
    return cuda_harness, sycl_harness


# 更新unified_converter.py中的测试harness生成
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        kernel_id = sys.argv[1]
        cuda_file = f"kernel_dataset/cuda/{kernel_id}_kernel.cu"
        sycl_file = f"kernel_dataset/sycl/{kernel_id}_kernel.dp.cpp"
        try:
            cuda_harness, sycl_harness = generate_real_test_harness(kernel_id, cuda_file, sycl_file)
            print(f"Generated real test harness for {kernel_id}")
            print("\nCUDA Harness (first 50 lines):")
            print('\n'.join(cuda_harness.split('\n')[:50]))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
