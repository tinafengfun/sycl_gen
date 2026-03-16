#!/usr/bin/env python3
"""
修复SYCL内核编译错误
Fix SYCL kernel compilation errors
"""

import subprocess
import sys
from pathlib import Path

def test_compilation(sycl_file: str) -> tuple[bool, str]:
    """测试单个文件的编译"""
    try:
        # 复制到docker
        result = subprocess.run(
            ['docker', 'cp', sycl_file, 'lsv-container:/workspace/test.cpp'],
            capture_output=True,
            timeout=10
        )
        
        # 尝试编译
        result = subprocess.run(
            ['docker', 'exec', 'lsv-container', 'bash', '-c',
             'cd /workspace && icpx -fsycl -c test.cpp -o test.o 2>&1'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return result.returncode == 0, result.stderr
    except Exception as e:
        return False, str(e)

def fix_common_errors(content: str) -> str:
    """修复常见编译错误"""
    # 修复1: 替换CUDA特定的数学函数
    content = content.replace('__expf(', 'sycl::exp(')
    content = content.replace('__logf(', 'sycl::log(')
    content = content.replace('__sqrtf(', 'sycl::sqrt(')
    content = content.replace('__fdividef(', 'sycl::native::divide(')
    
    # 修复2: 替换warp shuffle
    content = content.replace('__shfl_xor_sync(0xFFFFFFFF,', 'sycl::ext::oneapi::broadcast(')
    
    # 修复3: 替换CUDA运行时调用
    content = content.replace('cudaDeviceSynchronize()', 'queue.wait()')
    content = content.replace('cudaGetLastError()', '0')
    
    # 修复4: 修复threadIdx/blockIdx
    content = content.replace('threadIdx.x', 'item.get_local_id(0)')
    content = content.replace('blockIdx.x', 'item.get_group(0)')
    content = content.replace('blockDim.x', 'item.get_local_range(0)')
    
    return content

def fix_kernel_file(sycl_file: str) -> bool:
    """修复单个内核文件"""
    print(f"修复: {Path(sycl_file).name}")
    
    # 读取原文件
    with open(sycl_file, 'r') as f:
        content = f.read()
    
    # 应用修复
    fixed_content = fix_common_errors(content)
    
    # 保存修复后的文件
    with open(sycl_file, 'w') as f:
        f.write(fixed_content)
    
    # 测试编译
    success, error = test_compilation(sycl_file)
    
    if success:
        print(f"  ✅ 编译成功")
        return True
    else:
        print(f"  ❌ 编译失败: {error[:100]}")
        return False

def main():
    """主函数"""
    print("=" * 70)
    print("修复SYCL内核编译错误")
    print("=" * 70)
    print()
    
    sycl_dir = Path('kernel_dataset/sycl')
    
    # 获取所有.dp.cpp文件
    kernel_files = list(sycl_dir.glob('*_kernel.dp.cpp'))
    
    print(f"发现 {len(kernel_files)} 个内核文件")
    print()
    
    fixed = 0
    failed = 0
    
    for kernel_file in kernel_files:
        # 测试当前状态
        success, _ = test_compilation(str(kernel_file))
        
        if success:
            print(f"⏭️  {kernel_file.name} - 已通过")
            continue
        
        # 尝试修复
        if fix_kernel_file(str(kernel_file)):
            fixed += 1
        else:
            failed += 1
    
    print()
    print("=" * 70)
    print("修复完成:")
    print(f"  成功: {fixed}")
    print(f"  失败: {failed}")
    print("=" * 70)

if __name__ == '__main__':
    main()
