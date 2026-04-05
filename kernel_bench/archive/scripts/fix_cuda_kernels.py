import re
from pathlib import Path

KERNEL_DIR = Path('kernel_dataset/cuda')

# 修复 batch_norm: 添加缺少的头文件
batch_norm_file = KERNEL_DIR / 'batch_norm_kernel.cu'
if batch_norm_file.exists():
    content = batch_norm_file.read_text()
    # 修复 #include 行
    content = content.replace('#include\n', '#include "cuda_common.h"\n')
    batch_norm_file.write_text(content)
    print('✅ Fixed batch_norm_kernel.cu')

# 修复 policy_map: 添加 cstdio
policy_map_file = KERNEL_DIR / 'policy_map_kernel.cu'
if policy_map_file.exists():
    content = policy_map_file.read_text()
    if '#include <cstdio>' not in content and '#include <stdio.h>' not in content:
        content = content.replace('#include <cuda_runtime.h>', '#include <cuda_runtime.h>\n#include <cstdio>')
        policy_map_file.write_text(content)
        print('✅ Fixed policy_map_kernel.cu')

# 修复 winograd_filter_transform: 添加宏定义
winograd_file = KERNEL_DIR / 'winograd_filter_transform_kernel.cu'
if winograd_file.exists():
    content = winograd_file.read_text()
    if '#define ReportCUDAErrors' not in content:
        # 添加 ReportCUDAErrors 宏定义
        insert_text = '''#define ReportCUDAErrors(status) do { \\
    if (status != cudaSuccess) { \\
        printf("CUDA Error: %s at %s:%d\\n", cudaGetErrorString(status), __FILE__, __LINE__); \\
    } \\
} while(0)

'''
        content = content.replace('namespace lczero {', insert_text + 'namespace lczero {')
        winograd_file.write_text(content)
        print('✅ Fixed winograd_filter_transform_kernel.cu')

print('批量修复完成')
