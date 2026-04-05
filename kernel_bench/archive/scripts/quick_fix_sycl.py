#!/usr/bin/env python3
"""
快速修复SYCL文件 - 使用标准头文件模板
"""

import re
from pathlib import Path

def fix_sycl_file(sycl_file: Path) -> bool:
    """修复单个SYCL文件"""
    
    if not sycl_file.exists():
        print(f"❌ 文件不存在: {sycl_file}")
        return False
    
    content = sycl_file.read_text()
    
    # 检查是否已经修复
    if 'sycl_standard_header.h' in content:
        print(f"⏭️  已修复: {sycl_file.name}")
        return True
    
    # 1. 替换头文件包含
    # 移除旧的头文件包含
    content = re.sub(r'#include\s+["\u003c]neural/.*[\"\u003e]\n?', '', content)
    content = re.sub(r'#include\s+["\u003c]utils/.*[\"\u003e]\n?', '', content)
    content = re.sub(r'#include\s+["\u003c]cuda_common\.h[\"\u003e]\n?', '', content)
    content = re.sub(r'#include\s+["\u003c]winograd_helper\.inc[\"\u003e]\n?', '', content)
    
    # 2. 添加标准头文件
    header_include = '#include "include/sycl_standard_header.h"\n'
    
    # 在第一个#include后添加
    if '#include' in content:
        lines = content.split('\n')
        new_lines = []
        added = False
        for line in lines:
            new_lines.append(line)
            if not added and line.strip().startswith('#include'):
                new_lines.append(header_include)
                added = True
        content = '\n'.join(new_lines)
    else:
        # 在namespace前添加
        content = header_include + '\n' + content
    
    # 3. 替换命名空间 cudnn_backend -> sycldnn_backend
    content = content.replace('namespace cudnn_backend {', 
                             'namespace sycldnn_backend {')
    
    # 4. 移除重复的enum定义
    if 'enum ActivationFunction' in content:
        # 移除重复定义，保留头文件中的
        pattern = r'enum\s+ActivationFunction\s*\{[^}]+\};\s*\n?'
        content = re.sub(pattern, '', content, count=1)
    
    # 5. 修复错误处理
    content = content.replace('throw Exception(', 'printf("Error: ')
    content = re.sub(r'throw Exception\([^)]+\);', 
                     'printf("Error: exception\\n"); return;', content)
    
    # 6. 保存修复后的文件
    sycl_file.write_text(content)
    print(f"✅ 修复: {sycl_file.name}")
    return True


def batch_fix_sycl_files():
    """批量修复所有SYCL文件"""
    
    sycl_dir = Path('kernel_dataset/sycl')
    sycl_files = list(sycl_dir.glob('*_kernel.dp.cpp'))
    
    print(f"🚀 批量修复 {len(sycl_files)} 个SYCL文件")
    print()
    
    fixed = 0
    skipped = 0
    failed = 0
    
    for sycl_file in sycl_files:
        if fix_sycl_file(sycl_file):
            fixed += 1
        else:
            failed += 1
    
    print()
    print(f"✅ 修复: {fixed}")
    print(f"⏭️  跳过: {skipped}")
    print(f"❌ 失败: {failed}")


if __name__ == '__main__':
    batch_fix_sycl_files()
