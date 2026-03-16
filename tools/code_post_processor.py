#!/usr/bin/env python3
"""
Code Post-Processor for AI Generated SYCL Code
AI生成代码后处理器 - 自动修复常见问题

修复以下问题：
1. Lambda中错误使用模板参数
2. local_accessor构造函数
3. 其他常见语法错误
"""

import re
from pathlib import Path
from typing import List, Tuple

class CodePostProcessor:
    """代码后处理器"""
    
    def __init__(self):
        self.fixes_applied = []
    
    def process(self, code: str) -> str:
        """
        处理代码，修复常见问题
        
        Args:
            code: 原始代码
            
        Returns:
            修复后的代码
        """
        self.fixes_applied = []
        
        # 按顺序应用修复
        code = self._remove_markdown(code)
        code = self._fix_lambda_template_args(code)
        code = self._fix_local_accessor(code)
        code = self._fix_namespace_braces(code)
        code = self._fix_missing_semicolons(code)
        code = self._fix_cuda_error_handling(code)
        code = self._fix_abort_function(code)
        
        return code
    
    def _remove_markdown(self, code: str) -> str:
        """移除markdown标记"""
        # 移除 ```cpp 和 ```
        code = re.sub(r'^```cpp\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
        return code.strip()
    
    def _fix_lambda_template_args(self, code: str) -> str:
        """
        修复lambda中的模板参数问题
        
        问题：cgh.parallel_for<class FilterTransformKernel<T>>(...)
        修复：cgh.parallel_for<class FilterTransformKernel>(...)
        """
        # 查找 parallel_for 中的模板参数
        pattern = r'(cgh\.parallel_for\s*<)\s*class\s+(\w+)<[^\u003e]+>(\s*>)'
        
        def replace_template_arg(match):
            self.fixes_applied.append(f"Removed template args from lambda: {match.group(2)}")
            return f'{match.group(1)}class {match.group(2)}{match.group(3)}'
        
        code = re.sub(pattern, replace_template_arg, code)
        return code
    
    def _fix_local_accessor(self, code: str) -> str:
        """
        修复local_accessor构造函数
        
        问题：sycl::local_accessor<float, 1> arr(C, item.get_group())
        修复：sycl::local_accessor<float, 1> arr(sycl::range<1>(C), cgh)
        """
        # 这个修复需要在handler上下文中，可能需要更复杂的处理
        # 暂时添加一个提示
        if 'local_accessor' in code and 'item.get_group()' in code:
            self.fixes_applied.append("WARNING: local_accessor may need manual fix")
        return code
    
    def _fix_namespace_braces(self, code: str) -> str:
        """修复namespace大括号匹配"""
        # 计算namespace的大括号
        open_count = code.count('{')
        close_count = code.count('}')
        
        if open_count > close_count:
            # 需要添加关闭大括号
            diff = open_count - close_count
            code += '\n' + '}' * diff
            self.fixes_applied.append(f"Added {diff} missing closing braces")
        
        return code
    
    def _fix_missing_semicolons(self, code: str) -> str:
        """修复缺失的分号"""
        # 在namespace结束前添加分号（如果有需要）
        return code
    
    def _fix_cuda_error_handling(self, code: str) -> str:
        """移除或修复CUDA错误处理代码"""
        # 移除ReportCUDAErrors宏（如果存在且未正确定义）
        if 'ReportCUDAErrors' in code and 'cudaError_t' in code:
            # 替换为SYCL错误处理或移除
            code = re.sub(
                r'#define\s+ReportCUDAErrors.*?#endif',
                '// CUDA error handling removed - SYCL uses exceptions',
                code,
                flags=re.DOTALL
            )
            self.fixes_applied.append("Removed CUDA error handling macros")
        
        return code
    
    def _fix_abort_function(self, code: str) -> str:
        """
        修复错误的abort函数调用
        
        问题：AI生成 sycl::ext::oneapi::experimental::abort()
        修复：替换为 std::abort() 并添加头文件
        """
        if 'sycl::ext::oneapi::experimental::abort()' in code:
            code = code.replace(
                'sycl::ext::oneapi::experimental::abort()',
                'std::abort()'
            )
            self.fixes_applied.append("Fixed sycl::ext::oneapi::experimental::abort() -> std::abort()")
            
            # 检查是否已有cstdlib头文件
            if '#include <cstdlib>' not in code and '#include <stdlib.h>' not in code:
                # 在sycl头文件后添加cstdlib
                code = code.replace(
                    '#include <sycl/sycl.hpp>',
                    '#include <sycl/sycl.hpp>\n#include <cstdlib>'
                )
                self.fixes_applied.append("Added #include <cstdlib> for std::abort")
        
        return code
    
    def get_fixes_summary(self) -> List[str]:
        """获取应用的修复摘要"""
        return self.fixes_applied


class AdvancedCodeFixer:
    """高级代码修复器 - 处理更复杂的问题"""
    
    def __init__(self):
        self.fixes = []
    
    def fix_winograd_specific(self, code: str) -> str:
        """修复Winograd kernel的特定问题"""
        # 修复FilterTransformKernel等特定问题
        
        # 查找并修复 parallel_for 中的模板参数
        lines = code.split('\n')
        new_lines = []
        
        for line in lines:
            # 修复类似 cgh.parallel_for<class FilterTransformKernel<T>> 的问题
            if 'parallel_for' in line and '<class' in line and '<T>' in line:
                line = re.sub(r'<class\s+(\w+)<[^\u003e]+>(>>?)', r'<class \1\2', line)
                self.fixes.append("Fixed template args in parallel_for")
            
            new_lines.append(line)
        
        return '\n'.join(new_lines)
    
    def fix_shared_memory(self, code: str) -> str:
        """修复shared memory声明"""
        # 查找错误的shared memory使用
        if '__shared__' in code:
            # 需要转换为local_accessor，但这很复杂
            self.fixes.append("WARNING: __shared__ found - needs manual conversion to local_accessor")
        return code
    
    def fix_incomplete_code(self, code: str) -> str:
        """修复不完整的代码"""
        # 检查代码是否以不完整的行结束
        lines = code.split('\n')
        
        # 移除末尾的空行和不完整行
        while lines and (not lines[-1].strip() or lines[-1].strip().endswith(',')):
            if lines[-1].strip().endswith(','):
                self.fixes.append("Removed incomplete trailing line")
            lines.pop()
        
        # 确保代码完整
        code = '\n'.join(lines)
        
        return code


def post_process_code(code: str, kernel_id: str = "") -> Tuple[str, List[str]]:
    """
    后处理入口函数
    
    Args:
        code: 原始代码
        kernel_id: kernel标识（用于特定修复）
        
    Returns:
        (修复后的代码, 修复摘要列表)
    """
    processor = CodePostProcessor()
    advanced = AdvancedCodeFixer()
    
    # 基础修复
    code = processor.process(code)
    
    # 高级修复
    if 'winograd' in kernel_id.lower() or 'filter' in kernel_id.lower():
        code = advanced.fix_winograd_specific(code)
    
    code = advanced.fix_shared_memory(code)
    code = advanced.fix_incomplete_code(code)
    
    # 收集所有修复
    all_fixes = processor.get_fixes_summary() + advanced.fixes
    
    return code, all_fixes


if __name__ == "__main__":
    # 测试
    test_code = '''```cpp
    cgh.parallel_for<class FilterTransformKernel<T>>(
        sycl::nd_range<1>(...)
    ```'''
    
    result, fixes = post_process_code(test_code, "winograd_filter")
    print("Fixed code:")
    print(result)
    print("\nFixes applied:")
    for fix in fixes:
        print(f"  - {fix}")
