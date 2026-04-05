#!/usr/bin/env python3
"""
NaN Behavior Consistency Tester
NaN行为一致性测试模块

测试CUDA和SYCL对NaN/Inf的处理是否一致
"""

import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class NaNTestResult:
    """NaN测试结果"""
    operation: str
    input_a: float
    input_b: float
    cuda_result: float
    sycl_result: float
    consistent: bool
    note: str = ""


class NaNBehaviorTester:
    """NaN行为一致性测试器"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.remote_host = "root@10.112.229.160"
    
    def run_all_tests(self) -> Dict:
        """
        运行所有NaN行为测试
        
        Returns:
            测试结果字典
        """
        test_cases = [
            # (input_a, input_b, operation, description)
            (1.0, np.nan, "add", "1.0 + NaN"),
            (np.nan, 1.0, "add", "NaN + 1.0"),
            (np.nan, np.nan, "add", "NaN + NaN"),
            (np.inf, np.inf, "add", "inf + inf"),
            (-np.inf, -np.inf, "add", "-inf + -inf"),
            (np.inf, -np.inf, "add", "inf + -inf"),
            (0.0, 0.0, "div", "0.0 / 0.0"),
            (1.0, 0.0, "div", "1.0 / 0.0"),
            (-1.0, 0.0, "div", "-1.0 / 0.0"),
            (np.inf, 0.0, "div", "inf / 0.0"),
            (np.nan, 0.0, "mul", "NaN * 0.0"),
            (np.inf, 0.0, "mul", "inf * 0.0"),
            (np.nan, 1.0, "mul", "NaN * 1.0"),
            (np.nan, np.nan, "mul", "NaN * NaN"),
        ]
        
        results = []
        consistent_count = 0
        
        print("\n=== NaN Behavior Consistency Test ===\n")
        
        for input_a, input_b, operation, desc in test_cases:
            result = self.test_operation(input_a, input_b, operation, desc)
            results.append(result)
            
            if result.consistent:
                consistent_count += 1
                status = "✅ Consistent"
            else:
                status = "⚠️  Different"
            
            print(f"{status} {desc}")
            print(f"       CUDA: {self._format_float(result.cuda_result)}")
            print(f"       SYCL: {self._format_float(result.sycl_result)}")
            if result.note:
                print(f"       Note: {result.note}")
            print()
        
        total = len(results)
        consistency_rate = consistent_count / total if total > 0 else 0
        
        summary = {
            "total_tests": total,
            "consistent": consistent_count,
            "inconsistent": total - consistent_count,
            "consistency_rate": consistency_rate,
            "results": results
        }
        
        print(f"=== Summary ===")
        print(f"Total tests: {total}")
        print(f"Consistent: {consistent_count}")
        print(f"Inconsistent: {total - consistent_count}")
        print(f"Consistency rate: {consistency_rate*100:.1f}%")
        
        return summary
    
    def test_operation(self, input_a: float, input_b: float, 
                      operation: str, description: str) -> NaNTestResult:
        """测试单个操作"""
        # 运行CUDA测试
        cuda_result = self._run_cuda_nan_test(input_a, input_b, operation)
        
        # 运行SYCL测试
        sycl_result = self._run_sycl_nan_test(input_a, input_b, operation)
        
        # 比较结果
        consistent, note = self._compare_nan_results(
            cuda_result, sycl_result, input_a, input_b, operation
        )
        
        return NaNTestResult(
            operation=description,
            input_a=input_a,
            input_b=input_b,
            cuda_result=cuda_result,
            sycl_result=sycl_result,
            consistent=consistent,
            note=note
        )
    
    def _run_cuda_nan_test(self, a: float, b: float, operation: str) -> float:
        """在CUDA环境中运行NaN测试"""
        test_code = f'''
#include <cuda_runtime.h>
#include <math>
#include <iostream>

__global__ void nan_test(float* result, float a, float b) {{
    float res;
    if (strcmp("{operation}", "add") == 0) {{
        res = a + b;
    }} else if (strcmp("{operation}", "mul") == 0) {{
        res = a * b;
    }} else if (strcmp("{operation}", "div") == 0) {{
        res = a / b;
    }} else {{
        res = 0.0f;
    }}
    *result = res;
}}

int main() {{
    float *d_result;
    float h_result;
    cudaMalloc(&d_result, sizeof(float));
    
    float a = {a}f;
    float b = {b}f;
    
    nan_test<<<1, 1>>>(d_result, a, b);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << h_result << std::endl;
    
    cudaFree(d_result);
    return 0;
}}
'''
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(test_code)
                local_cu = f.name
            
            # 复制到远程并运行
            remote_cu = "/tmp/nan_test.cu"
            subprocess.run(['scp', local_cu, f'{self.remote_host}:{remote_cu}'],
                          capture_output=True, timeout=30)
            
            ssh_cmd = f'''
            ssh {self.remote_host} '
            docker exec cuda12.9-test bash -c "
            cd /workspace && 
            nvcc -O2 {remote_cu} -o nan_test && 
            ./nan_test
            "
            '
            '''
            
            result = subprocess.run(ssh_cmd, shell=True, capture_output=True,
                                  text=True, timeout=60)
            
            if result.returncode == 0:
                try:
                    return float(result.stdout.strip())
                except:
                    return float('nan')
            else:
                return float('nan')
                
        except:
            return float('nan')
        finally:
            import os
            try:
                os.unlink(local_cu)
            except:
                pass
    
    def _run_sycl_nan_test(self, a: float, b: float, operation: str) -> float:
        """在SYCL环境中运行NaN测试"""
        test_code = f'''
#include <sycl/sycl.hpp>
#include <math>
#include <iostream>

int main() {{
    sycl::queue q;
    
    float result = 0.0f;
    float* d_result = sycl::malloc_device<float>(1, q);
    
    float a_val = {a}f;
    float b_val = {b}f;
    
    q.single_task([&]() {{
        float res;
        if constexpr ("{operation}" == "add") {{
            res = a_val + b_val;
        }} else if constexpr ("{operation}" == "mul") {{
            res = a_val * b_val;
        }} else if constexpr ("{operation}" == "div") {{
            res = a_val / b_val;
        }} else {{
            res = 0.0f;
        }}
        *d_result = res;
    }}).wait();
    
    q.memcpy(&result, d_result, sizeof(float)).wait();
    
    std::cout << result << std::endl;
    
    sycl::free(d_result, q);
    return 0;
}}
'''
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(test_code)
                local_cpp = f.name
            
            # 复制到docker容器
            container_cpp = "/workspace/nan_test.cpp"
            subprocess.run(['docker', 'cp', local_cpp, f'lsv-container:{container_cpp}'],
                          capture_output=True, timeout=30)
            
            # 编译并运行
            compile_cmd = [
                'docker', 'exec', 'lsv-container', 'bash', '-c',
                f'cd /workspace && icpx -fsycl -O2 {container_cpp} -o nan_test'
            ]
            subprocess.run(compile_cmd, capture_output=True, timeout=60)
            
            run_cmd = ['docker', 'exec', 'lsv-container', '/workspace/nan_test']
            result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                try:
                    return float(result.stdout.strip())
                except:
                    return float('nan')
            else:
                return float('nan')
                
        except:
            return float('nan')
        finally:
            import os
            try:
                os.unlink(local_cpp)
            except:
                pass
    
    def _compare_nan_results(self, cuda_val: float, sycl_val: float,
                           input_a: float, input_b: float, operation: str) -> Tuple[bool, str]:
        """比较NaN结果，判断是否一致"""
        # 两者都是NaN -> 一致
        if np.isnan(cuda_val) and np.isnan(sycl_val):
            return True, "Both return NaN"
        
        # 两者都是Inf -> 检查符号
        if np.isinf(cuda_val) and np.isinf(sycl_val):
            if np.sign(cuda_val) == np.sign(sycl_val):
                return True, "Both return same signed Inf"
            else:
                return False, "Return Inf with different signs"
        
        # 一个是NaN，一个是数字 -> 不一致（宽松模式下可接受）
        if np.isnan(cuda_val) != np.isnan(sycl_val):
            return False, "One returns NaN, the other returns number"
        
        # 都是普通数字 -> 比较数值
        if abs(cuda_val - sycl_val) < 1e-5:
            return True, "Results match"
        else:
            return False, f"Results differ: {cuda_val} vs {sycl_val}"
    
    def _format_float(self, val: float) -> str:
        """格式化float显示"""
        if np.isnan(val):
            return "NaN"
        elif np.isinf(val):
            return "+Inf" if val > 0 else "-Inf"
        else:
            return f"{val:.6f}"


if __name__ == "__main__":
    tester = NaNBehaviorTester()
    results = tester.run_all_tests()
    
    # 如果有不一致的情况，打印建议
    if results["inconsistent"] > 0:
        print("\n⚠️  Found inconsistent NaN handling between CUDA and SYCL!")
        print("Recommendations:")
        print("- Use explicit NaN checks in kernel code")
        print("- Consider using sycl::isnan() for portable NaN detection")
        print("- Document any platform-specific NaN behavior")
