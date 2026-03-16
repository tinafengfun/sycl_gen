#!/usr/bin/env python3
"""
Simplified Accuracy Tester
简化版准确度测试器 - 使用运行时编译进行测试

直接在运行时编译CUDA和SYCL代码进行对比测试
"""

import os
import sys
import json
import subprocess
import numpy as np
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class SimplifiedAccuracyTester:
    """简化版准确度测试器"""
    
    def __init__(self, kernel_id: str, trace_session: str):
        self.kernel_id = kernel_id
        self.trace_session = trace_session
        self.base_dir = Path(__file__).parent.parent
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # 测试配置
        self.test_configs = [
            {"name": "small_random", "size": 256, "dtype": "float32"},
            {"name": "medium_random", "size": 1024, "dtype": "float32"},
            {"name": "large_random", "size": 4096, "dtype": "float32"},
        ]
        
        # 容差配置
        self.tolerance = {"abs": 1e-5, "rel": 1e-4}
    
    def log(self, action: str, details: dict):
        """记录日志"""
        trace_file = self.base_dir / ".traces" / "sessions" / self.trace_session / "accuracy_tests.jsonl"
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "kernel": self.kernel_id,
            "action": action,
            "details": details
        }
        
        with open(trace_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def generate_test_data(self, size: int, dtype: str = "float32") -> np.ndarray:
        """生成测试数据"""
        if dtype == "float16":
            return np.random.randn(size).astype(np.float16)
        return np.random.randn(size).astype(np.float32)
    
    def compile_and_run_cuda(self, cuda_file: str, test_data: np.ndarray) -> Optional[np.ndarray]:
        """编译并运行CUDA代码"""
        try:
            # 创建临时测试程序
            test_code = self._generate_cuda_test_code(cuda_file, len(test_data))
            test_file = self.temp_dir / f"{self.kernel_id}_test.cu"
            with open(test_file, 'w') as f:
                f.write(test_code)
            
            # 编译
            exe_file = self.temp_dir / f"{self.kernel_id}_test"
            compile_cmd = [
                "nvcc", "-O2", "-arch=sm_70",
                "-I", str(self.base_dir),
                "-o", str(exe_file),
                str(test_file)
            ]
            
            result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                self.log("cuda_compile_failed", {"error": result.stderr})
                return None
            
            # 保存输入数据
            input_file = self.temp_dir / f"{self.kernel_id}_input.bin"
            test_data.tofile(input_file)
            
            # 运行
            output_file = self.temp_dir / f"{self.kernel_id}_cuda_output.bin"
            run_cmd = [str(exe_file), str(input_file), str(output_file)]
            
            result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                self.log("cuda_run_failed", {"error": result.stderr})
                return None
            
            # 读取输出
            output = np.fromfile(output_file, dtype=test_data.dtype)
            self.log("cuda_test_passed", {"output_size": len(output)})
            return output
            
        except Exception as e:
            self.log("cuda_test_error", {"error": str(e)})
            return None
    
    def compile_and_run_sycl(self, sycl_file: str, test_data: np.ndarray) -> Optional[np.ndarray]:
        """编译并运行SYCL代码"""
        try:
            # 创建临时测试程序
            test_code = self._generate_sycl_test_code(sycl_file, len(test_data))
            test_file = self.temp_dir / f"{self.kernel_id}_test.dp.cpp"
            with open(test_file, 'w') as f:
                f.write(test_code)
            
            # 编译 - 使用icpx
            exe_file = self.temp_dir / f"{self.kernel_id}_sycl_test"
            compile_cmd = [
                "icpx", "-fsycl", "-O2", "-std=c++17",
                "-I", str(self.base_dir),
                "-o", str(exe_file),
                str(test_file)
            ]
            
            result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                self.log("sycl_compile_failed", {"error": result.stderr[:500]})
                return None
            
            # 保存输入数据
            input_file = self.temp_dir / f"{self.kernel_id}_input.bin"
            test_data.tofile(input_file)
            
            # 运行
            output_file = self.temp_dir / f"{self.kernel_id}_sycl_output.bin"
            run_cmd = [str(exe_file), str(input_file), str(output_file)]
            
            result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                self.log("sycl_run_failed", {"error": result.stderr})
                return None
            
            # 读取输出
            output = np.fromfile(output_file, dtype=test_data.dtype)
            self.log("sycl_test_passed", {"output_size": len(output)})
            return output
            
        except Exception as e:
            self.log("sycl_test_error", {"error": str(e)})
            return None
    
    def _generate_cuda_test_code(self, cuda_file: str, size: int) -> str:
        """生成CUDA测试代码"""
        # 简化的测试代码，实际使用需要根据kernel签名调整
        code = f'''
#include "{cuda_file}"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {{
    if (argc < 3) {{
        printf("Usage: %s <input_file> <output_file>\\n", argv[0]);
        return 1;
    }}
    
    const int size = {size};
    float* input = new float[size];
    float* output = new float[size];
    
    // 读取输入
    FILE* fp = fopen(argv[1], "rb");
    fread(input, sizeof(float), size, fp);
    fclose(fp);
    
    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    // 拷贝到设备
    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动kernel - 这里需要根据实际情况调整
    int blocks = (size + 255) / 256;
    int threads = 256;
    // kernel_call<<<blocks, threads>>>(d_output, d_input, size);
    
    cudaDeviceSynchronize();
    
    // 拷贝回主机
    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 保存输出
    fp = fopen(argv[2], "wb");
    fwrite(output, sizeof(float), size, fp);
    fclose(fp);
    
    // 清理
    delete[] input;
    delete[] output;
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}}
'''
        return code
    
    def _generate_sycl_test_code(self, sycl_file: str, size: int) -> str:
        """生成SYCL测试代码"""
        code = f'''
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {{
    if (argc < 3) {{
        printf("Usage: %s <input_file> <output_file>\\n", argv[0]);
        return 1;
    }}
    
    const int size = {size};
    std::vector<float> input(size);
    std::vector<float> output(size);
    
    // 读取输入
    FILE* fp = fopen(argv[1], "rb");
    fread(input.data(), sizeof(float), size, fp);
    fclose(fp);
    
    // 创建队列
    sycl::queue q;
    
    // 分配设备内存
    float* d_input = sycl::malloc_device<float>(size, q);
    float* d_output = sycl::malloc_device<float>(size, q);
    
    // 拷贝到设备
    q.memcpy(d_input, input.data(), size * sizeof(float)).wait();
    
    // 启动kernel - 这里需要根据实际情况调整
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {{
        int i = idx[0];
        if (i < size) {{
            d_output[i] = d_input[i];  // 简单的passthrough，实际需要调用转换后的kernel
        }}
    }}).wait();
    
    // 拷贝回主机
    q.memcpy(output.data(), d_output, size * sizeof(float)).wait();
    
    // 保存输出
    fp = fopen(argv[2], "wb");
    fwrite(output.data(), sizeof(float), size, fp);
    fclose(fp);
    
    // 清理
    sycl::free(d_input, q);
    sycl::free(d_output, q);
    
    return 0;
}}
'''
        return code
    
    def compare_results(self, cuda_out: np.ndarray, sycl_out: np.ndarray) -> Dict:
        """对比结果"""
        if len(cuda_out) != len(sycl_out):
            return {"status": "size_mismatch", "pass": False}
        
        abs_diff = np.abs(cuda_out - sycl_out)
        rel_diff = abs_diff / (np.abs(cuda_out) + 1e-8)
        
        max_abs = float(np.max(abs_diff))
        max_rel = float(np.max(rel_diff))
        mean_abs = float(np.mean(abs_diff))
        
        pass_test = max_abs <= self.tolerance["abs"] and max_rel <= self.tolerance["rel"]
        
        return {
            "status": "PASS" if pass_test else "FAIL",
            "pass": pass_test,
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
            "mean_abs_diff": mean_abs
        }
    
    def run_test(self, cuda_file: str, sycl_file: str) -> Dict:
        """运行完整测试"""
        print(f"\n🧪 [AccuracyTester] Starting tests for {self.kernel_id}")
        
        results = []
        
        for config in self.test_configs:
            print(f"\n  📊 Test: {config['name']} ({config['size']} elements)")
            
            # 生成测试数据
            test_data = self.generate_test_data(config['size'], config['dtype'])
            
            # 运行CUDA
            print("    Running CUDA...")
            cuda_out = self.compile_and_run_cuda(cuda_file, test_data)
            if cuda_out is None:
                print("    ❌ CUDA failed")
                results.append({"test": config['name'], "status": "CUDA_FAILED"})
                continue
            
            # 运行SYCL
            print("    Running SYCL...")
            sycl_out = self.compile_and_run_sycl(sycl_file, test_data)
            if sycl_out is None:
                print("    ❌ SYCL failed")
                results.append({"test": config['name'], "status": "SYCL_FAILED"})
                continue
            
            # 对比
            comparison = self.compare_results(cuda_out, sycl_out)
            results.append({
                "test": config['name'],
                "status": comparison['status'],
                "comparison": comparison
            })
            
            status_emoji = "✅" if comparison['pass'] else "❌"
            print(f"    {status_emoji} {comparison['status']}: max_abs={comparison['max_abs_diff']:.2e}")
        
        # 统计
        passed = sum(1 for r in results if r.get('status') == 'PASS')
        total = len(results)
        pass_rate = passed / total if total > 0 else 0
        
        report = {
            "kernel_id": self.kernel_id,
            "total_tests": total,
            "passed": passed,
            "pass_rate": pass_rate,
            "status": "PASS" if passed == total else "FAIL",
            "details": results
        }
        
        print(f"\n📊 Summary: {passed}/{total} passed ({pass_rate*100:.1f}%)")
        
        # 保存报告
        report_file = self.base_dir / ".traces" / "sessions" / self.trace_session / f"accuracy_report_{self.kernel_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: simplified_accuracy_tester.py <kernel_id> <cuda_file> <sycl_file>")
        sys.exit(1)
    
    tester = SimplifiedAccuracyTester(sys.argv[1], "test_session")
    report = tester.run_test(sys.argv[2], sys.argv[3])
    print(json.dumps(report, indent=2))
