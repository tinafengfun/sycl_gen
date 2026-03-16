#!/usr/bin/env python3
"""
Accuracy Test Framework for CUDA-to-SYCL Conversion
数值准确度测试框架

Features:
- 生成测试数据 (边界值、随机值、特殊值)
- 编译并执行CUDA和SYCL版本
- 对比输出结果
- 生成准确度报告
- Trace集成
"""

import os
import sys
import json
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class AccuracyTester:
    """准确度测试器"""
    
    def __init__(self, kernel_id: str, trace_session: str):
        self.kernel_id = kernel_id
        self.trace_session = trace_session
        self.base_dir = Path(__file__).parent.parent
        
        # 测试配置
        self.test_config = {
            "float32": {
                "abs_tol": 1e-5,
                "rel_tol": 1e-4,
                "max_mismatch_rate": 0.001  # 0.1%
            },
            "float16": {
                "abs_tol": 1e-3,
                "rel_tol": 1e-2,
                "max_mismatch_rate": 0.01   # 1%
            }
        }
        
        # 测试结果
        self.results = {
            "test_cases": [],
            "cuda_outputs": [],
            "sycl_outputs": [],
            "comparisons": [],
            "pass_rate": 0.0,
            "status": "pending"
        }
        
    def log_trace(self, action: str, details: Dict):
        """记录Trace日志"""
        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "session": self.trace_session,
            "kernel": self.kernel_id,
            "agent": "AccuracyTester",
            "action": action,
            "details": details
        }
        
        # 追加到trace日志
        trace_file = self.base_dir / ".traces" / "sessions" / self.trace_session / "accuracy_tests.jsonl"
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(trace_file, 'a') as f:
            f.write(json.dumps(trace_entry) + '\n')
    
    def generate_test_data(self, test_type: str, size: int, dtype: str = "float32") -> np.ndarray:
        """
        生成测试数据
        
        Args:
            test_type: "boundary", "random", "special"
            size: 数据大小
            dtype: "float32" or "float16"
        
        Returns:
            numpy array with test data
        """
        if dtype == "float16":
            dtype_np = np.float16
        else:
            dtype_np = np.float32
        
        if test_type == "boundary":
            # 边界值测试
            values = [0.0, 1.0, -1.0, 
                     np.finfo(dtype_np).max, 
                     np.finfo(dtype_np).min,
                     np.finfo(dtype_np).eps]
            data = np.tile(values, (size // len(values)) + 1)[:size].astype(dtype_np)
            
        elif test_type == "random_uniform":
            # 均匀分布随机值
            data = np.random.uniform(-1.0, 1.0, size).astype(dtype_np)
            
        elif test_type == "random_normal":
            # 正态分布随机值
            data = np.random.normal(0.0, 1.0, size).astype(dtype_np)
            
        elif test_type == "special":
            # 特殊值测试
            values = [np.inf, -np.inf, np.nan, 
                     np.finfo(dtype_np).tiny,
                     0.0, -0.0]
            data = np.tile(values, (size // len(values)) + 1)[:size].astype(dtype_np)
            
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        self.log_trace("generate_test_data", {
            "test_type": test_type,
            "size": size,
            "dtype": dtype
        })
        
        return data
    
    def run_cuda_test(self, test_data: np.ndarray, kernel_file: str) -> np.ndarray:
        """
        运行CUDA版本测试
        
        Args:
            test_data: 输入测试数据
            kernel_file: CUDA kernel文件
            
        Returns:
            CUDA输出结果
        """
        # 保存测试数据
        input_file = f"/tmp/{self.kernel_id}_cuda_input.bin"
        output_file = f"/tmp/{self.kernel_id}_cuda_output.bin"
        test_data.tofile(input_file)
        
        # 编译CUDA测试程序
        compile_cmd = [
            "nvcc", "-O2", "-arch=sm_70",
            "-o", f"/tmp/test_{self.kernel_id}_cuda",
            f"tests/cuda_wrappers/{self.kernel_id}_wrapper.cu"
        ]
        
        try:
            subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            self.log_trace("cuda_compile_error", {"error": str(e)})
            raise
        
        # 运行CUDA程序
        run_cmd = [f"/tmp/test_{self.kernel_id}_cuda", input_file, output_file]
        
        try:
            result = subprocess.run(run_cmd, check=True, capture_output=True, text=True)
            execution_time = float(result.stdout.strip())  # 假设输出执行时间
        except subprocess.CalledProcessError as e:
            self.log_trace("cuda_execution_error", {"error": str(e)})
            raise
        
        # 读取输出
        output_data = np.fromfile(output_file, dtype=test_data.dtype)
        
        self.log_trace("run_cuda_test", {
            "input_size": len(test_data),
            "output_size": len(output_data),
            "execution_time_ms": execution_time
        })
        
        return output_data
    
    def run_sycl_test(self, test_data: np.ndarray, kernel_file: str) -> np.ndarray:
        """
        运行SYCL版本测试
        
        Args:
            test_data: 输入测试数据
            kernel_file: SYCL kernel文件
            
        Returns:
            SYCL输出结果
        """
        # 保存测试数据
        input_file = f"/tmp/{self.kernel_id}_sycl_input.bin"
        output_file = f"/tmp/{self.kernel_id}_sycl_output.bin"
        test_data.tofile(input_file)
        
        # 通过B60容器编译和运行
        # 1. 同步文件到容器
        sync_cmd = [
            "docker", "cp", kernel_file,
            f"lsv-container:/workspace/{self.kernel_id}.dp.cpp"
        ]
        subprocess.run(sync_cmd, check=True)
        
        # 2. 编译SYCL
        compile_cmd = [
            "docker", "exec", "lsv-container",
            "icpx", "-fsycl", "-O2", "-std=c++17",
            "-o", f"/workspace/test_{self.kernel_id}",
            f"/workspace/{self.kernel_id}.dp.cpp"
        ]
        
        try:
            subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            self.log_trace("sycl_compile_error", {"error": str(e)})
            raise
        
        # 3. 运行SYCL程序
        run_cmd = [
            "docker", "exec", "lsv-container",
            f"/workspace/test_{self.kernel_id}",
            input_file, output_file
        ]
        
        try:
            result = subprocess.run(run_cmd, check=True, capture_output=True, text=True)
            execution_time = float(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            self.log_trace("sycl_execution_error", {"error": str(e)})
            raise
        
        # 4. 复制输出文件回来
        copy_cmd = [
            "docker", "cp",
            f"lsv-container:{output_file}", output_file
        ]
        subprocess.run(copy_cmd, check=True)
        
        # 读取输出
        output_data = np.fromfile(output_file, dtype=test_data.dtype)
        
        self.log_trace("run_sycl_test", {
            "input_size": len(test_data),
            "output_size": len(output_data),
            "execution_time_ms": execution_time
        })
        
        return output_data
    
    def compare_results(self, cuda_out: np.ndarray, sycl_out: np.ndarray, 
                       dtype: str) -> Dict:
        """
        对比CUDA和SYCL输出结果
        
        Args:
            cuda_out: CUDA输出
            sycl_out: SYCL输出
            dtype: 数据类型
            
        Returns:
            对比结果字典
        """
        config = self.test_config[dtype]
        
        # 确保大小相同
        if len(cuda_out) != len(sycl_out):
            return {
                "status": "size_mismatch",
                "cuda_size": len(cuda_out),
                "sycl_size": len(sycl_out)
            }
        
        # 计算差异
        abs_diff = np.abs(cuda_out - sycl_out)
        rel_diff = abs_diff / (np.abs(cuda_out) + 1e-8)
        
        # 统计
        total = len(cuda_out)
        abs_mismatches = np.sum(abs_diff > config["abs_tol"])
        rel_mismatches = np.sum(rel_diff > config["rel_tol"])
        
        # 取两者都超标的
        mismatches = np.sum((abs_diff > config["abs_tol"]) & 
                           (rel_diff > config["rel_tol"]))
        
        mismatch_rate = mismatches / total
        pass_threshold = config["max_mismatch_rate"]
        
        result = {
            "total_elements": int(total),
            "mismatches": int(mismatches),
            "mismatch_rate": float(mismatch_rate),
            "max_abs_diff": float(np.max(abs_diff)),
            "max_rel_diff": float(np.max(rel_diff)),
            "mean_abs_diff": float(np.mean(abs_diff)),
            "status": "PASS" if mismatch_rate <= pass_threshold else "FAIL"
        }
        
        self.log_trace("compare_results", result)
        
        return result
    
    def run_full_accuracy_test(self, kernel_file_cuda: str, kernel_file_sycl: str) -> Dict:
        """
        运行完整的准确度测试
        
        Args:
            kernel_file_cuda: CUDA kernel文件路径
            kernel_file_sycl: SYCL kernel文件路径
            
        Returns:
            完整测试结果报告
        """
        print(f"[AccuracyTester] Starting accuracy test for {self.kernel_id}")
        
        test_suite = {
            "boundary_small": ("boundary", 64, "float32"),
            "boundary_large": ("boundary", 1024, "float32"),
            "random_uniform": ("random_uniform", 4096, "float32"),
            "random_normal": ("random_normal", 4096, "float32"),
            "special_values": ("special", 128, "float32"),
            "fp16_test": ("random_uniform", 1024, "float16")
        }
        
        all_results = []
        
        for test_name, (test_type, size, dtype) in test_suite.items():
            print(f"[AccuracyTester] Running test: {test_name}")
            
            try:
                # 生成测试数据
                test_data = self.generate_test_data(test_type, size, dtype)
                
                # 运行CUDA版本
                cuda_out = self.run_cuda_test(test_data, kernel_file_cuda)
                
                # 运行SYCL版本
                sycl_out = self.run_sycl_test(test_data, kernel_file_sycl)
                
                # 对比结果
                comparison = self.compare_results(cuda_out, sycl_out, dtype)
                
                test_result = {
                    "test_name": test_name,
                    "test_type": test_type,
                    "size": size,
                    "dtype": dtype,
                    "comparison": comparison
                }
                
                all_results.append(test_result)
                
            except Exception as e:
                print(f"[AccuracyTester] Test {test_name} failed: {e}")
                all_results.append({
                    "test_name": test_name,
                    "status": "ERROR",
                    "error": str(e)
                })
        
        # 统计总体通过率
        passed = sum(1 for r in all_results 
                    if r.get("comparison", {}).get("status") == "PASS")
        total = len(all_results)
        
        final_report = {
            "kernel_id": self.kernel_id,
            "test_timestamp": datetime.now().isoformat(),
            "total_tests": total,
            "passed_tests": passed,
            "pass_rate": passed / total if total > 0 else 0,
            "overall_status": "PASS" if passed == total else "FAIL",
            "test_details": all_results
        }
        
        # 保存报告
        report_file = (self.base_dir / ".traces" / "sessions" / self.trace_session / 
                      f"accuracy_report_{self.kernel_id}.json")
        
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        self.log_trace("accuracy_test_complete", final_report)
        
        print(f"[AccuracyTester] Accuracy test complete: {final_report['overall_status']}")
        print(f"  Pass rate: {final_report['pass_rate']*100:.1f}% ({passed}/{total})")
        
        return final_report


def main():
    """命令行入口"""
    if len(sys.argv) < 4:
        print("Usage: accuracy_tester.py <kernel_id> <cuda_file> <sycl_file> <trace_session>")
        sys.exit(1)
    
    kernel_id = sys.argv[1]
    cuda_file = sys.argv[2]
    sycl_file = sys.argv[3]
    trace_session = sys.argv[4]
    
    tester = AccuracyTester(kernel_id, trace_session)
    report = tester.run_full_accuracy_test(cuda_file, sycl_file)
    
    # 输出结果
    print(json.dumps(report, indent=2))
    
    # 根据结果返回退出码
    sys.exit(0 if report["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
