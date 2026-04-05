#!/usr/bin/env python3
"""
Winograd Input Transform Accuracy Test Runner
运行完整的准确度测试流程

工作流程:
1. 生成测试数据
2. 编译SYCL测试程序
3. 运行SYCL测试（在B60）
4. 编译并运行CUDA测试（在远程CUDA环境）
5. 对比结果并生成报告
"""

import os
import sys
import json
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path

class WinogradAccuracyTest:
    def __init__(self, trace_session=None):
        self.base_dir = Path(__file__).parent.parent.parent
        self.test_dir = self.base_dir / "test" / "accuracy"
        self.results_dir = self.base_dir / "results" / "accuracy"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.trace_session = trace_session or f"winograd_acc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Test configurations
        self.test_configs = [
            {"N": 1, "C": 64, "dtype": "float", "layout": "nchw", "test_type": "random"},
            {"N": 1, "C": 64, "dtype": "float", "layout": "nchw", "test_type": "ones"},
            {"N": 1, "C": 64, "dtype": "float", "layout": "nchw", "test_type": "sequential"},
            {"N": 4, "C": 128, "dtype": "float", "layout": "nchw", "test_type": "random"},
            {"N": 1, "C": 64, "dtype": "float", "layout": "nhcw", "test_type": "random"},
        ]
        
        # Tolerance thresholds
        self.tolerance = {
            "float": {"abs": 1e-5, "rel": 1e-4},
            "half": {"abs": 1e-3, "rel": 1e-2}
        }
        
        self.results = {
            "session": self.trace_session,
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {"total": 0, "passed": 0, "failed": 0}
        }
    
    def log(self, message, level="INFO"):
        """打印日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def generate_test_data(self, config):
        """生成测试数据"""
        N, C = config["N"], config["C"]
        size = N * C * 8 * 8  # Input: N x C x 8 x 8
        
        test_type = config["test_type"]
        dtype = np.float32 if config["dtype"] == "float" else np.float16
        
        if test_type == "ones":
            data = np.ones(size, dtype=dtype)
        elif test_type == "zeros":
            data = np.zeros(size, dtype=dtype)
        elif test_type == "sequential":
            data = (np.arange(size) % 100 / 100.0).astype(dtype)
        elif test_type == "boundary":
            values = np.array([0.0, 1.0, -1.0, 1e-7, 1e7, -1e7], dtype=dtype)
            data = np.tile(values, (size // len(values)) + 1)[:size].astype(dtype)
        else:  # random
            np.random.seed(42)
            data = np.random.uniform(-1.0, 1.0, size).astype(dtype)
        
        # Save to file
        filename = f"input_N{N}_C{C}_{config['dtype']}_{test_type}.bin"
        filepath = self.results_dir / filename
        data.tofile(filepath)
        
        self.log(f"Generated test data: {filename} ({size} elements)")
        return filepath
    
    def compile_sycl_test(self):
        """编译SYCL测试程序"""
        self.log("Compiling SYCL test program...")
        
        # Sync files to container first
        self.log("Syncing source files to B60 container...")
        
        # Copy kernel file
        subprocess.run([
            "docker", "cp", 
            str(self.base_dir / "kernel_dataset" / "sycl" / "winograd_input_transform_kernel.dp.cpp"),
            "lsv-container:/workspace/kernel_file.dp.cpp"
        ], check=True)
        
        # Copy test harness
        subprocess.run([
            "docker", "cp",
            str(self.test_dir / "winograd_sycl_test.cpp"),
            "lsv-container:/workspace/test_harness.cpp"
        ], check=True)
        
        # Compile in container
        compile_cmd = """
cd /workspace && \
icpx -fsycl -O2 -std=c++17 -c kernel_file.dp.cpp -o /tmp/kernel.o && \
icpx -fsycl -O2 -std=c++17 test_harness.cpp /tmp/kernel.o -o /workspace/winograd_sycl_test
"""
        
        result = subprocess.run(
            ["docker", "exec", "lsv-container", "bash", "-c", compile_cmd],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            self.log(f"SYCL compilation failed:\n{result.stderr}", "ERROR")
            return False
        
        self.log("SYCL compilation successful")
        return True
    
    def run_sycl_test(self, config, input_file):
        """运行SYCL测试"""
        N, C = config["N"], config["C"]
        dtype = config["dtype"]
        layout = "nhcw" if config["layout"] == "nhcw" else "nchw"
        test_type = config["test_type"]
        
        # Copy input file to container
        input_filename = input_file.name
        container_input = f"/workspace/test_data/{input_filename}"
        container_output = f"/workspace/test_data/sycl_output_N{N}_C{C}_{dtype}_{test_type}.bin"
        local_output = self.results_dir / f"sycl_output_N{N}_C{C}_{dtype}_{test_type}.bin"
        
        # Create test_data directory in container
        subprocess.run(
            ["docker", "exec", "lsv-container", "mkdir", "-p", "/workspace/test_data"],
            check=True
        )
        
        # Copy input file to container
        subprocess.run(
            ["docker", "cp", str(input_file), f"lsv-container:{container_input}"],
            check=True
        )
        
        # Run test in B60
        test_binary = "/workspace/winograd_sycl_test"
        cmd = [
            "docker", "exec", "lsv-container",
            test_binary,
            str(N), str(C), dtype, layout, test_type,
            container_input, container_output
        ]
        
        self.log(f"Running SYCL test: N={N}, C={C}, dtype={dtype}, layout={layout}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            self.log(f"SYCL test failed:\n{result.stderr}", "ERROR")
            self.log(f"stdout:\n{result.stdout}", "ERROR")
            return None
        
        # Copy output back from container
        subprocess.run(
            ["docker", "cp", f"lsv-container:{container_output}", str(local_output)],
            check=True
        )
        
        self.log(f"SYCL test completed")
        return local_output
    
    def run_cuda_test(self, config, input_file):
        """运行CUDA测试（作为对比基准）"""
        N, C = config["N"], config["C"]
        dtype = config["dtype"]
        layout = "nhcw" if config["layout"] == "nhcw" else "nchw"
        test_type = config["test_type"]
        
        # Copy input file to remote CUDA environment
        input_filename = input_file.name
        remote_input = f"/tmp/test_data/{input_filename}"
        remote_output = f"/tmp/test_data/cuda_output_N{N}_C{C}_{dtype}_{test_type}.bin"
        local_output = self.results_dir / f"cuda_output_N{N}_C{C}_{dtype}_{test_type}.bin"
        
        # Create directory on remote host
        ssh_cmd = f"ssh root@10.112.229.160 'mkdir -p /tmp/test_data'"
        subprocess.run(ssh_cmd, shell=True, check=True)
        
        # Copy input file to remote host
        scp_cmd = f"scp {input_file} root@10.112.229.160:{remote_input}"
        subprocess.run(scp_cmd, shell=True, check=True)
        
        # Copy CUDA test files to remote host
        scp_kernel = f"scp {self.base_dir}/kernel_dataset/cuda/winograd_input_transform_kernel.cu root@10.112.229.160:/tmp/kernel.cu"
        scp_test = f"scp {self.test_dir}/winograd_cuda_test.cpp root@10.112.229.160:/tmp/test.cpp"
        subprocess.run(scp_kernel, shell=True, check=True)
        subprocess.run(scp_test, shell=True, check=True)
        
        # Copy files from remote host to docker container
        copy_to_docker = f"""
        ssh root@10.112.229.160 '
        docker cp /tmp/kernel.cu cuda12.9-test:/workspace/kernel.cu && \\
        docker cp /tmp/test.cpp cuda12.9-test:/workspace/test.cpp && \\
        docker cp {remote_input} cuda12.9-test:/workspace/input.bin && \\
        docker exec cuda12.9-test mkdir -p /workspace/test_data
        '
        """
        subprocess.run(copy_to_docker, shell=True, check=True)
        
        # Compile and run in docker container (GPU already configured at container creation)
        compile_and_run_cmd = f"""
        ssh root@10.112.229.160 '
        docker exec cuda12.9-test bash -c "
        cd /workspace && \\
        nvcc -O2 -std=c++17 -c kernel.cu -o kernel.o && \\
        nvcc -O2 -std=c++17 test.cpp kernel.o -o cuda_test && \\
        ./cuda_test {N} {C} {dtype} {layout} {test_type} /workspace/input.bin /workspace/test_data/output.bin
        "
        '
        """
        
        self.log(f"Running CUDA test: N={N}, C={C}, dtype={dtype}, layout={layout}")
        
        result = subprocess.run(compile_and_run_cmd, shell=True, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            self.log(f"CUDA test failed:\n{result.stderr}", "ERROR")
            self.log(f"stdout:\n{result.stdout}", "ERROR")
            return None
        
        # Copy output from docker to remote host
        copy_docker_to_host = f"docker cp cuda12.9-test:/workspace/test_data/output.bin {remote_output}"
        result = subprocess.run(
            ["ssh", "root@10.112.229.160", copy_docker_to_host],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            self.log(f"Failed to copy from docker to host: {result.stderr}", "ERROR")
            return None
        
        # Copy from remote host to local machine
        scp_back = f"scp root@10.112.229.160:{remote_output} {local_output}"
        result = subprocess.run(scp_back, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            self.log(f"Failed to copy from host to local: {result.stderr}", "ERROR")
            return None
        
        self.log(f"CUDA test completed")
        return local_output
    
    def compare_results(self, sycl_output, cuda_output, config):
        """对比SYCL和CUDA结果"""
        dtype = config["dtype"]
        
        # Load both outputs
        sycl_data = np.fromfile(sycl_output, dtype=np.float32)
        cuda_data = np.fromfile(cuda_output, dtype=np.float32)
        
        # Check sizes match
        if len(sycl_data) != len(cuda_data):
            self.log(f"Size mismatch: SYCL={len(sycl_data)}, CUDA={len(cuda_data)}", "ERROR")
            return False
        
        # Check for NaN/Inf
        nan_count_sycl = np.sum(np.isnan(sycl_data))
        inf_count_sycl = np.sum(np.isinf(sycl_data))
        nan_count_cuda = np.sum(np.isnan(cuda_data))
        inf_count_cuda = np.sum(np.isinf(cuda_data))
        
        if nan_count_sycl > 0 or inf_count_sycl > 0:
            self.log(f"SYCL: Found {nan_count_sycl} NaN and {inf_count_sycl} Inf values", "ERROR")
            return False
        
        if nan_count_cuda > 0 or inf_count_cuda > 0:
            self.log(f"CUDA: Found {nan_count_cuda} NaN and {inf_count_cuda} Inf values", "ERROR")
            return False
        
        # Calculate differences
        abs_diff = np.abs(sycl_data - cuda_data)
        rel_diff = abs_diff / (np.abs(cuda_data) + 1e-10)  # Add epsilon to avoid division by zero
        
        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)
        mean_abs_diff = np.mean(abs_diff)
        mean_rel_diff = np.mean(rel_diff)
        
        # Get tolerance thresholds
        tol = self.tolerance.get(dtype, self.tolerance["float"])
        
        # Count violations
        abs_violations = np.sum(abs_diff > tol["abs"])
        rel_violations = np.sum(rel_diff > tol["rel"])
        
        violation_rate = max(abs_violations, rel_violations) / len(sycl_data)
        max_mismatch_rate = tol.get("max_mismatch_rate", 0.001)  # Default 0.1%
        
        # Log comparison results
        self.log(f"Comparison Results:")
        self.log(f"  Max absolute difference: {max_abs_diff:.6e} (threshold: {tol['abs']:.0e})")
        self.log(f"  Max relative difference: {max_rel_diff:.6e} (threshold: {tol['rel']:.0e})")
        self.log(f"  Mean absolute difference: {mean_abs_diff:.6e}")
        self.log(f"  Mean relative difference: {mean_rel_diff:.6e}")
        self.log(f"  Violation rate: {violation_rate*100:.2f}% (threshold: {max_mismatch_rate*100:.1f}%)")
        
        # Determine pass/fail
        passed = (max_abs_diff <= tol["abs"] * 10 or max_rel_diff <= tol["rel"] * 10) and violation_rate <= max_mismatch_rate
        
        if passed:
            self.log(f"✅ Results match within tolerance")
        else:
            self.log(f"❌ Results exceed tolerance", "WARNING")
        
        return passed
    
    def run_all_tests(self):
        """运行所有测试"""
        self.log("="*60)
        self.log("Starting Winograd Accuracy Tests")
        self.log("="*60)
        
        # Compile SYCL test
        if not self.compile_sycl_test():
            self.log("Failed to compile SYCL test", "ERROR")
            return False
        
        # Run each test configuration
        for i, config in enumerate(self.test_configs):
            self.log(f"\nTest {i+1}/{len(self.test_configs)}: {config}")
            
            # Generate test data
            input_file = self.generate_test_data(config)
            
            # Run SYCL test
            output_file = self.run_sycl_test(config, input_file)
            
            if output_file is None:
                self.results["tests"].append({
                    "config": config,
                    "status": "failed",
                    "error": "SYCL execution failed"
                })
                self.results["summary"]["failed"] += 1
                continue
            
            # Run CUDA test as baseline
            cuda_output_file = self.run_cuda_test(config, input_file)
            
            if cuda_output_file is None:
                self.results["tests"].append({
                    "config": config,
                    "status": "failed",
                    "error": "CUDA execution failed"
                })
                self.results["summary"]["failed"] += 1
                continue
            
            # Compare SYCL vs CUDA results
            if self.compare_results(output_file, cuda_output_file, config):
                self.results["tests"].append({
                    "config": config,
                    "status": "passed",
                    "sycl_output": str(output_file),
                    "cuda_output": str(cuda_output_file)
                })
                self.results["summary"]["passed"] += 1
            else:
                self.results["tests"].append({
                    "config": config,
                    "status": "failed",
                    "sycl_output": str(output_file),
                    "cuda_output": str(cuda_output_file)
                })
                self.results["summary"]["failed"] += 1
            
            self.results["summary"]["total"] += 1
        
        # Generate report
        self.generate_report()
        
        return self.results["summary"]["failed"] == 0
    
    def generate_report(self):
        """生成测试报告"""
        report_file = self.results_dir / f"accuracy_report_{self.trace_session}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"\n{'='*60}")
        self.log("Test Summary")
        self.log(f"{'='*60}")
        self.log(f"Total tests: {self.results['summary']['total']}")
        self.log(f"Passed: {self.results['summary']['passed']}")
        self.log(f"Failed: {self.results['summary']['failed']}")
        self.log(f"Pass rate: {self.results['summary']['passed']/max(self.results['summary']['total'],1)*100:.1f}%")
        self.log(f"\nReport saved to: {report_file}")

if __name__ == "__main__":
    tester = WinogradAccuracyTest()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
