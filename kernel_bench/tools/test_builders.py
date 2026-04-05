#!/usr/bin/env python3
"""
Test suite for builder tools
测试builder工具的各个功能
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

# 添加tools目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from b60_sycl_builder import B60SyclBuilder
from remote_cuda_builder import RemoteCudaBuilder

class TestB60SyclBuilder:
    """测试B60 SYCL Builder"""
    
    def setup_method(self):
        """每个测试前执行"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.builder = B60SyclBuilder.__new__(B60SyclBuilder)
        self.builder.base_dir = self.test_dir
        self.builder.results_dir = self.test_dir / "results" / "b60"
        self.builder.scripts_dir = self.test_dir / "scripts" / "b60"
        self.builder.status_file = self.test_dir / ".build_status.json"
        self.builder.container = "lsv-container"
        self.builder.workspace = "/workspace"
        self.builder.compiler = "icpx"
        self.builder.compiler_flags = "-fsycl -O2 -std=c++17"
        
    def teardown_method(self):
        """每个测试后清理"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_directory_creation(self):
        """测试目录自动创建"""
        assert not self.builder.results_dir.exists()
        self.builder._ensure_directories()
        assert self.builder.results_dir.exists()
        assert self.builder.scripts_dir.exists()
        print("[PASS] Directory creation test")
    
    def test_timestamp_format(self):
        """测试时间戳格式 YYYYMMDD_HHMMSS"""
        timestamp = self.builder._get_timestamp()
        assert len(timestamp) == 15  # YYYYMMDD_HHMMSS
        assert timestamp[8] == '_'  # 分隔符
        assert timestamp[:8].isdigit()  # 日期部分
        assert timestamp[9:].isdigit()  # 时间部分
        print("[PASS] Timestamp format test")
    
    def test_build_script_generation(self):
        """测试编译脚本生成"""
        self.builder._ensure_directories()
        script_path = self.builder._generate_build_script(
            "test_kernel",
            "/workspace/test.dp.cpp",
            "20260228_143022"
        )
        
        assert script_path.exists()
        content = script_path.read_text()
        
        # 检查脚本内容
        assert "#!/bin/bash" in content
        assert "set -e" in content
        assert "set -x" in content
        assert "icpx -fsycl" in content
        assert "test_kernel" in content
        assert "20260228_143022" in content
        
        # 检查执行权限
        assert os.access(script_path, os.X_OK)
        print("[PASS] Build script generation test")
    
    def test_status_update(self):
        """测试状态文件更新"""
        self.builder._ensure_directories()
        
        # 更新状态
        self.builder._update_status(
            "add_vectors",
            "success",
            "20260228_143022",
            "kernel_dataset/sycl/add_vectors.dp.cpp",
            "results/b60/compile_add_vectors_20260228_143022.log",
            "scripts/b60/build_add_vectors_20260228_143022.sh",
            2.34
        )
        
        # 验证状态文件
        assert self.builder.status_file.exists()
        with open(self.builder.status_file) as f:
            data = json.load(f)
        
        assert "environments" in data
        assert "b60" in data["environments"]
        assert "add_vectors" in data["environments"]["b60"]["kernels"]
        assert data["environments"]["b60"]["kernels"]["add_vectors"]["status"] == "success"
        print("[PASS] Status update test")
    
    def test_status_statistics(self):
        """测试状态统计功能"""
        self.builder._ensure_directories()
        
        # 添加多个kernel状态
        for i, status in enumerate(["success", "success", "failed"]):
            self.builder._update_status(
                f"kernel_{i}",
                status,
                "20260228_143022",
                f"kernel_{i}.dp.cpp",
                f"results/b60/compile_{i}.log",
                f"scripts/b60/build_{i}.sh",
                1.0
            )
        
        with open(self.builder.status_file) as f:
            data = json.load(f)
        
        stats = data["environments"]["b60"]["statistics"]
        assert stats["total"] == 3
        assert stats["success"] == 2
        assert stats["failed"] == 1
        print("[PASS] Status statistics test")


class TestRemoteCudaBuilder:
    """测试Remote CUDA Builder"""
    
    def setup_method(self):
        """每个测试前执行"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.builder = RemoteCudaBuilder.__new__(RemoteCudaBuilder)
        self.builder.base_dir = self.test_dir
        self.builder.results_dir = self.test_dir / "results" / "cuda"
        self.builder.scripts_dir = self.test_dir / "scripts" / "cuda"
        self.builder.status_file = self.test_dir / ".build_status.json"
        self.builder.ssh_host = "root@10.112.229.160"
        self.builder.container = "cuda12.9-test"
        self.builder.workspace = "/workspace"
        self.builder.compiler = "/usr/local/cuda/bin/nvcc"
        self.builder.compiler_flags = "-O2 -arch=sm_70"
        
    def teardown_method(self):
        """每个测试后清理"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_cuda_script_generation(self):
        """测试CUDA编译脚本生成"""
        self.builder._ensure_directories()
        script_path = self.builder._generate_build_script(
            "test_kernel",
            "/workspace/kernel_dataset/cuda/test.cu",
            "20260228_143022"
        )
        
        assert script_path.exists()
        content = script_path.read_text()
        
        # 检查CUDA特定内容
        assert "/usr/local/cuda/bin/nvcc" in content
        assert "-O2 -arch=sm_70" in content
        assert "nvcc --version" in content
        print("[PASS] CUDA script generation test")
    
    def test_remote_status_update(self):
        """测试远程构建状态更新"""
        self.builder._ensure_directories()
        
        self.builder._update_status(
            "softmax",
            "success",
            "20260228_143022",
            "kernel_dataset/cuda/softmax_kernel.cu",
            "results/cuda/compile_softmax_20260228_143022.log",
            "scripts/cuda/build_softmax_20260228_143022.sh",
            5.67
        )
        
        with open(self.builder.status_file) as f:
            data = json.load(f)
        
        assert "remote_cuda" in data["environments"]
        cuda_env = data["environments"]["remote_cuda"]
        assert cuda_env["ssh_host"] == "root@10.112.229.160"
        assert cuda_env["container"] == "cuda12.9-test"
        print("[PASS] Remote status update test")


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("Builder Tools Test Suite")
    print("="*60)
    
    # B60 Tests
    print("\n--- B60 SYCL Builder Tests ---")
    b60_tests = TestB60SyclBuilder()
    
    tests = [
        ("Directory Creation", b60_tests.test_directory_creation),
        ("Timestamp Format", b60_tests.test_timestamp_format),
        ("Script Generation", b60_tests.test_build_script_generation),
        ("Status Update", b60_tests.test_status_update),
        ("Statistics", b60_tests.test_status_statistics),
    ]
    
    for name, test_func in tests:
        try:
            b60_tests.setup_method()
            test_func()
            b60_tests.teardown_method()
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            b60_tests.teardown_method()
    
    # CUDA Tests
    print("\n--- Remote CUDA Builder Tests ---")
    cuda_tests = TestRemoteCudaBuilder()
    
    cuda_test_funcs = [
        ("CUDA Script Generation", cuda_tests.test_cuda_script_generation),
        ("Remote Status Update", cuda_tests.test_remote_status_update),
    ]
    
    for name, test_func in cuda_test_funcs:
        try:
            cuda_tests.setup_method()
            test_func()
            cuda_tests.teardown_method()
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            cuda_tests.teardown_method()
    
    print("\n" + "="*60)
    print("Test Suite Completed")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
