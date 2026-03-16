#!/usr/bin/env python3
"""
快速准确度测试 - 使用现有测试基础设施
Quick accuracy test using existing test infrastructure
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime

def check_test_cuda_exists():
    """检查 test_cuda 目录是否存在"""
    test_cuda_dir = Path('test_cuda')
    if not test_cuda_dir.exists():
        print("❌ test_cuda 目录不存在")
        print("   请确保 test_cuda 目录已准备好")
        return False
    return True

def copy_to_remote():
    """拷贝代码到远程机器"""
    print("📤 拷贝 test_cuda 到远程机器...")
    result = subprocess.run(
        ['scp', '-r', 'test_cuda', 'root@10.112.229.160:/home/tianfeng/'],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"❌ 拷贝失败: {result.stderr}")
        return False
    print("✅ 拷贝成功")
    return True

def compile_and_test_cuda():
    """在远程CUDA容器中编译和测试"""
    print("🔨 在远程Docker中编译CUDA代码...")
    
    commands = """
cd /home/tianfeng/test_cuda
docker exec cuda12.9-test bash -c "cd /workspace/test_cuda && make clean && make all"
"""
    
    result = subprocess.run(
        ['ssh', 'root@10.112.229.160', commands],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode != 0:
        print(f"❌ 编译失败: {result.stderr}")
        return False
    
    print("✅ CUDA编译成功")
    
    # 运行测试
    print("🧪 运行CUDA测试...")
    run_commands = """
cd /home/tianfeng/test_cuda
docker exec cuda12.9-test bash -c "cd /workspace/test_cuda && make run"
"""
    
    result = subprocess.run(
        ['ssh', 'root@10.112.229.160', run_commands],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    print(result.stdout)
    if result.stderr:
        print(f"stderr: {result.stderr}")
    
    return result.returncode == 0

def collect_results():
    """收集测试结果"""
    print("📦 收集测试结果...")
    
    commands = """
cd /home/tianfeng/test_cuda
ls -lh reference_data/ 2>/dev/null || echo "No reference_data directory"
"""
    
    result = subprocess.run(
        ['ssh', 'root@10.112.229.160', commands],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    return True

def run_quick_accuracy_test():
    """运行快速准确度测试"""
    print("=" * 80)
    print("🧪 快速 CUDA 准确度测试")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 检查test_cuda
    if not check_test_cuda_exists():
        return
    
    # 拷贝到远程
    if not copy_to_remote():
        return
    
    # 编译和测试
    if not compile_and_test_cuda():
        print("❌ CUDA测试失败")
        return
    
    # 收集结果
    collect_results()
    
    print()
    print("=" * 80)
    print("✅ 快速测试完成")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == '__main__':
    run_quick_accuracy_test()
