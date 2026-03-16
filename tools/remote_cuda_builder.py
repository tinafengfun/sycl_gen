#!/usr/bin/env python3
"""
Remote CUDA Builder Tool - Improved Version
在远程节点的CUDA docker中进行CUDA kernel编译和测试

改进点:
- 健壮的多级目录创建（远程主机+容器）
- 改进的文件同步和验证
- 更好的错误处理和日志记录
- 路径安全检查
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class RemoteCudaBuilder:
    """远程CUDA编译工具"""
    
    def __init__(self, config_path: Optional[str] = None):
        # 远程配置
        self.ssh_host = "root@10.112.229.160"
        self.container = "cuda12.9-test"
        self.workspace = "/workspace"
        self.compiler = "/usr/local/cuda/bin/nvcc"
        self.compiler_flags = "-O2 -arch=sm_70"
        
        # 本地路径
        self.base_dir = Path(__file__).parent.parent
        self.results_dir = self.base_dir / "results" / "cuda"
        self.scripts_dir = self.base_dir / "scripts" / "cuda"
        self.status_file = self.base_dir / ".build_status.json"
        
        # 确保目录存在
        self._ensure_directories()
        
    def _ensure_directories(self):
        """创建必要的目录结构"""
        directories = [
            self.results_dir,
            self.scripts_dir,
            self.results_dir / "build_cuda"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
    def _get_timestamp(self) -> str:
        """生成时间戳 YYYYMMDD_HHMMSS"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _run_command(self, cmd: List[str], capture_output: bool = True,
                     check: bool = False) -> Tuple[int, str, str]:
        """运行本地shell命令"""
        print(f"[LOCAL] {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=capture_output, text=True)
        
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd,
                output=result.stdout, stderr=result.stderr
            )
        
        return result.returncode, result.stdout, result.stderr
    
    def _run_remote_command(self, cmd: str, capture_output: bool = True,
                           check: bool = False) -> Tuple[int, str, str]:
        """在远程主机运行命令"""
        full_cmd = ["ssh", self.ssh_host, cmd]
        print(f"[REMOTE] {cmd}")
        result = subprocess.run(full_cmd, capture_output=capture_output, text=True)
        
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, full_cmd,
                output=result.stdout, stderr=result.stderr
            )
        
        return result.returncode, result.stdout, result.stderr
    
    def _run_container_command(self, cmd: str, capture_output: bool = True) -> Tuple[int, str, str]:
        """在远程容器内运行命令"""
        container_cmd = f"docker exec {self.container} {cmd}"
        return self._run_remote_command(container_cmd, capture_output)
    
    def _validate_path(self, path: Path) -> bool:
        """验证路径是否安全"""
        try:
            path.resolve().relative_to(self.base_dir.resolve())
            return True
        except ValueError:
            return False
    
    def _generate_build_script(self, kernel_name: str, container_source: str,
                               timestamp: str) -> Path:
        """生成编译脚本"""
        script_path = self.scripts_dir / f"build_{kernel_name}_{timestamp}.sh"
        output_file = f"{self.workspace}/build_cuda/{kernel_name}.o"
        log_file = f"{self.workspace}/results/cuda/compile_{kernel_name}_{timestamp}.log"
        
        script_content = f"""#!/bin/bash
# CUDA Build Script - Generated {timestamp}
# Kernel: {kernel_name}
# Source: {container_source}

set -e
set -x

echo "=== CUDA Compilation in Remote Container ==="
echo "Timestamp: {timestamp}"
echo "Container: {self.container}"
echo "Kernel: {kernel_name}"
echo "Source: {container_source}"
echo "Compiler: {self.compiler}"
echo "Flags: {self.compiler_flags}"
echo ""

# 验证源文件存在
if [ ! -f "{container_source}" ]; then
    echo "[ERROR] Source file not found: {container_source}"
    ls -la $(dirname "{container_source}") || true
    exit 1
fi

# 创建输出目录
mkdir -p {self.workspace}/build_cuda
mkdir -p {self.workspace}/results/cuda

# 检查CUDA环境
echo "Checking CUDA environment..."
which nvcc
nvcc --version

# 编译
echo "Starting compilation..."
{self.compiler} {self.compiler_flags} \\
  -c "{container_source}" \\
  -o "{output_file}" \\
  2>&1 | tee "{log_file}"

EXIT_CODE=${{PIPESTATUS[0]}}
echo ""
echo "=== NVCC Exit Code: $EXIT_CODE ==="

if [ $EXIT_CODE -eq 0 ]; then
    echo "CUDA compilation successful!"
    if [ -f "{output_file}" ]; then
        echo "Output file:"
        ls -lh "{output_file}"
    else
        echo "[WARNING] Output file not found!"
        EXIT_CODE=1
    fi
else
    echo "CUDA compilation failed!"
fi

exit $EXIT_CODE
"""
        
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        return script_path
    
    def _update_status(self, kernel_name: str, status: str, timestamp: str,
                      source_file: str, local_log: str, script_file: str,
                      duration: float, error_msg: str = ""):
        """更新构建状态文件"""
        status_data = {}
        
        # 读取现有状态
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    status_data = json.load(f)
            except json.JSONDecodeError:
                status_data = {}
        
        # 确保基本结构
        if "metadata" not in status_data:
            status_data["metadata"] = {}
        if "environments" not in status_data:
            status_data["environments"] = {}
        if "remote_cuda" not in status_data["environments"]:
            status_data["environments"]["remote_cuda"] = {
                "type": "remote",
                "ssh_host": self.ssh_host,
                "container": self.container,
                "compiler": self.compiler,
                "kernels": {},
                "statistics": {"total": 0, "success": 0, "failed": 0}
            }
        
        # 更新元数据
        status_data["metadata"]["last_updated"] = datetime.now().isoformat()
        
        # 更新kernel状态
        kernel_info = {
            "status": status,
            "last_build": timestamp,
            "source_file": source_file,
            "local_log": local_log,
            "remote_log": f"{self.workspace}/results/cuda/compile_{kernel_name}_{timestamp}.log",
            "script_file": str(Path(script_file).relative_to(self.base_dir)),
            "duration_seconds": round(duration, 2)
        }
        
        if error_msg:
            kernel_info["error_summary"] = error_msg[:500]
        
        status_data["environments"]["remote_cuda"]["kernels"][kernel_name] = kernel_info
        
        # 更新统计
        kernels = status_data["environments"]["remote_cuda"]["kernels"]
        stats = status_data["environments"]["remote_cuda"]["statistics"]
        stats["total"] = len(kernels)
        stats["success"] = sum(1 for k in kernels.values() if k.get("status") == "success")
        stats["failed"] = sum(1 for k in kernels.values() if k.get("status") == "failed")
        
        # 写回文件
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to update status file: {e}")
    
    def check_connectivity(self) -> bool:
        """检查SSH和容器连接"""
        print(f"[CHECK] Checking connectivity to {self.ssh_host}")
        
        # 检查SSH命令
        code, _, _ = self._run_command(["which", "ssh"])
        if code != 0:
            print("[ERROR] SSH not found. Please install SSH client.")
            return False
        
        # 检查SSH连接
        code, stdout, stderr = self._run_remote_command("echo 'SSH_OK'")
        if code != 0 or "SSH_OK" not in stdout:
            print(f"[ERROR] SSH connection failed!")
            print(f"[HINT] Check: ssh {self.ssh_host}")
            print(f"[HINT] Make sure SSH key is configured: ssh-copy-id {self.ssh_host}")
            return False
        print(f"[OK] SSH connection successful")
        
        # 检查docker容器
        print(f"[CHECK] Checking container: {self.container}")
        code, stdout, stderr = self._run_remote_command(
            f"docker ps --filter 'name={self.container}' --format '{{{{.Names}}}}'"
        )
        
        if self.container not in stdout:
            print(f"[ERROR] Container '{self.container}' is not running!")
            print(f"[HINT] Start with: ssh {self.ssh_host} 'docker start {self.container}'")
            return False
        print(f"[OK] Container is running")
        
        # 检查NVCC
        print(f"[CHECK] Checking NVCC in container")
        code, stdout, stderr = self._run_container_command("which nvcc")
        if code != 0:
            print(f"[ERROR] NVCC not found in container!")
            return False
        print(f"[OK] NVCC available: {stdout.strip()}")
        
        return True
    
    def sync_to_remote(self, local_path: str) -> bool:
        """同步代码到远程主机
        
        流程:
        1. 在远程创建 /workspace/kernel_dataset/ 目录
        2. 复制 cuda/ 目录内容到 /workspace/kernel_dataset/cuda/
        3. 验证同步
        """
        print(f"[SYNC L1] Local -> Remote host...")
        
        src_path = self.base_dir / local_path
        if not src_path.exists():
            print(f"[ERROR] Source path not found: {src_path}")
            return False
        
        # 解析路径: kernel_dataset/cuda -> parent=kernel_dataset, dir=cuda
        parent_dir = str(Path(local_path).parent)  # "kernel_dataset"
        dir_name = str(Path(local_path).name)      # "cuda"
        
        # 1. 在远程创建父目录 /workspace/kernel_dataset
        remote_parent = f"{self.workspace}/{parent_dir}"
        print(f"[SYNC] Creating remote directory: {remote_parent}")
        code, _, _ = self._run_remote_command(f"mkdir -p {remote_parent}")
        if code != 0:
            print(f"[ERROR] Failed to create remote directory")
            return False
        
        # 2. 同步代码（复制目录内容，避免嵌套）
        # 使用 src_path/. 来复制目录内容
        print(f"[SYNC] Copying files...")
        code, stdout, stderr = self._run_command([
            "scp", "-r", f"{src_path}/.",
            f"{self.ssh_host}:{remote_parent}/{dir_name}"
        ])
        
        if code != 0:
            print(f"[ERROR] Failed to sync to remote: {stderr}")
            return False
        
        # 3. 验证同步
        print(f"[SYNC] Verifying remote sync at {remote_parent}/{dir_name}...")
        code, stdout, _ = self._run_remote_command(f"ls -la {remote_parent}/{dir_name}")
        if code != 0:
            print(f"[ERROR] Remote sync verification failed")
            return False
        
        file_count = len([l for l in stdout.split('\n') if l.strip() and not l.startswith('total')])
        print(f"[OK] Synced to remote host ({file_count} items)")
        return True
    
    def sync_to_container(self, local_path: str = "kernel_dataset/cuda") -> bool:
        """从远程主机同步到容器
        
        Args:
            local_path: 本地路径，如 "kernel_dataset/cuda"
        """
        print(f"[SYNC L2] Remote host -> Container...")
        
        # 解析路径
        parent_dir = str(Path(local_path).parent)  # "kernel_dataset"
        dir_name = str(Path(local_path).name)      # "cuda"
        
        # 1. 首先在容器内创建父目录
        dest_parent = f"{self.workspace}/{parent_dir}"
        print(f"[SYNC] Creating container directory: {dest_parent}")
        code, _, stderr = self._run_remote_command(
            f"docker exec {self.container} mkdir -p {dest_parent}"
        )
        
        if code != 0:
            print(f"[ERROR] Failed to create container directory: {stderr}")
            return False
        
        # 2. 复制文件（复制目录内容，避免嵌套）
        dest_dir = f"{dest_parent}/{dir_name}"
        cmd = f"docker cp {self.workspace}/{dir_name}/. {self.container}:{dest_dir}/"
        code, stdout, stderr = self._run_remote_command(cmd)
        
        if code != 0:
            print(f"[ERROR] Failed to sync to container: {stderr}")
            return False
        
        # 3. 验证同步
        print(f"[SYNC] Verifying container sync at {dest_dir}...")
        code, stdout, _ = self._run_container_command(f"ls -la {dest_dir}")
        if code != 0:
            print(f"[ERROR] Container sync verification failed")
            return False
        
        file_count = len([l for l in stdout.split('\n') if l.strip() and not l.startswith('total')])
        print(f"[OK] Synced to container ({file_count} items)")
        return True
    
    def compile_single(self, source_file: str) -> bool:
        """编译单个kernel文件"""
        source_path = Path(source_file)
        
        # 验证路径安全
        if not self._validate_path(source_path):
            print(f"[ERROR] Invalid source file path: {source_file}")
            return False
        
        if not source_path.exists():
            print(f"[ERROR] Source file not found: {source_file}")
            return False
        
        kernel_name = source_path.stem
        timestamp = self._get_timestamp()
        
        print(f"\n{'='*60}")
        print(f"[BUILD] Remote CUDA Compilation: {kernel_name}")
        print(f"[BUILD] Source: {source_file}")
        print(f"[BUILD] Timestamp: {timestamp}")
        print(f"[BUILD] Host: {self.ssh_host}")
        print(f"[BUILD] Container: {self.container}")
        print(f"{'='*60}\n")
        
        # 1. 检查连接
        if not self.check_connectivity():
            return False
        
        start_time = datetime.now()
        
        try:
            # 2. 生成编译脚本
            container_source = f"{self.workspace}/kernel_dataset/cuda/{source_path.name}"
            script_file = self._generate_build_script(kernel_name, container_source, timestamp)
            print(f"[SCRIPT] Generated: {script_file}")
            
            # 3. 同步到远程主机
            if not self.sync_to_remote("kernel_dataset/cuda"):
                return False
            
            # 4. 同步到容器
            if not self.sync_to_container("kernel_dataset/cuda"):
                return False
            
            # 5. 复制脚本到远程
            remote_script_dir = f"{self.workspace}/scripts/cuda"
            remote_script_path = f"{remote_script_dir}/build_{kernel_name}_{timestamp}.sh"
            
            # 创建远程脚本目录
            code, _, _ = self._run_remote_command(f"mkdir -p {remote_script_dir}")
            
            code, _, stderr = self._run_command([
                "scp", str(script_file),
                f"{self.ssh_host}:{remote_script_path}"
            ])
            
            if code != 0:
                print(f"[ERROR] Failed to copy script to remote: {stderr}")
                return False
            
            # 复制到容器
            container_script = f"{self.workspace}/build_{kernel_name}_{timestamp}.sh"
            code, _, stderr = self._run_remote_command(
                f"docker cp {remote_script_path} {self.container}:{container_script}"
            )
            
            if code != 0:
                print(f"[ERROR] Failed to copy script to container: {stderr}")
                return False
            
            # 6. 在容器内执行编译
            log_file = self.results_dir / f"compile_{kernel_name}_{timestamp}.log"
            print(f"[BUILD] Starting remote compilation...")
            
            code, stdout, stderr = self._run_remote_command(
                f"docker exec {self.container} bash {container_script}"
            )
            
            # 保存日志
            with open(log_file, 'w') as f:
                f.write(f"Exit code: {code}\n")
                f.write("=== STDOUT ===\n")
                f.write(stdout)
                if stderr:
                    f.write("\n=== STDERR ===\n")
                    f.write(stderr)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # 7. 回传编译产物
            if code == 0:
                print(f"[SUCCESS] Compilation completed in {duration:.2f}s")
                print(f"[LOG] Saved: {log_file}")
                
                # 回传build目录
                print(f"[FETCH] Retrieving build artifacts...")
                self._run_remote_command(
                    f"docker cp {self.container}:{self.workspace}/build_cuda/. "
                    f"{self.workspace}/results/cuda/"
                )
                self._run_command([
                    "scp", "-r",
                    f"{self.ssh_host}:{self.workspace}/results/cuda/*",
                    str(self.results_dir)
                ])
                
                # 验证输出文件
                output_file = self.results_dir / "build_cuda" / f"{kernel_name}.o"
                if output_file.exists():
                    print(f"[OUTPUT] {output_file}")
                
                self._update_status(
                    kernel_name, "success", timestamp,
                    str(source_path),
                    str(log_file),
                    str(script_file),
                    duration
                )
                return True
            else:
                print(f"[FAILED] Compilation failed (exit code: {code})")
                print(f"[LOG] Saved: {log_file}")
                
                error_msg = stderr if stderr else stdout
                self._update_status(
                    kernel_name, "failed", timestamp,
                    str(source_path),
                    str(log_file),
                    str(script_file),
                    duration,
                    error_msg
                )
                return False
                
        except Exception as e:
            print(f"[ERROR] Unexpected error during compilation: {e}")
            return False
    
    def compile_all(self) -> Dict[str, bool]:
        """批量编译所有CUDA kernel"""
        cuda_dir = self.base_dir / "kernel_dataset" / "cuda"
        
        if not cuda_dir.exists():
            print(f"[ERROR] Directory not found: {cuda_dir}")
            return {}
        
        # 获取所有.cu文件
        kernels = list(cuda_dir.glob("*.cu"))
        if not kernels:
            print(f"[WARNING] No .cu files found in {cuda_dir}")
            return {}
        
        print(f"[BATCH] Found {len(kernels)} kernels to compile\n")
        
        results = {}
        batch_timestamp = self._get_timestamp()
        batch_log = self.results_dir / f"batch_status_{batch_timestamp}.jsonl"
        
        for i, kernel_file in enumerate(kernels, 1):
            print(f"\n[{i}/{len(kernels)}] Processing: {kernel_file.name}")
            success = self.compile_single(str(kernel_file.relative_to(self.base_dir)))
            results[kernel_file.stem] = success
            
            # 记录批量状态
            try:
                with open(batch_log, 'a') as f:
                    json.dump({
                        "kernel": kernel_file.stem,
                        "status": "success" if success else "failed",
                        "time": datetime.now().isoformat(),
                        "index": i,
                        "total": len(kernels)
                    }, f)
                    f.write('\n')
            except Exception as e:
                print(f"[WARNING] Failed to write batch log: {e}")
        
        # 生成汇总报告
        self._generate_summary_report(results, batch_timestamp)
        
        return results
    
    def _generate_summary_report(self, results: Dict[str, bool], timestamp: str):
        """生成批量编译汇总报告"""
        summary_file = self.results_dir / f"summary_{timestamp}.json"
        
        success_count = sum(1 for v in results.values() if v)
        failed_count = len(results) - success_count
        
        summary = {
            "timestamp": timestamp,
            "environment": "remote_cuda",
            "host": self.ssh_host,
            "container": self.container,
            "total": len(results),
            "success": success_count,
            "failed": failed_count,
            "success_rate": f"{(success_count/len(results)*100):.1f}%" if results else "0%",
            "results": results,
            "generated_at": datetime.now().isoformat()
        }
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to write summary: {e}")
        
        print(f"\n{'='*60}")
        print(f"[SUMMARY] Remote CUDA Batch Compilation Report")
        print(f"{'='*60}")
        print(f"Host:      {self.ssh_host}")
        print(f"Container: {self.container}")
        print(f"Total:     {len(results)}")
        print(f"Success:   {success_count}")
        print(f"Failed:    {failed_count}")
        print(f"Rate:      {summary['success_rate']}")
        print(f"Report:    {summary_file}")
        print(f"{'='*60}\n")
    
    def get_status(self) -> Dict:
        """获取当前构建状态"""
        if not self.status_file.exists():
            return {}
        
        try:
            with open(self.status_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to read status file: {e}")
            return {}
    
    def print_status(self):
        """打印构建状态"""
        status = self.get_status()
        
        if not status or "environments" not in status:
            print("No build status available yet.")
            return
        
        cuda_status = status.get("environments", {}).get("remote_cuda", {})
        kernels = cuda_status.get("kernels", {})
        stats = cuda_status.get("statistics", {})
        
        print(f"\n{'='*60}")
        print(f"Remote CUDA Build Status")
        print(f"{'='*60}")
        print(f"SSH Host:  {cuda_status.get('ssh_host', 'N/A')}")
        print(f"Container: {cuda_status.get('container', 'N/A')}")
        print(f"Compiler:  {cuda_status.get('compiler', 'N/A')}")
        print(f"Total:     {stats.get('total', 0)}")
        print(f"Success:   {stats.get('success', 0)}")
        print(f"Failed:    {stats.get('failed', 0)}")
        print(f"{'='*60}\n")
        
        # 打印失败的kernel
        failed = [(name, info) for name, info in kernels.items() 
                  if info.get("status") == "failed"]
        if failed:
            print("Failed kernels:")
            for name, info in failed:
                error = info.get('error_summary', 'Unknown error')[:100]
                print(f"  - {name}: {error}")
            print()

def main():
    parser = argparse.ArgumentParser(
        description="Remote CUDA Builder - Compile CUDA kernels in remote docker container",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s compile kernel_dataset/cuda/add_vectors.cu
  %(prog)s compile-all
  %(prog)s status
        """
    )
    parser.add_argument(
        "command",
        choices=["compile", "compile-all", "status"],
        help="Command to execute"
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Kernel file to compile (for 'compile' command)"
    )
    
    args = parser.parse_args()
    
    builder = RemoteCudaBuilder()
    
    if args.command == "compile":
        if not args.file:
            print("[ERROR] Please specify a file to compile")
            print("Usage: compile <file_path>")
            sys.exit(1)
        success = builder.compile_single(args.file)
        sys.exit(0 if success else 1)
    
    elif args.command == "compile-all":
        results = builder.compile_all()
        failed = sum(1 for v in results.values() if not v)
        sys.exit(0 if failed == 0 else 1)
    
    elif args.command == "status":
        builder.print_status()
        sys.exit(0)

if __name__ == "__main__":
    main()
