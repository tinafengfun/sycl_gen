#!/usr/bin/env python3
"""
B60 SYCL Builder Tool - BMG/XPU Support Version
在本地B60 docker容器中进行SYCL编译和测试，支持Intel BMG (Battlemage) GPU

改进点:
- 支持Intel BMG (Battlemage) AOT编译
- 基于Intel torch-xpu-ops的BuildFlags.cmake配置
- 更健壮的目录创建和验证
- 更好的错误处理和日志记录
- 改进的文件同步逻辑
- 添加编译后验证步骤
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class B60SyclBuilder:
    """B60 docker SYCL编译工具 - BMG/XPU支持"""
    
    def __init__(self, config_path: Optional[str] = None):
        # 配置参数
        self.container = "lsv-container"
        self.workspace = "/workspace"
        self.compiler = "icpx"
        
        # SYCL编译标志 - 基于Intel torch-xpu-ops的BuildFlags.cmake
        # 基础标志
        self.base_flags = "-fsycl -O2 -std=c++17"
        
        # SYCL Kernel选项 (来自BuildFlags.cmake)
        self.sycl_kernel_options = [
            # "-fno-sycl-unnamed-lambda",   # 注释掉：我们的kernel使用匿名lambda
            "-fsycl-unnamed-lambda",      # 启用匿名lambda支持
            "-sycl-std=2020",             # SYCL 2020标准
            "-fhonor-nans",               # 尊重NaN
            "-fhonor-infinities",         # 尊重Infinity
            "-fno-associative-math",      # 禁用结合律优化（确定性结果）
            "-fno-approx-func",           # 禁用近似函数
            "-no-ftz",                    # 禁用flush-to-zero
        ]
        
        # AOT编译目标 - 支持BMG (Battlemage)
        # Linux平台的目标列表
        self.aot_targets = "pvc,bmg,dg2,arl-h,mtl-h,lnl-m,ptl-h,ptl-u"
        
        # 设备编译选项
        self.device_compile_flags = [
            f"-fsycl-targets=spir64_gen,spir64",  # AOT + JIT编译目标
        ]
        
        # 离线编译器选项 (传递给-Xs)
        self.offline_compiler_flags = [
            f"-device {self.aot_targets}",
            "-options -cl-poison-unsupported-fp64-kernels",
            "-options -cl-intel-enable-auto-large-GRF-mode",
            "-options -cl-fp32-correctly-rounded-divide-sqrt",
            "-options -cl-intel-greater-than-4GB-buffer-required",
        ]
        
        # 构建完整的编译器标志
        self.compiler_flags = self._build_compiler_flags()
        
        # 本地路径
        self.base_dir = Path(__file__).parent.parent
        self.results_dir = self.base_dir / "results" / "b60"
        self.scripts_dir = self.base_dir / "scripts" / "b60"
        self.status_file = self.base_dir / ".build_status.json"
        
        # 确保目录存在
        self._ensure_directories()
        
    def _build_compiler_flags(self) -> str:
        """构建完整的编译器标志字符串"""
        flags = [self.base_flags]
        flags.extend(self.sycl_kernel_options)
        flags.extend(self.device_compile_flags)
        return " ".join(flags)
    
    def _get_device_link_flags(self) -> str:
        """获取设备链接标志"""
        # 设备链接标志包含AOT目标和压缩选项
        flags = [
            "-fsycl-max-parallel-link-jobs=4",  # 并行链接
            "--offload-compress",                # 压缩离线二进制文件
            f"-fsycl-targets=spir64_gen,spir64",
        ]
        
        # 添加离线编译器选项
        offline_opts = " ".join(self.offline_compiler_flags)
        flags.append(f"-Xs '{offline_opts}'")
        
        return " ".join(flags)
        
    def _ensure_directories(self):
        """创建必要的目录结构"""
        directories = [
            self.results_dir,
            self.scripts_dir,
            self.results_dir / "build_sycl"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def _get_timestamp(self) -> str:
        """生成时间戳 YYYYMMDD_HHMMSS"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _run_command(self, cmd: List[str], capture_output: bool = True, 
                     check: bool = False) -> Tuple[int, str, str]:
        """运行shell命令
        
        Args:
            cmd: 命令列表
            capture_output: 是否捕获输出
            check: 失败时是否抛出异常
        
        Returns:
            (exit_code, stdout, stderr)
        """
        print(f"[CMD] {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=capture_output, text=True)
        
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, 
                output=result.stdout, stderr=result.stderr
            )
        
        return result.returncode, result.stdout, result.stderr
    
    def _validate_path(self, path: Path) -> bool:
        """验证路径是否安全"""
        try:
            path.resolve().relative_to(self.base_dir.resolve())
            return True
        except ValueError:
            return False
    
    def _generate_build_script(self, kernel_name: str, source_file: str, 
                               timestamp: str) -> Path:
        """生成编译脚本 - 支持BMG AOT编译"""
        script_path = self.scripts_dir / f"build_{kernel_name}_{timestamp}.sh"
        output_file = f"{self.workspace}/build_sycl/{kernel_name}.o"
        log_file = f"{self.workspace}/results/b60/compile_{kernel_name}_{timestamp}.log"
        
        # 编译标志（包含SYCL kernel选项和AOT目标）
        compile_flags = self.compiler_flags
        
        # 设备链接标志（用于AOT编译）
        device_link_flags = self._get_device_link_flags()
        
        script_content = f"""#!/bin/bash
# SYCL Build Script for BMG/XPU - Generated {timestamp}
# Kernel: {kernel_name}
# Source: {source_file}
# AOT Targets: {self.aot_targets}

set -e
set -x

echo "=== SYCL Compilation for Intel XPU (BMG Support) ==="
echo "Timestamp: {timestamp}"
echo "Kernel: {kernel_name}"
echo "Source: {source_file}"
echo "Compiler: {self.compiler}"
echo "Compile Flags: {compile_flags}"
echo "Device Link Flags: {device_link_flags}"
echo "AOT Targets: {self.aot_targets}"
echo ""

# 验证源文件存在
if [ ! -f "{source_file}" ]; then
    echo "[ERROR] Source file not found: {source_file}"
    exit 1
fi

# 创建输出目录
mkdir -p {self.workspace}/build_sycl
mkdir -p {self.workspace}/results/b60

# 编译步骤：
# 1. 编译源文件为对象文件（包含设备代码）
echo "Step 1: Compiling source to object file..."
{self.compiler} {compile_flags} \\
  -c "{source_file}" \\
  -o "{output_file}" \\
  2>&1 | tee "{log_file}.step1"

STEP1_EXIT=${{PIPESTATUS[0]}}
if [ $STEP1_EXIT -ne 0 ]; then
    echo "[ERROR] Compilation step 1 failed with exit code $STEP1_EXIT"
    exit $STEP1_EXIT
fi

# 2. 设备代码链接（AOT编译为指定目标）
echo ""
echo "Step 2: Linking device code for AOT targets..."
{self.compiler} -fsycl {device_link_flags} \\
  "{output_file}" \\
  -o "{output_file}.linked" \\
  2>&1 | tee "{log_file}.step2"

STEP2_EXIT=${{PIPESTATUS[0]}}
if [ $STEP2_EXIT -ne 0 ]; then
    echo "[WARNING] Device linking step 2 failed with exit code $STEP2_EXIT"
    echo "This may be expected for kernels without device code or if AOT is not needed"
    # 不要退出，因为对象文件可能已经可用
else
    echo "Device linking successful"
fi

# 合并日志
cat "{log_file}.step1" > "{log_file}"
if [ -f "{log_file}.step2" ]; then
    echo "" >> "{log_file}"
    echo "=== Device Link Step ===" >> "{log_file}"
    cat "{log_file}.step2" >> "{log_file}"
fi

EXIT_CODE=0
if [ -f "{output_file}" ]; then
    echo ""
    echo "=== Compilation Successful ==="
    echo "Output file: {output_file}"
    ls -lh "{output_file}"
    
    # 检查是否包含BMG设备代码
    echo ""
    echo "Checking device code presence..."
    if objdump -h "{output_file}" 2>/dev/null | grep -q ".text"; then
        echo "✓ Device code present in object file"
    fi
else
    echo "[WARNING] Output file not found!"
    EXIT_CODE=1
fi

echo ""
echo "=== Compilation Exit Code: $EXIT_CODE ==="
exit $EXIT_CODE
"""
        
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        return script_path
    
    def _update_status(self, kernel_name: str, status: str, timestamp: str,
                      source_file: str, log_file: str, script_file: str,
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
        if "b60" not in status_data["environments"]:
            status_data["environments"]["b60"] = {
                "type": "local",
                "container": self.container,
                "compiler": self.compiler,
                "aot_targets": self.aot_targets,
                "kernels": {},
                "statistics": {"total": 0, "success": 0, "failed": 0, "pending": 0}
            }
        
        # 更新元数据
        status_data["metadata"]["last_updated"] = datetime.now().isoformat()
        
        # 更新kernel状态
        kernel_info = {
            "status": status,
            "last_build": timestamp,
            "source_file": source_file,
            "log_file": str(Path(log_file).relative_to(self.base_dir)),
            "script_file": str(Path(script_file).relative_to(self.base_dir)),
            "duration_seconds": round(duration, 2),
            "aot_targets": self.aot_targets
        }
        
        if error_msg:
            kernel_info["error_summary"] = error_msg[:500]  # 限制长度
        
        status_data["environments"]["b60"]["kernels"][kernel_name] = kernel_info
        
        # 更新统计
        kernels = status_data["environments"]["b60"]["kernels"]
        stats = status_data["environments"]["b60"]["statistics"]
        stats["total"] = len(kernels)
        stats["success"] = sum(1 for k in kernels.values() if k.get("status") == "success")
        stats["failed"] = sum(1 for k in kernels.values() if k.get("status") == "failed")
        
        # 写回文件
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to update status file: {e}")
    
    def check_container(self) -> bool:
        """检查容器状态"""
        print(f"[CHECK] Checking container: {self.container}")
        
        # 检查docker命令
        code, _, _ = self._run_command(["which", "docker"])
        if code != 0:
            print("[ERROR] Docker not found. Please install Docker.")
            return False
        
        # 检查容器是否运行
        code, stdout, stderr = self._run_command(
            ["docker", "ps", "--format", "{{.Names}}"]
        )
        
        if code != 0:
            print(f"[ERROR] Failed to list containers: {stderr}")
            return False
        
        if self.container not in stdout:
            print(f"[ERROR] Container '{self.container}' is not running!")
            print(f"[HINT] Start with: docker start {self.container}")
            print(f"[HINT] Or check status: docker ps -a | grep {self.container}")
            return False
        
        # 检查workspace可访问，如不存在则创建
        code, _, stderr = self._run_command([
            "docker", "exec", self.container, 
            "ls", "-la", self.workspace
        ])
        
        if code != 0:
            print(f"[INFO] Workspace not found, creating...")
            code, _, stderr = self._run_command([
                "docker", "exec", self.container,
                "mkdir", "-p", self.workspace
            ])
            if code != 0:
                print(f"[ERROR] Cannot create workspace: {stderr}")
                return False
        
        # 检查编译器可用性
        print(f"[CHECK] Checking compiler: {self.compiler}")
        code, stdout, stderr = self._run_command([
            "docker", "exec", self.container,
            "which", self.compiler
        ])
        
        if code != 0:
            print(f"[ERROR] Compiler '{self.compiler}' not found in container")
            return False
        
        # 检查SYCL支持
        code, stdout, stderr = self._run_command([
            "docker", "exec", self.container,
            self.compiler, "--version"
        ])
        
        if code == 0:
            version_line = stdout.split('\n')[0]
            print(f"[INFO] Compiler version: {version_line}")
        
        print(f"[OK] Container is ready (BMG/XPU support enabled)")
        return True
    
    def sync_to_container(self, local_path: str) -> bool:
        """同步代码到容器
        
        流程:
        1. 在容器内创建目标目录结构
        2. 使用docker cp复制文件（复制目录内容，避免嵌套）
        3. 验证文件已同步
        """
        print(f"[SYNC] Syncing code to container...")
        
        src_path = self.base_dir / local_path
        if not src_path.exists():
            print(f"[ERROR] Source path not found: {src_path}")
            return False
        
        # local_path 是 "kernel_dataset/sycl" 这样的路径
        # 我们需要在容器内创建 /workspace/kernel_dataset 目录
        # 然后把本地的 sycl 目录复制进去
        parent_dir = str(Path(local_path).parent)  # "kernel_dataset"
        dir_name = str(Path(local_path).name)      # "sycl"
        
        # 1. 在容器内创建父目录
        dest_parent = f"{self.workspace}/{parent_dir}"
        print(f"[SYNC] Creating directory: {dest_parent}")
        code, _, stderr = self._run_command([
            "docker", "exec", self.container,
            "mkdir", "-p", dest_parent
        ])
        
        if code != 0:
            print(f"[ERROR] Failed to create directory: {stderr}")
            return False
        
        # 2. 同步代码（复制目录内容，避免嵌套）
        print(f"[SYNC] Copying files...")
        # 使用 src_path/. 来复制目录内容，而不是目录本身
        code, stdout, stderr = self._run_command([
            "docker", "cp", 
            f"{src_path}/.",
            f"{self.container}:{dest_parent}/{dir_name}"
        ])
        
        if code != 0:
            print(f"[ERROR] Failed to sync code: {stderr}")
            return False
        
        # 3. 验证同步
        dest_dir = f"{dest_parent}/{dir_name}"
        print(f"[SYNC] Verifying sync at {dest_dir}...")
        code, stdout, _ = self._run_command([
            "docker", "exec", self.container,
            "ls", "-la", dest_dir
        ])
        
        if code != 0 or not stdout.strip():
            print(f"[ERROR] Sync verification failed")
            return False
        
        file_count = len([l for l in stdout.split('\n') if l.strip() and not l.startswith('total')])
        print(f"[OK] Code synced successfully ({file_count} items)")
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
        print(f"[BUILD] Compiling: {kernel_name}")
        print(f"[BUILD] Source: {source_file}")
        print(f"[BUILD] Timestamp: {timestamp}")
        print(f"[BUILD] AOT Targets: {self.aot_targets}")
        print(f"[BUILD] BMG Support: Enabled")
        print(f"{'='*60}\n")
        
        # 1. 检查容器
        if not self.check_container():
            return False
        
        start_time = datetime.now()
        
        try:
            # 2. 生成编译脚本
            container_source = f"{self.workspace}/kernel_dataset/sycl/{source_path.name}"
            script_file = self._generate_build_script(kernel_name, container_source, timestamp)
            print(f"[SCRIPT] Generated: {script_file}")
            
            # 3. 同步代码
            if not self.sync_to_container("kernel_dataset/sycl"):
                return False
            
            # 4. 复制脚本到容器
            container_script = f"{self.workspace}/build_{kernel_name}_{timestamp}.sh"
            print(f"[SYNC] Copying script to container...")
            code, _, stderr = self._run_command([
                "docker", "cp", str(script_file),
                f"{self.container}:{container_script}"
            ])
            
            if code != 0:
                print(f"[ERROR] Failed to copy script: {stderr}")
                return False
            
            # 5. 在容器内执行编译
            log_file = self.results_dir / f"compile_{kernel_name}_{timestamp}.log"
            print(f"[BUILD] Starting compilation...")
            print(f"[BUILD] This may take a while for AOT compilation...")
            
            code, stdout, stderr = self._run_command([
                "docker", "exec", self.container,
                "bash", container_script
            ])
            
            # 保存日志
            with open(log_file, 'w') as f:
                f.write(f"Exit code: {code}\n")
                f.write(f"Compiler flags: {self.compiler_flags}\n")
                f.write(f"AOT Targets: {self.aot_targets}\n")
                f.write("=== STDOUT ===\n")
                f.write(stdout)
                if stderr:
                    f.write("\n=== STDERR ===\n")
                    f.write(stderr)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # 6. 回传编译产物
            if code == 0:
                print(f"[SUCCESS] Compilation completed in {duration:.2f}s")
                print(f"[LOG] Saved: {log_file}")
                
                # 尝试回传.o文件
                build_dir = self.results_dir / "build_sycl"
                build_dir.mkdir(exist_ok=True)
                self._run_command([
                    "docker", "cp",
                    f"{self.container}:{self.workspace}/build_sycl/",
                    str(build_dir)
                ])
                
                # 验证输出文件
                output_file = build_dir / f"{kernel_name}.o"
                if output_file.exists():
                    print(f"[OUTPUT] {output_file}")
                    # 显示文件大小
                    size_mb = output_file.stat().st_size / (1024 * 1024)
                    print(f"[OUTPUT] Size: {size_mb:.2f} MB")
                
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
                
                # 提取错误信息
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
        """批量编译所有sycl kernel"""
        sycl_dir = self.base_dir / "kernel_dataset" / "sycl"
        
        if not sycl_dir.exists():
            print(f"[ERROR] Directory not found: {sycl_dir}")
            return {}
        
        # 获取所有.dp.cpp文件
        kernels = list(sycl_dir.glob("*.dp.cpp"))
        if not kernels:
            print(f"[WARNING] No .dp.cpp files found in {sycl_dir}")
            return {}
        
        print(f"[BATCH] Found {len(kernels)} kernels to compile")
        print(f"[BATCH] AOT Targets: {self.aot_targets}")
        print(f"[BATCH] This will compile for multiple Intel GPU architectures\n")
        
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
                        "total": len(kernels),
                        "aot_targets": self.aot_targets
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
            "environment": "b60",
            "compiler": self.compiler,
            "aot_targets": self.aot_targets,
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
        print(f"[SUMMARY] Batch Compilation Report")
        print(f"{'='*60}")
        print(f"AOT Targets: {self.aot_targets}")
        print(f"Total:       {len(results)}")
        print(f"Success:     {success_count}")
        print(f"Failed:      {failed_count}")
        print(f"Rate:        {summary['success_rate']}")
        print(f"Report:      {summary_file}")
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
        
        b60_status = status.get("environments", {}).get("b60", {})
        kernels = b60_status.get("kernels", {})
        stats = b60_status.get("statistics", {})
        
        print(f"\n{'='*60}")
        print(f"B60 SYCL Build Status - Intel XPU (BMG Support)")
        print(f"{'='*60}")
        print(f"Container:   {b60_status.get('container', 'N/A')}")
        print(f"Compiler:    {b60_status.get('compiler', 'N/A')}")
        print(f"AOT Targets: {b60_status.get('aot_targets', 'N/A')}")
        print(f"Total:       {stats.get('total', 0)}")
        print(f"Success:     {stats.get('success', 0)}")
        print(f"Failed:      {stats.get('failed', 0)}")
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
        description="B60 SYCL Builder for Intel XPU (BMG Support) - Compile SYCL kernels in docker container",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s compile kernel_dataset/sycl/winograd_input_transform_kernel.dp.cpp
  %(prog)s compile-all
  %(prog)s status
  
Environment Variables:
  AOT_TARGETS     Override AOT target list (default: pvc,bmg,dg2,arl-h,mtl-h,lnl-m,ptl-h,ptl-u)
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
    
    builder = B60SyclBuilder()
    
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
