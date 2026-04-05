#!/usr/bin/env python3
"""
Progress Monitor & Test Executor
进度监控与测试执行器

核心功能：
1. 实时进度监控
2. 异步并行执行
3. 编译错误自动修复（LLM驱动）
4. 完整结果收集
"""

import asyncio
import time
import json
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import sys

sys.path.insert(0, str(Path(__file__).parent))

@dataclass
class TestPhase:
    """测试阶段信息"""
    name: str
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_seconds: float = 0.0
    details: Dict = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class TestProgress:
    """单个测试的进度信息"""
    test_id: str
    test_name: str
    status: str = "pending"
    phases: Dict[str, TestPhase] = field(default_factory=dict)
    current_phase: Optional[str] = None
    overall_progress: float = 0.0
    
    def start_phase(self, phase_name: str):
        """开始新阶段"""
        self.current_phase = phase_name
        self.phases[phase_name] = TestPhase(
            name=phase_name,
            status="running",
            start_time=time.time()
        )
        self.status = "running"
    
    def end_phase(self, phase_name: str, success: bool = True, details: Dict = None, error: str = None):
        """结束阶段"""
        if phase_name in self.phases:
            phase = self.phases[phase_name]
            phase.status = "completed" if success else "failed"
            phase.end_time = time.time()
            phase.duration_seconds = phase.end_time - phase.start_time
            if details:
                phase.details.update(details)
            if error:
                phase.error = error
    
    def get_duration(self) -> float:
        """获取总持续时间"""
        total = 0.0
        for phase in self.phases.values():
            total += phase.duration_seconds
        return total


class ProgressMonitor:
    """进度监控器"""
    
    def __init__(self, callback: Optional[Callable] = None):
        self.tests: Dict[str, TestProgress] = {}
        self.start_time = time.time()
        self.callback = callback or self._default_callback
        self.completed_tests = 0
        self.failed_tests = 0
    
    def _default_callback(self, progress: TestProgress):
        """默认进度回调 - 打印到控制台"""
        phase_progress_map = {
            "harness_generation": 10,
            "cuda_compilation": 25,
            "sycl_compilation": 40,
            "cuda_execution": 60,
            "sycl_execution": 80,
            "result_comparison": 95,
            "completed": 100
        }
        
        current = progress.current_phase
        percent = phase_progress_map.get(current, 0) if current else 0
        
        # 计算整体进度
        total_tests = len(self.tests)
        if total_tests > 0:
            completed_progress = sum(
                100 if t.status == "completed" else 
                phase_progress_map.get(t.current_phase, 0) if t.current_phase else 0
                for t in self.tests.values()
            ) / total_tests
        else:
            completed_progress = 0
        
        # 打印进度条
        bar_length = 30
        filled = int(bar_length * completed_progress / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        elapsed = time.time() - self.start_time
        
        print(f"\r⏳ [{bar}] {completed_progress:.1f}% | "
              f"Test: {progress.test_name[:20]:20} | "
              f"Phase: {current or 'waiting':15} | "
              f"Elapsed: {elapsed:.1f}s", end='', flush=True)
    
    def start_test(self, test_id: str, test_name: str):
        """开始新测试"""
        self.tests[test_id] = TestProgress(
            test_id=test_id,
            test_name=test_name
        )
    
    def update_phase(self, test_id: str, phase_name: str, 
                    status: str = "running", details: Dict = None):
        """更新阶段状态"""
        if test_id in self.tests:
            progress = self.tests[test_id]
            
            if status == "running" and phase_name not in progress.phases:
                progress.start_phase(phase_name)
            elif status in ["completed", "failed"]:
                success = (status == "completed")
                progress.end_phase(phase_name, success=success, details=details)
                
                if phase_name == "completed":
                    progress.status = "completed"
                    self.completed_tests += 1
                elif phase_name == "failed":
                    progress.status = "failed"
                    self.failed_tests += 1
            
            # 触发回调
            self.callback(progress)
    
    def get_summary(self) -> Dict:
        """获取进度摘要"""
        total = len(self.tests)
        completed = sum(1 for t in self.tests.values() if t.status == "completed")
        failed = sum(1 for t in self.tests.values() if t.status == "failed")
        running = total - completed - failed
        
        # 计算平均进度
        if total > 0:
            avg_progress = sum(
                100 if t.status == "completed" else 50
                for t in self.tests.values()
            ) / total
        else:
            avg_progress = 0
        
        return {
            "total_tests": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "average_progress": avg_progress,
            "elapsed_time": time.time() - self.start_time,
            "tests": {
                tid: {
                    "name": t.test_name,
                    "status": t.status,
                    "current_phase": t.current_phase,
                    "duration": t.get_duration()
                }
                for tid, t in self.tests.items()
            }
        }


class AsyncTestExecutor:
    """异步测试执行器"""
    
    def __init__(self, monitor: ProgressMonitor, max_llm_concurrency: int = 4):
        self.monitor = monitor
        self.llm_semaphore = asyncio.Semaphore(max_llm_concurrency)
        self.cuda_semaphore = asyncio.Semaphore(1)  # 单GPU
        self.sycl_semaphore = asyncio.Semaphore(1)  # 单GPU
        self.base_dir = Path(__file__).parent.parent
        self.remote_host = "root@10.112.229.160"
        
        # 导入LLM harness生成器
        from llm_harness_generator import LLMHarnessGenerator
        self.harness_generator = LLMHarnessGenerator()
    
    async def execute_test(self, test_id: str, kernel_code: str, 
                          test_config: Dict) -> Dict:
        """
        执行单个完整测试
        
        Returns:
            测试结果字典
        """
        test_name = test_config.get("name", test_id)
        self.monitor.start_test(test_id, test_name)
        
        result = {
            "test_id": test_id,
            "test_name": test_name,
            "status": "pending",
            "phases": {},
            "error": None
        }
        
        try:
            # Phase 1: 生成测试harness
            self.monitor.update_phase(test_id, "harness_generation", "running")
            
            async with self.llm_semaphore:
                harness = await self.harness_generator.generate_full_harness(
                    kernel_code, test_config
                )
            
            if not harness.success:
                raise Exception("Harness generation failed")
            
            self.monitor.update_phase(test_id, "harness_generation", "completed", {
                "llm_calls": harness.llm_calls,
                "generation_time": harness.generation_time,
                "cuda_lines": len(harness.cuda_code.split('\n')),
                "sycl_lines": len(harness.sycl_code.split('\n'))
            })
            
            result["phases"]["harness_generation"] = {
                "status": "completed",
                "llm_calls": harness.llm_calls,
                "duration": harness.generation_time
            }
            
            # Phase 2: 编译CUDA代码
            self.monitor.update_phase(test_id, "cuda_compilation", "running")
            
            async with self.cuda_semaphore:
                cuda_success = await self._compile_cuda_with_retry(
                    harness.cuda_code, test_id
                )
            
            if not cuda_success:
                raise Exception("CUDA compilation failed after retries")
            
            self.monitor.update_phase(test_id, "cuda_compilation", "completed")
            result["phases"]["cuda_compilation"] = {"status": "completed"}
            
            # Phase 3: 编译SYCL代码
            self.monitor.update_phase(test_id, "sycl_compilation", "running")
            
            async with self.sycl_semaphore:
                sycl_success = await self._compile_sycl_with_retry(
                    harness.sycl_code, test_id
                )
            
            if not sycl_success:
                raise Exception("SYCL compilation failed after retries")
            
            self.monitor.update_phase(test_id, "sycl_compilation", "completed")
            result["phases"]["sycl_compilation"] = {"status": "completed"}
            
            # Phase 4: 执行CUDA测试
            self.monitor.update_phase(test_id, "cuda_execution", "running")
            
            async with self.cuda_semaphore:
                cuda_output = await self._execute_cuda_test(test_id)
            
            if cuda_output is None:
                raise Exception("CUDA execution failed")
            
            self.monitor.update_phase(test_id, "cuda_execution", "completed", {
                "output_size": len(cuda_output) * 4  # float32 = 4 bytes
            })
            result["phases"]["cuda_execution"] = {"status": "completed"}
            
            # Phase 5: 执行SYCL测试
            self.monitor.update_phase(test_id, "sycl_execution", "running")
            
            async with self.sycl_semaphore:
                sycl_output = await self._execute_sycl_test(test_id)
            
            if sycl_output is None:
                raise Exception("SYCL execution failed")
            
            self.monitor.update_phase(test_id, "sycl_execution", "completed", {
                "output_size": len(sycl_output) * 4
            })
            result["phases"]["sycl_execution"] = {"status": "completed"}
            
            # Phase 6: 对比结果
            self.monitor.update_phase(test_id, "result_comparison", "running")
            
            comparison = self._compare_results(
                cuda_output, sycl_output, test_config
            )
            
            self.monitor.update_phase(test_id, "result_comparison", "completed")
            self.monitor.update_phase(test_id, "completed", "completed")
            
            result["phases"]["result_comparison"] = comparison
            result["status"] = "PASSED" if comparison["pass"] else "FAILED"
            result["comparison"] = comparison
            
        except Exception as e:
            self.monitor.update_phase(test_id, "failed", "failed", 
                                     {"error": str(e)})
            result["status"] = "FAILED"
            result["error"] = str(e)
            result["phases"]["error"] = {"message": str(e)}
        
        return result
    
    async def _compile_cuda_with_retry(self, code: str, test_id: str, 
                                      max_retries: int = 3) -> bool:
        """编译CUDA代码，带自动修复重试"""
        for attempt in range(max_retries):
            # 保存代码
            cu_file = f"/tmp/{test_id}_test.cu"
            with open(cu_file, 'w') as f:
                f.write(code)
            
            # 尝试编译
            success = await self._compile_cuda_remote(cu_file, test_id)
            
            if success:
                return True
            
            if attempt < max_retries - 1:
                print(f"\n  ⚠️  CUDA compilation failed, attempt {attempt + 1}/{max_retries}")
                print(f"  🔧 Trying to fix with LLM...")
                
                # 获取错误信息
                error_msg = await self._get_cuda_compile_error(cu_file, test_id)
                
                # 使用LLM修复
                async with self.llm_semaphore:
                    code = await self.harness_generator.fix_compilation_error(
                        code, error_msg, "cuda"
                    )
        
        return False
    
    async def _compile_sycl_with_retry(self, code: str, test_id: str,
                                      max_retries: int = 3) -> bool:
        """编译SYCL代码，带自动修复重试"""
        for attempt in range(max_retries):
            # 保存代码
            cpp_file = f"/tmp/{test_id}_test.cpp"
            with open(cpp_file, 'w') as f:
                f.write(code)
            
            # 尝试编译
            success = await self._compile_sycl_docker(cpp_file, test_id)
            
            if success:
                return True
            
            if attempt < max_retries - 1:
                print(f"\n  ⚠️  SYCL compilation failed, attempt {attempt + 1}/{max_retries}")
                print(f"  🔧 Trying to fix with LLM...")
                
                # 获取错误信息
                error_msg = await self._get_sycl_compile_error(cpp_file, test_id)
                
                # 使用LLM修复
                async with self.llm_semaphore:
                    code = await self.harness_generator.fix_compilation_error(
                        code, error_msg, "sycl"
                    )
        
        return False
    
    async def _compile_cuda_remote(self, cu_file: str, test_id: str) -> bool:
        """在远程CUDA环境编译"""
        try:
            # 复制到远程
            remote_cu = f"/tmp/{test_id}_test.cu"
            subprocess.run(['scp', cu_file, f'{self.remote_host}:{remote_cu}'],
                          capture_output=True, timeout=30)
            
            # 复制到docker容器
            docker_cp = f'ssh {self.remote_host} "docker cp {remote_cu} cuda12.9-test:/workspace/test.cu"'
            subprocess.run(docker_cp, shell=True, capture_output=True, timeout=30)
            
            # 编译
            compile_cmd = f'''
            ssh {self.remote_host} '
            docker exec cuda12.9-test bash -c "
            cd /workspace && 
            nvcc -O2 -Wno-deprecated-gpu-targets test.cu -o {test_id}_test
            "
            '
            '''
            
            result = subprocess.run(compile_cmd, shell=True, 
                                  capture_output=True, timeout=120)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"CUDA compile error: {e}")
            return False
    
    async def _compile_sycl_docker(self, cpp_file: str, test_id: str) -> bool:
        """在本地docker容器编译SYCL"""
        try:
            # 复制到容器
            subprocess.run(['docker', 'cp', cpp_file, 
                          f'lsv-container:/workspace/{test_id}_test.cpp'],
                         capture_output=True, timeout=30)
            
            # 编译
            compile_cmd = [
                'docker', 'exec', 'lsv-container', 'bash', '-c',
                f'cd /workspace && icpx -fsycl -O2 {test_id}_test.cpp -o {test_id}_test'
            ]
            
            result = subprocess.run(compile_cmd, capture_output=True, 
                                  text=True, timeout=120)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"SYCL compile error: {e}")
            return False
    
    async def _get_cuda_compile_error(self, cu_file: str, test_id: str) -> str:
        """获取CUDA编译错误"""
        try:
            compile_cmd = f'''
            ssh {self.remote_host} '
            docker exec cuda12.9-test bash -c "
            cd /workspace && 
            nvcc -O2 test.cu -o test 2>&1 || echo COMPILATION_FAILED
            "
            '
            '''
            
            result = subprocess.run(compile_cmd, shell=True,
                                  capture_output=True, text=True, timeout=60)
            
            return result.stderr if result.stderr else result.stdout
            
        except:
            return "Unknown compilation error"
    
    async def _get_sycl_compile_error(self, cpp_file: str, test_id: str) -> str:
        """获取SYCL编译错误"""
        try:
            compile_cmd = [
                'docker', 'exec', 'lsv-container', 'bash', '-c',
                f'cd /workspace && icpx -fsycl -O2 {test_id}_test.cpp -o {test_id}_test'
            ]
            
            result = subprocess.run(compile_cmd, capture_output=True,
                                  text=True, timeout=60)
            
            return result.stderr
            
        except:
            return "Unknown compilation error"
    
    async def _execute_cuda_test(self, test_id: str) -> Optional[np.ndarray]:
        """执行CUDA测试"""
        try:
            # 运行测试
            run_cmd = f'''
            ssh {self.remote_host} '
            docker exec cuda12.9-test bash -c "
            cd /workspace && ./{test_id}_test
            "
            '
            '''
            
            result = subprocess.run(run_cmd, shell=True,
                                  capture_output=True, timeout=60)
            
            if result.returncode != 0:
                return None
            
            # 复制输出文件回来
            docker_cp = f'ssh {self.remote_host} "docker cp cuda12.9-test:/workspace/test_output_cuda.bin /tmp/{test_id}_cuda_output.bin"'
            subprocess.run(docker_cp, shell=True, capture_output=True, timeout=30)
            
            scp_back = subprocess.run(
                ['scp', f'{self.remote_host}:/tmp/{test_id}_cuda_output.bin', 
                 f'/tmp/{test_id}_cuda_output.bin'],
                capture_output=True, timeout=30
            )
            
            # 读取输出
            output = np.fromfile(f'/tmp/{test_id}_cuda_output.bin', dtype=np.float32)
            return output
            
        except Exception as e:
            print(f"CUDA execution error: {e}")
            return None
    
    async def _execute_sycl_test(self, test_id: str) -> Optional[np.ndarray]:
        """执行SYCL测试"""
        try:
            # 运行测试
            run_cmd = ['docker', 'exec', 'lsv-container', f'/workspace/{test_id}_test']
            result = subprocess.run(run_cmd, capture_output=True, timeout=60)
            
            if result.returncode != 0:
                return None
            
            # 复制输出文件回来
            subprocess.run(['docker', 'cp', 
                          f'lsv-container:/workspace/test_output_sycl.bin',
                          f'/tmp/{test_id}_sycl_output.bin'],
                         capture_output=True, timeout=30)
            
            # 读取输出
            output = np.fromfile(f'/tmp/{test_id}_sycl_output.bin', dtype=np.float32)
            return output
            
        except Exception as e:
            print(f"SYCL execution error: {e}")
            return None
    
    def _compare_results(self, cuda_output: np.ndarray, sycl_output: np.ndarray,
                        test_config: Dict) -> Dict:
        """对比CUDA和SYCL结果"""
        dtype = test_config.get("dtype", "float32")
        
        # 容差配置
        tolerances = {
            "float32": {"abs": 1e-5, "rel": 1e-4},
            "bfloat16": {"abs": 1e-3, "rel": 1e-2},
            "float16": {"abs": 1e-3, "rel": 1e-2}
        }
        
        tol = tolerances.get(dtype, tolerances["float32"])
        
        # 计算误差
        abs_error = np.abs(cuda_output - sycl_output)
        rel_error = abs_error / (np.abs(cuda_output) + 1e-10)
        
        max_abs_error = float(np.max(abs_error))
        max_rel_error = float(np.max(rel_error))
        mean_abs_error = float(np.mean(abs_error))
        
        # 统计违规
        violations = np.sum((abs_error > tol["abs"]) & (rel_error > tol["rel"]))
        violation_rate = violations / len(cuda_output)
        
        # 判断是否通过
        passed = violation_rate < 0.001  # 0.1%容忍
        
        return {
            "pass": passed,
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "mean_abs_error": mean_abs_error,
            "violations": int(violations),
            "violation_rate": float(violation_rate),
            "tolerance": tol,
            "cuda_sample": cuda_output[:5].tolist(),
            "sycl_sample": sycl_output[:5].tolist()
        }


# 使用示例
async def example_usage():
    """示例用法"""
    # 创建进度监控器
    monitor = ProgressMonitor()
    
    # 创建执行器
    executor = AsyncTestExecutor(monitor, max_llm_concurrency=2)
    
    # 测试kernel
    kernel_code = '''
template <typename T>
__global__ void addOne_kernel(T* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 1.0f;
    }
}
'''
    
    # 测试配置
    test_config = {
        "name": "addOne_f32_test",
        "dtype": "float32",
        "N": 1,
        "C": 64,
        "H": 8,
        "W": 8,
        "data_gen": "random_uniform",
        "min_val": -1.0,
        "max_val": 1.0,
        "template_types": ["float"]
    }
    
    # 执行测试
    print("="*70)
    print("🧪 Running Example Test")
    print("="*70)
    
    result = await executor.execute_test("test_001", kernel_code, test_config)
    
    print("\n" + "="*70)
    print("📊 Test Result:")
    print("="*70)
    print(json.dumps(result, indent=2))
    
    # 获取进度摘要
    summary = monitor.get_summary()
    print(f"\n📈 Progress Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(example_usage())
