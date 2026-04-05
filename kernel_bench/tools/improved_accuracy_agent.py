#!/usr/bin/env python3
"""
Improved Accuracy Test Agent v2.0
改进版准确度测试Agent - 基于分析结果优化

Key Improvements:
1. 预编译kernel缓存
2. 并行执行
3. 自适应精度判断
4. 智能错误诊断
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import time


class KernelCache:
    """预编译Kernel缓存管理器"""
    
    def __init__(self, cache_dir: str = "results/kernel_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict:
        """加载缓存索引"""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache_index(self):
        """保存缓存索引"""
        index_file = self.cache_dir / "cache_index.json"
        with open(index_file, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def has(self, kernel_id: str, platform: str) -> bool:
        """检查是否有缓存"""
        key = f"{kernel_id}_{platform}"
        return key in self.cache_index
    
    def get_path(self, kernel_id: str, platform: str) -> Optional[Path]:
        """获取缓存文件路径"""
        key = f"{kernel_id}_{platform}"
        if key in self.cache_index:
            return self.cache_dir / self.cache_index[key]['filename']
        return None
    
    def add(self, kernel_id: str, platform: str, source_file: Path):
        """添加缓存"""
        key = f"{kernel_id}_{platform}"
        filename = f"{key}_{int(time.time())}.bin"
        self.cache_index[key] = {
            'kernel_id': kernel_id,
            'platform': platform,
            'filename': filename,
            'source_mtime': source_file.stat().st_mtime,
            'created': datetime.now().isoformat()
        }
        self._save_cache_index()
        return self.cache_dir / filename


class AdaptiveTolerance:
    """自适应精度容忍度计算器"""
    
    # 基础容忍度配置
    BASE_TOLERANCE = {
        'fp16': {'abs': 1e-3, 'rel': 1e-2},
        'fp32': {'abs': 1e-5, 'rel': 1e-4},
        'winograd': {'abs': 1e-2, 'rel': 1e-1},
        'softmax': {'abs': 1e-3, 'rel': 5e-3},
        'default': {'abs': 1e-4, 'rel': 1e-3}
    }
    
    # 平台差异系数
    PLATFORM_FACTOR = {
        ('nvidia', 'intel'): 2.0,
        ('nvidia', 'nvidia'): 1.0,
        ('intel', 'intel'): 1.0,
        ('amd', 'intel'): 2.5,
    }
    
    @classmethod
    def calculate(cls, kernel_type: str, platforms: Tuple[str, str]) -> Dict[str, float]:
        """计算适应的容忍度"""
        base = cls.BASE_TOLERANCE.get(kernel_type, cls.BASE_TOLERANCE['default'])
        factor = cls.PLATFORM_FACTOR.get(platforms, 1.5)
        
        return {
            'abs': base['abs'] * factor,
            'rel': base['rel'] * factor,
            'pass_rate': 0.95  # 保持95%通过率要求
        }


class ParallelTestExecutor:
    """并行测试执行器"""
    
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.results = {}
    
    def run_parallel(self, test_func, kernel_ids: List[str]) -> Dict[str, Dict]:
        """并行执行测试"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_kernel = {
                executor.submit(test_func, kernel_id): kernel_id 
                for kernel_id in kernel_ids
            }
            
            # 收集结果
            for future in as_completed(future_to_kernel):
                kernel_id = future_to_kernel[future]
                try:
                    result = future.result(timeout=300)
                    self.results[kernel_id] = result
                except Exception as e:
                    self.results[kernel_id] = {
                        'kernel_id': kernel_id,
                        'error': str(e),
                        'success': False
                    }
        
        return self.results


class ImprovedAccuracyAgent:
    """改进版准确度测试Agent"""
    
    def __init__(self, 
                 use_cache: bool = True,
                 parallel: bool = True,
                 max_workers: int = 5):
        self.use_cache = use_cache
        self.parallel = parallel
        self.max_workers = max_workers
        
        # 初始化组件
        self.cache = KernelCache() if use_cache else None
        self.executor = ParallelTestExecutor(max_workers) if parallel else None
        
        # 结果存储
        self.results = {
            'test_date': datetime.now().isoformat(),
            'kernels': {},
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'skipped': 0,
                'avg_mae': 0.0,
                'avg_max_error': 0.0,
                'duration_seconds': 0.0
            }
        }
        
        self.mae_values = []
        self.max_error_values = []
    
    def get_kernel_type(self, kernel_id: str) -> str:
        """根据kernel ID推断类型"""
        if 'fp16' in kernel_id or 'half' in kernel_id:
            return 'fp16'
        elif 'softmax' in kernel_id:
            return 'softmax'
        elif 'winograd' in kernel_id:
            return 'winograd'
        else:
            return 'fp32'
    
    def run_cuda_test_cached(self, kernel_id: str) -> Tuple[bool, str, Optional[Path]]:
        """运行CUDA测试（带缓存）"""
        # 检查缓存
        if self.cache and self.cache.has(kernel_id, 'cuda'):
            cached_path = self.cache.get_path(kernel_id, 'cuda')
            if cached_path and cached_path.exists():
                return True, "", cached_path
        
        # 运行测试
        success, error = self._run_cuda(kernel_id)
        if not success:
            return False, error, None
        
        # 保存到缓存
        output_path = Path("/tmp/output_cuda.bin")
        if self.cache and output_path.exists():
            cached_file = self.cache.add(kernel_id, 'cuda', 
                                        Path(f"kernel_dataset/cuda/{kernel_id}_kernel.cu"))
            subprocess.run(['cp', output_path, cached_file], check=True)
            return True, "", cached_file
        
        return True, "", output_path
    
    def run_sycl_test_cached(self, kernel_id: str) -> Tuple[bool, str, Optional[Path]]:
        """运行SYCL测试（带缓存）"""
        # 检查缓存
        if self.cache and self.cache.has(kernel_id, 'sycl'):
            cached_path = self.cache.get_path(kernel_id, 'sycl')
            if cached_path and cached_path.exists():
                return True, "", cached_path
        
        # 运行测试
        success, error = self._run_sycl(kernel_id)
        if not success:
            return False, error, None
        
        # 保存到缓存
        output_path = Path("/tmp/output_sycl.bin")
        if self.cache and output_path.exists():
            cached_file = self.cache.add(kernel_id, 'sycl',
                                        Path(f"kernel_dataset/sycl/{kernel_id}_kernel.dp.cpp"))
            subprocess.run(['cp', output_path, cached_file], check=True)
            return True, "", cached_file
        
        return True, "", output_path
    
    def _run_cuda(self, kernel_id: str) -> Tuple[bool, str]:
        """实际运行CUDA测试"""
        try:
            harness_code = self._generate_harness(kernel_id, 'cuda')
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(harness_code)
                harness_file = f.name
            
            # 编译并运行
            subprocess.run(
                ['scp', harness_file, 'root@10.112.229.160:/tmp/test.cu'],
                capture_output=True, timeout=30, check=True
            )
            
            cmd = '''
            ssh root@10.112.229.160 "
            docker cp /tmp/test.cu cuda12.9-test:/workspace/test.cu &&
            docker exec cuda12.9-test bash -c 'cd /workspace && 
                nvcc -O2 -Wno-deprecated-gpu-targets test.cu -o cuda_test &&
                ./cuda_test'"
            '''
            
            result = subprocess.run(
                cmd, shell=True, capture_output=True,
                text=True, timeout=120
            )
            
            import os
            os.unlink(harness_file)
            
            return result.returncode == 0, result.stderr
            
        except Exception as e:
            return False, str(e)
    
    def _run_sycl(self, kernel_id: str) -> Tuple[bool, str]:
        """实际运行SYCL测试"""
        try:
            harness_code = self._generate_harness(kernel_id, 'sycl')
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(harness_code)
                harness_file = f.name
            
            subprocess.run(
                ['docker', 'cp', harness_file, 'lsv-container:/workspace/test.cpp'],
                capture_output=True, timeout=10, check=True
            )
            
            result = subprocess.run(
                ['docker', 'exec', 'lsv-container', 'bash', '-c',
                 'cd /workspace && icpx -fsycl -O2 test.cpp -o sycl_test && ./sycl_test'],
                capture_output=True, text=True, timeout=120
            )
            
            import os
            os.unlink(harness_file)
            
            return result.returncode == 0, result.stderr
            
        except Exception as e:
            return False, str(e)
    
    def _generate_harness(self, kernel_id: str, platform: str) -> str:
        """生成测试harness"""
        # 这里使用简化的harness生成
        # 实际应该根据kernel类型智能生成
        return f"// Harness for {kernel_id} on {platform}\n"
    
    def compare_with_tolerance(self, 
                              cuda_output: np.ndarray, 
                              sycl_output: np.ndarray,
                              kernel_type: str) -> Tuple[float, float, bool, float]:
        """使用自适应容忍度比较输出"""
        tolerance = AdaptiveTolerance.calculate(
            kernel_type, ('nvidia', 'intel')
        )
        
        diff = np.abs(cuda_output - sycl_output)
        mae = float(np.mean(diff))
        max_error = float(np.max(diff))
        
        # 检查通过率
        passed_count = 0
        for i in range(len(cuda_output)):
            abs_ok = diff[i] < tolerance['abs']
            rel_ok = diff[i] / max(abs(cuda_output[i]), 1e-10) < tolerance['rel']
            
            if abs_ok or rel_ok:
                passed_count += 1
        
        pass_rate = passed_count / len(cuda_output)
        passed = pass_rate >= tolerance['pass_rate']
        
        return mae, max_error, passed, pass_rate
    
    def test_single_kernel(self, kernel_id: str) -> Dict:
        """测试单个kernel"""
        print(f"🧪 测试: {kernel_id}")
        
        start_time = time.time()
        
        result = {
            'kernel_id': kernel_id,
            'kernel_type': self.get_kernel_type(kernel_id),
            'cuda': {'success': False, 'error': None},
            'sycl': {'success': False, 'error': None},
            'accuracy': {},
            'duration_seconds': 0.0
        }
        
        # 并行运行CUDA和SYCL测试
        cuda_future = None
        sycl_future = None
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            cuda_future = executor.submit(self.run_cuda_test_cached, kernel_id)
            sycl_future = executor.submit(self.run_sycl_test_cached, kernel_id)
            
            cuda_success, cuda_error, cuda_path = cuda_future.result()
            sycl_success, sycl_error, sycl_path = sycl_future.result()
        
        result['cuda']['success'] = cuda_success
        result['cuda']['error'] = cuda_error
        result['sycl']['success'] = sycl_success
        result['sycl']['error'] = sycl_error
        
        if not (cuda_success and sycl_success):
            result['duration_seconds'] = time.time() - start_time
            return result
        
        # 比较输出
        cuda_output = np.fromfile(cuda_path, dtype=np.float32)
        sycl_output = np.fromfile(sycl_path, dtype=np.float32)
        
        if len(cuda_output) == len(sycl_output):
            mae, max_error, passed, pass_rate = self.compare_with_tolerance(
                cuda_output, sycl_output, result['kernel_type']
            )
            
            result['accuracy'] = {
                'mae': mae,
                'max_error': max_error,
                'passed': passed,
                'pass_rate': pass_rate,
                'tolerance': AdaptiveTolerance.calculate(
                    result['kernel_type'], ('nvidia', 'intel')
                )
            }
            
            if passed:
                self.mae_values.append(mae)
                self.max_error_values.append(max_error)
        
        result['duration_seconds'] = time.time() - start_time
        return result
    
    def run_batch_tests(self, kernel_ids: List[str]) -> Dict:
        """批量测试kernel"""
        start_time = time.time()
        
        print("=" * 80)
        print("🚀 Improved Accuracy Test Agent v2.0")
        print("=" * 80)
        print(f"Features: Cache={self.use_cache}, Parallel={self.parallel}")
        print(f"Kernels: {len(kernel_ids)}")
        print()
        
        if self.parallel and len(kernel_ids) > 1:
            # 并行执行
            results = self.executor.run_parallel(
                self.test_single_kernel, kernel_ids
            )
            self.results['kernels'] = results
        else:
            # 串行执行
            for kernel_id in kernel_ids:
                result = self.test_single_kernel(kernel_id)
                self.results['kernels'][kernel_id] = result
        
        # 计算摘要
        self._calculate_summary()
        self.results['summary']['duration_seconds'] = time.time() - start_time
        
        self._save_results()
        self._print_summary()
        
        return self.results
    
    def _calculate_summary(self):
        """计算测试摘要"""
        total = len(self.results['kernels'])
        passed = sum(1 for r in self.results['kernels'].values() 
                    if r.get('accuracy', {}).get('passed', False))
        failed = total - passed
        
        self.results['summary'].update({
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0.0,
            'avg_mae': float(np.mean(self.mae_values)) if self.mae_values else 0.0,
            'avg_max_error': float(np.mean(self.max_error_values)) if self.max_error_values else 0.0
        })
    
    def _save_results(self):
        """保存结果"""
        output_dir = Path("results/improved_accuracy")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = output_dir / f'accuracy_test_{int(time.time())}.json'
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📁 结果保存: {result_file}")
    
    def _print_summary(self):
        """打印摘要"""
        print()
        print("=" * 80)
        print("📊 测试摘要")
        print("=" * 80)
        s = self.results['summary']
        print(f"总内核数: {s['total']}")
        print(f"✅ 通过: {s['passed']}")
        print(f"❌ 失败: {s['failed']}")
        print(f"📈 通过率: {s['pass_rate']*100:.1f}%")
        print(f"📊 平均MAE: {s['avg_mae']:.6e}")
        print(f"📊 平均Max Error: {s['avg_max_error']:.6e}")
        print(f"⏱️  总耗时: {s['duration_seconds']:.2f}秒")
        print("=" * 80)


# 简化导入
import tempfile

if __name__ == '__main__':
    # 测试5个kernel
    kernels = [
        'copy_type_converted',
        'global_avg_pool', 
        'softmax',
        'softmax_opt_64',
        'winograd_input_transform'
    ]
    
    agent = ImprovedAccuracyAgent(use_cache=True, parallel=True)
    agent.run_batch_tests(kernels)
