#!/usr/bin/env python3
"""
并行执行所有 kernel 测试
Parallel Kernel Test Execution
"""

import sys
import os
import subprocess
import tempfile
import json
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, str(Path(__file__).parent / 'harnesses'))

from all_harnesses import ALL_HARNESSES
from phase5_batch4_harnesses import PHASE5_BATCH4_HARNESSES

ALL_KERNELS = {}
ALL_KERNELS.update(ALL_HARNESSES)
ALL_KERNELS.update(PHASE5_BATCH4_HARNESSES)

# Thread-safe results collection
results_lock = threading.Lock()
test_results = []

class ParallelTestRunner:
    def __init__(self, cuda_host="10.112.229.160", 
                 cuda_container="cuda12.9-test",
                 sycl_container="lsv-container",
                 max_workers=4):
        self.cuda_host = cuda_host
        self.cuda_container = cuda_container
        self.sycl_container = sycl_container
        self.max_workers = max_workers
        self.completed = 0
        self.total = 0
        
    def run_cuda(self, code: str, kernel_id: str) -> Tuple[bool, float, str]:
        """运行 CUDA，返回(成功, 耗时, 错误)"""
        start = time.time()
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(code)
                cuda_file = f.name
            
            # SCP to remote
            r = subprocess.run(['scp', cuda_file, f'root@{self.cuda_host}:/tmp/test.cu'],
                             capture_output=True, timeout=30)
            if r.returncode != 0:
                os.unlink(cuda_file)
                return False, time.time()-start, "SCP failed"
            
            # Compile and run
            cmd = f'''ssh root@{self.cuda_host} "docker cp /tmp/test.cu {self.cuda_container}:/workspace/test.cu && 
                     docker exec {self.cuda_container} bash -c 'cd /workspace && 
                     nvcc -O2 -Wno-deprecated-gpu-targets test.cu -o test 2>&1 && ./test'"'''
            
            r = subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
            os.unlink(cuda_file)
            
            if r.returncode != 0:
                return False, time.time()-start, f"CUDA error: {r.stderr.decode()[:100]}"
            
            # Copy output back
            subprocess.run(['ssh', f'root@{self.cuda_host}', 
                           f'docker cp {self.cuda_container}:/workspace/output_cuda.bin /tmp/'],
                          capture_output=True)
            subprocess.run(['scp', f'root@{self.cuda_host}:/tmp/output_cuda.bin', '/tmp/'],
                          capture_output=True)
            
            return True, time.time()-start, ""
            
        except Exception as e:
            return False, time.time()-start, str(e)
    
    def run_sycl(self, code: str, kernel_id: str) -> Tuple[bool, float, str]:
        """运行 SYCL，返回(成功, 耗时, 错误)"""
        start = time.time()
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(code)
                sycl_file = f.name
            
            # Copy to container
            r = subprocess.run(['docker', 'cp', sycl_file, f'{self.sycl_container}:/workspace/test.cpp'],
                              capture_output=True, timeout=10)
            if r.returncode != 0:
                os.unlink(sycl_file)
                return False, time.time()-start, "Docker cp failed"
            
            # Compile and run
            r = subprocess.run(['docker', 'exec', self.sycl_container, 'bash', '-c',
                               'cd /workspace && icpx -fsycl -O2 test.cpp -o test 2>&1 && ./test'],
                              capture_output=True, timeout=120)
            os.unlink(sycl_file)
            
            if r.returncode != 0:
                return False, time.time()-start, f"SYCL error: {r.stderr.decode()[:100]}"
            
            # Copy output back
            subprocess.run(['docker', 'cp', f'{self.sycl_container}:/workspace/output_sycl.bin', '/tmp/'],
                          capture_output=True)
            
            return True, time.time()-start, ""
            
        except Exception as e:
            return False, time.time()-start, str(e)
    
    def compare(self, kernel_id: str) -> Tuple[bool, float, float, str]:
        """对比输出"""
        try:
            cuda_file = '/tmp/output_cuda.bin'
            sycl_file = '/tmp/output_sycl.bin'
            
            if not os.path.exists(cuda_file):
                return False, 0, 0, "CUDA output missing"
            if not os.path.exists(sycl_file):
                return False, 0, 0, "SYCL output missing"
            
            cuda_out = np.fromfile(cuda_file, dtype=np.float32)
            sycl_out = np.fromfile(sycl_file, dtype=np.float32)
            
            if len(cuda_out) != len(sycl_out):
                return False, 0, 0, f"Size mismatch: {len(cuda_out)} vs {len(sycl_out)}"
            
            if len(cuda_out) == 0:
                return False, 0, 0, "Empty output"
            
            diff = np.abs(cuda_out - sycl_out)
            mae = float(np.mean(diff))
            max_err = float(np.max(diff))
            
            # Tolerance
            if any(x in kernel_id for x in ['half', 'fp16', 'converted']):
                passed = (mae < 1e-3) and (max_err < 1e-2)
            else:
                passed = (mae < 1e-4) and (max_err < 1e-3)
            
            return passed, mae, max_err, ""
            
        except Exception as e:
            return False, 0, 0, str(e)
    
    def test_single_kernel(self, kernel_id: str) -> Dict:
        """测试单个 kernel"""
        if kernel_id not in ALL_KERNELS:
            return {
                'kernel_id': kernel_id,
                'status': 'NOT_FOUND',
                'passed': False,
                'error': 'Harness not found'
            }
        
        harness = ALL_KERNELS[kernel_id]
        result = {
            'kernel_id': kernel_id,
            'status': 'UNKNOWN',
            'passed': False,
            'mae': 0.0,
            'max_err': 0.0,
            'cuda_time': 0.0,
            'sycl_time': 0.0,
            'total_time': 0.0,
            'error': ''
        }
        
        start_total = time.time()
        
        # CUDA
        cuda_ok, cuda_time, cuda_err = self.run_cuda(harness['cuda'], kernel_id)
        result['cuda_time'] = cuda_time
        if not cuda_ok:
            result['status'] = 'CUDA_FAILED'
            result['error'] = cuda_err
            result['total_time'] = time.time() - start_total
            return result
        
        # SYCL
        sycl_ok, sycl_time, sycl_err = self.run_sycl(harness['sycl'], kernel_id)
        result['sycl_time'] = sycl_time
        if not sycl_ok:
            result['status'] = 'SYCL_FAILED'
            result['error'] = sycl_err
            result['total_time'] = time.time() - start_total
            return result
        
        # Compare
        passed, mae, max_err, comp_err = self.compare(kernel_id)
        result['mae'] = mae
        result['max_err'] = max_err
        result['total_time'] = time.time() - start_total
        
        if comp_err:
            result['status'] = 'COMPARE_FAILED'
            result['error'] = comp_err
        elif passed:
            result['status'] = 'PASSED'
            result['passed'] = True
        else:
            result['status'] = 'FAILED'
        
        return result
    
    def run_all_parallel(self):
        """并行运行所有测试"""
        print("="*80)
        print("🚀 PARALLEL KERNEL TEST EXECUTION")
        print("="*80)
        print(f"\nTotal kernels: {len(ALL_KERNELS)}")
        print(f"Max parallel workers: {self.max_workers}")
        print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        kernel_ids = sorted(ALL_KERNELS.keys())
        self.total = len(kernel_ids)
        self.completed = 0
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_kernel = {
                executor.submit(self.test_single_kernel, kid): kid 
                for kid in kernel_ids
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_kernel):
                kernel_id = future_to_kernel[future]
                try:
                    result = future.result()
                    with results_lock:
                        test_results.append(result)
                        self.completed += 1
                        
                        # Progress output
                        status_icon = "✅" if result['passed'] else "❌"
                        print(f"[{self.completed}/{self.total}] {status_icon} {kernel_id:35s} "
                              f"MAE: {result['mae']:.2e} "
                              f"Time: {result['total_time']:.1f}s")
                        
                except Exception as e:
                    with results_lock:
                        test_results.append({
                            'kernel_id': kernel_id,
                            'status': 'EXCEPTION',
                            'passed': False,
                            'error': str(e)
                        })
                        self.completed += 1
        
        total_time = time.time() - start_time
        self.generate_final_report(total_time)
    
    def generate_final_report(self, total_time: float):
        """生成最终报告"""
        print("\n" + "="*80)
        print("📊 FINAL TEST REPORT")
        print("="*80)
        
        passed = sum(1 for r in test_results if r.get('passed'))
        failed = sum(1 for r in test_results if not r.get('passed'))
        total = len(test_results)
        
        print(f"\n⏱️  Total execution time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"\n📈 Results Summary:")
        print(f"  ✅ Passed:  {passed}/{total} ({passed/total*100:.1f}%)")
        print(f"  ❌ Failed:  {failed}/{total} ({failed/total*100:.1f}%)")
        
        # Sort by MAE
        sorted_results = sorted(test_results, key=lambda x: x.get('mae', 999))
        
        print(f"\n🏆 Top 10 Most Accurate:")
        for i, r in enumerate(sorted_results[:10], 1):
            if r.get('passed'):
                print(f"  {i:2d}. {r['kernel_id']:35s} MAE: {r['mae']:.2e}")
        
        if failed > 0:
            print(f"\n❌ Failed Kernels:")
            for r in sorted_results:
                if not r.get('passed'):
                    print(f"  - {r['kernel_id']:35s} Status: {r['status']}")
        
        # Save JSON results
        json_file = f"reports/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_kernels': total,
                'passed': passed,
                'failed': failed,
                'results': test_results
            }, f, indent=2)
        
        print(f"\n💾 Results saved: {json_file}")
        print("="*80)


def main():
    runner = ParallelTestRunner(max_workers=4)
    runner.run_all_parallel()
    
    # Return code based on success
    passed = sum(1 for r in test_results if r.get('passed'))
    return 0 if passed >= 25 else 1


if __name__ == '__main__':
    sys.exit(main())
