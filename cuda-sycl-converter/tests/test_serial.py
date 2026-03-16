#!/usr/bin/env python3
"""
串行执行所有 kernel 测试（避免文件冲突）
Serial Kernel Test Execution
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

sys.path.insert(0, str(Path(__file__).parent / 'harnesses'))

from all_harnesses import ALL_HARNESSES
from phase5_batch4_harnesses import PHASE5_BATCH4_HARNESSES

ALL_KERNELS = {}
ALL_KERNELS.update(ALL_HARNESSES)
ALL_KERNELS.update(PHASE5_BATCH4_HARNESSES)

class SerialTestRunner:
    def __init__(self, cuda_host="10.112.229.160", 
                 cuda_container="cuda12.9-test",
                 sycl_container="lsv-container"):
        self.cuda_host = cuda_host
        self.cuda_container = cuda_container
        self.sycl_container = sycl_container
        self.test_results = []
        
    def run_cuda(self, code: str, kernel_id: str, tmp_id: str) -> Tuple[bool, float, str]:
        """运行 CUDA，返回(成功, 耗时, 错误)"""
        start = time.time()
        try:
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(code)
                cuda_file = f.name
            
            # SCP with unique name
            r = subprocess.run(['scp', cuda_file, f'root@{self.cuda_host}:/tmp/{tmp_id}.cu'],
                             capture_output=True, timeout=30)
            os.unlink(cuda_file)
            
            if r.returncode != 0:
                return False, time.time()-start, "SCP failed"
            
            # Compile and run
            cmd = f'''ssh root@{self.cuda_host} "docker cp /tmp/{tmp_id}.cu {self.cuda_container}:/workspace/test.cu && 
                     docker exec {self.cuda_container} bash -c 'cd /workspace && 
                     nvcc -O2 -Wno-deprecated-gpu-targets test.cu -o test 2>&1 && ./test'"'''
            
            r = subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
            
            if r.returncode != 0:
                err = r.stderr.decode()[:200] if r.stderr else "Unknown CUDA error"
                return False, time.time()-start, err
            
            # Copy output back with unique name
            r1 = subprocess.run(['ssh', f'root@{self.cuda_host}', 
                               f'docker cp {self.cuda_container}:/workspace/output_cuda.bin /tmp/{tmp_id}_cuda.bin'],
                              capture_output=True)
            r2 = subprocess.run(['scp', f'root@{self.cuda_host}:/tmp/{tmp_id}_cuda.bin', f'/tmp/{tmp_id}_cuda.bin'],
                              capture_output=True)
            
            if r1.returncode != 0 or r2.returncode != 0:
                return False, time.time()-start, "Failed to copy CUDA output"
            
            return True, time.time()-start, ""
            
        except Exception as e:
            return False, time.time()-start, str(e)
    
    def run_sycl(self, code: str, kernel_id: str, tmp_id: str) -> Tuple[bool, float, str]:
        """运行 SYCL，返回(成功, 耗时, 错误)"""
        start = time.time()
        try:
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(code)
                sycl_file = f.name
            
            # Copy to container
            r = subprocess.run(['docker', 'cp', sycl_file, f'{self.sycl_container}:/workspace/test.cpp'],
                              capture_output=True, timeout=10)
            os.unlink(sycl_file)
            
            if r.returncode != 0:
                return False, time.time()-start, "Docker cp failed"
            
            # Compile and run
            r = subprocess.run(['docker', 'exec', self.sycl_container, 'bash', '-c',
                               'cd /workspace && icpx -fsycl -O2 test.cpp -o test 2>&1 && ./test'],
                              capture_output=True, timeout=120)
            
            if r.returncode != 0:
                err = r.stderr.decode()[:200] if r.stderr else "Unknown SYCL error"
                return False, time.time()-start, err
            
            # Copy output back with unique name
            unique_sycl_file = f'/tmp/{tmp_id}_sycl.bin'
            r = subprocess.run(['docker', 'cp', f'{self.sycl_container}:/workspace/output_sycl.bin', 
                               unique_sycl_file],
                              capture_output=True)
            
            if r.returncode != 0:
                return False, time.time()-start, "Failed to copy SYCL output"
            
            return True, time.time()-start, ""
            
        except Exception as e:
            return False, time.time()-start, str(e)
    
    def compare(self, kernel_id: str, tmp_id: str) -> Tuple[bool, float, float, str]:
        """对比输出"""
        try:
            cuda_file = f'/tmp/{tmp_id}_cuda.bin'
            sycl_file = f'/tmp/{tmp_id}_sycl.bin'
            
            if not os.path.exists(cuda_file):
                return False, 0, 0, "CUDA output missing"
            if not os.path.exists(sycl_file):
                return False, 0, 0, "SYCL output missing"
            
            cuda_out = np.fromfile(cuda_file, dtype=np.float32)
            sycl_out = np.fromfile(sycl_file, dtype=np.float32)
            
            # Cleanup
            try:
                os.unlink(cuda_file)
                os.unlink(sycl_file)
            except:
                pass
            
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
    
    def test_single_kernel(self, kernel_id: str, idx: int, total: int) -> Dict:
        """测试单个 kernel"""
        print(f"\n[{idx}/{total}] Testing: {kernel_id}")
        
        if kernel_id not in ALL_KERNELS:
            return {
                'kernel_id': kernel_id,
                'status': 'NOT_FOUND',
                'passed': False,
                'error': 'Harness not found'
            }
        
        harness = ALL_KERNELS[kernel_id]
        tmp_id = f"{kernel_id}_{int(time.time()*1000)%10000}"
        
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
        
        # Generate ONE tmp_id for both CUDA and SYCL
        tmp_id = f"{kernel_id}_{int(time.time()*1000)%10000}"
        
        # CUDA
        print(f"  🚀 CUDA...", end='', flush=True)
        cuda_ok, cuda_time, cuda_err = self.run_cuda(harness['cuda'], kernel_id, tmp_id)
        result['cuda_time'] = cuda_time
        
        if not cuda_ok:
            print(f" ❌ ({cuda_time:.1f}s)")
            result['status'] = 'CUDA_FAILED'
            result['error'] = cuda_err[:100]
            result['total_time'] = time.time() - start_total
            return result
        print(f" ✅ ({cuda_time:.1f}s)")
        
        # SYCL
        print(f"  🚀 SYCL...", end='', flush=True)
        sycl_ok, sycl_time, sycl_err = self.run_sycl(harness['sycl'], kernel_id, tmp_id)
        result['sycl_time'] = sycl_time
        
        if not sycl_ok:
            print(f" ❌ ({sycl_time:.1f}s)")
            result['status'] = 'SYCL_FAILED'
            result['error'] = sycl_err[:100]
            result['total_time'] = time.time() - start_total
            return result
        print(f" ✅ ({sycl_time:.1f}s)")
        
        # Compare
        print(f"  📊 Compare...", end='', flush=True)
        passed, mae, max_err, comp_err = self.compare(kernel_id, tmp_id)
        result['mae'] = mae
        result['max_err'] = max_err
        result['total_time'] = time.time() - start_total
        
        if comp_err:
            print(f" ❌ {comp_err[:50]}")
            result['status'] = 'COMPARE_FAILED'
            result['error'] = comp_err[:100]
        elif passed:
            print(f" ✅ PASS (MAE: {mae:.2e})")
            result['status'] = 'PASSED'
            result['passed'] = True
        else:
            print(f" ❌ FAIL (MAE: {mae:.2e})")
            result['status'] = 'FAILED'
        
        return result
    
    def run_all_serial(self):
        """串行运行所有测试"""
        print("="*80)
        print("🚀 SERIAL KERNEL TEST EXECUTION")
        print("="*80)
        print(f"\nTotal kernels: {len(ALL_KERNELS)}")
        print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        kernel_ids = sorted(ALL_KERNELS.keys())
        total = len(kernel_ids)
        start_time = time.time()
        
        for i, kernel_id in enumerate(kernel_ids, 1):
            try:
                result = self.test_single_kernel(kernel_id, i, total)
                self.test_results.append(result)
            except Exception as e:
                print(f"\n  ❌ Exception: {e}")
                self.test_results.append({
                    'kernel_id': kernel_id,
                    'status': 'EXCEPTION',
                    'passed': False,
                    'error': str(e)[:100]
                })
        
        total_time = time.time() - start_time
        self.generate_final_report(total_time)
    
    def generate_final_report(self, total_time: float):
        """生成最终报告"""
        print("\n" + "="*80)
        print("📊 FINAL TEST REPORT")
        print("="*80)
        
        passed = sum(1 for r in self.test_results if r.get('passed'))
        failed = sum(1 for r in self.test_results if not r.get('passed'))
        total = len(self.test_results)
        
        print(f"\n⏱️  Total execution time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"\n📈 Results Summary:")
        print(f"  ✅ Passed:  {passed}/{total} ({passed/total*100:.1f}%)")
        print(f"  ❌ Failed:  {failed}/{total} ({failed/total*100:.1f}%)")
        
        # Sort by MAE
        sorted_results = sorted(self.test_results, key=lambda x: x.get('mae', 999))
        
        # Passed kernels
        passed_kernels = [r for r in sorted_results if r.get('passed')]
        if passed_kernels:
            print(f"\n✅ Passed Kernels ({len(passed_kernels)}):")
            for r in passed_kernels:
                print(f"  - {r['kernel_id']:35s} MAE: {r['mae']:.2e}  Max: {r['max_err']:.2e}")
        
        # Failed kernels
        failed_kernels = [r for r in sorted_results if not r.get('passed')]
        if failed_kernels:
            print(f"\n❌ Failed Kernels ({len(failed_kernels)}):")
            for r in failed_kernels:
                print(f"  - {r['kernel_id']:35s} Status: {r['status']:15s} Error: {r.get('error', 'N/A')[:40]}")
        
        # Save JSON
        json_file = f"reports/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_kernels': total,
                'passed': passed,
                'failed': failed,
                'results': self.test_results
            }, f, indent=2)
        
        print(f"\n💾 Results saved: {json_file}")
        
        # Generate markdown report
        self.generate_markdown_report(total_time, passed, failed)
        
        print("="*80)
    
    def generate_markdown_report(self, total_time: float, passed: int, failed: int):
        """生成 Markdown 报告"""
        report = []
        report.append("# CUDA to SYCL Kernel Conversion Test Results")
        report.append(f"\n**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n## Summary\n")
        report.append(f"- **Total Kernels:** {len(self.test_results)}")
        report.append(f"- **Passed:** {passed} ({passed/len(self.test_results)*100:.1f}%)")
        report.append(f"- **Failed:** {failed} ({failed/len(self.test_results)*100:.1f}%)")
        report.append(f"- **Execution Time:** {total_time:.1f}s ({total_time/60:.1f} min)")
        
        # Detailed table
        report.append(f"\n## Detailed Results\n")
        report.append("| Kernel | Status | MAE | Max Error | CUDA(s) | SYCL(s) |")
        report.append("|--------|--------|-----|-----------|---------|---------|")
        
        for r in sorted(self.test_results, key=lambda x: x['kernel_id']):
            status = "✅ PASS" if r.get('passed') else f"❌ {r.get('status')}"
            mae = f"{r.get('mae', 0):.2e}" if r.get('mae', 0) > 0 else "-"
            max_err = f"{r.get('max_err', 0):.2e}" if r.get('max_err', 0) > 0 else "-"
            cuda_t = f"{r.get('cuda_time', 0):.1f}"
            sycl_t = f"{r.get('sycl_time', 0):.1f}"
            report.append(f"| {r['kernel_id']} | {status} | {mae} | {max_err} | {cuda_t} | {sycl_t} |")
        
        # Save
        md_file = f"reports/FINAL_TEST_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(md_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"📄 Markdown report: {md_file}")


def main():
    runner = SerialTestRunner()
    runner.run_all_serial()
    
    passed = sum(1 for r in runner.test_results if r.get('passed'))
    return 0 if passed >= 20 else 1


if __name__ == '__main__':
    sys.exit(main())
