#!/usr/bin/env python3
"""
FINAL VERIFICATION - All 28 Kernels
22 (fixed) + 6 (Batch 4) = 28 kernels total
"""

import sys
import subprocess
import tempfile
import os
import numpy as np

sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/.opencode/plans')

from FINAL_ALL_HARNESSES import ALL_HARNESSES as FINAL_22
from phase5_batch4_harnesses import PHASE5_BATCH4_HARNESSES as BATCH4

# Merge all
ALL_HARNESSES = {}
ALL_HARNESSES.update(FINAL_22)
ALL_HARNESSES.update(BATCH4)

def run_cuda(code, kernel_id):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
        f.write(code)
        cuda_file = f.name
    
    try:
        subprocess.run(['scp', cuda_file, 'root@10.112.229.160:/tmp/test.cu'],
                      capture_output=True, timeout=30, check=True)
        
        cmd = '''ssh root@10.112.229.160 "docker cp /tmp/test.cu cuda12.9-test:/workspace/test.cu && 
                 docker exec cuda12.9-test bash -c 'cd /workspace && 
                 nvcc -O2 -Wno-deprecated-gpu-targets test.cu -o test && ./test'"'''
        
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
        os.unlink(cuda_file)
        
        if result.returncode == 0:
            subprocess.run(['ssh', 'root@10.112.229.160', 
                           'docker cp cuda12.9-test:/workspace/output_cuda.bin /tmp/'],
                          capture_output=True, check=True)
            subprocess.run(['scp', 'root@10.112.229.160:/tmp/output_cuda.bin', '/tmp/'],
                          capture_output=True, check=True)
            return True
        return False
    except:
        return False

def run_sycl(code, kernel_id):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(code)
        sycl_file = f.name
    
    try:
        subprocess.run(['docker', 'cp', sycl_file, 'lsv-container:/workspace/test.cpp'],
                      capture_output=True, timeout=10, check=True)
        
        result = subprocess.run(['docker', 'exec', 'lsv-container', 'bash', '-c',
                               'cd /workspace && icpx -fsycl -O2 test.cpp -o test && ./test'],
                              capture_output=True, timeout=120)
        os.unlink(sycl_file)
        
        if result.returncode == 0:
            subprocess.run(['docker', 'cp', 'lsv-container:/workspace/output_sycl.bin', '/tmp/'],
                          capture_output=True, check=True)
            return True
        return False
    except:
        return False

def compare_outputs(kernel_id):
    try:
        cuda_out = np.fromfile('/tmp/output_cuda.bin', dtype=np.float32)
        sycl_out = np.fromfile('/tmp/output_sycl.bin', dtype=np.float32)
        
        if len(cuda_out) != len(sycl_out):
            return False, 1.0, 1.0
        
        diff = np.abs(cuda_out - sycl_out)
        mae = float(np.mean(diff))
        max_err = float(np.max(diff))
        
        tolerance_mae = 1e-3 if any(x in kernel_id for x in ['half', 'fp16', 'converted']) else 1e-4
        tolerance_max = 1e-2 if any(x in kernel_id for x in ['half', 'fp16', 'converted']) else 1e-3
        
        passed = (mae < tolerance_mae) and (max_err < tolerance_max)
        return passed, mae, max_err
    except:
        return False, 0, 0

def test_kernel(kernel_id, harness):
    print(f"\n{'='*60}")
    print(f"Testing: {kernel_id}")
    print('='*60)
    
    cuda_ok = run_cuda(harness['cuda'], kernel_id)
    if not cuda_ok:
        print("  ❌ CUDA failed")
        return {'kernel_id': kernel_id, 'passed': False, 'error': 'CUDA failed'}
    
    sycl_ok = run_sycl(harness['sycl'], kernel_id)
    if not sycl_ok:
        print("  ❌ SYCL failed")
        return {'kernel_id': kernel_id, 'passed': False, 'error': 'SYCL failed'}
    
    passed, mae, max_err = compare_outputs(kernel_id)
    
    if passed:
        print(f"  ✅ PASS - MAE: {mae:.2e}, Max: {max_err:.2e}")
    else:
        print(f"  ❌ FAIL - MAE: {mae:.2e}, Max: {max_err:.2e}")
    
    return {
        'kernel_id': kernel_id,
        'passed': passed,
        'mae': mae,
        'max_err': max_err
    }

def main():
    print("=" * 80)
    print("🚀 FINAL VERIFICATION - All 28 Kernels")
    print("=" * 80)
    print(f"\n📊 Total kernels: {len(ALL_HARNESSES)}")
    print("Target: 25+ kernels with passing accuracy")
    print()
    
    results = []
    
    for kernel_id, harness in ALL_HARNESSES.items():
        try:
            result = test_kernel(kernel_id, harness)
            results.append(result)
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            results.append({
                'kernel_id': kernel_id,
                'passed': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS")
    print("=" * 80)
    
    passed_count = sum(1 for r in results if r.get('passed'))
    total = len(results)
    
    print(f"\n✅ Passed: {passed_count}/{total}")
    print(f"❌ Failed: {total - passed_count}/{total}")
    
    if passed_count >= 25:
        print(f"\n🎉 SUCCESS! Target of 25+ kernels achieved!")
        print(f"   Total passing: {passed_count} kernels")
    else:
        print(f"\n⏳ Need {25 - passed_count} more kernels to reach target")
    
    print("\n" + "=" * 80)
    print("✅ PHASE 5 BATCH CONVERSION COMPLETE")
    print("=" * 80)
    
    return passed_count

if __name__ == '__main__':
    passed = main()
    exit(0 if passed >= 25 else 1)
