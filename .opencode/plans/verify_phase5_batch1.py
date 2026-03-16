#!/usr/bin/env python3
"""
Phase 5 Batch 1 - Accuracy Verification
验证5个新内核的准确度
"""

import sys
import subprocess
import tempfile
import os
import numpy as np

sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/.opencode/plans')
from phase5_batch1_harnesses import PHASE5_BATCH1_HARNESSES

def run_and_compare(kernel_id, harness):
    """Run both CUDA and SYCL and compare outputs"""
    print(f"\n{'='*60}")
    print(f"Testing: {kernel_id}")
    print('='*60)
    
    # Run CUDA
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
        f.write(harness['cuda'])
        cuda_file = f.name
    
    subprocess.run(['scp', cuda_file, 'root@10.112.229.160:/tmp/test.cu'],
                  capture_output=True, timeout=30, check=True)
    
    cmd = '''ssh root@10.112.229.160 "docker cp /tmp/test.cu cuda12.9-test:/workspace/test.cu && 
             docker exec cuda12.9-test bash -c 'cd /workspace && 
             nvcc -O2 -Wno-deprecated-gpu-targets test.cu -o test && ./test'"'''
    subprocess.run(cmd, shell=True, capture_output=True, timeout=120, check=True)
    os.unlink(cuda_file)
    
    # Copy CUDA output back
    subprocess.run(['ssh', 'root@10.112.229.160', 
                   'docker cp cuda12.9-test:/workspace/output_cuda.bin /tmp/'],
                  capture_output=True, check=True)
    subprocess.run(['scp', 'root@10.112.229.160:/tmp/output_cuda.bin', '/tmp/'],
                  capture_output=True, check=True)
    
    # Run SYCL
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(harness['sycl'])
        sycl_file = f.name
    
    subprocess.run(['docker', 'cp', sycl_file, 'lsv-container:/workspace/test.cpp'],
                  capture_output=True, timeout=10, check=True)
    subprocess.run(['docker', 'exec', 'lsv-container', 'bash', '-c',
                   'cd /workspace && icpx -fsycl -O2 test.cpp -o test && ./test'],
                  capture_output=True, timeout=120, check=True)
    os.unlink(sycl_file)
    
    # Copy SYCL output
    subprocess.run(['docker', 'cp', 'lsv-container:/workspace/output_sycl.bin', '/tmp/'],
                  capture_output=True, check=True)
    
    # Compare
    cuda_out = np.fromfile('/tmp/output_cuda.bin', dtype=np.float32)
    sycl_out = np.fromfile('/tmp/output_sycl.bin', dtype=np.float32)
    
    if len(cuda_out) != len(sycl_out):
        print(f"❌ Size mismatch: CUDA={len(cuda_out)}, SYCL={len(sycl_out)}")
        return False, 0, 0
    
    diff = np.abs(cuda_out - sycl_out)
    mae = float(np.mean(diff))
    max_err = float(np.max(diff))
    
    # Use appropriate tolerance based on kernel type
    if kernel_id in ['copy_type_converted']:
        # Type conversion can have precision differences
        tolerance_mae = 1e-3
        tolerance_max = 1e-2
    else:
        tolerance_mae = 1e-5
        tolerance_max = 1e-4
    
    passed = (mae < tolerance_mae) and (max_err < tolerance_max)
    
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - MAE: {mae:.2e}, Max Error: {max_err:.2e}")
    print(f"       (Tolerance: MAE < {tolerance_mae}, Max < {tolerance_max})")
    
    return passed, mae, max_err

def main():
    print("=" * 80)
    print("🎯 Phase 5 Batch 1 - Accuracy Verification")
    print("=" * 80)
    
    results = []
    for kernel_id, harness in PHASE5_BATCH1_HARNESSES.items():
        try:
            passed, mae, max_err = run_and_compare(kernel_id, harness)
            results.append({
                'kernel_id': kernel_id,
                'passed': passed,
                'mae': mae,
                'max_err': max_err
            })
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'kernel_id': kernel_id,
                'passed': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Accuracy Summary")
    print("=" * 80)
    
    passed_count = sum(1 for r in results if r.get('passed'))
    total = len(results)
    
    for r in results:
        status = "✅" if r.get('passed') else "❌"
        print(f"{status} {r['kernel_id']:30s} MAE: {r.get('mae', 0):.2e} Max: {r.get('max_err', 0):.2e}")
    
    print("=" * 80)
    print(f"\n总计: {passed_count}/{total} 个内核通过准确度验证")
    
    # Update progress
    old_count = 7
    new_total = old_count + passed_count
    remaining = max(0, 25 - new_total)
    
    print(f"\n📈 进度更新:")
    print(f"  已有: {old_count} 个内核")
    print(f"  新增: {passed_count} 个内核")
    print(f"  总计: {new_total} 个内核")
    print(f"  目标: 25 个内核")
    print(f"  剩余: {remaining} 个")
    
    if passed_count == total:
        print("\n🎉 恭喜！Phase 5 Batch 1 全部通过！")
        print("\n下一步:")
        print("  1. 继续 Phase 5 Batch 2 (再转换5个内核)")
        print("  2. 更新 RealAccuracyTester 包含新 harnesses")
    
    return results

if __name__ == '__main__':
    main()
