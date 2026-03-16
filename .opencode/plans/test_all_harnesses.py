#!/usr/bin/env python3
"""
Phase 5 Complete - All Batches Testing
测试所有 12+5+5=17 个已有内核 + 新增的 Batch 2+3 内核
"""

import sys
import subprocess
import tempfile
import os
import numpy as np

sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/.opencode/plans')

# Import all harnesses
from phase1_fixed_harnesses import FIXED_HARNESSES as PHASE1
from phase2_improved_harnesses import PHASE2_IMPROVED_HARNESSES as PHASE2
from phase5_batch1_harnesses import PHASE5_BATCH1_HARNESSES as BATCH1
from phase5_batch2_harnesses import PHASE5_BATCH2_HARNESSES as BATCH2
from phase5_batch3_harnesses import PHASE5_BATCH3_HARNESSES as BATCH3

# Merge all
ALL_HARNESSES = {}
ALL_HARNESSES.update(PHASE1)
ALL_HARNESSES.update(PHASE2)
ALL_HARNESSES.update(BATCH1)
ALL_HARNESSES.update(BATCH2)
ALL_HARNESSES.update(BATCH3)

def run_cuda(code, kernel_id):
    """Run CUDA code and return output file"""
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
    except Exception as e:
        print(f"  CUDA error: {e}")
        return False

def run_sycl(code, kernel_id):
    """Run SYCL code and return output file"""
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
    except Exception as e:
        print(f"  SYCL error: {e}")
        return False

def compare_outputs(kernel_id):
    """Compare CUDA and SYCL outputs"""
    try:
        cuda_out = np.fromfile('/tmp/output_cuda.bin', dtype=np.float32)
        sycl_out = np.fromfile('/tmp/output_sycl.bin', dtype=np.float32)
        
        if len(cuda_out) != len(sycl_out):
            return False, 1.0, 1.0, f"Size mismatch: {len(cuda_out)} vs {len(sycl_out)}"
        
        diff = np.abs(cuda_out - sycl_out)
        mae = float(np.mean(diff))
        max_err = float(np.max(diff))
        
        # Adjust tolerance for type conversion kernels
        if 'half' in kernel_id or 'fp16' in kernel_id or 'converted' in kernel_id:
            tolerance_mae = 1e-3
            tolerance_max = 1e-2
        else:
            tolerance_mae = 1e-4
            tolerance_max = 1e-3
        
        passed = (mae < tolerance_mae) and (max_err < tolerance_max)
        return passed, mae, max_err, None
    except Exception as e:
        return False, 0, 0, str(e)

def test_kernel(kernel_id, harness):
    """Test a single kernel"""
    print(f"\n{'='*60}")
    print(f"Testing: {kernel_id}")
    print('='*60)
    
    # Run CUDA
    print("  Running CUDA...")
    cuda_ok = run_cuda(harness['cuda'], kernel_id)
    if not cuda_ok:
        print("  ❌ CUDA failed")
        return {'kernel_id': kernel_id, 'passed': False, 'error': 'CUDA failed'}
    
    # Run SYCL
    print("  Running SYCL...")
    sycl_ok = run_sycl(harness['sycl'], kernel_id)
    if not sycl_ok:
        print("  ❌ SYCL failed")
        return {'kernel_id': kernel_id, 'passed': False, 'error': 'SYCL failed'}
    
    # Compare
    print("  Comparing outputs...")
    passed, mae, max_err, error = compare_outputs(kernel_id)
    
    if passed:
        print(f"  ✅ PASS - MAE: {mae:.2e}, Max: {max_err:.2e}")
    else:
        print(f"  ❌ FAIL - MAE: {mae:.2e}, Max: {max_err:.2e}")
        if error:
            print(f"     Error: {error}")
    
    return {
        'kernel_id': kernel_id,
        'passed': passed,
        'mae': mae,
        'max_err': max_err,
        'error': error
    }

def main():
    print("=" * 80)
    print("🚀 Phase 5 Complete - Testing All Harnesses")
    print("=" * 80)
    print(f"\n📊 Total kernels to test: {len(ALL_HARNESSES)}")
    print()
    
    results = []
    
    # Test all kernels
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
    print("📊 FINAL SUMMARY")
    print("=" * 80)
    
    passed_count = sum(1 for r in results if r.get('passed'))
    total = len(results)
    
    print(f"\n✅ Passed: {passed_count}/{total}")
    print(f"❌ Failed: {total - passed_count}/{total}")
    print()
    
    # Group by phase
    phases = {
        'Phase 1': [k for k in PHASE1.keys()],
        'Phase 2': [k for k in PHASE2.keys()],
        'Batch 1': [k for k in BATCH1.keys()],
        'Batch 2': [k for k in BATCH2.keys()],
        'Batch 3': [k for k in BATCH3.keys()]
    }
    
    for phase, kernels in phases.items():
        print(f"\n{phase}:")
        for kernel_id in kernels:
            result = next((r for r in results if r['kernel_id'] == kernel_id), None)
            if result:
                status = "✅" if result.get('passed') else "❌"
                print(f"  {status} {kernel_id}")
    
    print("\n" + "=" * 80)
    print(f"🎯 Progress: {passed_count}/25+ kernels completed")
    print("=" * 80)
    
    # Save consolidated harnesses
    print("\n💾 Saving consolidated harnesses...")
    with open('/home/intel/tianfeng/opencode_bench/.opencode/plans/all_harnesses_consolidated.py', 'w') as f:
        f.write("#!/usr/bin/env python3\n")
        f.write('"""\nConsolidated harnesses - Phase 5 Complete\n"""\n\n')
        f.write('ALL_HARNESSES = {\n')
        
        for kernel_id, harness in ALL_HARNESSES.items():
            f.write(f"    '{kernel_id}': {{\n")
            f.write(f"        'cuda': '''\n")
            f.write(harness['cuda'])
            f.write("''',\n")
            f.write(f"        'sycl': '''\n")
            f.write(harness['sycl'])
            f.write("'''\n")
            f.write("    },\n")
        
        f.write('}\n')
    
    print("✅ Saved to: all_harnesses_consolidated.py")
    
    return results

if __name__ == '__main__':
    results = main()
