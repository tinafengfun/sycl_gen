#!/usr/bin/env python3
"""
Debug version of Phase 5 Batch 1 test with detailed error logging
"""

import sys
import subprocess
import tempfile
import os
sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/.opencode/plans')

from phase5_batch1_harnesses import PHASE5_BATCH1_HARNESSES

def run_cuda_debug(kernel_id, code):
    """Debug version with error output"""
    print(f"\n🔍 Debugging CUDA for {kernel_id}...")
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(code)
            cuda_file = f.name
        
        # Copy to remote
        result = subprocess.run(['scp', cuda_file, 'root@10.112.229.160:/tmp/test.cu'],
                      capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"  ❌ SCP failed: {result.stderr}")
            return False
        
        # Compile and run
        cmd = '''ssh root@10.112.229.160 "docker cp /tmp/test.cu cuda12.9-test:/workspace/test.cu && 
                 docker exec cuda12.9-test bash -c 'cd /workspace && 
                 nvcc -O2 -Wno-deprecated-gpu-targets test.cu -o test 2>&1 && ./test'"'''
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        os.unlink(cuda_file)
        
        if result.returncode != 0:
            print(f"  ❌ CUDA execution failed:")
            print(f"     stdout: {result.stdout}")
            print(f"     stderr: {result.stderr}")
            return False
        else:
            print(f"  ✅ CUDA success")
            return True
            
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False

def run_sycl_debug(kernel_id, code):
    """Debug version with error output"""
    print(f"🔍 Debugging SYCL for {kernel_id}...")
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(code)
            sycl_file = f.name
        
        # Copy to container
        result = subprocess.run(['docker', 'cp', sycl_file, 'lsv-container:/workspace/test.cpp'],
                      capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print(f"  ❌ Docker cp failed: {result.stderr}")
            return False
        
        # Compile and run
        result = subprocess.run(['docker', 'exec', 'lsv-container', 'bash', '-c',
                               'cd /workspace && icpx -fsycl -O2 test.cpp -o test 2>&1 && ./test'],
                              capture_output=True, text=True, timeout=120)
        os.unlink(sycl_file)
        
        if result.returncode != 0:
            print(f"  ❌ SYCL execution failed:")
            print(f"     stdout: {result.stdout}")
            print(f"     stderr: {result.stderr}")
            return False
        else:
            print(f"  ✅ SYCL success")
            return True
            
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False

def main():
    print("=" * 80)
    print("🔍 Phase 5 Batch 1 - Debug Test")
    print("=" * 80)
    
    for kernel_id in PHASE5_BATCH1_HARNESSES:
        print(f"\n{'='*60}")
        print(f"Testing: {kernel_id}")
        print('='*60)
        
        harness = PHASE5_BATCH1_HARNESSES[kernel_id]
        
        # Test CUDA
        cuda_ok = run_cuda_debug(kernel_id, harness['cuda'])
        
        # Test SYCL
        sycl_ok = run_sycl_debug(kernel_id, harness['sycl'])
        
        if cuda_ok and sycl_ok:
            print(f"\n✅ {kernel_id}: Both CUDA and SYCL passed!")
        else:
            print(f"\n❌ {kernel_id}: Failed (CUDA: {cuda_ok}, SYCL: {sycl_ok})")

if __name__ == '__main__':
    main()
