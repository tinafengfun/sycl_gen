#!/usr/bin/env python3
"""
CUDA-SYCL Kernel Accuracy Test Suite
CUDA-SYCL 内核准确度测试套件

This script runs accuracy tests for CUDA to SYCL kernel conversions.
此脚本运行 CUDA 到 SYCL 内核转换的准确度测试。

Usage:
  python3 run_tests.py [options]

Options:
  --kernel KERNEL    Test specific kernel (default: all)
  --list            List all available kernels
  --verify          Run verification checks
  --parallel N      Run tests in parallel (default: 3)
  --tolerance       Set error tolerance (default: 1e-4)
"""

import sys
import os
import subprocess
import tempfile
import numpy as np
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add harnesses to path (从 src/harnesses 目录导入)
HARNESS_PATH = Path(__file__).parent.parent / 'src' / 'harnesses'
sys.path.insert(0, str(HARNESS_PATH))

from all_harnesses import ALL_HARNESSES
from batch4_harnesses import PHASE5_BATCH4_HARNESSES

# Merge all harnesses
ALL_KERNELS = {}
ALL_KERNELS.update(ALL_HARNESSES)
ALL_KERNELS.update(PHASE5_BATCH4_HARNESSES)


class KernelAccuracyTester:
    """Kernel accuracy tester for CUDA-SYCL comparison"""
    
    def __init__(self, cuda_host: str = "10.112.229.160",
                 cuda_container: str = "cuda12.9-test",
                 sycl_container: str = "lsv-container",
                 tolerance_mae: float = 1e-4,
                 tolerance_max: float = 1e-3):
        self.cuda_host = cuda_host
        self.cuda_container = cuda_container
        self.sycl_container = sycl_container
        self.tolerance_mae = tolerance_mae
        self.tolerance_max = tolerance_max
        
    def run_cuda(self, code: str) -> bool:
        """Compile and run CUDA code"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(code)
                cuda_file = f.name
            
            # Copy to remote CUDA host
            subprocess.run(['scp', cuda_file, f'root@{self.cuda_host}:/tmp/test.cu'],
                         capture_output=True, timeout=30, check=True)
            
            # Compile and run in container
            cmd = f'''ssh root@{self.cuda_host} "docker cp /tmp/test.cu {self.cuda_container}:/workspace/test.cu && 
                     docker exec {self.cuda_container} bash -c 'cd /workspace && 
                     nvcc -O2 -Wno-deprecated-gpu-targets test.cu -o test && ./test'"'''
            
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
            os.unlink(cuda_file)
            
            if result.returncode == 0:
                # Copy output back
                subprocess.run(['ssh', f'root@{self.cuda_host}', 
                               f'docker cp {self.cuda_container}:/workspace/output_cuda.bin /tmp/'],
                              capture_output=True, check=True)
                subprocess.run(['scp', f'root@{self.cuda_host}:/tmp/output_cuda.bin', '/tmp/'],
                              capture_output=True, check=True)
                return True
            return False
        except Exception as e:
            print(f"CUDA error: {e}")
            return False
    
    def run_sycl(self, code: str) -> bool:
        """Compile and run SYCL code"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(code)
                sycl_file = f.name
            
            # Copy to SYCL container
            subprocess.run(['docker', 'cp', sycl_file, f'{self.sycl_container}:/workspace/test.cpp'],
                          capture_output=True, timeout=10, check=True)
            
            # Compile and run
            result = subprocess.run(['docker', 'exec', self.sycl_container, 'bash', '-c',
                                   'cd /workspace && icpx -fsycl -O2 test.cpp -o test && ./test'],
                                  capture_output=True, timeout=120)
            os.unlink(sycl_file)
            
            if result.returncode == 0:
                # Copy output back
                subprocess.run(['docker', 'cp', f'{self.sycl_container}:/workspace/output_sycl.bin', '/tmp/'],
                              capture_output=True, check=True)
                return True
            return False
        except Exception as e:
            print(f"SYCL error: {e}")
            return False
    
    def compare_outputs(self, kernel_id: str) -> Tuple[bool, float, float]:
        """Compare CUDA and SYCL outputs"""
        try:
            cuda_out = np.fromfile('/tmp/output_cuda.bin', dtype=np.float32)
            sycl_out = np.fromfile('/tmp/output_sycl.bin', dtype=np.float32)
            
            if len(cuda_out) != len(sycl_out):
                return False, 1.0, 1.0
            
            diff = np.abs(cuda_out - sycl_out)
            mae = float(np.mean(diff))
            max_err = float(np.max(diff))
            
            # Adjust tolerance for half-precision kernels
            if any(x in kernel_id for x in ['half', 'fp16', 'converted']):
                tolerance_mae = 1e-3
                tolerance_max = 1e-2
            else:
                tolerance_mae = self.tolerance_mae
                tolerance_max = self.tolerance_max
            
            passed = (mae < tolerance_mae) and (max_err < tolerance_max)
            return passed, mae, max_err
        except Exception as e:
            return False, 0.0, 0.0
    
    def test_kernel(self, kernel_id: str) -> Dict:
        """Test a single kernel"""
        print(f"\n{'='*60}")
        print(f"Testing: {kernel_id}")
        print('='*60)
        
        if kernel_id not in ALL_KERNELS:
            return {'kernel_id': kernel_id, 'passed': False, 'error': 'Kernel not found'}
        
        harness = ALL_KERNELS[kernel_id]
        
        # Run CUDA
        print("  Running CUDA...")
        cuda_ok = self.run_cuda(harness['cuda'])
        if not cuda_ok:
            print("  ❌ CUDA failed")
            return {'kernel_id': kernel_id, 'passed': False, 'error': 'CUDA failed'}
        
        # Run SYCL
        print("  Running SYCL...")
        sycl_ok = self.run_sycl(harness['sycl'])
        if not sycl_ok:
            print("  ❌ SYCL failed")
            return {'kernel_id': kernel_id, 'passed': False, 'error': 'SYCL failed'}
        
        # Compare
        print("  Comparing outputs...")
        passed, mae, max_err = self.compare_outputs(kernel_id)
        
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
    
    def run_all_tests(self) -> List[Dict]:
        """Run all kernel tests"""
        print("=" * 80)
        print("🚀 CUDA-SYCL Kernel Accuracy Test Suite")
        print("=" * 80)
        print(f"\nTotal kernels: {len(ALL_KERNELS)}")
        print()
        
        results = []
        for kernel_id in sorted(ALL_KERNELS.keys()):
            try:
                result = self.test_kernel(kernel_id)
                results.append(result)
            except Exception as e:
                print(f"  ❌ Exception: {e}")
                results.append({
                    'kernel_id': kernel_id,
                    'passed': False,
                    'error': str(e)
                })
        
        # Summary
        self.print_summary(results)
        return results
    
    def print_summary(self, results: List[Dict]):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        
        passed_count = sum(1 for r in results if r.get('passed'))
        total = len(results)
        
        print(f"\n✅ Passed: {passed_count}/{total}")
        print(f"❌ Failed: {total - passed_count}/{total}")
        
        if passed_count >= 25:
            print(f"\n🎉 SUCCESS! Target of 25+ kernels achieved!")
        else:
            print(f"\n⏳ Need {25 - passed_count} more kernels to reach target")
        
        # List passed kernels
        print("\n✅ Passed kernels:")
        for r in results:
            if r.get('passed'):
                print(f"  - {r['kernel_id']}")
        
        # List failed kernels
        failed = [r for r in results if not r.get('passed')]
        if failed:
            print("\n❌ Failed kernels:")
            for r in failed:
                print(f"  - {r['kernel_id']}: {r.get('error', 'Unknown error')}")
        
        print("\n" + "=" * 80)


def list_kernels():
    """List all available kernels"""
    print("=" * 80)
    print("📋 Available Kernels")
    print("=" * 80)
    print()
    
    categories = {
        'Vector Operations': ['add_vectors', 'add_vectors_hnc_nhc', 'add_bias_nchw', 'add_bias_batched'],
        'Data Conversion': ['copy_type_converted', 'nchw_to_nhwc'],
        'Normalization': ['batch_norm', 'layer_norm', 'global_scale', 'global_scale_fp16_nhwc'],
        'Pooling': ['global_avg_pool', 'global_avg_pool_nhwc_fp16'],
        'Winograd': ['winograd_input_transform', 'winograd_filter_transform', 
                    'winograd_output_transform', 'winograd_output_relu_input',
                    'winograd_output_se_relu_input', 'output_input_transform_fp16_shmem'],
        'Attention': ['softmax', 'softmax_opt_64', 'policy_map', 'promotion_logits',
                     'preprocess_attention_body', 'input_gating', 'gen_offset_pointers',
                     'se_layer_nhwc'],
    }
    
    for category, kernels in categories.items():
        print(f"\n{category}:")
        for k in kernels:
            status = "✅" if k in ALL_KERNELS else "❌"
            print(f"  {status} {k}")
    
    print(f"\n{'='*80}")
    print(f"Total: {len(ALL_KERNELS)} kernels")
    print('='*80)


def main():
    parser = argparse.ArgumentParser(
        description='CUDA-SYCL Kernel Accuracy Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python3 run_tests.py
  
  # Test specific kernel
  python3 run_tests.py --kernel add_vectors
  
  # List all kernels
  python3 run_tests.py --list
        """
    )
    
    parser.add_argument('--kernel', type=str, help='Test specific kernel')
    parser.add_argument('--list', action='store_true', help='List all kernels')
    parser.add_argument('--tolerance', type=float, default=1e-4, help='Error tolerance')
    parser.add_argument('--cuda-host', type=str, default='10.112.229.160', help='CUDA host IP')
    parser.add_argument('--cuda-container', type=str, default='cuda12.9-test', help='CUDA container name')
    parser.add_argument('--sycl-container', type=str, default='lsv-container', help='SYCL container name')
    
    args = parser.parse_args()
    
    if args.list:
        list_kernels()
        return
    
    tester = KernelAccuracyTester(
        cuda_host=args.cuda_host,
        cuda_container=args.cuda_container,
        sycl_container=args.sycl_container,
        tolerance_mae=args.tolerance
    )
    
    if args.kernel:
        if args.kernel not in ALL_KERNELS:
            print(f"Error: Kernel '{args.kernel}' not found")
            print(f"Use --list to see available kernels")
            return 1
        result = tester.test_kernel(args.kernel)
        return 0 if result['passed'] else 1
    else:
        results = tester.run_all_tests()
        passed = sum(1 for r in results if r.get('passed'))
        return 0 if passed >= 25 else 1


if __name__ == '__main__':
    sys.exit(main())
