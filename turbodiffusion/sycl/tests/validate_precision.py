#!/usr/bin/env python3
"""
Precision validation for SYCL kernels vs PyTorch reference implementations.

This script validates the numerical correctness of SYCL Flash Attention and
Sparse Attention implementations against PyTorch reference implementations.
"""

import torch
import numpy as np
import sys
import os

# Ensure the module can be found
sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/turbodiffusion-sycl')
sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/turbodiffusion-sycl/operators')

from turbodiffusion_sycl import FlashAttentionSYCL, SparseAttentionSYCL


def validate_flash_attention():
    """Validate Flash Attention numerical correctness."""
    print("=" * 60)
    print("Validating Flash Attention")
    print("=" * 60)
    
    # Test configurations
    configs = [
        {'B': 2, 'H': 12, 'S': 1024, 'D': 128},
        {'B': 1, 'H': 12, 'S': 2048, 'D': 128},
        {'B': 2, 'H': 8, 'S': 1024, 'D': 64},  # GQA
    ]
    
    results = []
    for config in configs:
        B, H, S, D = config['B'], config['H'], config['S'], config['D']
        print(f"\nConfig: B={B}, H={H}, S={S}, D={D}")
        
        try:
            # Generate test tensors
            torch.manual_seed(42)
            q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cpu')
            k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cpu')
            v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cpu')
            
            # PyTorch reference (using standard attention)
            with torch.no_grad():
                scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(D)
                attn = torch.softmax(scores, dim=-1)
                ref_output = torch.matmul(attn, v)
            
            # SYCL implementation
            try:
                fa = FlashAttentionSYCL(head_dim=D, num_heads=H)
                # Move to XPU
                q_xpu = q.to('xpu')
                k_xpu = k.to('xpu')
                v_xpu = v.to('xpu')
                sycl_output = fa(q_xpu, k_xpu, v_xpu)
                
                # Move back to CPU for comparison
                sycl_output = sycl_output.cpu()
                
                # Compare
                max_error = (ref_output.float() - sycl_output.float()).abs().max().item()
                mean_error = (ref_output.float() - sycl_output.float()).abs().mean().item()
                
                print(f"  Max error: {max_error:.6e}")
                print(f"  Mean error: {mean_error:.6e}")
                
                # Check if within tolerance (BF16 has ~1e-3 precision)
                passed = max_error < 1e-2 and not torch.isnan(sycl_output).any()
                print(f"  Status: {'PASS' if passed else 'FAIL'}")
                
                results.append({
                    'config': config,
                    'max_error': max_error,
                    'mean_error': mean_error,
                    'passed': passed
                })
            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    'config': config,
                    'error': str(e),
                    'passed': False
                })
        except Exception as e:
            print(f"  Error setting up test: {e}")
            results.append({
                'config': config,
                'error': str(e),
                'passed': False
            })
    
    return results


def validate_sparse_attention():
    """Validate Sparse Attention numerical correctness."""
    print("\n" + "=" * 60)
    print("Validating Sparse Attention")
    print("=" * 60)
    
    # Test configurations with different topk ratios
    configs = [
        {'B': 2, 'H': 8, 'S': 512, 'D': 64, 'topk': 0.5},
        {'B': 1, 'H': 12, 'S': 1024, 'D': 64, 'topk': 0.3},
    ]
    
    results = []
    for config in configs:
        B, H, S, D, topk = config['B'], config['H'], config['S'], config['D'], config['topk']
        print(f"\nConfig: B={B}, H={H}, S={S}, D={D}, topk={topk}")
        
        try:
            # Generate test tensors
            torch.manual_seed(42)
            q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cpu')
            k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cpu')
            v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cpu')
            
            # PyTorch reference (using standard attention)
            with torch.no_grad():
                scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(D)
                attn = torch.softmax(scores, dim=-1)
                ref_output = torch.matmul(attn, v)
            
            # SYCL implementation
            try:
                sa = SparseAttentionSYCL(head_dim=D, topk=topk)
                # Move to XPU
                q_xpu = q.to('xpu')
                k_xpu = k.to('xpu')
                v_xpu = v.to('xpu')
                sycl_output = sa(q_xpu, k_xpu, v_xpu)
                
                # Move back to CPU for comparison
                sycl_output = sycl_output.cpu()
                
                # Compare
                max_error = (ref_output.float() - sycl_output.float()).abs().max().item()
                mean_error = (ref_output.float() - sycl_output.float()).abs().mean().item()
                
                print(f"  Max error: {max_error:.6e}")
                print(f"  Mean error: {mean_error:.6e}")
                
                # Check if within tolerance
                passed = max_error < 1e-2 and not torch.isnan(sycl_output).any()
                print(f"  Status: {'PASS' if passed else 'FAIL'}")
                
                results.append({
                    'config': config,
                    'max_error': max_error,
                    'mean_error': mean_error,
                    'passed': passed
                })
            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    'config': config,
                    'error': str(e),
                    'passed': False
                })
        except Exception as e:
            print(f"  Error setting up test: {e}")
            results.append({
                'config': config,
                'error': str(e),
                'passed': False
            })
    
    return results


def generate_report(results_flash, results_sparse):
    """Generate precision validation report."""
    # Ensure results directory exists
    os.makedirs('/home/intel/tianfeng/opencode_bench/turbodiffusion-sycl/results', exist_ok=True)
    report_path = '/home/intel/tianfeng/opencode_bench/turbodiffusion-sycl/results/precision_validation.md'
    
    with open(report_path, 'w') as f:
        f.write("# SYCL Kernel Precision Validation Report\n\n")
        f.write(f"**Date**: {torch.__version__}\n")
        f.write(f"**PyTorch Version**: {torch.__version__}\n")
        f.write(f"**Device**: Intel XPU\n\n")
        
        # Flash Attention results
        f.write("## Flash Attention\n\n")
        f.write("| Config | Max Error | Mean Error | Status |\n")
        f.write("|--------|-----------|------------|--------|\n")
        for r in results_flash:
            if 'error' in r:
                f.write(f"| {r['config']} | N/A | N/A | ERROR: {r['error'][:30]}... |\n")
            else:
                cfg = r['config']
                cfg_str = f"B={cfg['B']}, H={cfg['H']}, S={cfg['S']}, D={cfg['D']}"
                status = "✅ PASS" if r['passed'] else "❌ FAIL"
                f.write(f"| {cfg_str} | {r['max_error']:.6e} | {r['mean_error']:.6e} | {status} |\n")
        
        # Sparse Attention results
        f.write("\n## Sparse Attention\n\n")
        f.write("| Config | Max Error | Mean Error | Status |\n")
        f.write("|--------|-----------|------------|--------|\n")
        for r in results_sparse:
            if 'error' in r:
                f.write(f"| {r['config']} | N/A | N/A | ERROR: {r['error'][:30]}... |\n")
            else:
                cfg = r['config']
                cfg_str = f"B={cfg['B']}, H={cfg['H']}, S={cfg['S']}, D={cfg['D']}, topk={cfg['topk']}"
                status = "✅ PASS" if r['passed'] else "❌ FAIL"
                f.write(f"| {cfg_str} | {r['max_error']:.6e} | {r['mean_error']:.6e} | {status} |\n")
        
        # Summary
        f.write("\n## Summary\n\n")
        flash_passed = sum(1 for r in results_flash if r.get('passed', False))
        flash_total = len(results_flash)
        sparse_passed = sum(1 for r in results_sparse if r.get('passed', False))
        sparse_total = len(results_sparse)
        
        f.write(f"- **Flash Attention**: {flash_passed}/{flash_total} tests passed\n")
        f.write(f"- **Sparse Attention**: {sparse_passed}/{sparse_total} tests passed\n")
        
        if flash_passed == flash_total and sparse_passed == sparse_total:
            f.write("\n✅ **All validation tests passed!**\n")
        else:
            f.write("\n⚠️ **Some tests failed**\n")
    
    print(f"\nReport saved to: {report_path}")


def main():
    print("SYCL Kernel Precision Validation")
    print("=" * 60)
    
    # Run validations
    flash_results = validate_flash_attention()
    sparse_results = validate_sparse_attention()
    
    # Generate report
    generate_report(flash_results, sparse_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    flash_passed = sum(1 for r in flash_results if r.get('passed', False))
    sparse_passed = sum(1 for r in sparse_results if r.get('passed', False))
    print(f"Flash Attention: {flash_passed}/{len(flash_results)} tests passed")
    print(f"Sparse Attention: {sparse_passed}/{len(sparse_results)} tests passed")
    
    # Return exit code
    if flash_passed == len(flash_results) and sparse_passed == len(sparse_results):
        print("\n🎉 All precision validation tests passed!")
        return 0
    else:
        print("\n⚠️  Some precision validation tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
