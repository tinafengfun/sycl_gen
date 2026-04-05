#!/usr/bin/env python3
"""
Compare SYCL implementation against CUDA reference.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/TurboDiffusion/turbodiffusion')

from turbodiffusion_sycl import FlashAttentionSYCL, SparseAttentionSYCL


def compare_attention_outputs():
    """Generate same inputs, run on both platforms, compare outputs."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    print("Comparing attention outputs between SYCL and reference...")
    
    # Generate test tensors
    batch_size = 2
    num_heads = 12
    seq_len = 1024
    head_dim = 128
    
    # Create reference on CPU first
    q_cpu = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k_cpu = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v_cpu = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Reference: PyTorch native attention (CPU)
    scale = head_dim ** -0.5
    attn_cpu = torch.matmul(q_cpu, k_cpu.transpose(-2, -1)) * scale
    attn_weights_cpu = torch.softmax(attn_cpu, dim=-1)
    output_ref = torch.matmul(attn_weights_cpu, v_cpu)
    
    # SYCL implementation
    device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
    q_sycl = q_cpu.to(device)
    k_sycl = k_cpu.to(device)
    v_sycl = v_cpu.to(device)
    
    output_sycl = FlashAttentionSYCL.apply(q_sycl, k_sycl, v_sycl)
    output_sycl_cpu = output_sycl.cpu()
    
    # Compare
    max_diff = torch.max(torch.abs(output_ref - output_sycl_cpu)).item()
    mean_diff = torch.mean(torch.abs(output_ref - output_sycl_cpu)).item()
    
    print(f"\nFlash Attention Comparison:")
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")
    print(f"  Shape: {output_ref.shape}")
    
    # Test sparse attention
    print("\n\nSparse Attention Comparison:")
    topk = 0.2
    output_sparse_sycl = SparseAttentionSYCL.apply(q_sycl, k_sycl, v_sycl, topk)
    output_sparse_cpu = output_sparse_sycl.cpu()
    
    max_diff_sparse = torch.max(torch.abs(output_ref - output_sparse_cpu)).item()
    mean_diff_sparse = torch.mean(torch.abs(output_ref - output_sparse_cpu)).item()
    
    print(f"  Max difference: {max_diff_sparse:.6e}")
    print(f"  Mean difference: {mean_diff_sparse:.6e}")
    
    return {
        'flash_attention': {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'passed': max_diff < 1e-3
        },
        'sparse_attention': {
            'max_diff': max_diff_sparse,
            'mean_diff': mean_diff_sparse,
            'passed': max_diff_sparse < 1e-2  # Sparse has higher tolerance
        }
    }


def main():
    print("=" * 60)
    print("SYCL vs Reference Implementation Comparison")
    print("=" * 60)
    
    results = compare_attention_outputs()
    
    # Generate report
    report_lines = [
        "# SYCL Implementation Comparison Report\n",
        "## Flash Attention\n",
        f"- Max difference: {results['flash_attention']['max_diff']:.6e}\n",
        f"- Mean difference: {results['flash_attention']['mean_diff']:.6e}\n",
        f"- Status: {'PASS' if results['flash_attention']['passed'] else 'FAIL'}\n",
        "\n## Sparse Attention\n",
        f"- Max difference: {results['sparse_attention']['max_diff']:.6e}\n",
        f"- Mean difference: {results['sparse_attention']['mean_diff']:.6e}\n",
        f"- Status: {'PASS' if results['sparse_attention']['passed'] else 'FAIL'}\n",
        "\n## Notes\n",
        "- Tolerance: 1e-3 for Flash Attention, 1e-2 for Sparse Attention\n",
        "- Differences expected due to floating point precision and algorithm variations\n"
    ]
    
    # Save report
    from pathlib import Path
    results_dir = Path('/home/intel/tianfeng/opencode_bench/turbodiffusion-sycl/results')
    results_dir.mkdir(exist_ok=True)
    
    report_path = results_dir / 'comparison_report.md'
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    
    print(f"\n\nReport saved to: {report_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    all_passed = all(r['passed'] for r in results.values())
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")


if __name__ == '__main__':
    main()
