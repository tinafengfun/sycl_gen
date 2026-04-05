#!/usr/bin/env python3
"""
Benchmark SYCL kernels against PyTorch baseline.
"""

import torch
import time
import json
import sys
sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/TurboDiffusion/turbodiffusion')

from turbodiffusion_sycl import FlashAttentionSYCL, SparseAttentionSYCL


def benchmark_flash_attention():
    """Benchmark flash attention for different sequence lengths."""
    results = {}
    
    device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
    
    for seq_len in [1024, 2048, 4096, 8192]:
        print(f"Benchmarking Flash Attention with seq_len={seq_len}")
        
        # Create tensors
        batch_size = 2
        num_heads = 12
        head_dim = 128
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        
        # Warmup
        for _ in range(10):
            _ = FlashAttentionSYCL.apply(q, k, v)
        
        # Benchmark
        torch.xpu.synchronize() if device.type == 'xpu' else None
        start = time.time()
        
        num_iters = 100
        for _ in range(num_iters):
            out = FlashAttentionSYCL.apply(q, k, v)
        
        torch.xpu.synchronize() if device.type == 'xpu' else None
        elapsed = time.time() - start
        
        throughput = (batch_size * seq_len * num_iters) / elapsed
        
        results[f"seq_{seq_len}"] = {
            "time_ms": elapsed * 1000 / num_iters,
            "throughput_tokens_per_sec": throughput,
            "memory_gb": torch.xpu.memory_allocated() / 1e9 if device.type == 'xpu' else 0
        }
    
    return results


def benchmark_sparse_attention():
    """Benchmark sparse attention for different sequence lengths."""
    results = {}
    
    device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
    topk_ratios = [0.1, 0.2, 0.3]
    
    for seq_len in [1024, 2048, 4096, 8192]:
        print(f"Benchmarking Sparse Attention with seq_len={seq_len}")
        
        # Create tensors
        batch_size = 2
        num_heads = 12
        head_dim = 128
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        
        seq_results = {}
        for topk in topk_ratios:
            # Warmup
            for _ in range(10):
                _ = SparseAttentionSYCL.apply(q, k, v, topk)
            
            # Benchmark
            torch.xpu.synchronize() if device.type == 'xpu' else None
            start = time.time()
            
            num_iters = 100
            for _ in range(num_iters):
                out = SparseAttentionSYCL.apply(q, k, v, topk)
            
            torch.xpu.synchronize() if device.type == 'xpu' else None
            elapsed = time.time() - start
            
            throughput = (batch_size * seq_len * num_iters) / elapsed
            
            seq_results[f"topk_{topk}"] = {
                "time_ms": elapsed * 1000 / num_iters,
                "throughput_tokens_per_sec": throughput
            }
        
        results[f"seq_{seq_len}"] = seq_results
    
    return results


def main():
    print("Starting SYCL Benchmark Suite")
    print("=" * 50)
    
    results = {
        'flash_attention': benchmark_flash_attention(),
        'sparse_attention': benchmark_sparse_attention(),
    }
    
    # Ensure results directory exists
    results_dir = Path('/home/intel/tianfeng/opencode_bench/turbodiffusion-sycl/results')
    results_dir.mkdir(exist_ok=True)
    
    output_path = results_dir / 'benchmark_sycl.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark complete. Results saved to {output_path}")
    
    # Print summary
    print("\nSummary:")
    print("-" * 50)
    for kernel_type, data in results.items():
        print(f"\n{kernel_type.upper()}:")
        for config, metrics in data.items():
            print(f"  {config}: {metrics}")


if __name__ == '__main__':
    from pathlib import Path
    main()
