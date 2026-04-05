#!/usr/bin/env python3
"""
Detailed XPU performance benchmarking and profiling.
Uses PyTorch reference implementation since SYCL ops module is not compiled.
"""

import torch
import time
import json
import sys
import math

# Add the package to path
sys.path.insert(0, '/workspace/turbodiffusion-sycl')

# Import with warning suppression
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from turbodiffusion_sycl.attention import FlashAttentionSYCL, SparseAttentionSYCL


def benchmark_flash_attention_detailed():
    """Benchmark Flash Attention with different configurations."""
    print("=" * 70)
    print("Flash Attention Performance Benchmark")
    print("=" * 70)
    
    configs = [
        # (B, H, S, D, name)
        (1, 12, 1024, 128, "Wan2.1-1.3B-default"),
        (1, 12, 2048, 128, "2x-sequence"),
        (1, 12, 4096, 128, "4x-sequence"),
        (2, 12, 1024, 128, "2x-batch"),
        (1, 8, 1024, 64, "GQA-small"),
    ]
    
    results = []
    device = torch.device('xpu')
    
    # Check if SYCL ops are actually available
    sycl_available = False
    try:
        import turbodiffusion_sycl_ops
        sycl_available = True
        print("✓ SYCL operations module loaded")
    except ImportError:
        print("⚠ SYCL operations module not available - using PyTorch reference")
    
    for B, H, S, D, name in configs:
        print(f"\nConfig: {name} (B={B}, H={H}, S={S}, D={D})")
        
        # Create tensors
        q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
        
        # Warmup - use standard PyTorch attention as reference
        for _ in range(10):
            # Standard scaled dot-product attention
            scale = 1.0 / math.sqrt(D)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(scores, dim=-1)
            _ = torch.matmul(attn_weights, v)
        torch.xpu.synchronize()
        
        # Benchmark
        num_iters = 50  # Reduced for faster execution
        start = time.perf_counter()
        for _ in range(num_iters):
            scale = 1.0 / math.sqrt(D)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, v)
        torch.xpu.synchronize()
        elapsed = time.perf_counter() - start
        
        # Metrics
        time_per_iter = elapsed / num_iters * 1000  # ms
        # FLOPs for attention: 2 * B * H * S * S * D (Q@K^T) + 2 * B * H * S * S * D (Attn@V)
        # = 4 * B * H * S^2 * D
        flops = 4 * B * H * S * S * D
        throughput = flops / (time_per_iter / 1000) / 1e12  # TFLOPS
        # Memory traffic: Q + K + V reads + output write = 4 * B * H * S * D bytes (BF16=2 bytes each)
        memory_traffic_bytes = B * H * S * D * 4 * 2  # 4 tensors (Q,K,V,Out), 2 bytes each
        memory_bw = memory_traffic_bytes / (time_per_iter / 1000) / 1e9  # GB/s
        
        print(f"  Time: {time_per_iter:.3f} ms/iter")
        print(f"  Throughput: {throughput:.2f} TFLOPS")
        print(f"  Memory BW: {memory_bw:.2f} GB/s")
        print(f"  Implementation: {'SYCL' if sycl_available else 'PyTorch'}")
        
        results.append({
            'config': name,
            'B': B, 'H': H, 'S': S, 'D': D,
            'time_ms': time_per_iter,
            'tflops': throughput,
            'memory_bw_gbps': memory_bw,
            'implementation': 'SYCL' if sycl_available else 'PyTorch'
        })
    
    return results


def benchmark_sparse_attention_detailed():
    """Benchmark Sparse Attention with different topk ratios."""
    print("\n" + "=" * 70)
    print("Sparse Attention Performance Benchmark")
    print("=" * 70)
    
    configs = [
        # (B, H, S, D, topk, name)
        (1, 12, 1024, 128, 0.2, "topk-0.2"),
        (1, 12, 1024, 128, 0.1, "topk-0.1"),
        (1, 12, 2048, 128, 0.2, "2x-seq-topk-0.2"),
    ]
    
    results = []
    device = torch.device('xpu')
    
    sycl_available = False
    try:
        import turbodiffusion_sycl_ops
        sycl_available = True
    except ImportError:
        pass
    
    for B, H, S, D, topk, name in configs:
        print(f"\nConfig: {name} (B={B}, H={H}, S={S}, D={D}, topk={topk})")
        
        q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
        
        # Warmup - simplified sparse attention using topk
        for _ in range(10):
            # Compute full attention but mask out based on block importance
            scale = 1.0 / math.sqrt(D)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # Simplified sparse masking: keep only topk fraction of attention scores per query
            k_val = int(S * topk)
            topk_scores, _ = torch.topk(scores, k_val, dim=-1)
            threshold = topk_scores[:, :, :, -1:]
            mask = scores >= threshold
            scores_masked = scores.masked_fill(~mask, float('-inf'))
            attn_weights = torch.softmax(scores_masked, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, 0.0)
            _ = torch.matmul(attn_weights, v)
        torch.xpu.synchronize()
        
        # Benchmark
        num_iters = 50
        start = time.perf_counter()
        for _ in range(num_iters):
            scale = 1.0 / math.sqrt(D)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            k_val = int(S * topk)
            topk_scores, _ = torch.topk(scores, k_val, dim=-1)
            threshold = topk_scores[:, :, :, -1:]
            mask = scores >= threshold
            scores_masked = scores.masked_fill(~mask, float('-inf'))
            attn_weights = torch.softmax(scores_masked, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, 0.0)
            out = torch.matmul(attn_weights, v)
        torch.xpu.synchronize()
        elapsed = time.perf_counter() - start
        
        time_per_iter = elapsed / num_iters * 1000
        # For sparse attention, only topk fraction of compute
        flops_sparse = 4 * B * H * S * S * D * topk
        throughput = flops_sparse / (time_per_iter / 1000) / 1e12
        speedup_theoretical = 1.0 / topk
        
        print(f"  Time: {time_per_iter:.3f} ms/iter")
        print(f"  Throughput: {throughput:.2f} TFLOPS (sparse)")
        print(f"  Theoretical speedup vs dense: {speedup_theoretical:.1f}x")
        print(f"  Implementation: {'SYCL' if sycl_available else 'PyTorch'}")
        
        results.append({
            'config': name,
            'topk': topk,
            'time_ms': time_per_iter,
            'tflops_sparse': throughput,
            'speedup_theoretical': speedup_theoretical,
            'implementation': 'SYCL' if sycl_available else 'PyTorch'
        })
    
    return results


def profile_with_intel_tools():
    """Profile using Intel profiling tools if available."""
    print("\n" + "=" * 70)
    print("Intel Profiling Tools")
    print("=" * 70)
    
    try:
        import subprocess
        
        # Check for available tools
        tools_check = []
        
        # Check sycl-ls
        try:
            result = subprocess.run(['sycl-ls'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                tools_check.append("✓ sycl-ls: Available")
                print("SYCL Devices:")
                print(result.stdout)
            else:
                tools_check.append("✗ sycl-ls: Not available")
        except:
            tools_check.append("✗ sycl-ls: Not available")
        
        # Check vtune
        try:
            result = subprocess.run(['which', 'vtune'], capture_output=True, text=True)
            if result.returncode == 0:
                tools_check.append("✓ VTune: Available")
            else:
                tools_check.append("✗ VTune: Not available")
        except:
            tools_check.append("✗ VTune: Not available")
        
        # Document available tools and commands
        profile_info = f"""
Profiling Tools Status:
{chr(10).join(tools_check)}

Recommended Profiling Commands:
# Profile GPU hotspots
vtune -collect gpu-hotspots -result-dir vtune_flash_attention -- python benchmark_kernels_xpu.py

# Profile memory bandwidth
vtune -collect memory-access -result-dir vtune_memory -- python benchmark_kernels_xpu.py

# GPU metrics
vtune -collect gpu-compute-performance -result-dir vtune_compute -- python benchmark_kernels_xpu.py

# Check device properties
sycl-ls -v

# Level Zero tracing
export ZE_ENABLE_TRACING_LAYER=1
export ZET_ENABLE_PROGRAM_DEBUGGING=1
"""
        print(profile_info)
        
        with open('/workspace/turbodiffusion-sycl/results/profiling_commands.sh', 'w') as f:
            f.write(profile_info)
        
        return tools_check
        
    except Exception as e:
        print(f"Error checking profiling tools: {e}")
        return []


def generate_performance_report(flash_results, sparse_results):
    """Generate performance report."""
    report = {
        'device': 'Intel XPU (Battlemage G21)',
        'date': '2025-04-03',
        'device_info': {},
        'flash_attention': flash_results,
        'sparse_attention': sparse_results
    }
    
    # Try to get device info
    try:
        if torch.xpu.is_available():
            report['device_info']['name'] = torch.xpu.get_device_name()
            report['device_info']['available'] = True
    except:
        report['device_info']['available'] = False
    
    output_path = '/workspace/turbodiffusion-sycl/results/performance_xpu.json'
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nPerformance report saved to: {output_path}")
    
    # Generate markdown summary
    md_path = '/workspace/turbodiffusion-sycl/results/PERFORMANCE_XPU.md'
    with open(md_path, 'w') as f:
        f.write("# XPU Performance Benchmark Results\n\n")
        f.write("## Device Information\n\n")
        f.write(f"- **Device**: Intel XPU (Battlemage G21)\n")
        f.write(f"- **Date**: 2025-04-03\n")
        f.write(f"- **PyTorch**: {torch.__version__}\n")
        f.write(f"- **Implementation**: {flash_results[0].get('implementation', 'Unknown')}\n\n")
        
        f.write("## Flash Attention\n\n")
        f.write("| Config | B | H | S | D | Time (ms) | TFLOPS | Memory BW (GB/s) |\n")
        f.write("|--------|---|---|---|---|-----------|--------|------------------|\n")
        for r in flash_results:
            f.write(f"| {r['config']} | {r['B']} | {r['H']} | {r['S']} | {r['D']} | {r['time_ms']:.3f} | {r['tflops']:.2f} | {r['memory_bw_gbps']:.2f} |\n")
        
        f.write("\n## Sparse Attention\n\n")
        f.write("| Config | Top-k | Time (ms) | Sparse TFLOPS | Theoretical Speedup |\n")
        f.write("|--------|-------|-----------|---------------|---------------------|\n")
        for r in sparse_results:
            f.write(f"| {r['config']} | {r['topk']} | {r['time_ms']:.3f} | {r['tflops_sparse']:.2f} | {r['speedup_theoretical']:.1f}x |\n")
        
        f.write("\n## Performance Analysis\n\n")
        f.write("### Theoretical Peak Performance\n\n")
        f.write("Intel Battlemage G21 specifications:\n")
        f.write("- XMX (Matrix Engine): ~100 TFLOPS (BF16)\n")
        f.write("- Vector Engine: ~20 TFLOPS (FP32)\n")
        f.write("- Memory Bandwidth: ~560 GB/s\n\n")
        
        f.write("### Observed Performance\n\n")
        if flash_results:
            max_tflops = max(r['tflops'] for r in flash_results)
            f.write(f"- **Max Achieved TFLOPS**: {max_tflops:.2f}\n")
            f.write(f"- **XMX Efficiency**: {(max_tflops / 100 * 100):.1f}%\n\n")
        
        f.write("### Optimization Recommendations\n\n")
        f.write("1. **Memory Bandwidth**: Flash Attention is typically memory-bound\n")
        f.write("2. **Tile Sizes**: Optimize block sizes for Xe2 architecture\n")
        f.write("3. **Occupancy**: Maximize work-group occupancy\n")
        f.write("4. **XMX Utilization**: Ensure kernels use DPAS instructions\n\n")
    
    print(f"Markdown report saved to: {md_path}")


def main():
    print("Intel XPU Performance Benchmark")
    print("=" * 70)
    
    # Check device
    if not torch.xpu.is_available():
        print("WARNING: XPU not available. Results will be invalid.")
        return
    
    print(f"Device: {torch.xpu.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print()
    
    # Run benchmarks
    flash_results = benchmark_flash_attention_detailed()
    sparse_results = benchmark_sparse_attention_detailed()
    
    # Profiling info
    profile_with_intel_tools()
    
    # Generate report
    generate_performance_report(flash_results, sparse_results)
    
    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)
    
    print("\nNext Steps:")
    print("1. Review results in: /workspace/turbodiffusion-sycl/results/PERFORMANCE_XPU.md")
    print("2. Run Intel profiling tools for detailed analysis")
    print("3. Implement SYCL kernels to achieve better performance")


if __name__ == '__main__':
    main()
