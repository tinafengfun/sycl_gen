#!/usr/bin/env python3
"""
TurboDiffusion SYCL 增量优化测试框架
Incremental Optimization Testing Framework

Usage:
    python3 optimization_benchmark.py --phase baseline
    python3 optimization_benchmark.py --phase xmx
    python3 optimization_benchmark.py --phase memory
    python3 optimization_benchmark.py --all

Requirements:
    - PyTorch with XPU support
    - turbodiffusion_sycl module built and installed
"""

import argparse
import torch
import numpy as np
import time
import json
import sys
from datetime import datetime
from pathlib import Path

# Add paths
sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/TurboDiffusion/turbodiffusion')
sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/turbodiffusion-sycl')

class OptimizationBenchmark:
    """Benchmark framework for incremental optimization testing."""
    
    def __init__(self, device='xpu', output_dir='./results'):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
        print(f"\n{'='*70}")
        print(f"TurboDiffusion SYCL Optimization Benchmark")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"PyTorch: {torch.__version__}")
        print(f"Date: {datetime.now().isoformat()}")
        print(f"{'='*70}\n")
        
    def test_flash_attention_accuracy(self, kernel_name='baseline'):
        """Test Flash Attention numerical accuracy."""
        print(f"\n[Phase: {kernel_name}] Testing Flash Attention Accuracy...")
        
        from turbodiffusion_sycl import FlashAttentionSYCL
        
        # Test configurations
        test_cases = [
            {'B': 1, 'H': 12, 'S': 1024, 'D': 128, 'name': 'Wan2.1-default'},
            {'B': 2, 'H': 12, 'S': 512, 'D': 128, 'name': '2x-batch'},
            {'B': 1, 'H': 8, 'S': 1024, 'D': 64, 'name': 'GQA-8head'},
        ]
        
        results = []
        
        for config in test_cases:
            B, H, S, D = config['B'], config['H'], config['S'], config['D']
            
            # Create test tensors
            torch.manual_seed(42)
            q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cpu')
            k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cpu')
            v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cpu')
            
            # PyTorch reference (standard attention)
            with torch.no_grad():
                scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(D)
                attn = torch.softmax(scores, dim=-1)
                ref_output = torch.matmul(attn, v)
            
            # SYCL implementation
            try:
                fa = FlashAttentionSYCL(head_dim=D, num_heads=H)
                sycl_output = fa(q, k, v)
                
                # Compute errors
                max_error = (ref_output.float() - sycl_output.float()).abs().max().item()
                mean_error = (ref_output.float() - sycl_output.float()).abs().mean().item()
                rel_error = max_error / (ref_output.abs().max().item() + 1e-8)
                
                # Check for NaN/Inf
                has_nan = torch.isnan(sycl_output).any().item()
                has_inf = torch.isinf(sycl_output).any().item()
                
                # Accuracy criterion: max_error < 0.01 for BF16
                passed = max_error < 0.01 and not has_nan and not has_inf
                
                result = {
                    'config': config['name'],
                    'shape': f"[{B},{H},{S},{D}]",
                    'max_error': max_error,
                    'mean_error': mean_error,
                    'rel_error': rel_error,
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'passed': passed
                }
                
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {config['name']:20s} | Max Error: {max_error:.6f} | {status}")
                
                if has_nan:
                    print(f"    WARNING: NaN detected!")
                if has_inf:
                    print(f"    WARNING: Inf detected!")
                    
            except Exception as e:
                result = {
                    'config': config['name'],
                    'error': str(e),
                    'passed': False
                }
                print(f"  {config['name']:20s} | ERROR: {e}")
            
            results.append(result)
        
        self.results[f'{kernel_name}_accuracy'] = results
        return results
    
    def test_flash_attention_performance(self, kernel_name='baseline'):
        """Test Flash Attention performance."""
        print(f"\n[Phase: {kernel_name}] Testing Flash Attention Performance...")
        
        from turbodiffusion_sycl import FlashAttentionSYCL
        
        configs = [
            (1, 12, 1024, 128, 'Wan2.1-1.3B'),
            (1, 12, 2048, 128, '2x-seq'),
            (1, 12, 4096, 128, '4x-seq'),
            (2, 12, 1024, 128, '2x-batch'),
        ]
        
        results = []
        
        for B, H, S, D, name in configs:
            # Create tensors on device
            q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=self.device)
            k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=self.device)
            v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=self.device)
            
            # Initialize module
            fa = FlashAttentionSYCL(head_dim=D, num_heads=H)
            
            # Warmup
            for _ in range(10):
                _ = fa(q, k, v)
            if self.device.type == 'xpu':
                torch.xpu.synchronize()
            elif self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            num_iters = 10
            start = time.perf_counter()
            for _ in range(num_iters):
                out = fa(q, k, v)
            if self.device.type == 'xpu':
                torch.xpu.synchronize()
            elif self.device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            # Compute metrics
            time_per_iter = elapsed / num_iters * 1000  # ms
            flops = 2 * B * H * S * S * D  # Simplified FLOPs
            tflops = flops / (time_per_iter / 1000) / 1e12
            
            result = {
                'config': name,
                'shape': f"[{B},{H},{S},{D}]",
                'time_ms': time_per_iter,
                'tflops': tflops
            }
            results.append(result)
            
            print(f"  {name:20s} | Time: {time_per_iter:.3f} ms | {tflops:.2f} TFLOPS")
        
        self.results[f'{kernel_name}_performance'] = results
        return results
    
    def test_sparse_attention_accuracy(self, kernel_name='baseline'):
        """Test Sparse Attention accuracy."""
        print(f"\n[Phase: {kernel_name}] Testing Sparse Attention Accuracy...")
        
        from turbodiffusion_sycl import SparseAttentionSYCL
        
        configs = [
            {'B': 1, 'H': 12, 'S': 1024, 'D': 128, 'topk': 0.2, 'name': 'topk0.2'},
            {'B': 1, 'H': 12, 'S': 1024, 'D': 128, 'topk': 0.1, 'name': 'topk0.1'},
        ]
        
        results = []
        
        for config in configs:
            B, H, S, D, topk = config['B'], config['H'], config['S'], config['D'], config['topk']
            
            torch.manual_seed(42)
            q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cpu')
            k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cpu')
            v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cpu')
            
            # PyTorch reference (dense attention)
            with torch.no_grad():
                scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(D)
                attn = torch.softmax(scores, dim=-1)
                ref_output = torch.matmul(attn, v)
            
            # SYCL sparse attention
            try:
                sa = SparseAttentionSYCL(head_dim=D, topk=topk)
                sycl_output = sa(q, k, v)
                
                max_error = (ref_output.float() - sycl_output.float()).abs().max().item()
                mean_error = (ref_output.float() - sycl_output.float()).abs().mean().item()
                
                passed = max_error < 0.05  # Sparse attention has higher tolerance
                
                result = {
                    'config': config['name'],
                    'max_error': max_error,
                    'mean_error': mean_error,
                    'passed': passed
                }
                
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {config['name']:20s} | Max Error: {max_error:.6f} | {status}")
                
            except Exception as e:
                result = {'config': config['name'], 'error': str(e), 'passed': False}
                print(f"  {config['name']:20s} | ERROR: {e}")
            
            results.append(result)
        
        self.results[f'{kernel_name}_sparse_accuracy'] = results
        return results
    
    def compare_phases(self, phases=['baseline', 'xmx', 'memory', 'workgroup']):
        """Compare results across optimization phases."""
        print(f"\n\n{'='*70}")
        print("OPTIMIZATION COMPARISON")
        print(f"{'='*70}\n")
        
        # Load results from previous runs if available
        comparison = {}
        
        for phase in phases:
            result_file = self.output_dir / f'{phase}_results.json'
            if result_file.exists():
                with open(result_file) as f:
                    comparison[phase] = json.load(f)
        
        if not comparison:
            print("No previous results found for comparison.")
            return
        
        # Compare performance
        print("Performance Comparison (TFLOPS):")
        print("-" * 70)
        print(f"{'Config':<20} {'Baseline':>12} {'XMX':>12} {'Memory':>12} {'Speedup':>12}")
        print("-" * 70)
        
        baseline_perf = comparison.get('baseline', {}).get('baseline_performance', [])
        
        for i, base_res in enumerate(baseline_perf):
            config_name = base_res['config']
            base_tflops = base_res['tflops']
            
            row = f"{config_name:<20} {base_tflops:>12.2f}"
            
            for phase in phases[1:]:
                if phase in comparison:
                    phase_perf = comparison[phase].get(f'{phase}_performance', [])
                    if i < len(phase_perf):
                        phase_tflops = phase_perf[i]['tflops']
                        row += f" {phase_tflops:>12.2f}"
                        if phase == phases[-1]:
                            speedup = phase_tflops / base_tflops if base_tflops > 0 else 0
                            row += f" {speedup:>12.2f}x"
                    else:
                        row += f" {'N/A':>12}"
            
            print(row)
        
        print("-" * 70)
    
    def save_results(self, phase):
        """Save results to file."""
        output_file = self.output_dir / f'{phase}_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")
    
    def generate_report(self):
        """Generate final optimization report."""
        report_file = self.output_dir / 'OPTIMIZATION_REPORT.md'
        
        with open(report_file, 'w') as f:
            f.write("# TurboDiffusion SYCL Optimization Report\n\n")
            f.write(f"**Date**: {datetime.now().isoformat()}\n\n")
            
            for key, data in self.results.items():
                f.write(f"## {key}\n\n")
                if isinstance(data, list):
                    f.write("| Config | Metric | Value | Status |\n")
                    f.write("|--------|--------|-------|--------|\n")
                    for item in data:
                        config = item.get('config', 'N/A')
                        if 'max_error' in item:
                            f.write(f"| {config} | Max Error | {item['max_error']:.6f} | {'✓' if item.get('passed') else '✗'} |\n")
                        elif 'time_ms' in item:
                            f.write(f"| {config} | Time (ms) | {item['time_ms']:.3f} | {item.get('tflops', 0):.2f} TFLOPS |\n")
                f.write("\n")
        
        print(f"✓ Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='TurboDiffusion SYCL Optimization Benchmark')
    parser.add_argument('--phase', choices=['baseline', 'xmx', 'memory', 'workgroup', 'precision', 'all'],
                       default='baseline', help='Optimization phase to test')
    parser.add_argument('--device', default='xpu', help='Device to use (xpu/cuda/cpu)')
    parser.add_argument('--output-dir', default='./results', help='Output directory for results')
    parser.add_argument('--compare', action='store_true', help='Compare all phases')
    
    args = parser.parse_args()
    
    benchmark = OptimizationBenchmark(device=args.device, output_dir=args.output_dir)
    
    if args.compare:
        benchmark.compare_phases()
        return
    
    if args.phase == 'all':
        phases = ['baseline', 'xmx', 'memory', 'workgroup']
    else:
        phases = [args.phase]
    
    for phase in phases:
        print(f"\n{'='*70}")
        print(f"RUNNING PHASE: {phase.upper()}")
        print(f"{'='*70}")
        
        # Run tests
        benchmark.test_flash_attention_accuracy(phase)
        benchmark.test_flash_attention_performance(phase)
        benchmark.test_sparse_attention_accuracy(phase)
        
        # Save results
        benchmark.save_results(phase)
        
        # Clear results for next phase
        benchmark.results = {}
    
    # Generate final report
    benchmark.generate_report()
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
