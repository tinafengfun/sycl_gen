#!/usr/bin/env python3
"""
GPU Inference Comparison: Intel XPU (SYCL) vs NVIDIA CUDA

This script compares video generation performance between:
- Intel B60 (Battlemage G21) with SYCL optimization
- NVIDIA GPU with CUDA

Test Configuration:
- Model: Wan2.1 1.3B T2V (simplified)
- Prompt: "a cat playing with a colorful ball in a sunny garden"
- Resolution: 480p
- Steps: 4 (distilled model)
- Seed: 42 (for reproducibility)

Author: TurboDiffusion Team
Date: 2026-04-02
"""

import sys
import os
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Configuration
TEST_PROMPT = "a cat playing with a colorful ball in a sunny garden"
RESOLUTION = "480p"
NUM_FRAMES = 16  # Reduced for faster testing
NUM_STEPS = 4
SEED = 42
BATCH_SIZE = 1

print("="*80)
print("GPU Inference Comparison: Intel XPU (SYCL) vs NVIDIA CUDA")
print("="*80)

# Detect device
if torch.cuda.is_available():
    device = torch.device('cuda')
    device_name = torch.cuda.get_device_name(0)
    backend = "CUDA"
    print(f"\n✓ NVIDIA GPU detected: {device_name}")
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = torch.device('xpu')
    device_name = torch.xpu.get_device_name()
    backend = "XPU"
    print(f"\n✓ Intel XPU detected: {device_name}")
else:
    device = torch.device('cpu')
    device_name = "CPU"
    backend = "CPU"
    print(f"\n⚠️  No GPU detected, using CPU")

print(f"Backend: {backend}")
print(f"Device: {device}")

# Import SYCL optimization if on XPU
if backend == "XPU":
    sys.path.insert(0, '/workspace/turbodiffusion-sycl/operators')
    import os
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib/python3.12/dist-packages/torch/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
    import turbodiffusion_sycl_ops as sycl_ops
    print("✓ SYCL custom operators loaded")

# Model components
class WanRMSNorm(nn.Module):
    """RMSNorm with backend-specific optimization."""
    def __init__(self, dim, eps=1e-7, use_custom_op=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_custom_op = use_custom_op and backend == "XPU"
    
    def forward(self, x):
        if self.use_custom_op and x.is_xpu:
            # Use SYCL custom operator
            return sycl_ops.rmsnorm_forward(x.float(), self.weight, self.eps)
        else:
            # Standard PyTorch implementation
            dtype = x.dtype
            x_fp32 = x.float()
            mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
            rms = torch.sqrt(mean_square + self.eps)
            x_normalized = x_fp32 / rms
            output = x_normalized * self.weight
            return output.to(dtype)

class WanLayerNorm(nn.Module):
    """LayerNorm with backend-specific optimization."""
    def __init__(self, dim, eps=1e-5, use_custom_op=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.use_custom_op = use_custom_op and backend == "XPU"
    
    def forward(self, x):
        if self.use_custom_op and x.is_xpu:
            # Use SYCL custom operator
            return sycl_ops.layernorm_forward(x.float(), self.weight, self.bias, self.eps)
        else:
            # Standard PyTorch implementation
            dtype = x.dtype
            x_fp32 = x.float()
            mean = x_fp32.mean(dim=-1, keepdim=True)
            var = x_fp32.var(dim=-1, keepdim=True, unbiased=False)
            x_normalized = (x_fp32 - mean) / torch.sqrt(var + self.eps)
            output = x_normalized * self.weight + self.bias
            return output.to(dtype)

class WanTransformerBlock(nn.Module):
    """Simplified Wan2.1 transformer block."""
    def __init__(self, dim=1536, num_heads=12, use_custom_op=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Self-attention norms
        self.self_attn_norm_q = WanRMSNorm(dim, eps=1e-7, use_custom_op=use_custom_op)
        self.self_attn_norm_k = WanRMSNorm(dim, eps=1e-7, use_custom_op=use_custom_op)
        
        # Cross-attention norms
        self.cross_attn_norm_q = WanRMSNorm(dim, eps=1e-7, use_custom_op=use_custom_op)
        self.cross_attn_norm_k = WanRMSNorm(dim, eps=1e-7, use_custom_op=use_custom_op)
        
        # FFN norm
        self.norm3 = WanLayerNorm(dim, eps=1e-5, use_custom_op=use_custom_op)
        
        # Linear layers
        self.self_attn_q = nn.Linear(dim, dim, bias=True)
        self.self_attn_k = nn.Linear(dim, dim, bias=True)
        self.self_attn_v = nn.Linear(dim, dim, bias=True)
        self.self_attn_o = nn.Linear(dim, dim, bias=True)
        
        self.cross_attn_q = nn.Linear(dim, dim, bias=True)
        self.cross_attn_k = nn.Linear(dim, dim, bias=True)
        self.cross_attn_v = nn.Linear(dim, dim, bias=True)
        self.cross_attn_o = nn.Linear(dim, dim, bias=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x, text_emb=None):
        # Self-attention
        normed_q = self.self_attn_norm_q(x)
        normed_k = self.self_attn_norm_k(x)
        
        q = self.self_attn_q(normed_q)
        k = self.self_attn_k(normed_k)
        v = self.self_attn_v(x)
        
        # Simplified attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = self.self_attn_o(attn_out)
        
        # Cross-attention (if text provided)
        if text_emb is not None:
            cross_q = self.cross_attn_norm_q(x)
            cross_k = self.cross_attn_norm_k(text_emb)
            
            cq = self.cross_attn_q(cross_q)
            ck = self.cross_attn_k(cross_k)
            cv = self.cross_attn_v(text_emb)
            
            cross_attn = torch.matmul(cq, ck.transpose(-2, -1)) / (self.dim ** 0.5)
            cross_attn = torch.softmax(cross_attn, dim=-1)
            cross_out = torch.matmul(cross_attn, cv)
            cross_out = self.cross_attn_o(cross_out)
        else:
            cross_out = 0
        
        # FFN
        normed3 = self.norm3(x)
        ffn_out = self.ffn(normed3)
        
        # Residual
        return x + attn_out + cross_out + ffn_out

class WanModel(nn.Module):
    """Simplified Wan2.1 model."""
    def __init__(self, num_blocks=5, dim=1536, use_custom_op=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            WanTransformerBlock(dim=dim, use_custom_op=use_custom_op)
            for _ in range(num_blocks)
        ])
        self.final_norm = WanLayerNorm(dim, eps=1e-5, use_custom_op=use_custom_op)
        self.output_proj = nn.Linear(dim, dim)
    
    def forward(self, x, text_emb=None):
        for block in self.blocks:
            x = block(x, text_emb)
        x = self.final_norm(x)
        return self.output_proj(x)

def create_text_embedding(prompt, dim=1536):
    """Create dummy text embedding for testing."""
    torch.manual_seed(SEED)
    # Simulate T5 embedding: [batch, seq_len, dim]
    return torch.randn(BATCH_SIZE, 77, dim)

def create_latent_noise(shape):
    """Create latent noise for generation."""
    torch.manual_seed(SEED)
    return torch.randn(*shape)

def benchmark_inference(use_custom_op=False, num_iterations=10):
    """Benchmark inference performance."""
    print(f"\n{'='*80}")
    print(f"Benchmarking: {'SYCL Optimized' if use_custom_op else 'Standard PyTorch'}")
    print(f"{'='*80}")
    
    # Create model
    model = WanModel(num_blocks=5, dim=1536, use_custom_op=use_custom_op).to(device)
    model.eval()
    
    # Create input
    latent_shape = (BATCH_SIZE, 256, 1536)  # Simplified latent
    text_emb = create_text_embedding(TEST_PROMPT).to(device)
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        x = create_latent_noise(latent_shape).to(device)
        with torch.no_grad():
            _ = model(x, text_emb)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'xpu':
        torch.xpu.synchronize()
    
    # Benchmark
    print(f"Running {num_iterations} iterations...")
    times = []
    
    for i in range(num_iterations):
        x = create_latent_noise(latent_shape).to(device)
        
        start = time.time()
        with torch.no_grad():
            output = model(x, text_emb)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'xpu':
            torch.xpu.synchronize()
        
        elapsed = time.time() - start
        times.append(elapsed)
        
        if i == 0:
            first_output = output.cpu().clone()
    
    # Statistics
    times = np.array(times)
    mean_time = times.mean()
    std_time = times.std()
    min_time = times.min()
    max_time = times.max()
    
    print(f"\nResults ({num_iterations} iterations):")
    print(f"  Mean: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  Min:  {min_time*1000:.2f} ms")
    print(f"  Max:  {max_time*1000:.2f} ms")
    print(f"  FPS:  {1.0/mean_time:.2f}")
    
    return {
        'mean_ms': mean_time * 1000,
        'std_ms': std_time * 1000,
        'min_ms': min_time * 1000,
        'max_ms': max_time * 1000,
        'fps': 1.0 / mean_time,
        'output': first_output if 'first_output' in locals() else None
    }

def main():
    """Main comparison function."""
    print(f"\nTest Configuration:")
    print(f"  Prompt: {TEST_PROMPT}")
    print(f"  Device: {device_name}")
    print(f"  Backend: {backend}")
    print(f"  Model: Wan2.1 1.3B (simplified)")
    print(f"  Blocks: 5")
    
    # Test without optimization
    baseline_results = benchmark_inference(use_custom_op=False, num_iterations=10)
    
    # Test with optimization (only on XPU)
    if backend == "XPU":
        optimized_results = benchmark_inference(use_custom_op=True, num_iterations=10)
        
        # Calculate speedup
        speedup = baseline_results['mean_ms'] / optimized_results['mean_ms']
        
        print(f"\n{'='*80}")
        print("COMPARISON RESULTS")
        print(f"{'='*80}")
        print(f"Baseline (PyTorch):     {baseline_results['mean_ms']:.2f} ms")
        print(f"Optimized (SYCL):       {optimized_results['mean_ms']:.2f} ms")
        print(f"Speedup:                {speedup:.2f}x")
        print(f"FPS improvement:        {optimized_results['fps']/baseline_results['fps']:.2f}x")
        
        # Accuracy check
        if baseline_results['output'] is not None and optimized_results['output'] is not None:
            max_error = (baseline_results['output'] - optimized_results['output']).abs().max().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                baseline_results['output'].flatten(),
                optimized_results['output'].flatten(),
                dim=0
            ).item()
            
            print(f"\nAccuracy:")
            print(f"  Max error: {max_error:.2e}")
            print(f"  Cosine similarity: {cos_sim:.6f}")
            print(f"  Status: {'✅ PASSED' if max_error < 1e-3 else '❌ FAILED'}")
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'device': device_name,
            'backend': backend,
            'prompt': TEST_PROMPT,
            'baseline': baseline_results,
            'optimized': optimized_results,
            'speedup': speedup
        }
        
        output_dir = Path('/workspace/turbodiffusion-sycl/tests/comparison')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / f'{backend.lower()}_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
        
        print(f"\nResults saved to: {output_dir}/{backend.lower()}_results.json")
        
    else:
        print(f"\n{'='*80}")
        print(f"{backend} Results:")
        print(f"{'='*80}")
        print(f"Mean inference time: {baseline_results['mean_ms']:.2f} ms")
        print(f"FPS: {baseline_results['fps']:.2f}")
        
        # Save results for comparison
        results = {
            'timestamp': datetime.now().isoformat(),
            'device': device_name,
            'backend': backend,
            'prompt': TEST_PROMPT,
            'baseline': baseline_results
        }
        
        output_dir = Path('/workspace/turbodiffusion-sycl/tests/comparison')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / f'{backend.lower()}_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
        
        print(f"\nResults saved to: {output_dir}/{backend.lower()}_results.json")

if __name__ == "__main__":
    main()
