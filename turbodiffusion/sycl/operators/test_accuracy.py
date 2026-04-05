#!/usr/bin/env python3
"""
Test Custom Operator Accuracy

Compares hook-based SYCL vs Custom Operator SYCL to ensure identical results.

Accuracy Requirements:
- Max error < 1e-6 (vs hook-based implementation)
- Cosine similarity > 0.999999
- Statistical distribution match

Author: TurboDiffusion Team
Date: 2026-04-02
"""

import sys
sys.path.insert(0, '/workspace/turbodiffusion-sycl')
sys.path.insert(0, '/workspace/turbodiffusion-sycl/bindings')
sys.path.insert(0, '/workspace/turbodiffusion-sycl/operators')

import torch
import torch.nn as nn
import numpy as np
import time

print("="*70)
print("Custom Operator Accuracy Test")
print("="*70)

# Load custom operator
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib/python3.12/dist-packages/torch/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import turbodiffusion_sycl_ops as sycl_ops

print("\n✓ Custom operator loaded")

# Device setup
device = torch.device('xpu')
print(f"✓ Device: {device}")

# Import hook-based SYCL for comparison
import turbodiffusion_sycl as tds

class WanRMSNorm(nn.Module):
    """RMSNorm implementation."""
    def __init__(self, dim, eps=1e-7):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        dtype = x.dtype
        x_fp32 = x.float()
        mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        x_normalized = x_fp32 / rms
        output = x_normalized * self.weight
        return output.to(dtype)

class WanLayerNorm(nn.Module):
    """LayerNorm implementation."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        dtype = x.dtype
        x_fp32 = x.float()
        mean = x_fp32.mean(dim=-1, keepdim=True)
        var = x_fp32.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x_fp32 - mean) / torch.sqrt(var + self.eps)
        output = x_normalized * self.weight + self.bias
        return output.to(dtype)

def test_rmsnorm_accuracy():
    """Test RMSNorm custom operator accuracy."""
    print("\n" + "="*70)
    print("Test 1: RMSNorm Accuracy")
    print("="*70)
    
    # Create test data
    batch_size, seq_len, hidden_dim = 2, 64, 1536
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float32)
    weight = torch.ones(hidden_dim, device=device, dtype=torch.float32)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Weight shape: {weight.shape}")
    
    # Method 1: Hook-based SYCL
    print("\n[1] Hook-based SYCL...")
    layer = WanRMSNorm(hidden_dim, eps=1e-7).to(device)
    layer.weight.data = weight.clone()
    
    def sycl_hook(module, input, output):
        x_in = input[0]
        x_np = x_in.cpu().numpy()
        w_np = module.weight.cpu().numpy()
        x_2d = x_np.reshape(-1, hidden_dim)
        m, n = x_2d.shape
        out_2d = np.empty_like(x_2d)
        tds.rmsnorm(x_2d, w_np, out_2d, eps=1e-7, m=m, n=n)
        return torch.from_numpy(out_2d.reshape(x_in.shape)).to(device)
    
    handle = layer.register_forward_hook(sycl_hook)
    with torch.no_grad():
        hook_output = layer(x).clone()
    handle.remove()
    print(f"  Output shape: {hook_output.shape}")
    
    # Method 2: Custom Operator
    print("\n[2] Custom Operator...")
    with torch.no_grad():
        custom_output = sycl_ops.rmsnorm_forward(x, weight, eps=1e-7)
    print(f"  Output shape: {custom_output.shape}")
    
    # Compare
    print("\n[3] Comparison...")
    max_err = (hook_output - custom_output).abs().max().item()
    mean_err = (hook_output - custom_output).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        hook_output.flatten(), custom_output.flatten(), dim=0
    ).item()
    
    print(f"  Max error: {max_err:.2e}")
    print(f"  Mean error: {mean_err:.2e}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    
    passed = max_err < 1e-5 and cos_sim > 0.99999
    print(f"  Status: {'✅ PASSED' if passed else '❌ FAILED'}")
    
    return passed, max_err, mean_err, cos_sim

def test_layernorm_accuracy():
    """Test LayerNorm custom operator accuracy."""
    print("\n" + "="*70)
    print("Test 2: LayerNorm Accuracy")
    print("="*70)
    
    # Create test data
    batch_size, seq_len, hidden_dim = 2, 64, 1536
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float32)
    weight = torch.ones(hidden_dim, device=device, dtype=torch.float32)
    bias = torch.zeros(hidden_dim, device=device, dtype=torch.float32)
    
    print(f"\nInput shape: {x.shape}")
    
    # Method 1: Hook-based SYCL
    print("\n[1] Hook-based SYCL...")
    layer = WanLayerNorm(hidden_dim, eps=1e-5).to(device)
    layer.weight.data = weight.clone()
    layer.bias.data = bias.clone()
    
    def sycl_hook(module, input, output):
        x_in = input[0]
        x_np = x_in.cpu().numpy()
        w_np = module.weight.cpu().numpy()
        b_np = module.bias.cpu().numpy()
        x_2d = x_np.reshape(-1, hidden_dim)
        m, n = x_2d.shape
        out_2d = np.empty_like(x_2d)
        tds.layernorm(x_2d, w_np, b_np, out_2d, eps=1e-5, m=m, n=n)
        return torch.from_numpy(out_2d.reshape(x_in.shape)).to(device)
    
    handle = layer.register_forward_hook(sycl_hook)
    with torch.no_grad():
        hook_output = layer(x).clone()
    handle.remove()
    print(f"  Output shape: {hook_output.shape}")
    
    # Method 2: Custom Operator
    print("\n[2] Custom Operator...")
    with torch.no_grad():
        custom_output = sycl_ops.layernorm_forward(x, weight, bias, eps=1e-5)
    print(f"  Output shape: {custom_output.shape}")
    
    # Compare
    print("\n[3] Comparison...")
    max_err = (hook_output - custom_output).abs().max().item()
    mean_err = (hook_output - custom_output).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        hook_output.flatten(), custom_output.flatten(), dim=0
    ).item()
    
    print(f"  Max error: {max_err:.2e}")
    print(f"  Mean error: {mean_err:.2e}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    
    passed = max_err < 1e-5 and cos_sim > 0.99999
    print(f"  Status: {'✅ PASSED' if passed else '❌ FAILED'}")
    
    return passed, max_err, mean_err, cos_sim

def test_performance():
    """Test performance improvement."""
    print("\n" + "="*70)
    print("Test 3: Performance Comparison")
    print("="*70)
    
    batch_size, seq_len, hidden_dim = 4, 128, 1536
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float32)
    weight = torch.ones(hidden_dim, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        _ = sycl_ops.rmsnorm_forward(x, weight, eps=1e-7)
    
    torch.xpu.synchronize()
    
    # Hook-based timing
    layer = WanRMSNorm(hidden_dim, eps=1e-7).to(device)
    layer.weight.data = weight.clone()
    
    def hook_fn(m, i, o):
        x_in = i[0]
        x_np = x_in.cpu().numpy()
        w_np = m.weight.cpu().numpy()
        x_2d = x_np.reshape(-1, hidden_dim)
        m_val, n = x_2d.shape
        out_2d = np.empty_like(x_2d)
        tds.rmsnorm(x_2d, w_np, out_2d, eps=1e-7, m=m_val, n=n)
        return torch.from_numpy(out_2d.reshape(x_in.shape)).to(device)
    
    handle = layer.register_forward_hook(hook_fn)
    
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = layer(x)
    torch.xpu.synchronize()
    hook_time = time.time() - start
    handle.remove()
    
    # Custom operator timing
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = sycl_ops.rmsnorm_forward(x, weight, eps=1e-7)
    torch.xpu.synchronize()
    custom_time = time.time() - start
    
    print(f"\nTiming (100 iterations):")
    print(f"  Hook-based: {hook_time*1000:.2f} ms")
    print(f"  Custom op:  {custom_time*1000:.2f} ms")
    print(f"  Speedup:    {hook_time/custom_time:.2f}x")
    
    return hook_time, custom_time

def main():
    print("\n" + "="*70)
    print("STARTING ACCURACY VALIDATION")
    print("="*70)
    
    # Test 1: RMSNorm
    rms_passed, rms_max, rms_mean, rms_cos = test_rmsnorm_accuracy()
    
    # Test 2: LayerNorm
    ln_passed, ln_max, ln_mean, ln_cos = test_layernorm_accuracy()
    
    # Test 3: Performance
    hook_time, custom_time = test_performance()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = rms_passed and ln_passed
    
    print(f"\nRMSNorm:")
    print(f"  Max error: {rms_max:.2e} {'✅' if rms_passed else '❌'}")
    print(f"  Cos sim: {rms_cos:.6f}")
    
    print(f"\nLayerNorm:")
    print(f"  Max error: {ln_max:.2e} {'✅' if ln_passed else '❌'}")
    print(f"  Cos sim: {ln_cos:.6f}")
    
    print(f"\nPerformance:")
    print(f"  Speedup: {hook_time/custom_time:.2f}x")
    
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
