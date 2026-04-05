#!/usr/bin/env python3
"""
Verification Script: Check if SYCL is really being called for zero-error layers

This script adds explicit logging to verify SYCL kernel execution.
"""

import sys
sys.path.insert(0, '/workspace/turbodiffusion-sycl')
sys.path.insert(0, '/workspace/turbodiffusion-sycl/bindings')

import torch
import torch.nn as nn
import numpy as np
import turbodiffusion_sycl as tds

print("="*70)
print("VERIFICATION: Are zero-error layers really calling SYCL?")
print("="*70)

device = torch.device('xpu')

class WanRMSNorm(nn.Module):
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

# Load checkpoint
checkpoint = torch.load("/intel/hf_models/WAN2.1/checkpoints/TurboWan2.1-T2V-1.3B-480P.pth", map_location='cpu')
state_dict = checkpoint.get('state_dict', checkpoint)

# Test input with specific marker values
x = torch.randn(2, 64, 1536, device=device, dtype=torch.bfloat16)
x[0, 0, 0] = 3.14159  # Marker value

print(f"\nTest input shape: {x.shape}")
print(f"Marker value at [0,0,0]: {x[0,0,0].item():.6f}")

def test_layer_with_tracking(name, layer_class, weight_key, is_rmsnorm=True):
    """Test a layer with detailed execution tracking."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    
    # Get weight
    weight = state_dict[weight_key].to(device)
    print(f"Weight: {weight.shape}")
    
    # Create layer
    if is_rmsnorm:
        layer = WanRMSNorm(weight.shape[0], eps=1e-7).to(device)
        layer.weight.data = weight.clone()
    else:
        layer = WanLayerNorm(weight.shape[0], eps=1e-5).to(device)
        layer.weight.data = weight.clone()
        bias_key = weight_key.replace('.weight', '.bias')
        if bias_key in state_dict:
            layer.bias.data = state_dict[bias_key].to(device).clone()
    
    layer.eval()
    
    # Baseline (PyTorch only)
    print("\n[1] PyTorch baseline:")
    with torch.no_grad():
        baseline = layer(x).clone()
    print(f"  Output[0,0,0]: {baseline[0,0,0].item():.6f}")
    print(f"  Output[0,0,1]: {baseline[0,0,1].item():.6f}")
    
    # With SYCL hook - with detailed tracking
    print("\n[2] With SYCL hook (TRACKING ENABLED):")
    
    sycl_called = [False]
    sycl_input_sample = [None]
    sycl_output_sample = [None]
    
    def tracking_sycl_hook(module, input, output):
        print(f"  ✓ Hook was called!")
        sycl_called[0] = True
        
        x_input = input[0] if isinstance(input, (list, tuple)) else input
        original_shape = x_input.shape
        original_dtype = x_input.dtype
        
        print(f"  Input shape: {original_shape}, dtype: {original_dtype}")
        print(f"  Input[0,0,0]: {x_input[0,0,0].item():.6f}")
        sycl_input_sample[0] = x_input[0,0,0].item()
        
        # Convert to numpy
        x_np = x_input.float().cpu().numpy()
        x_2d = x_np.reshape(-1, module.dim)
        m, n = x_2d.shape
        output_2d = np.empty_like(x_2d)
        
        print(f"  Calling tds.rmsnorm({'rms' if is_rmsnorm else 'layernorm'})(m={m}, n={n})...")
        
        if is_rmsnorm:
            weight_np = module.weight.detach().cpu().float().numpy()
            tds.rmsnorm(x_2d, weight_np, output_2d, eps=module.eps, m=m, n=n)
        else:
            weight_np = module.weight.detach().cpu().float().numpy()
            bias_np = module.bias.detach().cpu().float().numpy()
            tds.layernorm(x_2d, weight_np, bias_np, output_2d, eps=module.eps, m=m, n=n)
        
        # Convert back
        output_np = output_2d.reshape(original_shape)
        result = torch.from_numpy(output_np).to(device=x_input.device)
        if original_dtype == torch.bfloat16:
            result = result.bfloat16()
        
        print(f"  SYCL output[0,0,0]: {result[0,0,0].item():.6f}")
        sycl_output_sample[0] = result[0,0,0].item()
        
        return result
    
    handle = layer.register_forward_hook(tracking_sycl_hook)
    
    with torch.no_grad():
        hooked = layer(x).clone()
    
    handle.remove()
    
    print(f"\n  Final output[0,0,0]: {hooked[0,0,0].item():.6f}")
    print(f"  SYCL was called: {sycl_called[0]}")
    
    # Compare
    print("\n[3] Comparison:")
    baseline_f = baseline.float()
    hooked_f = hooked.float()
    
    max_err = (baseline_f - hooked_f).abs().max().item()
    mean_err = (baseline_f - hooked_f).abs().mean().item()
    
    print(f"  Max error: {max_err:.2e}")
    print(f"  Mean error: {mean_err:.2e}")
    
    # Analysis
    print("\n[4] Analysis:")
    if not sycl_called[0]:
        print("  ❌ SYCL WAS NOT CALLED - Hook not triggered!")
    elif max_err == 0.0:
        print("  ⚠️  SYCL was called but error is 0.0")
        print("     Possible reasons:")
        print("     - Input values produce identical results")
        print("     - Computation happens to match exactly")
        print(f"     - PyTorch: {baseline[0,0,0].item():.6f}")
        print(f"     - SYCL: {sycl_output_sample[0]:.6f}")
        if abs(baseline[0,0,0].item() - sycl_output_sample[0]) > 0.001:
            print("     ❌ SYCL output differs from final output!")
            print("     → Hook return value not being used!")
    else:
        print(f"  ✓ SYCL called and produced different result")
        print(f"    Input: {sycl_input_sample[0]:.6f}")
        print(f"    PyTorch: {baseline[0,0,0].item():.6f}")
        print(f"    SYCL: {sycl_output_sample[0]:.6f}")
    
    return sycl_called[0], max_err

# Test all layers
print("\n" + "="*70)
print("TESTING ALL LAYER TYPES WITH TRACKING")
print("="*70)

results = {}

# 1. self_attn.norm_q
name = "self_attn.norm_q"
called, err = test_layer_with_tracking(
    name, WanRMSNorm, 'blocks.0.self_attn.norm_q.weight', True
)
results[name] = {'called': called, 'error': err}

# 2. cross_attn.norm_q (zero error case)
name = "cross_attn.norm_q"
called, err = test_layer_with_tracking(
    name, WanRMSNorm, 'blocks.0.cross_attn.norm_q.weight', True
)
results[name] = {'called': called, 'error': err}

# 3. norm3 LayerNorm (zero error case)
name = "norm3 (LayerNorm)"
called, err = test_layer_with_tracking(
    name, WanLayerNorm, 'blocks.0.norm3.weight', False
)
results[name] = {'called': called, 'error': err}

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

for name, data in results.items():
    status = "✓" if data['called'] else "✗"
    print(f"{status} {name}:")
    print(f"   SYCL called: {data['called']}")
    print(f"   Error: {data['error']:.2e}")

print("\n" + "="*70)
