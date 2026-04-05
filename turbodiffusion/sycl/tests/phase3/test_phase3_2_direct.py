#!/usr/bin/env python3
"""
Phase 3.2 Direct: Test RMSNorm directly without Q projection

This tests SYCL RMSNorm in isolation to verify accuracy.
"""

import sys
sys.path.insert(0, '/workspace/turbodiffusion-sycl')
sys.path.insert(0, '/workspace/turbodiffusion-sycl/bindings')

import torch
import torch.nn as nn
import numpy as np
import turbodiffusion_sycl as tds

print("="*70)
print("Phase 3.2 Direct: RMSNorm-Only Test")
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

# Load weight from checkpoint
checkpoint = torch.load("/intel/hf_models/WAN2.1/checkpoints/TurboWan2.1-T2V-1.3B-480P.pth", map_location='cpu')
state_dict = checkpoint.get('state_dict', checkpoint)
weight = state_dict['blocks.0.self_attn.norm_q.weight'].to(device)

# Create model with just RMSNorm
model = WanRMSNorm(weight.shape[0], eps=1e-7).to(device)
model.weight.data = weight.clone()

# Test input
x = torch.randn(2, 64, 1536, device=device, dtype=torch.bfloat16)
print(f"\nInput shape: {x.shape}, dtype: {x.dtype}")
print(f"Weight shape: {model.weight.shape}")

# Baseline (PyTorch only)
print("\n[1] PyTorch baseline:")
with torch.no_grad():
    baseline = model(x).clone()
print(f"  Output[0,0,:5]: {baseline[0,0,:5]}")

# With SYCL hook
print("\n[2] With SYCL hook:")
sycl_called = [0]

def sycl_hook(module, input, output):
    sycl_called[0] += 1
    x_input = input[0]
    
    # Convert to numpy and call SYCL
    x_np = x_input.float().cpu().numpy()
    weight_np = module.weight.detach().cpu().float().numpy()
    x_2d = x_np.reshape(-1, module.dim)
    m, n = x_2d.shape
    output_2d = np.empty_like(x_2d)
    
    tds.rmsnorm(x_2d, weight_np, output_2d, eps=module.eps, m=m, n=n)
    
    # Convert back
    output_np = output_2d.reshape(x_input.shape)
    result = torch.from_numpy(output_np).to(device=x_input.device).bfloat16()
    
    print(f"  SYCL computed output[0,0,:5]: {result[0,0,:5]}")
    return result

handle = model.register_forward_hook(sycl_hook)

with torch.no_grad():
    hooked = model(x).clone()

print(f"  Final output[0,0,:5]: {hooked[0,0,:5]}")
print(f"  SYCL was called: {sycl_called[0]} times")

handle.remove()

# Compare
print("\n[3] Comparison:")
max_err = (baseline.float() - hooked.float()).abs().max().item()
mean_err = (baseline.float() - hooked.float()).abs().mean().item()
cos_sim = torch.nn.functional.cosine_similarity(
    baseline.flatten(), hooked.flatten(), dim=0
).item()

print(f"  Max error: {max_err:.2e}")
print(f"  Mean error: {mean_err:.2e}")
print(f"  Cosine similarity: {cos_sim:.6f}")
print(f"  Status: {'✅ PASSED' if max_err < 1e-3 else '❌ FAILED'}")

# Additional check: are they identical?
identical = torch.allclose(baseline, hooked)
print(f"\n  Outputs identical: {identical}")
if identical:
    print("  ⚠️  WARNING: Outputs are identical - SYCL result not used!")
else:
    print("  ✓ Outputs are different - SYCL result is being used")
