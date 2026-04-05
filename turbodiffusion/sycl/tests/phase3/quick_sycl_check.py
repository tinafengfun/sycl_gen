#!/usr/bin/env python3
"""
Quick SYCL Verification Check
=============================

Fast sanity check to verify SYCL is actually executing.
Run this first before the full debug suite.

Usage in B60 container:
    python3 /workspace/turbodiffusion-sycl/tests/phase3/quick_sycl_check.py
"""

import sys
sys.path.insert(0, '/workspace/turbodiffusion-sycl')
sys.path.insert(0, '/workspace/turbodiffusion-sycl/bindings')

import numpy as np
import torch

print("="*60)
print("QUICK SYCL VERIFICATION CHECK")
print("="*60)

# Check 1: SYCL bindings available
try:
    import turbodiffusion_sycl as tds
    if not tds.is_available():
        print("[FAIL] SYCL bindings not available")
        sys.exit(1)
    info = tds.get_device_info()
    print(f"[OK] SYCL device: {info.get('name', 'Unknown')}")
except Exception as e:
    print(f"[FAIL] Cannot import SYCL: {e}")
    sys.exit(1)

# Check 2: SYCL produces different output than reference
def pytorch_rmsnorm(x, weight, eps=1e-6):
    dtype = x.dtype
    x_fp32 = x.float()
    mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_square + eps)
    x_normalized = x_fp32 / rms
    output = x_normalized * weight
    return output.to(dtype)

print("\n[TEST] Creating unique test input...")
torch.manual_seed(123)
x = torch.randn(2, 16, 64, dtype=torch.bfloat16)
x[0, 0, 0] = 3.14159  # Unique marker
weight = torch.ones(64, dtype=torch.bfloat16)
weight[0] = 2.0

print(f"  Input marker: {x[0,0,0].item():.6f}")
print(f"  Weight[0]: {weight[0].item():.6f}")

# PyTorch reference
pytorch_out = pytorch_rmsnorm(x, weight)
print(f"  PyTorch output[0,0,0]: {pytorch_out[0,0,0].item():.6f}")

# SYCL implementation
print("\n[TEST] Calling SYCL kernel...")
x_np = x.float().numpy()
weight_np = weight.float().numpy()
x_2d = x_np.reshape(-1, 64)
output_2d = np.empty_like(x_2d)

tds.rmsnorm(x_2d, weight_np, output_2d, eps=1e-6, m=x_2d.shape[0], n=64)

sycl_out = torch.from_numpy(output_2d.reshape(2, 16, 64)).bfloat16()
print(f"  SYCL output[0,0,0]: {sycl_out[0,0,0].item():.6f}")

# Compare
max_error = (pytorch_out.float() - sycl_out.float()).abs().max().item()
print(f"\n[RESULT] Max error: {max_error:.2e}")

if max_error == 0.0:
    print("[FAIL] Error is exactly 0.0 - SYCL may not be executing!")
    print("       Run full debug script for detailed analysis.")
    sys.exit(1)
elif max_error < 1e-7:
    print("[WARN] Error is extremely small (< 1e-7)")
    print("       This is suspicious - check full debug output.")
    sys.exit(1)
else:
    print("[PASS] SYCL is producing different outputs (error={:.2e})".format(max_error))
    print("       SYCL kernel appears to be executing correctly.")
    sys.exit(0)
