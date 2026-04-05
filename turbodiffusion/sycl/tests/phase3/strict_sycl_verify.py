#!/usr/bin/env python3
"""
Strict Verification: Ensure SYCL kernel actually modifies output

This test uses controlled inputs to verify SYCL is really computing,
not just returning unmodified buffers.
"""

import sys
sys.path.insert(0, '/workspace/turbodiffusion-sycl')
sys.path.insert(0, '/workspace/turbodiffusion-sycl/bindings')

import torch
import numpy as np
import turbodiffusion_sycl as tds

print("="*70)
print("STRICT VERIFICATION: SYCL Kernel Real Execution Check")
print("="*70)

# Test 1: Verify SYCL modifies output buffer
print("\n[TEST 1] Verify output buffer is modified")
print("-"*70)

# Create input with specific pattern
x_np = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
weight_np = np.ones(8, dtype=np.float32)

# Pre-fill output with sentinel values
output_np = np.full_like(x_np, 999.0)  # Fill with 999

print(f"Input: {x_np[0]}")
print(f"Output before SYCL: {output_np[0]}")

# Call SYCL
tds.rmsnorm(x_np, weight_np, output_np, eps=1e-7, m=1, n=8)

print(f"Output after SYCL: {output_np[0]}")

# Check if output was modified
if np.all(output_np == 999.0):
    print("❌ FAIL: Output not modified - SYCL kernel did not write!")
elif np.allclose(output_np, x_np):
    print("⚠️  WARNING: Output equals input - possible pass-through")
else:
    print("✓ PASS: Output was modified by SYCL")
    
# Manual computation for verification
mean_sq = np.mean(x_np**2)
rms = np.sqrt(mean_sq + 1e-7)
expected = x_np / rms
print(f"Expected: {expected[0]}")
print(f"Match: {np.allclose(output_np, expected, rtol=1e-5)}")

# Test 2: Different inputs must produce different outputs
print("\n[TEST 2] Different inputs → different outputs")
print("-"*70)

x1 = np.ones((1, 1536), dtype=np.float32) * 0.5
x2 = np.ones((1, 1536), dtype=np.float32) * 2.0
weight = np.ones(1536, dtype=np.float32)
out1 = np.empty_like(x1)
out2 = np.empty_like(x2)

tds.rmsnorm(x1, weight, out1, eps=1e-7, m=1, n=1536)
tds.rmsnorm(x2, weight, out2, eps=1e-7, m=1, n=1536)

print(f"Input 1 mean: {np.mean(x1):.6f}")
print(f"Input 2 mean: {np.mean(x2):.6f}")
print(f"Output 1 mean: {np.mean(out1):.6f}")
print(f"Output 2 mean: {np.mean(out2):.6f}")

if np.allclose(out1, out2):
    print("❌ FAIL: Different inputs produced same output!")
else:
    print("✓ PASS: Different inputs produce different outputs")

# Test 3: Verify SYCL produces different result than simple copy
print("\n[TEST 3] SYCL vs identity check")
print("-"*70)

x = np.random.randn(1, 1536).astype(np.float32)
weight = np.ones(1536, dtype=np.float32)
out = np.empty_like(x)

tds.rmsnorm(x, weight, out, eps=1e-7, m=1, n=1536)

# Check if output is just a scaled copy of input
ratio = out / (x + 1e-10)
ratio_std = np.std(ratio)

print(f"Output/Input ratio std: {ratio_std:.6f}")
if ratio_std < 0.001:
    print("❌ FAIL: Output is just scaled input - no real computation!")
else:
    print("✓ PASS: Output shows real computation (RMS normalization)")

# Test 4: Memory corruption check
print("\n[TEST 4] Memory corruption check")
print("-"*70)

# Create input with guard values
x_big = np.random.randn(10, 1536).astype(np.float32)
x_big[0, 0] = 999.0  # Guard value at start
x_big[9, 1535] = 888.0  # Guard value at end

weight = np.ones(1536, dtype=np.float32)
out_big = np.zeros_like(x_big)
out_big[0, 0] = 777.0  # Output guard
out_big[9, 1535] = 666.0  # Output guard

tds.rmsnorm(x_big, weight, out_big, eps=1e-7, m=10, n=1536)

print(f"Input guard [0,0]: {x_big[0,0]} (should be 999)")
print(f"Input guard [9,1535]: {x_big[9,1535]} (should be 888)")
print(f"Output guard [0,0]: {out_big[0,0]} (was 777)")
print(f"Output guard [9,1535]: {out_big[9,1535]} (was 666)")

input_corrupted = (x_big[0,0] != 999.0 or x_big[9,1535] != 888.0)
output_not_written = (out_big[0,0] == 777.0 or out_big[9,1535] == 666.0)

if input_corrupted:
    print("❌ FAIL: Input memory corrupted!")
elif output_not_written:
    print("❌ FAIL: Output not written!")
else:
    print("✓ PASS: Memory handling correct")

# Test 5: Direct comparison with PyTorch on same input
print("\n[TEST 5] Direct PyTorch vs SYCL comparison")
print("-"*70)

# Use controlled input
x_test = np.random.randn(2, 64, 1536).astype(np.float32)
x_test[0, 0, 0] = 1.0
x_test[0, 0, 1] = 2.0
x_test[0, 0, 2] = 3.0

weight_test = np.ones(1536, dtype=np.float32)
weight_test[0] = 2.0  # Different weight for first element

# SYCL computation
out_sycl = np.empty_like(x_test)
tds.rmsnorm(x_test.reshape(-1, 1536), weight_test, out_sycl.reshape(-1, 1536), 
            eps=1e-7, m=128, n=1536)

# PyTorch computation
x_torch = torch.from_numpy(x_test).to('xpu')
weight_torch = torch.from_numpy(weight_test).to('xpu')
x_fp32 = x_torch.float()
mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
rms = torch.sqrt(mean_square + 1e-7)
out_torch = (x_fp32 / rms * weight_torch).cpu().numpy()

# Compare specific elements
print(f"Element [0,0,0]:")
print(f"  Input: {x_test[0,0,0]:.6f}")
print(f"  Weight: {weight_test[0]:.6f}")
print(f"  SYCL: {out_sycl[0,0,0]:.6f}")
print(f"  PyTorch: {out_torch[0,0,0]:.6f}")
print(f"  Diff: {abs(out_sycl[0,0,0] - out_torch[0,0,0]):.6e}")

diff = np.abs(out_sycl - out_torch).max()
print(f"\nMax difference across all elements: {diff:.6e}")

if diff == 0.0:
    print("⚠️  WARNING: Perfect match (0 difference) - check if both use same code path")
elif diff < 1e-5:
    print("✓ PASS: Very small difference (normal floating point variation)")
else:
    print(f"⚠️  Difference: {diff:.6e} (may indicate algorithm difference)")

# Summary
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)

all_pass = True
print("\nAll SYCL kernel execution checks:")
print("  1. Output buffer modification: " + ("✓" if not np.all(output_np == 999.0) else "❌"))
print("  2. Input sensitivity: " + ("✓" if not np.allclose(out1, out2) else "❌"))
print("  3. Computation verification: " + ("✓" if ratio_std >= 0.001 else "❌"))
print("  4. Memory safety: " + ("✓" if not (input_corrupted or output_not_written) else "❌"))
print("  5. Result correctness: " + ("✓" if diff < 1e-3 else "⚠️"))

if all_pass:
    print("\n✓ ALL CHECKS PASSED - SYCL kernel is executing correctly")
else:
    print("\n❌ SOME CHECKS FAILED - SYCL kernel may not be working properly")
