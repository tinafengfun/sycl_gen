#!/usr/bin/env python3
"""
Phase 2 Test Suite (Fixed): Intel GPU Native Testing

Tests SYCL kernel replacement against PyTorch XPU reference.
Both run on Intel GPU (B60/Xe2) for fair comparison.

Tests:
- Phase 2.1: head.norm (LayerNorm)
- Phase 2.2: blocks[*].norm2 (LayerNorm)
- Phase 2.3: blocks[*].norm1 (LayerNorm)  
- Phase 2.4: RMSNorm (norm_q, norm_k)

Author: TurboDiffusion-SYCL Team
Date: 2026-04-01
"""

import sys
import os
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, '/workspace/turbodiffusion-sycl')
sys.path.insert(0, '/workspace/turbodiffusion-sycl/bindings')
sys.path.insert(0, '/workspace/turbodiffusion-sycl/hooks')

print("="*70)
print("Phase 2 Test Suite: Intel GPU Native (XPU vs SYCL)")
print("="*70)

# Check PyTorch XPU
try:
    import torch
    import torch.nn as nn
    print(f"\n✓ PyTorch {torch.__version__}")
    
    if torch.cuda.is_available():
        print("⚠️  CUDA available (unexpected on Intel GPU)")
    
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f"✓ Intel XPU available: {torch.xpu.get_device_name()}")
        HAS_XPU = True
    else:
        print("⚠️  XPU not available, using CPU")
        HAS_XPU = False
except ImportError:
    print("✗ PyTorch not available")
    sys.exit(1)

# Import SYCL
try:
    import turbodiffusion_sycl as tds
    if not tds.is_available():
        print("\n✗ SYCL bindings not available!")
        sys.exit(1)
    
    info = tds.get_device_info()
    print(f"✓ SYCL Device: {info['name']}")
except Exception as e:
    print(f"\n✗ Failed to import SYCL: {e}")
    sys.exit(1)

from hooks import SyclDispatcher

# Device selection
if HAS_XPU:
    DEVICE = torch.device('xpu')
else:
    DEVICE = torch.device('cpu')

print(f"\nUsing device: {DEVICE}")

# ============================================================================
# Test 1: LayerNorm (head.norm, blocks[*].norm1, blocks[*].norm2)
# ============================================================================

def test_layernorm(name, input_shape, eps=1e-5):
    """Test LayerNorm: PyTorch XPU vs SYCL."""
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"{'='*70}")
    
    # Create test data
    x_np = np.random.randn(*input_shape).astype(np.float32)
    dim = input_shape[-1]
    
    # PyTorch XPU reference
    x_torch = torch.from_numpy(x_np).to(DEVICE)
    layernorm_torch = nn.LayerNorm(dim, eps=eps).to(DEVICE)
    
    with torch.no_grad():
        torch_output = layernorm_torch(x_torch)
    
    if HAS_XPU:
        torch.xpu.synchronize()
    
    torch_result = torch_output.cpu().numpy()
    
    # SYCL implementation
    x_2d = x_np.reshape(-1, dim)
    m, n = x_2d.shape
    output_2d = np.empty_like(x_2d)
    gamma = np.ones(n, dtype=np.float32)
    beta = np.zeros(n, dtype=np.float32)
    
    tds.layernorm(x_2d, gamma, beta, output_2d, eps=eps, m=m, n=n)
    sycl_result = output_2d.reshape(input_shape)
    
    # Compare
    max_error = np.abs(torch_result - sycl_result).max()
    
    torch_flat = torch_result.flatten()
    sycl_flat = sycl_result.flatten()
    cos_sim = np.dot(torch_flat, sycl_flat) / (
        np.linalg.norm(torch_flat) * np.linalg.norm(sycl_flat) + 1e-10
    )
    
    passed = max_error < 1e-3 and cos_sim >= 0.999
    
    print(f"Input shape: {input_shape}")
    print(f"PyTorch XPU output shape: {torch_result.shape}")
    print(f"SYCL output shape: {sycl_result.shape}")
    print(f"Max error: {max_error:.2e}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Status: {'✅ PASSED' if passed else '❌ FAILED'}")
    
    return {
        'test': name,
        'passed': bool(passed),
        'max_error': float(max_error),
        'cosine_similarity': float(cos_sim)
    }

# ============================================================================
# Test 2: RMSNorm (norm_q, norm_k)
# ============================================================================

def test_rmsnorm(name, input_shape, eps=1e-6):
    """Test RMSNorm: PyTorch (manual) vs SYCL."""
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"{'='*70}")
    
    # Create test data
    x_np = np.random.randn(*input_shape).astype(np.float32)
    dim = input_shape[-1]
    
    # PyTorch XPU reference (manual RMSNorm since PyTorch doesn't have built-in)
    x_torch = torch.from_numpy(x_np).to(DEVICE)
    weight_torch = torch.ones(dim, device=DEVICE)
    
    with torch.no_grad():
        mean_square = (x_torch ** 2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + eps)
        x_normalized = x_torch / rms
        torch_output = x_normalized * weight_torch
    
    if HAS_XPU:
        torch.xpu.synchronize()
    
    torch_result = torch_output.cpu().numpy()
    
    # SYCL implementation
    x_2d = x_np.reshape(-1, dim)
    m, n = x_2d.shape
    output_2d = np.empty_like(x_2d)
    weight = np.ones(n, dtype=np.float32)
    
    tds.rmsnorm(x_2d, weight, output_2d, eps=eps, m=m, n=n)
    sycl_result = output_2d.reshape(input_shape)
    
    # Compare
    max_error = np.abs(torch_result - sycl_result).max()
    
    torch_flat = torch_result.flatten()
    sycl_flat = sycl_result.flatten()
    cos_sim = np.dot(torch_flat, sycl_flat) / (
        np.linalg.norm(torch_flat) * np.linalg.norm(sycl_flat) + 1e-10
    )
    
    passed = max_error < 1e-3 and cos_sim >= 0.999
    
    print(f"Input shape: {input_shape}")
    print(f"PyTorch XPU output shape: {torch_result.shape}")
    print(f"SYCL output shape: {sycl_result.shape}")
    print(f"Max error: {max_error:.2e}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Status: {'✅ PASSED' if passed else '❌ FAILED'}")
    
    return {
        'test': name,
        'passed': bool(passed),
        'max_error': float(max_error),
        'cosine_similarity': float(cos_sim)
    }

# ============================================================================
# Main Test Suite
# ============================================================================

def main():
    """Run all Phase 2 tests."""
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'phase': '2_fixed',
        'device': str(DEVICE),
        'sycl_device': info['name'],
        'tests': []
    }
    
    all_passed = True
    
    # Phase 2.1: head.norm
    print("\n" + "="*70)
    print("PHASE 2.1: head.norm (LayerNorm)")
    print("="*70)
    
    result = test_layernorm('head.norm', (2, 64, 1536))
    results['tests'].append(result)
    if not result['passed']:
        all_passed = False
    
    # Phase 2.2: blocks[*].norm2
    print("\n" + "="*70)
    print("PHASE 2.2: blocks[*].norm2 (LayerNorm)")
    print("="*70)
    
    # Single block
    result = test_layernorm('blocks[0].norm2', (2, 64, 1536))
    results['tests'].append(result)
    if not result['passed']:
        all_passed = False
    
    result = test_layernorm('blocks[15].norm2', (2, 64, 1536))
    results['tests'].append(result)
    if not result['passed']:
        all_passed = False
    
    # Multiple blocks (simulated with larger batch)
    result = test_layernorm('blocks[0-9].norm2', (20, 64, 1536))
    results['tests'].append(result)
    if not result['passed']:
        all_passed = False
    
    # Phase 2.3: blocks[*].norm1
    print("\n" + "="*70)
    print("PHASE 2.3: blocks[*].norm1 (LayerNorm - Self-Attention)")
    print("="*70)
    
    result = test_layernorm('blocks[0].norm1', (2, 64, 1536))
    results['tests'].append(result)
    if not result['passed']:
        all_passed = False
    
    result = test_layernorm('blocks[29].norm1', (2, 64, 1536))
    results['tests'].append(result)
    if not result['passed']:
        all_passed = False
    
    result = test_layernorm('all_blocks.norm1', (60, 32, 1536))
    results['tests'].append(result)
    if not result['passed']:
        all_passed = False
    
    # Phase 2.4: RMSNorm
    print("\n" + "="*70)
    print("PHASE 2.4: RMSNorm (Attention Q/K)")
    print("="*70)
    
    result = test_rmsnorm('norm_q', (2, 64, 12, 128))  # per-head
    results['tests'].append(result)
    if not result['passed']:
        all_passed = False
    
    result = test_rmsnorm('norm_k', (2, 64, 12, 128))
    results['tests'].append(result)
    if not result['passed']:
        all_passed = False
    
    result = test_rmsnorm('combined_QK', (24, 64, 128))
    results['tests'].append(result)
    if not result['passed']:
        all_passed = False
    
    # Summary
    print("\n" + "="*70)
    print("Phase 2 Test Summary (Fixed)")
    print("="*70)
    
    passed_count = sum(1 for t in results['tests'] if t['passed'])
    total_count = len(results['tests'])
    
    print(f"Device: {DEVICE}")
    print(f"SYCL Device: {info['name']}")
    print(f"Tests passed: {passed_count}/{total_count}")
    print(f"Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    max_errors = [t['max_error'] for t in results['tests']]
    print(f"\nError Statistics:")
    print(f"  Max: {max(max_errors):.2e}")
    print(f"  Mean: {sum(max_errors)/len(max_errors):.2e}")
    print(f"  Min: {min(max_errors):.2e}")
    
    results['summary'] = {
        'total_tests': total_count,
        'passed_tests': passed_count,
        'overall_passed': all_passed,
        'max_error': float(max(max_errors)),
        'mean_error': float(sum(max_errors)/len(max_errors)),
        'min_error': float(min(max_errors))
    }
    
    # Save results
    output_dir = '/workspace/turbodiffusion-sycl/tests/phase2'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'phase2_fixed_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
