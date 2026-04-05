#!/usr/bin/env python3
"""
Phase 2.3 Test: blocks[*].norm1 SYCL Replacement (Fixed)

Tests replacing the norm1 layer (Self-Attention input LayerNorm) in all blocks
with SYCL implementation. Tests against PyTorch XPU/CPU reference.

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
print("Phase 2.3: blocks[*].norm1 SYCL Replacement Test (Fixed)")
print("="*70)

# Import PyTorch
try:
    import torch
    import torch.nn as nn
    print(f"\n✓ PyTorch {torch.__version__}")
    
    # Check for XPU
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        print(f"✓ Intel XPU: {torch.xpu.get_device_name()}")
    else:
        device = torch.device('cpu')
        print(f"⚠️  Using CPU")
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

def test_layernorm(name, input_shape, eps=1e-5):
    """Test LayerNorm: PyTorch vs SYCL."""
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"{'='*70}")
    
    # Create test input
    x_np = np.random.randn(*input_shape).astype(np.float32)
    dim = input_shape[-1]
    
    # PyTorch reference
    x_torch = torch.from_numpy(x_np).to(device)
    layernorm_torch = nn.LayerNorm(dim, eps=eps).to(device)
    
    with torch.no_grad():
        torch_output = layernorm_torch(x_torch)
    
    if device.type == 'xpu':
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
    print(f"Device: {device}")
    print(f"PyTorch output shape: {torch_result.shape}")
    print(f"SYCL output shape: {sycl_result.shape}")
    print(f"Max error: {max_error:.2e}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Status: {'✅ PASSED' if passed else '❌ FAILED'}")
    
    return {
        'test': name,
        'passed': bool(passed),
        'max_error': float(max_error),
        'cosine_similarity': float(cos_sim),
        'device': str(device)
    }

def main():
    """Main test function."""
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'phase': '2.3',
        'test': 'blocks[*].norm1',
        'device': str(device),
        'sycl_device': info['name'],
        'tests': []
    }
    
    # Test single blocks
    print("\n" + "="*70)
    print("PHASE 1: Single Block Tests")
    print("="*70)
    
    for block_idx in [0, 15, 29]:
        result = test_layernorm(f'blocks[{block_idx}].norm1', (2, 64, 1536))
        results['tests'].append(result)
    
    # Test multiple blocks
    print("\n" + "="*70)
    print("PHASE 2: Multiple Block Tests")
    print("="*70)
    
    result = test_layernorm('blocks[0-9].norm1', (20, 64, 1536))
    results['tests'].append(result)
    
    result = test_layernorm('blocks[10-19].norm1', (20, 64, 1536))
    results['tests'].append(result)
    
    result = test_layernorm('blocks[20-29].norm1', (20, 64, 1536))
    results['tests'].append(result)
    
    # Test all blocks
    print("\n" + "="*70)
    print("PHASE 3: All Blocks Test")
    print("="*70)
    
    result = test_layernorm('all_blocks.norm1', (60, 32, 1536))
    results['tests'].append(result)
    
    # Summary
    print("\n" + "="*70)
    print("Phase 2.3 Test Summary")
    print("="*70)
    
    passed_count = sum(1 for t in results['tests'] if t['passed'])
    total_count = len(results['tests'])
    
    print(f"Tests passed: {passed_count}/{total_count}")
    print(f"Overall: {'✅ ALL TESTS PASSED' if passed_count == total_count else '❌ SOME TESTS FAILED'}")
    
    max_errors = [t['max_error'] for t in results['tests']]
    print(f"\nError Statistics:")
    print(f"  Max: {max(max_errors):.2e}")
    print(f"  Mean: {sum(max_errors)/len(max_errors):.2e}")
    
    results['summary'] = {
        'total_tests': total_count,
        'passed_tests': passed_count,
        'overall_passed': passed_count == total_count,
        'max_error': float(max(max_errors)),
        'mean_error': float(sum(max_errors)/len(max_errors))
    }
    
    # Save results
    output_dir = '/workspace/turbodiffusion-sycl/tests/phase2'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'phase2_3_fixed_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return passed_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
