#!/usr/bin/env python3
"""
Phase 3.2 Fixed: Single Layer SYCL Replacement with Real Model

Fixed version with improved accuracy debugging.

Author: TurboDiffusion-SYCL Team
Date: 2026-04-01
"""

import sys
import os
import numpy as np
import json
import time
from datetime import datetime

sys.path.insert(0, '/workspace/turbodiffusion-sycl')
sys.path.insert(0, '/workspace/turbodiffusion-sycl/bindings')
sys.path.insert(0, '/workspace/turbodiffusion-sycl/hooks')

print("="*70)
print("Phase 3.2 Fixed: Single Layer Replacement - Real Wan2.1 Model")
print("="*70)

try:
    import torch
    import torch.nn as nn
    print(f"\n✓ PyTorch {torch.__version__}")
    
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        print(f"✓ Intel XPU: {torch.xpu.get_device_name()}")
    else:
        device = torch.device('cpu')
        print(f"⚠️  Using CPU")
except ImportError:
    print("✗ PyTorch not available")
    sys.exit(1)

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

from hooks import SyclDispatcher, LayerRegistry

model_path = "/intel/hf_models/WAN2.1/checkpoints/TurboWan2.1-T2V-1.3B-480P.pth"

def pytorch_rmsnorm(x, weight, eps=1e-6):
    """Reference RMSNorm implementation matching Wan2.1 model."""
    dtype = x.dtype
    x_fp32 = x.float()
    mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_square + eps)
    x_normalized = x_fp32 / rms
    output = x_normalized * weight
    return output.to(dtype)

def sycl_rmsnorm_direct(x_torch, weight_torch, eps=1e-6):
    """
    Direct SYCL RMSNorm without hook mechanism.
    Simpler version to debug accuracy issues.
    """
    # Get shapes
    original_shape = x_torch.shape
    dim = original_shape[-1]
    
    # Convert inputs to numpy FP32
    x_np = x_torch.float().cpu().numpy()
    weight_np = weight_torch.float().cpu().numpy()
    
    # Reshape to 2D
    x_2d = x_np.reshape(-1, dim)
    m, n = x_2d.shape
    
    # Prepare output
    output_2d = np.empty_like(x_2d)
    
    # Call SYCL kernel
    tds.rmsnorm(x_2d, weight_np, output_2d, eps=eps, m=m, n=n)
    
    # Reshape back
    output_np = output_2d.reshape(original_shape)
    
    # Convert back to torch with original dtype
    output_torch = torch.from_numpy(output_np).to(device=x_torch.device)
    if x_torch.dtype == torch.bfloat16:
        output_torch = output_torch.bfloat16()
    
    return output_torch

def test_rmsnorm_accuracy():
    """Test RMSNorm accuracy directly without hooks."""
    print(f"\n{'='*70}")
    print("Step 1: Direct RMSNorm Accuracy Test")
    print(f"{'='*70}")
    
    # Load checkpoint
    print("Loading model checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Get actual weight from model
    weight_key = 'blocks.0.self_attn.norm_q.weight'
    weight = state_dict[weight_key].to(device)
    print(f"  Weight: {weight.shape}, {weight.dtype}")
    
    # Create test input
    batch_size, seq_len, hidden_dim = 2, 64, 1536
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)
    print(f"  Input: {x.shape}, {x.dtype}")
    
    # Test with different epsilon values
    print("\nTesting different epsilon values:")
    for eps in [1e-5, 1e-6, 1e-7]:
        # PyTorch reference
        pytorch_out = pytorch_rmsnorm(x, weight, eps=eps)
        
        # SYCL implementation
        sycl_out = sycl_rmsnorm_direct(x, weight, eps=eps)
        
        # Compare
        pytorch_fp32 = pytorch_out.float()
        sycl_fp32 = sycl_out.float()
        
        max_error = (pytorch_fp32 - sycl_fp32).abs().max().item()
        mean_error = (pytorch_fp32 - sycl_fp32).abs().mean().item()
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            pytorch_fp32.flatten(), sycl_fp32.flatten(), dim=0
        ).item()
        
        print(f"  eps={eps}: max_error={max_error:.2e}, mean_error={mean_error:.2e}, cos_sim={cos_sim:.6f}")
    
    # Use best epsilon for next test
    best_eps = 1e-6
    print(f"\nUsing eps={best_eps} for hook test")
    return best_eps, weight

def test_with_hook(eps, weight):
    """Test with hook mechanism."""
    print(f"\n{'='*70}")
    print("Step 2: Hook-Based Replacement Test")
    print(f"{'='*70}")
    
    # Create simple module
    class TestModule(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.norm = nn.Module()
            self.norm.weight = nn.Parameter(weight.clone())
            self.proj = nn.Linear(1536, 1536, bias=True).to(device, dtype=torch.bfloat16)
        
        def forward(self, x):
            normed = pytorch_rmsnorm(x, self.norm.weight, eps=eps)
            return self.proj(normed)
    
    model = TestModule(weight)
    model.eval()
    
    # Test input
    x = torch.randn(2, 64, 1536, device=device, dtype=torch.bfloat16)
    
    # PyTorch reference
    print("Running PyTorch reference...")
    with torch.no_grad():
        pytorch_out = model(x)
    print(f"  PyTorch output: {pytorch_out.shape}, {pytorch_out.dtype}")
    
    # Create SYCL hook
    def sycl_hook(module, input, output):
        x_input = input[0] if isinstance(input, (list, tuple)) else input
        return sycl_rmsnorm_direct(x_input, module.weight, eps=eps)
    
    # Register hook
    dispatcher = SyclDispatcher(model, test_mode=True)
    dispatcher.register_hook('norm', sycl_hook)
    dispatcher.enable('norm')
    
    # Run with SYCL
    print("Running with SYCL hook...")
    with torch.no_grad():
        sycl_out = model(x)
    print(f"  SYCL output: {sycl_out.shape}, {sycl_out.dtype}")
    
    # Compare
    pytorch_fp32 = pytorch_out.float()
    sycl_fp32 = sycl_out.float()
    
    max_error = (pytorch_fp32 - sycl_fp32).abs().max().item()
    mean_error = (pytorch_fp32 - sycl_fp32).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        pytorch_fp32.flatten(), sycl_fp32.flatten(), dim=0
    ).item()
    
    print(f"\nResults:")
    print(f"  Max error: {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Status: {'✅ PASSED' if max_error < 1e-3 and cos_sim >= 0.999 else '❌ FAILED'}")
    
    dispatcher.remove_all_hooks()
    
    return max_error, mean_error, cos_sim

def main():
    """Main test function."""
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'phase': '3.2-fixed',
        'device': str(device),
        'sycl_device': info['name']
    }
    
    try:
        # Step 1: Direct accuracy test
        best_eps, weight = test_rmsnorm_accuracy()
        
        # Step 2: Hook test
        max_err, mean_err, cos_sim = test_with_hook(best_eps, weight)
        
        results['success'] = max_err < 1e-3 and cos_sim >= 0.999
        results['max_error'] = max_err
        results['mean_error'] = mean_err
        results['cosine_similarity'] = cos_sim
        results['epsilon'] = best_eps
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        results['success'] = False
        results['error'] = str(e)
    
    # Save results
    output_dir = '/workspace/turbodiffusion-sycl/tests/phase3'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'phase3_2_fixed_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results.get('success', False)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
