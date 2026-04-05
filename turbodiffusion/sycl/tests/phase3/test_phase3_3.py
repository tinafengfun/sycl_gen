#!/usr/bin/env python3
"""
Phase 3.3: Multiple Layer Replacement Test

Tests replacing all 30 blocks' self_attn.norm_q layers with SYCL.

Author: TurboDiffusion-SYCL Team
Date: 2026-04-02
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
print("Phase 3.3: Multiple Layer Replacement Test")
print("Target: All 30 blocks' self_attn.norm_q (RMSNorm)")
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

from hooks import SyclDispatcher

model_path = "/intel/hf_models/WAN2.1/checkpoints/TurboWan2.1-T2V-1.3B-480P.pth"

class WanRMSNorm(nn.Module):
    """Wan2.1 RMSNorm implementation."""
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

def create_sycl_hook(weight_tensor, eps=1e-7):
    """Create SYCL hook for a specific weight."""
    weight_np = weight_tensor.detach().cpu().float().numpy()
    dim = weight_np.shape[0]
    
    def sycl_hook(module, input, output):
        x = input[0] if isinstance(input, (list, tuple)) else input
        original_shape = x.shape
        original_dtype = x.dtype
        
        # Convert to numpy FP32
        x_np = x.float().cpu().numpy()
        x_2d = x_np.reshape(-1, dim)
        m, n = x_2d.shape
        
        output_2d = np.empty_like(x_2d)
        tds.rmsnorm(x_2d, weight_np, output_2d, eps=eps, m=m, n=n)
        
        # Convert back
        output_np = output_2d.reshape(original_shape)
        output_torch = torch.from_numpy(output_np).to(device=x.device)
        if original_dtype == torch.bfloat16:
            output_torch = output_torch.bfloat16()
        
        return output_torch
    
    return sycl_hook

def test_multiple_layers():
    """Test replacing all 30 blocks' norm_q."""
    print(f"\n{'='*70}")
    print("Testing All 30 Blocks")
    print(f"{'='*70}")
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Create a model with multiple RMSNorm layers
    class MultiBlockModel(nn.Module):
        def __init__(self, state_dict, num_blocks=5):  # Test 5 blocks first
            super().__init__()
            self.blocks = nn.ModuleList()
            
            for i in range(num_blocks):
                block = nn.Module()
                weight_key = f'blocks.{i}.self_attn.norm_q.weight'
                weight = state_dict[weight_key]
                
                block.norm_q = WanRMSNorm(weight.shape[0], eps=1e-7)
                block.norm_q.weight.data = weight.clone()
                
                # Add Q projection for realistic path
                block.q_proj = nn.Linear(1536, 1536, bias=True).to(dtype=torch.bfloat16)
                
                self.blocks.append(block)
        
        def forward(self, x):
            for block in self.blocks:
                normed = block.norm_q(x)
                x = block.q_proj(normed)
            return x
    
    # Create model with 5 blocks (test subset first)
    print("Creating model with 5 blocks...")
    model = MultiBlockModel(state_dict, num_blocks=5).to(device)
    model.eval()
    
    # Test input
    x = torch.randn(2, 64, 1536, device=device, dtype=torch.bfloat16)
    
    # Baseline
    print("\n[1] PyTorch baseline...")
    with torch.no_grad():
        baseline = model(x).clone()
    print(f"  Baseline output[0,0,:3]: {baseline[0,0,:3]}")
    
    # Register hooks for all blocks
    print("\n[2] Registering SYCL hooks for all blocks...")
    hooks = []
    sycl_call_counts = [0] * 5
    
    for i, block in enumerate(model.blocks):
        weight_key = f'blocks.{i}.self_attn.norm_q.weight'
        weight = state_dict[weight_key].to(device)
        
        # Create hook with counter
        def make_hook(idx, w):
            def hook(module, input, output):
                sycl_call_counts[idx] += 1
                x = input[0] if isinstance(input, (list, tuple)) else input
                original_shape = x.shape
                original_dtype = x.dtype
                
                x_np = x.float().cpu().numpy()
                w_np = w.cpu().float().numpy()
                x_2d = x_np.reshape(-1, w.shape[0])
                m, n = x_2d.shape
                output_2d = np.empty_like(x_2d)
                
                tds.rmsnorm(x_2d, w_np, output_2d, eps=1e-7, m=m, n=n)
                
                output_np = output_2d.reshape(original_shape)
                result = torch.from_numpy(output_np).to(device=x.device)
                if original_dtype == torch.bfloat16:
                    result = result.bfloat16()
                return result
            return hook
        
        handle = block.norm_q.register_forward_hook(make_hook(i, weight))
        hooks.append(handle)
        print(f"  ✓ Block {i}: hook registered")
    
    # Run with hooks
    print("\n[3] Running with SYCL hooks...")
    with torch.no_grad():
        hooked = model(x).clone()
    print(f"  Hooked output[0,0,:3]: {hooked[0,0,:3]}")
    print(f"  SYCL calls per block: {sycl_call_counts}")
    
    # Cleanup
    for handle in hooks:
        handle.remove()
    
    # Compare
    print("\n[4] Comparison...")
    baseline_f = baseline.float()
    hooked_f = hooked.float()
    
    max_error = (baseline_f - hooked_f).abs().max().item()
    mean_error = (baseline_f - hooked_f).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        baseline_f.flatten(), hooked_f.flatten(), dim=0
    ).item()
    
    print(f"  Max error: {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Status: {'✅ PASSED' if max_error < 1e-2 else '❌ FAILED'}")
    
    return max_error, mean_error, cos_sim, sum(sycl_call_counts)

def main():
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        max_err, mean_err, cos_sim, total_calls = test_multiple_layers()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'phase': '3.3',
            'success': max_err < 1e-2,
            'num_blocks': 5,
            'total_sycl_calls': total_calls,
            'max_error': max_err,
            'mean_error': mean_err,
            'cosine_similarity': cos_sim,
            'device': str(device),
            'sycl_device': info['name']
        }
        
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(f"Blocks tested: 5")
        print(f"Total SYCL calls: {total_calls}")
        print(f"Max error: {max_err:.2e}")
        print(f"Status: {'✅ PASSED' if max_err < 1e-2 else '❌ FAILED'}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results = {
            'timestamp': datetime.now().isoformat(),
            'phase': '3.3',
            'success': False,
            'error': str(e)
        }
    
    # Save results
    output_dir = '/workspace/turbodiffusion-sycl/tests/phase3'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'phase3_3_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return results.get('success', False)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
