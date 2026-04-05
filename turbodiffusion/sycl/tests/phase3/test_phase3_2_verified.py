#!/usr/bin/env python3
"""
Phase 3.2 Verified: Single Layer SYCL Replacement with Correct Hook

Fixed version with verified hook execution.
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
print("Phase 3.2 Verified: Single Layer Replacement with Real Model")
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

def test_with_verification():
    """Test with explicit verification that SYCL is called."""
    print(f"\n{'='*70}")
    print("Test with Execution Verification")
    print(f"{'='*70}")
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    weight_key = 'blocks.0.self_attn.norm_q.weight'
    weight = state_dict[weight_key].to(device)
    print(f"  Loaded weight: {weight.shape}, {weight.dtype}")
    
    # Create model
    class TestModel(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.norm_q = WanRMSNorm(weight.shape[0], eps=1e-7)
            self.norm_q.weight.data = weight.clone()
            self.q_proj = nn.Linear(1536, 1536, bias=True).to(device, dtype=torch.bfloat16)
        
        def forward(self, x):
            normed = self.norm_q(x)
            return self.q_proj(normed)
    
    model = TestModel(weight)
    model.eval()
    
    # Test input
    x = torch.randn(2, 64, 1536, device=device, dtype=torch.bfloat16)
    
    # First, run without hook to get baseline
    print("\n[1] Baseline (no hook)...")
    with torch.no_grad():
        baseline_output = model(x).clone()
    print(f"  Baseline output[0,0,0]: {baseline_output[0,0,0].item():.6f}")
    
    # Now create SYCL hook with tracking
    sycl_call_count = [0]
    
    def sycl_hook(module, input, output):
        """Hook that definitely calls SYCL."""
        sycl_call_count[0] += 1
        print(f"    [Hook Call #{sycl_call_count[0]}]")
        
        x_input = input[0] if isinstance(input, (list, tuple)) else input
        original_shape = x_input.shape
        original_dtype = x_input.dtype
        
        # Convert to numpy
        x_np = x_input.float().cpu().numpy()
        weight_np = module.weight.detach().cpu().float().numpy()
        
        # Reshape
        x_2d = x_np.reshape(-1, module.dim)
        m, n = x_2d.shape
        output_2d = np.empty_like(x_2d)
        
        # Call SYCL
        print(f"    Calling tds.rmsnorm({m}, {n})...")
        tds.rmsnorm(x_2d, weight_np, output_2d, eps=module.eps, m=m, n=n)
        
        # Convert back
        output_np = output_2d.reshape(original_shape)
        output_torch = torch.from_numpy(output_np).to(device=x_input.device)
        if original_dtype == torch.bfloat16:
            output_torch = output_torch.bfloat16()
        
        print(f"    SYCL output[0,0,0]: {output_torch[0,0,0].item():.6f}")
        return output_torch
    
    # Register hook
    print("\n[2] Registering hook...")
    dispatcher = SyclDispatcher(model, test_mode=True)
    
    # Get the actual norm_q module and register hook on it
    norm_q_module = model.norm_q
    handle = norm_q_module.register_forward_hook(sycl_hook)
    
    print(f"  Hook registered on norm_q")
    
    # Run with hook
    print("\n[3] Running with hook...")
    with torch.no_grad():
        hooked_output = model(x).clone()
    
    print(f"  Hooked output[0,0,0]: {hooked_output[0,0,0].item():.6f}")
    print(f"  SYCL was called {sycl_call_count[0]} times")
    
    # Compare
    print("\n[4] Comparison...")
    if sycl_call_count[0] == 0:
        print("  ❌ FAIL: SYCL was never called!")
        return False
    
    baseline_f = baseline_output.float()
    hooked_f = hooked_output.float()
    
    max_error = (baseline_f - hooked_f).abs().max().item()
    mean_error = (baseline_f - hooked_f).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        baseline_f.flatten(), hooked_f.flatten(), dim=0
    ).item()
    
    print(f"  Max error: {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    
    # Cleanup
    handle.remove()
    
    passed = max_error < 1e-3 and cos_sim >= 0.999
    print(f"  Status: {'✅ PASSED' if passed else '❌ FAILED'}")
    
    return passed, max_error, mean_error, cos_sim, sycl_call_count[0]

def main():
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        passed, max_err, mean_err, cos_sim, call_count = test_with_verification()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'phase': '3.2-verified',
            'success': passed,
            'sycl_calls': call_count,
            'max_error': max_err,
            'mean_error': mean_err,
            'cosine_similarity': cos_sim,
            'device': str(device),
            'sycl_device': info['name']
        }
        
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(f"SYCL calls: {call_count}")
        print(f"Max error: {max_err:.2e}")
        print(f"Status: {'✅ PASSED' if passed else '❌ FAILED'}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results = {
            'timestamp': datetime.now().isoformat(),
            'phase': '3.2-verified',
            'success': False,
            'error': str(e)
        }
    
    # Save results
    output_dir = '/workspace/turbodiffusion-sycl/tests/phase3'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'phase3_2_verified_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return results.get('success', False)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
