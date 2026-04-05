#!/usr/bin/env python3
"""
Phase 3.4: All Norm Layer Types Test

Tests all 5 types of norm layers in Wan2.1:
- self_attn.norm_q (RMSNorm)
- self_attn.norm_k (RMSNorm)
- cross_attn.norm_q (RMSNorm)
- cross_attn.norm_k (RMSNorm)
- norm3 (LayerNorm)

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
print("Phase 3.4: All Norm Layer Types Test")
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

model_path = "/intel/hf_models/WAN2.1/checkpoints/TurboWan2.1-T2V-1.3B-480P.pth"

class WanRMSNorm(nn.Module):
    """RMSNorm for attention Q/K."""
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
    """LayerNorm for FFN (norm3)."""
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

def test_norm_layer(name, layer, x, weight_key, state_dict, is_rmsnorm=True):
    """Test a single norm layer with SYCL replacement."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    
    # Get weight from checkpoint
    weight = state_dict[weight_key].to(device)
    print(f"  Weight: {weight.shape}, {weight.dtype}")
    
    # Setup layer
    if is_rmsnorm:
        layer.weight.data = weight.clone()
    else:
        # LayerNorm has weight and bias
        layer.weight.data = weight.clone()
        bias_key = weight_key.replace('.weight', '.bias')
        if bias_key in state_dict:
            layer.bias.data = state_dict[bias_key].to(device).clone()
    
    layer = layer.to(device)
    layer.eval()
    
    # Baseline
    print("  [1] PyTorch baseline...")
    with torch.no_grad():
        baseline = layer(x).clone()
    print(f"      Output[0,0,:3]: {baseline[0,0,:3]}")
    
    # SYCL hook
    print("  [2] With SYCL hook...")
    sycl_called = [0]
    
    def sycl_hook(module, input, output):
        sycl_called[0] += 1
        x_input = input[0] if isinstance(input, (list, tuple)) else input
        original_shape = x_input.shape
        original_dtype = x_input.dtype
        
        # Convert to numpy
        x_np = x_input.float().cpu().numpy()
        x_2d = x_np.reshape(-1, module.dim)
        m, n = x_2d.shape
        output_2d = np.empty_like(x_2d)
        
        if is_rmsnorm:
            # RMSNorm
            weight_np = module.weight.detach().cpu().float().numpy()
            tds.rmsnorm(x_2d, weight_np, output_2d, eps=module.eps, m=m, n=n)
        else:
            # LayerNorm
            weight_np = module.weight.detach().cpu().float().numpy()
            bias_np = module.bias.detach().cpu().float().numpy()
            gamma = weight_np
            beta = bias_np
            tds.layernorm(x_2d, gamma, beta, output_2d, eps=module.eps, m=m, n=n)
        
        # Convert back
        output_np = output_2d.reshape(original_shape)
        result = torch.from_numpy(output_np).to(device=x_input.device)
        if original_dtype == torch.bfloat16:
            result = result.bfloat16()
        return result
    
    handle = layer.register_forward_hook(sycl_hook)
    
    with torch.no_grad():
        hooked = layer(x).clone()
    
    handle.remove()
    
    print(f"      Output[0,0,:3]: {hooked[0,0,:3]}")
    print(f"      SYCL called: {sycl_called[0]} times")
    
    # Compare
    print("  [3] Comparison...")
    baseline_f = baseline.float()
    hooked_f = hooked.float()
    
    max_err = (baseline_f - hooked_f).abs().max().item()
    mean_err = (baseline_f - hooked_f).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        baseline_f.flatten(), hooked_f.flatten(), dim=0
    ).item()
    
    print(f"      Max error: {max_err:.2e}")
    print(f"      Mean error: {mean_err:.2e}")
    print(f"      Cosine similarity: {cos_sim:.6f}")
    print(f"      Status: {'✅' if max_err < 1e-2 else '⚠️ '}")
    
    return {
        'name': name,
        'max_error': max_err,
        'mean_error': mean_err,
        'cosine_similarity': cos_sim,
        'sycl_calls': sycl_called[0]
    }

def main():
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Create test input
    x = torch.randn(2, 64, 1536, device=device, dtype=torch.bfloat16)
    print(f"Test input: {x.shape}, {x.dtype}")
    
    # Test all 5 norm types
    results = {
        'timestamp': datetime.now().isoformat(),
        'phase': '3.4',
        'device': str(device),
        'tests': []
    }
    
    print("\n" + "="*70)
    print("Testing All 5 Norm Layer Types")
    print("="*70)
    
    # 1. self_attn.norm_q
    layer = WanRMSNorm(1536, eps=1e-7)
    result = test_norm_layer('self_attn.norm_q', layer, x, 
                            'blocks.0.self_attn.norm_q.weight', state_dict, True)
    results['tests'].append(result)
    
    # 2. self_attn.norm_k
    layer = WanRMSNorm(1536, eps=1e-7)
    result = test_norm_layer('self_attn.norm_k', layer, x,
                            'blocks.0.self_attn.norm_k.weight', state_dict, True)
    results['tests'].append(result)
    
    # 3. cross_attn.norm_q
    layer = WanRMSNorm(1536, eps=1e-7)
    result = test_norm_layer('cross_attn.norm_q', layer, x,
                            'blocks.0.cross_attn.norm_q.weight', state_dict, True)
    results['tests'].append(result)
    
    # 4. cross_attn.norm_k
    layer = WanRMSNorm(1536, eps=1e-7)
    result = test_norm_layer('cross_attn.norm_k', layer, x,
                            'blocks.0.cross_attn.norm_k.weight', state_dict, True)
    results['tests'].append(result)
    
    # 5. norm3 (LayerNorm)
    layer = WanLayerNorm(1536, eps=1e-5)
    result = test_norm_layer('norm3 (LayerNorm)', layer, x,
                            'blocks.0.norm3.weight', state_dict, False)
    results['tests'].append(result)
    
    # Summary
    print("\n" + "="*70)
    print("Phase 3.4 Summary")
    print("="*70)
    
    all_passed = all(t['max_error'] < 1e-2 for t in results['tests'])
    
    print(f"\nResults for all 5 norm types:")
    for t in results['tests']:
        status = "✅" if t['max_error'] < 1e-2 else "⚠️"
        print(f"  {status} {t['name']}: max_err={t['max_error']:.2e}, cos_sim={t['cosine_similarity']:.6f}")
    
    print(f"\nOverall: {'✅ All passed' if all_passed else '⚠️ Some exceeded threshold'}")
    
    results['summary'] = {
        'total_tests': len(results['tests']),
        'passed': sum(1 for t in results['tests'] if t['max_error'] < 1e-2),
        'all_passed': all_passed
    }
    
    # Save results
    output_dir = '/workspace/turbodiffusion-sycl/tests/phase3'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'phase3_4_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
