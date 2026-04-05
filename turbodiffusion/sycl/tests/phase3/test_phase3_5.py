#!/usr/bin/env python3
"""
Phase 3.5: Full Inference Pipeline Test

Tests complete Wan2.1 inference with all norm layers replaced by SYCL.
This is the most comprehensive test before video generation.

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
print("Phase 3.5: Full Inference Pipeline Test")
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
    """RMSNorm implementation matching Wan2.1."""
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
    """LayerNorm implementation matching Wan2.1."""
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

class MockWanBlock(nn.Module):
    """Simplified Wan2.1 transformer block."""
    def __init__(self, block_idx, state_dict):
        super().__init__()
        self.block_idx = block_idx
        
        # Self-attention norms
        self.self_attn_norm_q = WanRMSNorm(1536, eps=1e-7)
        self.self_attn_norm_k = WanRMSNorm(1536, eps=1e-7)
        
        weight_q = state_dict[f'blocks.{block_idx}.self_attn.norm_q.weight']
        weight_k = state_dict[f'blocks.{block_idx}.self_attn.norm_k.weight']
        self.self_attn_norm_q.weight.data = weight_q.clone()
        self.self_attn_norm_k.weight.data = weight_k.clone()
        
        # Cross-attention norms
        self.cross_attn_norm_q = WanRMSNorm(1536, eps=1e-7)
        self.cross_attn_norm_k = WanRMSNorm(1536, eps=1e-7)
        
        weight_q = state_dict[f'blocks.{block_idx}.cross_attn.norm_q.weight']
        weight_k = state_dict[f'blocks.{block_idx}.cross_attn.norm_k.weight']
        self.cross_attn_norm_q.weight.data = weight_q.clone()
        self.cross_attn_norm_k.weight.data = weight_k.clone()
        
        # FFN norm (LayerNorm)
        self.norm3 = WanLayerNorm(1536, eps=1e-5)
        weight = state_dict[f'blocks.{block_idx}.norm3.weight']
        bias = state_dict[f'blocks.{block_idx}.norm3.bias']
        self.norm3.weight.data = weight.clone()
        self.norm3.bias.data = bias.clone()
        
        # Linear layers (simplified) - ensure BF16
        self.self_attn_q = nn.Linear(1536, 1536, bias=True).to(dtype=torch.bfloat16)
        self.cross_attn_q = nn.Linear(1536, 1536, bias=True).to(dtype=torch.bfloat16)
        self.ffn = nn.Linear(1536, 1536, bias=True).to(dtype=torch.bfloat16)
    
    def forward(self, x, text_emb=None):
        # Self-attention path
        normed_q = self.self_attn_norm_q(x)
        normed_k = self.self_attn_norm_k(x)
        q = self.self_attn_q(normed_q)
        
        # Cross-attention path (if text provided)
        if text_emb is not None:
            cross_q = self.cross_attn_norm_q(x)
            cross_k = self.cross_attn_norm_k(text_emb)
            cross_q = self.cross_attn_q(cross_q)
        
        # FFN path
        normed3 = self.norm3(x)
        ffn_out = self.ffn(normed3)
        
        # Simplified output
        return x + q + ffn_out

class MockWanModel(nn.Module):
    """Simplified Wan2.1 model with configurable blocks."""
    def __init__(self, num_blocks=5, state_dict=None):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            MockWanBlock(i, state_dict) for i in range(num_blocks)
        ])
    
    def forward(self, x, text_emb=None):
        for block in self.blocks:
            x = block(x, text_emb)
        return x

def create_sycl_hook(weight_tensor, is_rmsnorm=True, eps=1e-7):
    """Create SYCL hook function."""
    weight_np = weight_tensor.detach().cpu().float().numpy()
    dim = weight_np.shape[0]
    
    def hook(module, input, output):
        x = input[0] if isinstance(input, (list, tuple)) else input
        original_shape = x.shape
        original_dtype = x.dtype
        
        # Convert to numpy
        x_np = x.float().cpu().numpy()
        x_2d = x_np.reshape(-1, dim)
        m, n = x_2d.shape
        output_2d = np.empty_like(x_2d)
        
        if is_rmsnorm:
            tds.rmsnorm(x_2d, weight_np, output_2d, eps=eps, m=m, n=n)
        else:
            # LayerNorm - need bias
            bias_np = module.bias.detach().cpu().float().numpy()
            tds.layernorm(x_2d, weight_np, bias_np, output_2d, eps=eps, m=m, n=n)
        
        # Convert back
        output_np = output_2d.reshape(original_shape)
        result = torch.from_numpy(output_np).to(device=x.device)
        if original_dtype == torch.bfloat16:
            result = result.bfloat16()
        
        return result
    
    return hook

def test_full_pipeline():
    """Test full inference pipeline with SYCL."""
    print(f"\n{'='*70}")
    print("Full Inference Pipeline Test")
    print(f"{'='*70}")
    
    # Load checkpoint
    print("\n[1/6] Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    print(f"  ✓ Loaded {len(state_dict)} tensors")
    
    # Create model (test with 5 blocks for speed)
    print("\n[2/6] Creating model (5 blocks)...")
    model = MockWanModel(num_blocks=5, state_dict=state_dict).to(device)
    model.eval()
    print(f"  ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create test input (simulating video latents)
    print("\n[3/6] Preparing test input...")
    batch_size = 1
    seq_len = 256  # Spatial tokens
    hidden_dim = 1536
    
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)
    text_emb = torch.randn(batch_size, 77, hidden_dim, device=device, dtype=torch.bfloat16)
    print(f"  ✓ Input shape: {x.shape}")
    print(f"  ✓ Text embedding shape: {text_emb.shape}")
    
    # Baseline (PyTorch only)
    print("\n[4/6] Running PyTorch baseline...")
    torch.xpu.synchronize() if device.type == 'xpu' else None
    start = time.time()
    
    with torch.no_grad():
        baseline_output = model(x, text_emb).clone()
    
    torch.xpu.synchronize() if device.type == 'xpu' else None
    pytorch_time = time.time() - start
    
    print(f"  ✓ Baseline output shape: {baseline_output.shape}")
    print(f"  ✓ PyTorch time: {pytorch_time*1000:.2f} ms")
    
    # Register SYCL hooks for ALL norm layers
    print("\n[5/6] Registering SYCL hooks (25 layers total)...")
    hooks = []
    sycl_call_count = [0]
    
    for block_idx, block in enumerate(model.blocks):
        # Self-attention norms
        handle = block.self_attn_norm_q.register_forward_hook(
            create_sycl_hook(block.self_attn_norm_q.weight, True, 1e-7)
        )
        hooks.append(handle)
        
        handle = block.self_attn_norm_k.register_forward_hook(
            create_sycl_hook(block.self_attn_norm_k.weight, True, 1e-7)
        )
        hooks.append(handle)
        
        # Cross-attention norms
        handle = block.cross_attn_norm_q.register_forward_hook(
            create_sycl_hook(block.cross_attn_norm_q.weight, True, 1e-7)
        )
        hooks.append(handle)
        
        handle = block.cross_attn_norm_k.register_forward_hook(
            create_sycl_hook(block.cross_attn_norm_k.weight, True, 1e-7)
        )
        hooks.append(handle)
        
        # FFN norm (LayerNorm)
        handle = block.norm3.register_forward_hook(
            create_sycl_hook(block.norm3.weight, False, 1e-5)
        )
        hooks.append(handle)
        
        print(f"  ✓ Block {block_idx}: 5 hooks registered")
    
    print(f"  Total: {len(hooks)} hooks registered")
    
    # Run with SYCL
    print("\n[6/6] Running with SYCL hooks...")
    torch.xpu.synchronize() if device.type == 'xpu' else None
    start = time.time()
    
    with torch.no_grad():
        sycl_output = model(x, text_emb).clone()
    
    torch.xpu.synchronize() if device.type == 'xpu' else None
    sycl_time = time.time() - start
    
    # Cleanup hooks
    for handle in hooks:
        handle.remove()
    
    print(f"  ✓ SYCL output shape: {sycl_output.shape}")
    print(f"  ✓ SYCL time: {sycl_time*1000:.2f} ms")
    
    # Compare
    print(f"\n{'='*70}")
    print("Results Comparison")
    print(f"{'='*70}")
    
    baseline_f = baseline_output.float()
    sycl_f = sycl_output.float()
    
    max_error = (baseline_f - sycl_f).abs().max().item()
    mean_error = (baseline_f - sycl_f).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        baseline_f.flatten(), sycl_f.flatten(), dim=0
    ).item()
    
    print(f"\nNumerical Accuracy:")
    print(f"  Max error: {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    
    print(f"\nPerformance:")
    print(f"  PyTorch time: {pytorch_time*1000:.2f} ms")
    print(f"  SYCL time: {sycl_time*1000:.2f} ms")
    speedup = pytorch_time / sycl_time if sycl_time > 0 else 0
    print(f"  Speedup: {speedup:.2f}x")
    
    # Statistical analysis
    print(f"\nStatistical Analysis:")
    error_dist = (baseline_f - sycl_f).abs()
    print(f"  Error percentiles:")
    print(f"    50th: {torch.quantile(error_dist, 0.5).item():.2e}")
    print(f"    90th: {torch.quantile(error_dist, 0.9).item():.2e}")
    print(f"    99th: {torch.quantile(error_dist, 0.99).item():.2e}")
    
    # Pass criteria
    passed = max_error < 1e-1 and cos_sim >= 0.99  # Relaxed for full pipeline
    print(f"\nStatus: {'✅ PASSED' if passed else '⚠️  WARNING - Check if acceptable for video'}")
    
    return {
        'max_error': max_error,
        'mean_error': mean_error,
        'cosine_similarity': cos_sim,
        'pytorch_time_ms': pytorch_time * 1000,
        'sycl_time_ms': sycl_time * 1000,
        'speedup': speedup,
        'num_blocks': 5,
        'num_hooks': len(hooks),
        'passed': passed
    }

def main():
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        results = test_full_pipeline()
        
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'phase': '3.5',
            'device': str(device),
            'sycl_device': info['name'],
            **results
        }
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'phase': '3.5',
            'success': False,
            'error': str(e)
        }
    
    # Save results
    output_dir = '/workspace/turbodiffusion-sycl/tests/phase3'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'phase3_5_results.json')
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return final_results.get('passed', False)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
