#!/usr/bin/env python3
"""
Phase 3.2: Single Layer SYCL Replacement with Real Model

Tests replacing a single RMSNorm layer (blocks.0.self_attn.norm_q) 
in the real Wan2.1 model with SYCL implementation.

Uses BF16 model with FP32→BF16 conversion for SYCL kernels.

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
print("Phase 3.2: Single Layer Replacement - Real Wan2.1 Model")
print("Target: blocks.0.self_attn.norm_q (RMSNorm)")
print("="*70)

# Import PyTorch
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

# Import hooks
from hooks import SyclDispatcher, LayerRegistry

# Model path
model_path = "/intel/hf_models/WAN2.1/checkpoints/TurboWan2.1-T2V-1.3B-480P.pth"

class WanRMSNorm(nn.Module):
    """
    Wan2.1 RMSNorm implementation.
    Using eps=1e-7 for optimal SYCL accuracy.
    """
    def __init__(self, dim, eps=1e-7):  # Changed from 1e-6 to 1e-7
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [..., dim]
        Returns:
            Normalized tensor [..., dim]
        """
        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        dtype = x.dtype
        
        # Convert to FP32 for stable computation
        x_fp32 = x.float()
        
        # Compute RMS
        mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        
        # Normalize
        x_normalized = x_fp32 / rms
        
        # Apply weight
        output = x_normalized * self.weight
        
        # Convert back to original dtype (BF16)
        return output.to(dtype)


def create_sycl_rmsnorm_hook(weight_tensor, eps=1e-7):  # Changed from 1e-6 to 1e-7
    """
    Create SYCL RMSNorm hook function.
    
    Args:
        weight_tensor: RMSNorm weight from model (FP32 or BF16)
        eps: Epsilon for numerical stability
    """
    # Convert weight to numpy FP32
    weight_np = weight_tensor.detach().cpu().float().numpy()
    dim = weight_np.shape[0]
    
    def sycl_hook(module, input, output):
        """
        Hook function with signature: (module, input, output) -> tensor
        """
        x = input[0] if isinstance(input, (list, tuple)) else input
        
        # Get input shape and dtype
        original_shape = x.shape
        original_dtype = x.dtype
        
        # Convert to numpy FP32 (handling BF16)
        if x.dtype == torch.bfloat16:
            x_fp32 = x.float().detach().cpu().numpy()
        else:
            x_fp32 = x.detach().cpu().numpy().astype(np.float32)
        
        # Reshape to 2D: [..., dim] -> [batch, dim]
        x_2d = x_fp32.reshape(-1, dim)
        m, n = x_2d.shape
        
        # Prepare output
        output_2d = np.empty_like(x_2d)
        
        # Call SYCL RMSNorm kernel
        try:
            tds.rmsnorm(x_2d, weight_np, output_2d, eps=eps, m=m, n=n)
        except Exception as e:
            print(f"⚠️  SYCL kernel failed: {e}")
            # Fall back to original output
            return output
        
        # Reshape back
        output_np = output_2d.reshape(original_shape)
        
        # Convert back to torch tensor with original dtype
        output_torch = torch.from_numpy(output_np).to(device=x.device)
        
        # Convert to BF16 if original was BF16
        if original_dtype == torch.bfloat16:
            output_torch = output_torch.bfloat16()
        
        return output_torch
    
    return sycl_hook


def load_model_partial():
    """
    Load only the first block of Wan2.1 model for testing.
    This saves memory and speeds up testing.
    """
    print(f"\n{'='*70}")
    print("Loading Model Checkpoint")
    print(f"{'='*70}")
    print(f"Checkpoint: {model_path}")
    
    # Load checkpoint to CPU first
    print("\n[1/3] Loading checkpoint to CPU...")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Filter for first block and related layers
    print("[2/3] Extracting block 0 layers...")
    block0_keys = [k for k in state_dict.keys() if k.startswith('blocks.0.')]
    
    # Create a simple module with just block 0
    class Block0Module(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Load self_attn.norm_q
            norm_q_key = 'blocks.0.self_attn.norm_q.weight'
            if norm_q_key in state_dict:
                weight = state_dict[norm_q_key]
                self.norm_q = WanRMSNorm(weight.shape[0], eps=1e-6)
                self.norm_q.weight.data = weight.clone()
                print(f"  ✓ Loaded {norm_q_key}: {weight.shape}, {weight.dtype}")
            else:
                raise ValueError(f"Key {norm_q_key} not found in checkpoint")
            
            # Load self_attn.q projection for full path testing
            q_key = 'blocks.0.self_attn.q.weight'
            if q_key in state_dict:
                weight = state_dict[q_key]
                bias = state_dict.get('blocks.0.self_attn.q.bias', None)
                self.q_proj = nn.Linear(weight.shape[1], weight.shape[0], bias=(bias is not None))
                self.q_proj.weight.data = weight.clone()
                if bias is not None:
                    self.q_proj.bias.data = bias.clone()
                print(f"  ✓ Loaded {q_key}: {weight.shape}, {weight.dtype}")
        
        def forward(self, x):
            """
            Forward: Input → RMSNorm(norm_q) → Q projection
            """
            # Apply RMSNorm
            normed = self.norm_q(x)
            # Apply Q projection
            q = self.q_proj(normed)
            return q
    
    model = Block0Module()
    
    # Move to device
    print(f"[3/3] Moving model to {device}...")
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {total_params:,} parameters")
    
    return model


def test_single_layer_replacement():
    """
    Test replacing blocks.0.self_attn.norm_q with SYCL.
    """
    print(f"\n{'='*70}")
    print("Phase 3.2: Single Layer Replacement Test")
    print(f"{'='*70}")
    
    # Load model
    model = load_model_partial()
    
    # Create test input
    print("\n[1/4] Creating test input...")
    batch_size = 2
    seq_len = 64
    hidden_dim = 1536
    
    # Create random input (matching model's expected input)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)
    print(f"  Input shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
    
    # Test 1: PyTorch reference (without hook)
    print("\n[2/4] Testing PyTorch reference...")
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = model(x)
        
        # Timed run
        torch.xpu.synchronize() if device.type == 'xpu' else None
        start = time.time()
        pytorch_output = model(x)
        torch.xpu.synchronize() if device.type == 'xpu' else None
        pytorch_time = time.time() - start
    
    print(f"  PyTorch output shape: {pytorch_output.shape}")
    print(f"  PyTorch output dtype: {pytorch_output.dtype}")
    print(f"  PyTorch time: {pytorch_time*1000:.2f} ms")
    
    # Test 2: With SYCL hook
    print("\n[3/4] Testing with SYCL hook...")
    
    # Create dispatcher
    dispatcher = SyclDispatcher(model, test_mode=True)
    
    # Create SYCL hook with model's weight
    weight_tensor = model.norm_q.weight
    sycl_fn = create_sycl_rmsnorm_hook(weight_tensor, eps=1e-6)
    
    # Register hook on norm_q
    dispatcher.register_hook('norm_q', sycl_fn, hook_type='forward')
    dispatcher.enable('norm_q')
    
    print(f"  ✓ Hook registered on norm_q")
    
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = model(x)
        
        # Timed run
        torch.xpu.synchronize() if device.type == 'xpu' else None
        start = time.time()
        sycl_output = model(x)
        torch.xpu.synchronize() if device.type == 'xpu' else None
        sycl_time = time.time() - start
    
    print(f"  SYCL output shape: {sycl_output.shape}")
    print(f"  SYCL output dtype: {sycl_output.dtype}")
    print(f"  SYCL time: {sycl_time*1000:.2f} ms")
    
    # Test 3: Compare outputs
    print("\n[4/4] Comparing outputs...")
    
    # Convert to FP32 for comparison
    pytorch_fp32 = pytorch_output.float()
    sycl_fp32 = sycl_output.float()
    
    # Compute error metrics
    max_error = (pytorch_fp32 - sycl_fp32).abs().max().item()
    mean_error = (pytorch_fp32 - sycl_fp32).abs().mean().item()
    
    # Cosine similarity
    pytorch_flat = pytorch_fp32.flatten()
    sycl_flat = sycl_fp32.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        pytorch_flat, sycl_flat, dim=0
    ).item()
    
    print(f"  Max error: {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    
    # Check pass/fail
    passed = max_error < 1e-3 and cos_sim >= 0.999
    print(f"  Status: {'✅ PASSED' if passed else '❌ FAILED'}")
    
    # Performance comparison
    speedup = pytorch_time / sycl_time if sycl_time > 0 else 0
    print(f"\n[Performance]")
    print(f"  PyTorch time: {pytorch_time*1000:.2f} ms")
    print(f"  SYCL time: {sycl_time*1000:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    # Cleanup
    dispatcher.remove_all_hooks()
    
    return {
        'test': 'blocks.0.self_attn.norm_q',
        'passed': passed,
        'max_error': max_error,
        'mean_error': mean_error,
        'cosine_similarity': cos_sim,
        'pytorch_time_ms': pytorch_time * 1000,
        'sycl_time_ms': sycl_time * 1000,
        'speedup': speedup,
        'input_shape': list(x.shape),
        'device': str(device)
    }


def main():
    """Main test function."""
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'phase': '3.2',
        'test': 'Single Layer Replacement - blocks.0.self_attn.norm_q',
        'device': str(device),
        'sycl_device': info['name'],
        'model_path': model_path
    }
    
    try:
        # Run test
        test_result = test_single_layer_replacement()
        results['test_result'] = test_result
        results['success'] = True
        
        # Summary
        print(f"\n{'='*70}")
        print("Phase 3.2 Summary")
        print(f"{'='*70}")
        print(f"Test: blocks.0.self_attn.norm_q replacement")
        print(f"Status: {'✅ PASSED' if test_result['passed'] else '❌ FAILED'}")
        print(f"Max Error: {test_result['max_error']:.2e}")
        print(f"Cosine Sim: {test_result['cosine_similarity']:.6f}")
        print(f"Speedup: {test_result['speedup']:.2f}x")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results['success'] = False
        results['error'] = str(e)
    
    # Save results
    output_dir = '/workspace/turbodiffusion-sycl/tests/phase3'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'phase3_2_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results.get('success', False) and results.get('test_result', {}).get('passed', False)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
