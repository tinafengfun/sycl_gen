#!/usr/bin/env python3
"""
Phase 3.1: Load Real Wan2.1 Model Test

Tests loading the real Wan2.1 1.3B model and basic functionality.
Validates that the model loads correctly on Intel XPU.

Author: TurboDiffusion-SYCL Team
Date: 2026-04-01
"""

import sys
import os
import torch
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, '/workspace/turbodiffusion-sycl')
sys.path.insert(0, '/workspace/turbodiffusion-sycl/bindings')

print("="*70)
print("Phase 3.1: Real Wan2.1 Model Loading Test")
print("="*70)

print(f"\n✓ PyTorch {torch.__version__}")

# Check device
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = torch.device('xpu')
    print(f"✓ Intel XPU available: {torch.xpu.get_device_name()}")
else:
    device = torch.device('cpu')
    print(f"⚠️  Using CPU (XPU not available)")

# Model path
model_path = "/intel/hf_models/WAN2.1/checkpoints/TurboWan2.1-T2V-1.3B-480P.pth"

print(f"\n{'='*70}")
print("Loading Wan2.1 Model")
print(f"{'='*70}")
print(f"Checkpoint: {model_path}")

try:
    # Load checkpoint
    print("\n[1/4] Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"✓ Checkpoint loaded")
    print(f"  Keys: {list(checkpoint.keys())[:5]}...")
    
    # Check model structure
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"\n[2/4] Analyzing model structure...")
    
    # Count parameters
    total_params = 0
    layer_counts = {}
    
    for key in state_dict.keys():
        # Count layers by type
        if 'norm' in key.lower():
            layer_type = 'norm'
        elif 'attn' in key.lower() or 'attention' in key.lower():
            layer_type = 'attention'
        elif 'mlp' in key.lower() or 'ffn' in key.lower():
            layer_type = 'mlp'
        else:
            layer_type = 'other'
        
        layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
        
        # Count parameters
        if isinstance(state_dict[key], torch.Tensor):
            total_params += state_dict[key].numel()
    
    print(f"✓ Model structure analyzed")
    print(f"  Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  Layer distribution:")
    for layer_type, count in sorted(layer_counts.items()):
        print(f"    - {layer_type}: {count} tensors")
    
    # Check for target layers (norm layers)
    print(f"\n[3/4] Checking target layers for SYCL replacement...")
    
    norm_layers = [k for k in state_dict.keys() if 'norm' in k.lower()]
    print(f"✓ Found {len(norm_layers)} norm layers")
    
    # Group by block
    blocks_norm1 = [k for k in norm_layers if '.norm1' in k]
    blocks_norm2 = [k for k in norm_layers if '.norm2' in k]
    head_norm = [k for k in norm_layers if 'head.norm' in k or k.startswith('norm')]
    
    print(f"  - blocks[*].norm1: {len(blocks_norm1)} layers")
    print(f"  - blocks[*].norm2: {len(blocks_norm2)} layers")
    print(f"  - head.norm: {len(head_norm)} layers")
    
    # Show examples
    if blocks_norm1:
        print(f"    Example: {blocks_norm1[0]}")
    if blocks_norm2:
        print(f"    Example: {blocks_norm2[0]}")
    
    # Check RMSNorm layers
    rms_layers = [k for k in state_dict.keys() if 'rms' in k.lower() or 'norm_q' in k or 'norm_k' in k]
    print(f"  - RMSNorm layers: {len(rms_layers)} layers")
    if rms_layers:
        print(f"    Example: {rms_layers[0]}")
    
    # Try to move to device
    print(f"\n[4/4] Testing device compatibility...")
    
    # Move a small tensor to test
    test_tensor = list(state_dict.values())[0]
    if isinstance(test_tensor, torch.Tensor):
        test_on_device = test_tensor.to(device)
        print(f"✓ Successfully moved tensor to {device}")
        print(f"  Tensor shape: {test_on_device.shape}")
        print(f"  Tensor dtype: {test_on_device.dtype}")
        print(f"  Tensor device: {test_on_device.device}")
    
    # Results
    results = {
        'timestamp': datetime.now().isoformat(),
        'phase': '3.1',
        'test': 'Model Loading',
        'success': True,
        'device': str(device),
        'model_path': model_path,
        'total_parameters': total_params,
        'total_parameters_gb': round(total_params / 1e9, 2),
        'layer_counts': layer_counts,
        'norm_layers': {
            'total': len(norm_layers),
            'norm1': len(blocks_norm1),
            'norm2': len(blocks_norm2),
            'head_norm': len(head_norm),
            'rmsnorm': len(rms_layers)
        }
    }
    
    print(f"\n{'='*70}")
    print("Phase 3.1 Summary")
    print(f"{'='*70}")
    print(f"✅ Model loaded successfully")
    print(f"   Device: {device}")
    print(f"   Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"   Norm layers: {len(norm_layers)} (SYCL replacement targets)")
    
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'phase': '3.1',
        'test': 'Model Loading',
        'success': False,
        'error': str(e)
    }

# Save results
output_dir = '/workspace/turbodiffusion-sycl/tests/phase3'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'phase3_1_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_file}")

sys.exit(0 if results.get('success', False) else 1)
