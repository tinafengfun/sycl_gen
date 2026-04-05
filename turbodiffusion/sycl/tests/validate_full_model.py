#!/usr/bin/env python3
"""
Full model validation - compare SYCL vs CUDA/PyTorch video generation.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/TurboDiffusion/turbodiffusion')

from inference.modify_model import select_model, replace_attention, replace_linear_norm
from turbodiffusion_sycl import replace_attention_sycl, replace_norm_sycl
from rcm.utils.model_utils import load_state_dict

def generate_video_with_pytorch(model, prompt, device='cpu'):
    """Generate video using standard PyTorch (reference)."""
    model = model.to(device).eval()
    
    # TODO: Implement actual video generation
    # For now, run a few forward passes and capture intermediate activations
    
    # Create dummy input (normally would be from text encoder)
    B = 1
    L = 1024  # sequence length
    C = 1536  # model dim
    x = torch.randn(B, L, C, dtype=torch.bfloat16, device=device)
    
    # Run through a few layers
    outputs = []
    with torch.no_grad():
        for i, block in enumerate(model.blocks[:3]):  # Test first 3 blocks
            x_out = block(x, seq_lens=torch.tensor([L]), freqs=None)
            outputs.append(x_out.cpu().float().numpy())
            x = x_out
    
    return outputs

def generate_video_with_sycl(model, prompt, device='xpu'):
    """Generate video using SYCL kernels."""
    # Replace attention and norm with SYCL versions
    replace_attention_sycl(model, attention_type='sparse', topk=0.2)
    replace_norm_sycl(model)
    
    model = model.to(device).eval()
    
    # Same test as above
    B = 1
    L = 1024
    C = 1536
    x = torch.randn(B, L, C, dtype=torch.bfloat16, device=device)
    
    outputs = []
    with torch.no_grad():
        for i, block in enumerate(model.blocks[:3]):
            x_out = block(x, seq_lens=torch.tensor([L]), freqs=None)
            outputs.append(x_out.cpu().float().numpy())
            x = x_out
    
    return outputs

def compare_activations(ref_outputs, sycl_outputs, tolerance=1e-2):
    """Compare activation tensors between reference and SYCL."""
    print("\nComparing activations:")
    print("-" * 60)
    
    all_passed = True
    for i, (ref, sycl) in enumerate(zip(ref_outputs, sycl_outputs)):
        max_diff = np.max(np.abs(ref - sycl))
        mean_diff = np.mean(np.abs(ref - sycl))
        rel_diff = max_diff / (np.abs(ref).max() + 1e-8)
        
        passed = max_diff < tolerance
        status = "PASS" if passed else "FAIL"
        
        print(f"Block {i}:")
        print(f"  Max absolute diff: {max_diff:.6e}")
        print(f"  Mean absolute diff: {mean_diff:.6e}")
        print(f"  Relative diff: {rel_diff:.6e}")
        print(f"  Status: {status}")
        
        if not passed:
            all_passed = False
            # Print statistics about where errors are largest
            diff = np.abs(ref - sycl)
            print(f"  Error percentiles: 50th={np.percentile(diff, 50):.2e}, "
                  f"90th={np.percentile(diff, 90):.2e}, 99th={np.percentile(diff, 99):.2e}")
    
    return all_passed

def main():
    print("Full Model Validation: SYCL vs PyTorch")
    print("=" * 60)
    
    # Check if model checkpoint exists
    model_path = '/intel/hf_models/WAN2.1/checkpoints/TurboWan2.1-T2V-1.3B-480P.pth'
    
    print(f"\nLoading model...")
    model_ref = select_model("Wan2.1-1.3B")
    
    try:
        state_dict = load_state_dict(model_path)
        model_ref.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {model_path}")
    except FileNotFoundError:
        print(f"Warning: Checkpoint not found at {model_path}")
        print("Running with random weights for testing...")
    
    # Test 1: PyTorch reference (CPU)
    print("\n" + "=" * 60)
    print("Test 1: PyTorch Reference (CPU)")
    print("=" * 60)
    model_cpu = select_model("Wan2.1-1.3B")
    ref_outputs = generate_video_with_pytorch(model_cpu, "test prompt", device='cpu')
    print(f"Generated {len(ref_outputs)} reference outputs")
    
    # Test 2: SYCL implementation
    print("\n" + "=" * 60)
    print("Test 2: SYCL Implementation (XPU)")
    print("=" * 60)
    model_sycl = select_model("Wan2.1-1.3B")
    sycl_outputs = generate_video_with_sycl(model_sycl, "test prompt", device='xpu')
    print(f"Generated {len(sycl_outputs)} SYCL outputs")
    
    # Compare
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    passed = compare_activations(ref_outputs, sycl_outputs)
    
    # Generate report
    report_path = '/home/intel/tianfeng/opencode_bench/turbodiffusion-sycl/results/full_model_validation.md'
    with open(report_path, 'w') as f:
        f.write("# Full Model Validation Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"Overall: {'PASS' if passed else 'FAIL'}\n\n")
        f.write("## Details\n\n")
        for i, (ref, sycl) in enumerate(zip(ref_outputs, sycl_outputs)):
            max_diff = np.max(np.abs(ref - sycl))
            mean_diff = np.mean(np.abs(ref - sycl))
            f.write(f"### Block {i}\n")
            f.write(f"- Max absolute diff: {max_diff:.6e}\n")
            f.write(f"- Mean absolute diff: {mean_diff:.6e}\n")
            f.write(f"- Status: {'PASS' if max_diff < 1e-2 else 'FAIL'}\n\n")
    
    print(f"\nReport saved to: {report_path}")
    print("\n" + "=" * 60)
    print(f"Final Result: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)
    
    return 0 if passed else 1

if __name__ == '__main__':
    exit(main())
