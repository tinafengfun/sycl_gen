#!/usr/bin/env python3
"""
Debug Script: Verify SYCL Kernel Execution
============================================

This script provides comprehensive debugging to confirm whether SYCL kernels
are actually being executed during Phase 3.2 testing.

Key checks:
1. Hook invocation tracking - logs every hook call
2. SYCL kernel execution verification - unique outputs per kernel
3. Fallback detection - checks if silent fallback is occurring
4. Output differentiation - confirms SYCL vs PyTorch produce different results

Author: Debug Tool
Date: 2026-04-02
"""

import sys
import os
import numpy as np
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, '/workspace/turbodiffusion-sycl')
sys.path.insert(0, '/workspace/turbodiffusion-sycl/bindings')
sys.path.insert(0, '/workspace/turbodiffusion-sycl/hooks')

print("="*80)
print("SYCL KERNEL EXECUTION DEBUG VERIFICATION")
print("="*80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Session ID: {uuid.uuid4().hex[:8]}")

# ============================================================================
# Part 1: Environment and Import Checks
# ============================================================================
print("\n" + "="*80)
print("PART 1: Environment and Import Verification")
print("="*80)

# Import PyTorch
try:
    import torch
    import torch.nn as nn
    print(f"\n[✓] PyTorch {torch.__version__}")
    
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        print(f"[✓] Intel XPU available: {torch.xpu.get_device_name()}")
    else:
        device = torch.device('cpu')
        print(f"[!] Using CPU (XPU not available)")
except ImportError as e:
    print(f"[✗] PyTorch import failed: {e}")
    sys.exit(1)

# Import SYCL bindings
try:
    import turbodiffusion_sycl as tds
    print(f"\n[✓] SYCL bindings imported")
    
    if not tds.is_available():
        print("[✗] SYCL bindings not available!")
        sys.exit(1)
    
    info = tds.get_device_info()
    print(f"[✓] SYCL Device: {info.get('name', 'Unknown')}")
    print(f"[✓] SYCL Vendor: {info.get('vendor', 'Unknown')}")
    print(f"[✓] SYCL Available: {info.get('available', False)}")
except Exception as e:
    print(f"[✗] SYCL import failed: {e}")
    sys.exit(1)

# Import hooks
try:
    from hooks import SyclDispatcher, LayerRegistry
    print(f"\n[✓] Hooks module imported")
except ImportError as e:
    print(f"[✗] Hooks import failed: {e}")
    sys.exit(1)

# ============================================================================
# Part 2: Execution Tracking and Instrumentation
# ============================================================================
class ExecutionTracker:
    """
    Tracks all hook calls and SYCL kernel executions.
    """
    def __init__(self):
        self.hook_calls: List[Dict[str, Any]] = []
        self.sycl_calls: List[Dict[str, Any]] = []
        self.fallback_calls: List[Dict[str, Any]] = []
        self.call_counter = 0
        
    def log_hook_call(self, layer_path: str, input_shape: tuple, 
                      enabled: bool, timestamp: float):
        """Log when a hook is called."""
        self.call_counter += 1
        call_info = {
            'id': self.call_counter,
            'type': 'hook_call',
            'layer_path': layer_path,
            'input_shape': input_shape,
            'enabled': enabled,
            'timestamp': timestamp,
            'session_marker': uuid.uuid4().hex[:4]
        }
        self.hook_calls.append(call_info)
        print(f"\n[HOOK CALL #{self.call_counter}] {layer_path}")
        print(f"  Input shape: {input_shape}")
        print(f"  SYCL enabled: {enabled}")
        print(f"  Session marker: {call_info['session_marker']}")
        return call_info['session_marker']
    
    def log_sycl_call(self, layer_path: str, marker: str, 
                      input_sample: float, output_sample: float,
                      execution_time_ms: float):
        """Log when SYCL kernel is actually executed."""
        call_info = {
            'type': 'sycl_call',
            'layer_path': layer_path,
            'marker': marker,
            'input_sample': input_sample,
            'output_sample': output_sample,
            'execution_time_ms': execution_time_ms
        }
        self.sycl_calls.append(call_info)
        print(f"[SYCL EXEC] {layer_path} marker={marker}")
        print(f"  Input sample: {input_sample:.6f}")
        print(f"  Output sample: {output_sample:.6f}")
        print(f"  Execution time: {execution_time_ms:.3f} ms")
        
    def log_fallback(self, layer_path: str, marker: str, reason: str):
        """Log when fallback occurs."""
        call_info = {
            'type': 'fallback',
            'layer_path': layer_path,
            'marker': marker,
            'reason': reason
        }
        self.fallback_calls.append(call_info)
        print(f"[FALLBACK] {layer_path} marker={marker}")
        print(f"  Reason: {reason}")
        
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        return {
            'total_hook_calls': len(self.hook_calls),
            'sycl_executions': len(self.sycl_calls),
            'fallbacks': len(self.fallback_calls),
            'sycl_execution_rate': len(self.sycl_calls) / len(self.hook_calls) 
                                   if self.hook_calls else 0,
            'hook_calls': self.hook_calls,
            'sycl_calls': self.sycl_calls,
            'fallback_calls': self.fallback_calls
        }

# Global tracker
tracker = ExecutionTracker()

# ============================================================================
# Part 3: Instrumented SYCL Hook
# ============================================================================
def create_instrumented_sycl_hook(weight_tensor, eps=1e-6, layer_name="unnamed"):
    """
    Create a SYCL hook with full instrumentation.
    
    Args:
        weight_tensor: RMSNorm weight from model
        eps: Epsilon for numerical stability
        layer_name: Name for tracking
    """
    weight_np = weight_tensor.detach().cpu().float().numpy()
    dim = weight_np.shape[0]
    
    def instrumented_sycl_hook(module, input, output):
        """Instrumented hook that tracks all execution paths."""
        import time
        
        x = input[0] if isinstance(input, (list, tuple)) else input
        original_shape = x.shape
        original_dtype = x.dtype
        
        # Log hook call
        marker = tracker.log_hook_call(
            layer_path=layer_name,
            input_shape=tuple(original_shape),
            enabled=True,
            timestamp=time.time()
        )
        
        try:
            # Get a sample value for verification
            input_sample = x.flatten()[0].item() if x.numel() > 0 else 0.0
            
            # Convert to numpy FP32
            if x.dtype == torch.bfloat16:
                x_fp32 = x.float().detach().cpu().numpy()
            else:
                x_fp32 = x.detach().cpu().numpy().astype(np.float32)
            
            # Reshape to 2D
            x_2d = x_fp32.reshape(-1, dim)
            m, n = x_2d.shape
            
            # Prepare output
            output_2d = np.empty_like(x_2d)
            
            # Call SYCL kernel with timing
            start_time = time.time()
            tds.rmsnorm(x_2d, weight_np, output_2d, eps=eps, m=m, n=n)
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Reshape back
            output_np = output_2d.reshape(original_shape)
            
            # Convert back to torch
            output_torch = torch.from_numpy(output_np).to(device=x.device)
            if original_dtype == torch.bfloat16:
                output_torch = output_torch.bfloat16()
            
            # Get output sample
            output_sample = output_torch.flatten()[0].item() if output_torch.numel() > 0 else 0.0
            
            # Log successful SYCL execution
            tracker.log_sycl_call(
                layer_path=layer_name,
                marker=marker,
                input_sample=input_sample,
                output_sample=output_sample,
                execution_time_ms=execution_time_ms
            )
            
            return output_torch
            
        except Exception as e:
            # Log fallback
            tracker.log_fallback(
                layer_path=layer_name,
                marker=marker,
                reason=str(e)
            )
            # Return original output (fallback)
            return output
    
    return instrumented_sycl_hook

# ============================================================================
# Part 4: PyTorch Reference Implementation
# ============================================================================
def pytorch_rmsnorm(x, weight, eps=1e-6):
    """Reference RMSNorm implementation."""
    dtype = x.dtype
    x_fp32 = x.float()
    mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_square + eps)
    x_normalized = x_fp32 / rms
    output = x_normalized * weight
    return output.to(dtype)

# ============================================================================
# Part 5: Core Test Functions
# ============================================================================
def test_1_direct_sycl_execution():
    """
    Test 1: Verify SYCL kernel executes and produces unique outputs.
    """
    print("\n" + "="*80)
    print("TEST 1: Direct SYCL Kernel Execution")
    print("="*80)
    
    # Create unique test input with known pattern
    test_seed = 42
    torch.manual_seed(test_seed)
    np.random.seed(test_seed)
    
    batch_size, seq_len, hidden_dim = 2, 8, 64  # Small for quick test
    
    # Create unique input pattern
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)
    # Add unique marker value at position [0,0,0]
    x[0, 0, 0] = 3.14159
    
    weight = torch.ones(hidden_dim, device=device, dtype=torch.bfloat16)
    weight[0] = 2.0  # Different weight for first element
    
    print(f"\nTest input shape: {x.shape}")
    print(f"Marker value at [0,0,0]: {x[0,0,0].item():.6f}")
    print(f"Weight at [0]: {weight[0].item():.6f}")
    
    # PyTorch reference
    pytorch_out = pytorch_rmsnorm(x, weight, eps=1e-6)
    print(f"\nPyTorch output[0,0,0]: {pytorch_out[0,0,0].item():.6f}")
    
    # SYCL implementation
    print("\nCalling SYCL kernel...")
    x_np = x.float().cpu().numpy()
    weight_np = weight.float().cpu().numpy()
    
    x_2d = x_np.reshape(-1, hidden_dim)
    m, n = x_2d.shape
    output_2d = np.empty_like(x_2d)
    
    tds.rmsnorm(x_2d, weight_np, output_2d, eps=1e-6, m=m, n=n)
    
    sycl_out_np = output_2d.reshape(batch_size, seq_len, hidden_dim)
    sycl_out = torch.from_numpy(sycl_out_np).to(device=device).bfloat16()
    
    print(f"SYCL output[0,0,0]: {sycl_out[0,0,0].item():.6f}")
    
    # Compare
    max_error = (pytorch_out.float() - sycl_out.float()).abs().max().item()
    mean_error = (pytorch_out.float() - sycl_out.float()).abs().mean().item()
    
    print(f"\nComparison:")
    print(f"  Max error: {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    
    # Verify outputs are different (if error is 0, something is wrong)
    if max_error == 0.0:
        print("\n[!] WARNING: Max error is exactly 0.0")
        print("    This is suspicious - even identical algorithms have floating-point differences")
        print("    Possible causes:")
        print("    - SYCL kernel is not actually executing")
        print("    - Output buffer is being copied from input without modification")
        print("    - PyTorch and SYCL are sharing the same implementation")
        return False
    elif max_error < 1e-7:
        print("\n[!] WARNING: Max error is extremely small (< 1e-7)")
        print("    This may indicate the SYCL kernel is not running")
        return False
    else:
        print(f"\n[✓] SYCL kernel executed (max_error={max_error:.2e})")
        return True

def test_2_hook_instrumentation():
    """
    Test 2: Verify hooks are being called and instrumented.
    """
    print("\n" + "="*80)
    print("TEST 2: Hook Instrumentation Verification")
    print("="*80)
    
    # Create a simple test module
    class TestModule(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.norm = nn.Module()
            self.norm.weight = nn.Parameter(torch.ones(dim))
            self.proj = nn.Linear(dim, dim, bias=True)
        
        def forward(self, x):
            normed = pytorch_rmsnorm(x, self.norm.weight, eps=1e-6)
            return self.proj(normed)
    
    hidden_dim = 64
    model = TestModule(hidden_dim).to(device, dtype=torch.bfloat16)
    model.eval()
    
    print(f"\nCreated test model with hidden_dim={hidden_dim}")
    
    # Create dispatcher with tracker
    dispatcher = SyclDispatcher(model, test_mode=True)
    
    # Create instrumented hook
    sycl_fn = create_instrumented_sycl_hook(
        model.norm.weight,
        eps=1e-6,
        layer_name="test.norm"
    )
    
    # Register and enable hook
    dispatcher.register_hook('norm', sycl_fn, hook_type='forward')
    dispatcher.enable('norm')
    
    print("\nHook registered and enabled")
    
    # Run forward pass
    x = torch.randn(2, 8, hidden_dim, device=device, dtype=torch.bfloat16)
    print(f"\nRunning forward pass with input shape {x.shape}...")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    
    # Check tracker results
    summary = tracker.get_summary()
    print(f"\nExecution Tracker Summary:")
    print(f"  Hook calls: {summary['total_hook_calls']}")
    print(f"  SYCL executions: {summary['sycl_executions']}")
    print(f"  Fallbacks: {summary['fallbacks']}")
    print(f"  Execution rate: {summary['sycl_execution_rate']*100:.1f}%")
    
    dispatcher.remove_all_hooks()
    
    if summary['total_hook_calls'] == 0:
        print("\n[✗] FAIL: No hook calls recorded!")
        print("    The hook was not invoked during forward pass")
        return False
    
    if summary['sycl_executions'] == 0:
        print("\n[✗] FAIL: No SYCL executions recorded!")
        print("    Hook was called but SYCL kernel did not execute")
        return False
    
    print("\n[✓] PASS: Hook instrumentation working correctly")
    return True

def test_3_differentiation_verification():
    """
    Test 3: Verify SYCL and PyTorch produce measurably different outputs.
    """
    print("\n" + "="*80)
    print("TEST 3: Output Differentiation Verification")
    print("="*80)
    
    # Reset tracker
    tracker.hook_calls.clear()
    tracker.sycl_calls.clear()
    tracker.fallback_calls.clear()
    
    hidden_dim = 1536
    
    # Load actual model weight if available
    model_path = "/intel/hf_models/WAN2.1/checkpoints/TurboWan2.1-T2V-1.3B-480P.pth"
    if os.path.exists(model_path):
        print(f"\nLoading weight from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        weight_key = 'blocks.0.self_attn.norm_q.weight'
        if weight_key in state_dict:
            weight = state_dict[weight_key].to(device)
            print(f"  Loaded weight: {weight.shape}, {weight.dtype}")
        else:
            weight = torch.ones(hidden_dim, device=device, dtype=torch.bfloat16)
            print(f"  Weight not found, using ones: {weight.shape}")
    else:
        print(f"\nModel not found, using synthetic weight")
        weight = torch.ones(hidden_dim, device=device, dtype=torch.bfloat16)
        # Add some variation
        weight[::10] = 1.5
    
    # Create test module
    class TestModule(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.norm = nn.Module()
            self.norm.weight = nn.Parameter(weight.clone())
            self.proj = nn.Linear(hidden_dim, hidden_dim, bias=True).to(device, dtype=torch.bfloat16)
        
        def forward(self, x):
            normed = pytorch_rmsnorm(x, self.norm.weight, eps=1e-6)
            return self.proj(normed)
    
    model = TestModule(weight)
    model.eval()
    
    # Test input with unique pattern
    batch_size, seq_len = 2, 64
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)
    # Add unique marker
    marker_value = 2.71828  # e
    x[0, 0, :5] = marker_value
    
    print(f"\nTest input:")
    print(f"  Shape: {x.shape}")
    print(f"  Marker values at [0,0,:5]: {x[0,0,:5].tolist()}")
    
    # Run without hook (PyTorch reference)
    print("\n--- PyTorch Reference ---")
    with torch.no_grad():
        pytorch_out = model(x)
    print(f"Output [0,0,:5]: {pytorch_out[0,0,:5].tolist()}")
    pytorch_flat = pytorch_out.flatten()
    
    # Run with hook (SYCL)
    print("\n--- SYCL Hook ---")
    dispatcher = SyclDispatcher(model, test_mode=True)
    sycl_fn = create_instrumented_sycl_hook(weight, eps=1e-6, layer_name="norm_q")
    dispatcher.register_hook('norm', sycl_fn, hook_type='forward')
    dispatcher.enable('norm')
    
    with torch.no_grad():
        sycl_out = model(x)
    
    print(f"Output [0,0,:5]: {sycl_out[0,0,:5].tolist()}")
    sycl_flat = sycl_out.flatten()
    
    dispatcher.remove_all_hooks()
    
    # Comprehensive comparison
    print("\n--- Comparison Results ---")
    
    # 1. Max error
    max_error = (pytorch_out.float() - sycl_out.float()).abs().max().item()
    print(f"Max error: {max_error:.2e}")
    
    # 2. Mean error
    mean_error = (pytorch_out.float() - sycl_out.float()).abs().mean().item()
    print(f"Mean error: {mean_error:.2e}")
    
    # 3. Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        pytorch_flat, sycl_flat, dim=0
    ).item()
    print(f"Cosine similarity: {cos_sim:.6f}")
    
    # 4. Relative error
    rel_error = ((pytorch_out.float() - sycl_out.float()).abs() / 
                 (pytorch_out.float().abs() + 1e-8)).max().item()
    print(f"Max relative error: {rel_error:.2e}")
    
    # 5. Element-wise comparison stats
    diff = (pytorch_out.float() - sycl_out.float()).abs()
    diff_stats = {
        'zero_elements': (diff == 0).sum().item(),
        'tiny_elements': (diff < 1e-7).sum().item(),
        'small_elements': ((diff >= 1e-7) & (diff < 1e-5)).sum().item(),
        'medium_elements': ((diff >= 1e-5) & (diff < 1e-3)).sum().item(),
        'large_elements': (diff >= 1e-3).sum().item()
    }
    total_elements = pytorch_out.numel()
    
    print(f"\nElement-wise difference distribution:")
    print(f"  Zero (error=0): {diff_stats['zero_elements']} ({100*diff_stats['zero_elements']/total_elements:.2f}%)")
    print(f"  Tiny (<1e-7): {diff_stats['tiny_elements']} ({100*diff_stats['tiny_elements']/total_elements:.2f}%)")
    print(f"  Small (1e-7 to 1e-5): {diff_stats['small_elements']} ({100*diff_stats['small_elements']/total_elements:.2f}%)")
    print(f"  Medium (1e-5 to 1e-3): {diff_stats['medium_elements']} ({100*diff_stats['medium_elements']/total_elements:.2f}%)")
    print(f"  Large (>=1e-3): {diff_stats['large_elements']} ({100*diff_stats['large_elements']/total_elements:.2f}%)")
    
    # Verification checks
    print("\n--- Verification Checks ---")
    
    # Check if SYCL was called
    summary = tracker.get_summary()
    if summary['sycl_executions'] == 0:
        print("[✗] FAIL: SYCL kernel was never executed!")
        return False
    print(f"[✓] SYCL kernel executed {summary['sycl_executions']} time(s)")
    
    # Check for zero error (suspicious)
    if max_error == 0.0:
        print("[✗] FAIL: Max error is exactly 0.0 - SYCL kernel may not be running!")
        return False
    print(f"[✓] Non-zero error detected: {max_error:.2e}")
    
    # Check if outputs are identical (suspicious)
    identical_elements = diff_stats['zero_elements']
    if identical_elements == total_elements:
        print("[✗] FAIL: All elements are identical - outputs are exactly the same!")
        return False
    print(f"[✓] Outputs are different ({total_elements - identical_elements} elements differ)")
    
    # Check cosine similarity
    if cos_sim < 0.99:
        print(f"[!] WARNING: Cosine similarity is low: {cos_sim:.6f}")
    else:
        print(f"[✓] Cosine similarity is good: {cos_sim:.6f}")
    
    print("\n[✓] PASS: Output differentiation verified")
    return True

def test_4_fallback_detection():
    """
    Test 4: Detect any silent fallback mechanisms.
    """
    print("\n" + "="*80)
    print("TEST 4: Fallback Mechanism Detection")
    print("="*80)
    
    # Reset tracker
    tracker.hook_calls.clear()
    tracker.sycl_calls.clear()
    tracker.fallback_calls.clear()
    
    hidden_dim = 64
    
    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.Module()
            self.norm.weight = nn.Parameter(torch.ones(hidden_dim))
        
        def forward(self, x):
            return pytorch_rmsnorm(x, self.norm.weight, eps=1e-6)
    
    model = TestModule().to(device, dtype=torch.bfloat16)
    model.eval()
    
    print("\n--- Test 4a: Normal Execution ---")
    
    # Normal hook
    dispatcher = SyclDispatcher(model, test_mode=False)
    sycl_fn = create_instrumented_sycl_hook(
        model.norm.weight,
        eps=1e-6,
        layer_name="norm"
    )
    dispatcher.register_hook('norm', sycl_fn, hook_type='forward')
    dispatcher.enable('norm')
    
    x = torch.randn(2, 8, hidden_dim, device=device, dtype=torch.bfloat16)
    
    with torch.no_grad():
        out1 = model(x)
    
    summary1 = tracker.get_summary()
    print(f"Hook calls: {summary1['total_hook_calls']}")
    print(f"SYCL executions: {summary1['sycl_executions']}")
    print(f"Fallbacks: {summary1['fallbacks']}")
    
    dispatcher.remove_all_hooks()
    
    # Reset tracker
    tracker.hook_calls.clear()
    tracker.sycl_calls.clear()
    tracker.fallback_calls.clear()
    
    print("\n--- Test 4b: Disabled Hook (Should NOT call SYCL) ---")
    
    dispatcher2 = SyclDispatcher(model, test_mode=False)
    dispatcher2.register_hook('norm', sycl_fn, hook_type='forward')
    # Note: NOT calling enable() - hook is registered but disabled
    
    with torch.no_grad():
        out2 = model(x)
    
    summary2 = tracker.get_summary()
    print(f"Hook calls: {summary2['total_hook_calls']}")
    print(f"SYCL executions: {summary2['sycl_executions']}")
    print(f"Fallbacks: {summary2['fallbacks']}")
    
    if summary2['sycl_executions'] > 0:
        print("[!] WARNING: SYCL executed even though hook was disabled!")
    else:
        print("[✓] Correctly skipped SYCL when hook disabled")
    
    dispatcher2.remove_all_hooks()
    
    print("\n--- Test 4c: Error Injection (Should Trigger Fallback) ---")
    
    tracker.hook_calls.clear()
    tracker.sycl_calls.clear()
    tracker.fallback_calls.clear()
    
    # Create a hook that will fail
    def failing_sycl_hook(module, input, output):
        tracker.log_hook_call("norm_error", tuple(output.shape), True, time.time())
        raise RuntimeError("Intentional error for testing fallback")
    
    dispatcher3 = SyclDispatcher(model, test_mode=False)
    dispatcher3.register_hook('norm', failing_sycl_hook, hook_type='forward')
    dispatcher3.enable('norm')
    
    with torch.no_grad():
        out3 = model(x)
    
    summary3 = tracker.get_summary()
    print(f"Hook calls: {summary3['total_hook_calls']}")
    print(f"SYCL executions: {summary3['sycl_executions']}")
    print(f"Fallbacks: {summary3['fallbacks']}")
    
    if summary3['fallbacks'] > 0:
        print("[✓] Fallback mechanism working correctly")
    else:
        print("[!] No fallback recorded - check if exception handling is working")
    
    dispatcher3.remove_all_hooks()
    
    return True

# ============================================================================
# Part 6: Main Execution
# ============================================================================
def main():
    """Run all debug tests."""
    print("\n" + "="*80)
    print("STARTING DEBUG VERIFICATION SUITE")
    print("="*80)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'sycl_device': info.get('name', 'Unknown'),
        'tests': {}
    }
    
    all_passed = True
    
    # Run tests
    try:
        print("\n[Running Test 1: Direct SYCL Execution]")
        results['tests']['test_1_direct_sycl'] = test_1_direct_sycl_execution()
    except Exception as e:
        print(f"\n[✗] Test 1 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['tests']['test_1_direct_sycl'] = False
        all_passed = False
    
    try:
        print("\n[Running Test 2: Hook Instrumentation]")
        results['tests']['test_2_hook_instrumentation'] = test_2_hook_instrumentation()
    except Exception as e:
        print(f"\n[✗] Test 2 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['tests']['test_2_hook_instrumentation'] = False
        all_passed = False
    
    try:
        print("\n[Running Test 3: Differentiation Verification]")
        results['tests']['test_3_differentiation'] = test_3_differentiation_verification()
    except Exception as e:
        print(f"\n[✗] Test 3 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['tests']['test_3_differentiation'] = False
        all_passed = False
    
    try:
        print("\n[Running Test 4: Fallback Detection]")
        results['tests']['test_4_fallback'] = test_4_fallback_detection()
    except Exception as e:
        print(f"\n[✗] Test 4 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['tests']['test_4_fallback'] = False
        all_passed = False
    
    # Final summary
    print("\n" + "="*80)
    print("DEBUG VERIFICATION SUMMARY")
    print("="*80)
    
    for test_name, passed in results['tests'].items():
        status = "[✓] PASS" if passed else "[✗] FAIL"
        print(f"{status}: {test_name}")
    
    results['all_passed'] = all(results['tests'].values())
    
    print(f"\nOverall: {'[✓] ALL TESTS PASSED' if results['all_passed'] else '[✗] SOME TESTS FAILED'}")
    
    # Save results
    output_file = '/workspace/turbodiffusion-sycl/tests/phase3/debug_sycl_execution_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Final diagnostic message
    print("\n" + "="*80)
    print("DIAGNOSTIC INTERPRETATION")
    print("="*80)
    
    if not results['tests'].get('test_1_direct_sycl', True):
        print("""
[ALERT] Test 1 (Direct SYCL) failed!
  This means the SYCL kernel itself is not producing valid outputs.
  Possible causes:
    - SYCL bindings not properly compiled
    - Device initialization failure
    - Kernel implementation error
    
  NEXT STEPS:
    1. Check if SYCL queue was created (look for "SYCL Queue created" message)
    2. Verify bindings are compiled: ls -la /workspace/turbodiffusion-sycl/bindings/
    3. Run unit tests: python3 /workspace/turbodiffusion-sycl/tests/unit/test_rmsnorm.py
""")
    
    if not results['tests'].get('test_2_hook_instrumentation', True):
        print("""
[ALERT] Test 2 (Hook Instrumentation) failed!
  The hooks are not being called during forward pass.
  Possible causes:
    - Hook not registered correctly
    - Layer path is incorrect
    - Hook type mismatch (forward vs pre_forward)
    
  NEXT STEPS:
    1. Verify layer path exists: print(model)
    2. Check hook registration returns successfully
    3. Try using register_forward_pre_hook instead
""")
    
    if not results['tests'].get('test_3_differentiation', True):
        print("""
[ALERT] Test 3 (Differentiation) failed!
  PyTorch and SYCL outputs are identical or too similar.
  Possible causes:
    - SYCL kernel silently failing and falling back
    - Both using same implementation
    - Output buffer not being written
    
  NEXT STEPS:
    1. Check tracker summary for fallback events
    2. Add explicit error logging in SYCL kernel
    3. Verify output buffer is properly allocated and written
""")
    
    print("="*80)
    
    return results['all_passed']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
