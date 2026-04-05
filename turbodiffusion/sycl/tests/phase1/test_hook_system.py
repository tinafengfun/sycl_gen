"""
Phase 1.1 Test: Hook System Validation

Tests the basic functionality of the SYCL hook system without requiring
actual SYCL kernel bindings. Uses mock SYCL functions to verify the
dispatcher works correctly.

Author: TurboDiffusion-SYCL Migration Team
Date: 2026-04-01
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict
import json
from datetime import datetime

# Import our hook system
from hooks import SyclDispatcher, LayerRegistry
from hooks.layer_adapters import SyclLayerNorm, SyclRMSNorm
from hooks.fallback import FallbackManager, FallbackPolicy, FallbackReason


class MockSyclKernels:
    """
    Mock SYCL kernels for testing the hook system.
    
    These simulate SYCL kernel behavior without requiring actual bindings.
    """
    
    @staticmethod
    def mock_layernorm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Mock LayerNorm that adds small noise to test validation."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + eps)
        # Add small controlled noise to simulate SYCL vs CUDA difference
        noise = torch.randn_like(normalized) * 1e-5
        return normalized + noise
    
    @staticmethod
    def mock_rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Mock RMSNorm that adds small noise."""
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        normalized = x / rms
        noise = torch.randn_like(normalized) * 1e-5
        return normalized + noise


class SimpleTestModel(nn.Module):
    """Simple model for testing hooks."""
    
    def __init__(self, dim: int = 64, num_blocks: int = 2):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        
        # Create simple blocks similar to Wan2.1 structure
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(dim),
                'norm2': nn.LayerNorm(dim),
                'linear': nn.Linear(dim, dim)
            })
            for _ in range(num_blocks)
        ])
        
        self.head = nn.ModuleDict({
            'norm': nn.LayerNorm(dim),
            'linear': nn.Linear(dim, dim)
        })
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through model."""
        for i, block in enumerate(self.blocks):
            # Simple transformer-like block
            x = block['norm1'](x)
            x = block['linear'](x)
            x = block['norm2'](x)
        
        x = self.head['norm'](x)
        x = self.head['linear'](x)
        return x


def test_01_dispatcher_basic():
    """Test 01: Basic dispatcher functionality."""
    print("\n" + "="*60)
    print("Test 01: Dispatcher Basic Functionality")
    print("="*60)
    
    # Create simple model
    model = SimpleTestModel(dim=64, num_blocks=2)
    model.eval()
    
    # Create dispatcher
    dispatcher = SyclDispatcher(model, test_mode=True)
    
    # Register mock hooks
    def mock_hook_fn(x):
        return MockSyclKernels.mock_layernorm(x)
    
    dispatcher.register_hook('head.norm', mock_hook_fn)
    
    # Test without enabling
    x = torch.randn(2, 4, 64)  # batch, seq, dim
    with torch.no_grad():
        output_no_hook = model(x)
    
    print(f"✓ Model forward without hooks works")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output_no_hook.shape}")
    
    # Enable hook
    dispatcher.enable('head.norm')
    
    with torch.no_grad():
        output_with_hook = model(x)
    
    print(f"✓ Model forward with hooks works")
    print(f"  Hook calls: {dispatcher.stats['hook_calls']}")
    print(f"  SYCL calls: {dispatcher.stats['sycl_calls']}")
    
    # Verify dispatcher tracked correctly
    assert dispatcher.stats['hook_calls'] == 2  # One per forward
    assert dispatcher.stats['sycl_calls'] == 1  # Only head.norm enabled
    
    dispatcher.print_stats()
    dispatcher.remove_all_hooks()
    
    print("✓ Test 01 PASSED")
    return True


def test_02_layer_registry():
    """Test 02: Layer registry paths."""
    print("\n" + "="*60)
    print("Test 02: Layer Registry")
    print("="*60)
    
    # Test LayerRegistry methods
    paths = [
        LayerRegistry.get_head_norm(),
        LayerRegistry.get_block_norm1(0),
        LayerRegistry.get_block_norm2(5),
        LayerRegistry.get_block_norm3(10),
        LayerRegistry.get_block_rms_norm_q(0),
        LayerRegistry.get_block_rms_norm_k(15),
    ]
    
    print("Generated paths:")
    for path in paths:
        print(f"  - {path}")
    
    # Test all norm layers generation
    all_layers = LayerRegistry.get_all_norm_layers(num_blocks=3)
    print(f"\nTotal layers for 3 blocks: {len(all_layers)}")
    for layer in all_layers:
        print(f"  - {layer}")
    
    print("✓ Test 02 PASSED")
    return True


def test_03_validation():
    """Test 03: Output validation in test mode."""
    print("\n" + "="*60)
    print("Test 03: Output Validation")
    print("="*60)
    
    model = SimpleTestModel(dim=64, num_blocks=1)
    model.eval()
    
    # Create dispatcher with test mode
    dispatcher = SyclDispatcher(model, test_mode=True)
    
    # Register hook with small error
    def mock_sycl_with_error(x):
        # Add controlled error
        return x + 0.0005  # Small error within threshold
    
    dispatcher.register_hook('head.norm', mock_sycl_with_error)
    dispatcher.enable('head.norm')
    
    # Run forward
    x = torch.randn(2, 4, 64)
    with torch.no_grad():
        output = model(x)
    
    # Check validation was performed
    assert 'head.norm' in dispatcher.reference_outputs
    assert 'head.norm' in dispatcher.sycl_outputs
    
    # Get validation report
    report = dispatcher.get_validation_report()
    print(f"\nValidation Report:")
    print(f"  Total layers validated: {report['summary']['total_layers']}")
    print(f"  Total validations: {report['summary']['total_validations']}")
    print(f"  Overall max error: {report['summary']['overall_max_error']:.2e}")
    
    assert report['summary']['overall_max_error'] < 1e-3, "Error exceeds threshold"
    
    dispatcher.remove_all_hooks()
    print("✓ Test 03 PASSED")
    return True


def test_04_fallback_mechanism():
    """Test 04: Fallback mechanism."""
    print("\n" + "="*60)
    print("Test 04: Fallback Mechanism")
    print("="*60)
    
    # Create fallback manager
    policy = FallbackPolicy(
        auto_fallback=True,
        max_error_threshold=1e-3,
        log_failures=True
    )
    manager = FallbackManager(policy)
    
    # Test fallback on error
    class FailingSyclFn:
        """Mock SYCL function that fails."""
        def __init__(self):
            self.call_count = 0
        
        def __call__(self, x):
            self.call_count += 1
            if self.call_count == 1:
                raise RuntimeError("Mock SYCL failure")
            return x
    
    failing_fn = FailingSyclFn()
    
    model = SimpleTestModel(dim=64, num_blocks=1)
    dispatcher = SyclDispatcher(model, test_mode=False)
    dispatcher.register_hook('head.norm', failing_fn)
    dispatcher.enable('head.norm')
    
    # First call should trigger fallback
    x = torch.randn(2, 4, 64)
    with torch.no_grad():
        output1 = model(x)
    
    print(f"After first call (with failure):")
    print(f"  Hook calls: {dispatcher.stats['hook_calls']}")
    print(f"  SYCL calls: {dispatcher.stats['sycl_calls']}")
    print(f"  CUDA fallbacks: {dispatcher.stats['cuda_fallbacks']}")
    print(f"  Errors: {len(dispatcher.stats['errors'])}")
    
    assert dispatcher.stats['cuda_fallbacks'] == 1
    assert len(dispatcher.stats['errors']) == 1
    
    dispatcher.remove_all_hooks()
    print("✓ Test 04 PASSED")
    return True


def test_05_temporary_enable():
    """Test 05: Temporary enable context manager."""
    print("\n" + "="*60)
    print("Test 05: Temporary Enable")
    print("="*60)
    
    model = SimpleTestModel(dim=64, num_blocks=1)
    
    dispatcher = SyclDispatcher(model)
    dispatcher.register_hook('head.norm', lambda x: x * 2)
    dispatcher.register_hook('blocks.0.norm1', lambda x: x * 3)
    
    # Enable only head.norm
    dispatcher.enable('head.norm')
    
    with torch.no_grad():
        output1 = model(torch.randn(2, 4, 64))
    
    print(f"Before temporary: enabled = {dispatcher.enabled_layers}")
    
    # Temporarily enable both
    with dispatcher.temporary_enable(['head.norm', 'blocks.0.norm1']):
        with torch.no_grad():
            output2 = model(torch.randn(2, 4, 64))
        print(f"During temporary: enabled = {dispatcher.enabled_layers}")
    
    print(f"After temporary: enabled = {dispatcher.enabled_layers}")
    
    # Verify restored
    assert dispatcher.enabled_layers == {'head.norm'}
    
    dispatcher.remove_all_hooks()
    print("✓ Test 05 PASSED")
    return True


def test_06_layer_adapters():
    """Test 06: Layer adapters creation."""
    print("\n" + "="*60)
    print("Test 06: Layer Adapters")
    print("="*60)
    
    # Test SyclLayerNorm
    pytorch_ln = nn.LayerNorm(64, eps=1e-5)
    sycl_ln = SyclLayerNorm.from_layernorm(pytorch_ln)
    
    x = torch.randn(2, 4, 64)
    
    with torch.no_grad():
        pytorch_out = pytorch_ln(x)
        sycl_out = sycl_ln(x)
    
    error = (pytorch_out - sycl_out).abs().max().item()
    print(f"LayerNorm adapter error: {error:.2e}")
    assert error < 1e-6, "LayerNorm adapter mismatch"
    
    # Test can_adapt
    from hooks.layer_adapters import LayerAdapterFactory
    
    can_adapt, layer_type = LayerAdapterFactory.can_adapt(pytorch_ln)
    print(f"Can adapt LayerNorm: {can_adapt}, type: {layer_type}")
    assert can_adapt and layer_type == 'layernorm'
    
    linear = nn.Linear(64, 64)
    can_adapt, layer_type = LayerAdapterFactory.can_adapt(linear)
    print(f"Can adapt Linear: {can_adapt}, type: {layer_type}")
    assert can_adapt and layer_type == 'linear'
    
    print("✓ Test 06 PASSED")
    return True


def run_all_tests() -> Dict:
    """Run all Phase 1.1 tests."""
    print("\n" + "="*70)
    print("PHASE 1.1: Hook System Validation Tests")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Dispatcher Basic", test_01_dispatcher_basic),
        ("Layer Registry", test_02_layer_registry),
        ("Output Validation", test_03_validation),
        ("Fallback Mechanism", test_04_fallback_mechanism),
        ("Temporary Enable", test_05_temporary_enable),
        ("Layer Adapters", test_06_layer_adapters),
    ]
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': [],
        'passed': 0,
        'failed': 0,
        'total': len(tests)
    }
    
    for test_name, test_fn in tests:
        try:
            success = test_fn()
            results['tests'].append({
                'name': test_name,
                'status': 'PASSED' if success else 'FAILED',
                'error': None
            })
            if success:
                results['passed'] += 1
            else:
                results['failed'] += 1
        except Exception as e:
            print(f"\n✗ Test '{test_name}' FAILED with exception:")
            print(f"  {str(e)}")
            traceback.print_exc()
            results['tests'].append({
                'name': test_name,
                'status': 'FAILED',
                'error': str(e)
            })
            results['failed'] += 1
    
    # Print summary
    print("\n" + "="*70)
    print("PHASE 1.1 TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {results['passed']/results['total']*100:.1f}%")
    
    if results['failed'] == 0:
        print("\n🎉 ALL TESTS PASSED - Phase 1.1 Complete!")
    else:
        print("\n⚠️  Some tests failed - Please review")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    import traceback
    
    # Run tests
    results = run_all_tests()
    
    # Save results to file
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'phase1_1_test_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Exit with appropriate code
    sys.exit(0 if results['failed'] == 0 else 1)
