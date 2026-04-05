"""
SYCL Dispatcher - Zero-intrusive hook system for TurboDiffusion

This module provides a non-intrusive way to integrate SYCL kernels with PyTorch
using hooks. No modifications to the original model are required.

Author: TurboDiffusion-SYCL Migration Team
Date: 2026-04-01
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager
import warnings
import numpy as np


class SyclDispatcher:
    """
    Dispatcher for SYCL kernel integration using PyTorch hooks.
    
    This class allows intercepting specific layers in a PyTorch model to
    validate or replace with SYCL implementations without modifying the 
    model structure.
    
    Attributes:
        model: The PyTorch model to hook
        hooks: Dictionary of registered hooks
        enabled_layers: Set of currently enabled SYCL layers
        test_mode: Whether to validate outputs against PyTorch reference
        reference_outputs: Storage for validation data
    
    Example:
        >>> dispatcher = SyclDispatcher(model)
        >>> dispatcher.register_hook('blocks.0.norm1', sycl_layernorm_fn)
        >>> dispatcher.enable('blocks.0.norm1')
        >>> output = model(input)  # Compares SYCL vs PyTorch for blocks.0.norm1
    """
    
    def __init__(self, model: nn.Module, test_mode: bool = False):
        """
        Initialize the SYCL dispatcher.
        
        Args:
            model: The PyTorch model to manage
            test_mode: If True, validate SYCL outputs against PyTorch reference
        """
        self.model = model
        self.hooks: Dict[str, Any] = {}  # layer_path -> hook_handle
        self.enabled_layers: set = set()
        self.test_mode = test_mode
        self.reference_outputs: Dict[str, List[torch.Tensor]] = {}
        self.sycl_outputs: Dict[str, List[torch.Tensor]] = {}
        
        # Get reference device (XPU or CPU)
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            self.reference_device = torch.device('xpu')
        else:
            self.reference_device = torch.device('cpu')
        
        # Statistics
        self.stats = {
            'hook_calls': 0,
            'sycl_calls': 0,
            'validation_failures': 0,  # Changed from cuda_fallbacks
            'errors': []
        }
    
    def _get_layer(self, layer_path: str) -> nn.Module:
        """
        Get a layer by its path (e.g., 'blocks.0.norm1').
        
        Args:
            layer_path: Dot-separated path to the layer
            
        Returns:
            The module at the specified path
            
        Raises:
            AttributeError: If path doesn't exist
        """
        parts = layer_path.split('.')
        module = self.model
        for part in parts:
            # Handle list indexing (e.g., blocks.0 -> blocks[0])
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    def _set_layer(self, layer_path: str, new_module: nn.Module) -> None:
        """
        Set a layer by its path.
        
        Args:
            layer_path: Dot-separated path to the layer
            new_module: Module to set at the path
        """
        parts = layer_path.split('.')
        parent = self.model
        for part in parts[:-1]:
            # Handle list indexing
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        # Handle final part
        last_part = parts[-1]
        if last_part.isdigit():
            parent[int(last_part)] = new_module
        else:
            setattr(parent, last_part, new_module)
    
    def register_hook(
        self, 
        layer_path: str, 
        sycl_fn: Callable,
        hook_type: str = 'forward'
    ) -> None:
        """
        Register a hook on a specific layer.
        
        Args:
            layer_path: Path to the layer (e.g., 'blocks.0.norm1')
            sycl_fn: Function with signature (module, input, output) -> tensor
            hook_type: Type of hook ('forward', 'pre_forward')
        """
        layer = self._get_layer(layer_path)
        
        def hook_fn(module, input, output):
            """Hook function that intercepts layer execution."""
            self.stats['hook_calls'] += 1
            
            if layer_path not in self.enabled_layers:
                # SYCL not enabled for this layer, use original
                return output
            
            try:
                # Run SYCL kernel
                self.stats['sycl_calls'] += 1
                # Call sycl_fn with correct signature (module, input, output)
                sycl_output = sycl_fn(module, input, output)
                
                # Validate if in test mode
                if self.test_mode:
                    self._validate_output(
                        output, sycl_output, layer_path, input[0]
                    )
                
                return sycl_output
                
            except Exception as e:
                # On error, record failure and use original output
                self.stats['validation_failures'] += 1
                self.stats['errors'].append({
                    'layer': layer_path,
                    'error': str(e),
                    'input_shape': input[0].shape if input else None
                })
                warnings.warn(
                    f"SYCL validation failed for {layer_path}, using PyTorch output: {e}"
                )
                return output
        
        # Register the hook
        if hook_type == 'forward':
            handle = layer.register_forward_hook(hook_fn)
        elif hook_type == 'pre_forward':
            handle = layer.register_forward_pre_hook(hook_fn)
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")
        
        self.hooks[layer_path] = {
            'handle': handle,
            'layer': layer,
            'sycl_fn': sycl_fn,
            'hook_type': hook_type
        }
        
        print(f"[Hook] Registered {hook_type} hook on {layer_path}")
    
    def _validate_output(
        self, 
        ref_output: torch.Tensor, 
        sycl_output: torch.Tensor,
        layer_path: str,
        input_tensor: torch.Tensor
    ) -> None:
        """
        Validate SYCL output against PyTorch reference.
        
        Args:
            ref_output: Output from PyTorch reference implementation
            sycl_output: Output from SYCL implementation
            layer_path: Path to the layer being validated
            input_tensor: Input tensor for context
        """
        # Store outputs for later analysis
        if layer_path not in self.reference_outputs:
            self.reference_outputs[layer_path] = []
            self.sycl_outputs[layer_path] = []
        
        self.reference_outputs[layer_path].append(ref_output.detach().cpu())
        self.sycl_outputs[layer_path].append(sycl_output.detach().cpu())
        
        # Compute error metrics
        max_error = (ref_output - sycl_output).abs().max().item()
        mean_error = (ref_output - sycl_output).abs().mean().item()
        
        # Cosine similarity
        ref_flat = ref_output.flatten()
        sycl_flat = sycl_output.flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            ref_flat, sycl_flat, dim=0
        ).item()
        
        # Log validation results
        print(f"[Validate] {layer_path}:")
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Output shape: {ref_output.shape}")
        print(f"  Max error: {max_error:.2e}")
        print(f"  Mean error: {mean_error:.2e}")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        
        # Check thresholds
        if max_error > 1e-3:
            warnings.warn(
                f"{layer_path}: Max error {max_error:.2e} exceeds threshold 1e-3"
            )
        if cos_sim < 0.999:
            warnings.warn(
                f"{layer_path}: Cosine similarity {cos_sim:.6f} below threshold 0.999"
            )
    
    def enable(self, layer_path: str) -> None:
        """
        Enable SYCL validation for a specific layer.
        
        Args:
            layer_path: Path to the layer to enable
        """
        if layer_path not in self.hooks:
            raise ValueError(f"No hook registered for {layer_path}")
        self.enabled_layers.add(layer_path)
        print(f"[Enable] SYCL validation for {layer_path}")
    
    def disable(self, layer_path: str) -> None:
        """
        Disable SYCL validation for a specific layer.
        
        Args:
            layer_path: Path to the layer to disable
        """
        self.enabled_layers.discard(layer_path)
        print(f"[Disable] SYCL validation for {layer_path}")
    
    def enable_pattern(self, pattern: str) -> None:
        """
        Enable SYCL for all layers matching a pattern.
        
        Args:
            pattern: Glob pattern (e.g., 'blocks.*.norm1')
        """
        import fnmatch
        matched = []
        for layer_path in self.hooks.keys():
            if fnmatch.fnmatch(layer_path, pattern):
                self.enable(layer_path)
                matched.append(layer_path)
        print(f"[Enable Pattern] Matched {len(matched)} layers: {pattern}")
    
    def enable_all(self) -> None:
        """Enable SYCL for all registered hooks."""
        for layer_path in self.hooks.keys():
            self.enable(layer_path)
    
    def disable_all(self) -> None:
        """Disable all SYCL validations."""
        self.enabled_layers.clear()
        print("[Disable All] All SYCL validations disabled")
    
    @contextmanager
    def temporary_enable(self, layer_paths: List[str]):
        """
        Context manager to temporarily enable layers.
        
        Args:
            layer_paths: List of layer paths to enable
            
        Example:
            >>> with dispatcher.temporary_enable(['blocks.0.norm1']):
            ...     output = model(input)
        """
        previous = self.enabled_layers.copy()
        for path in layer_paths:
            self.enable(path)
        try:
            yield
        finally:
            self.enabled_layers = previous
    
    def remove_hook(self, layer_path: str) -> None:
        """
        Remove a hook from a layer.
        
        Args:
            layer_path: Path to the layer
        """
        if layer_path in self.hooks:
            self.hooks[layer_path]['handle'].remove()
            del self.hooks[layer_path]
            self.enabled_layers.discard(layer_path)
            print(f"[Remove] Hook removed from {layer_path}")
    
    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        for layer_path in list(self.hooks.keys()):
            self.remove_hook(layer_path)
        print("[Remove All] All hooks removed")
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Generate a validation report for test mode.
        
        Returns:
            Dictionary containing validation statistics
        """
        if not self.test_mode:
            return {"error": "Test mode was not enabled"}
        
        report = {
            'layers': {},
            'summary': {
                'total_layers': len(self.reference_outputs),
                'total_validations': sum(
                    len(v) for v in self.reference_outputs.values()
                ),
                'max_errors': [],
                'mean_errors': [],
                'cosine_sims': []
            }
        }
        
        for layer_path in self.reference_outputs:
            ref_vals = self.reference_outputs[layer_path]
            sycl_vals = self.sycl_outputs[layer_path]
            
            errors = [(c - s).abs() for c, s in zip(ref_vals, sycl_vals)]
            max_errs = [e.max().item() for e in errors]
            mean_errs = [e.mean().item() for e in errors]
            
            report['layers'][layer_path] = {
                'num_validations': len(ref_vals),
                'max_error': max(max_errs),
                'mean_error': sum(mean_errs) / len(mean_errs),
                'max_error_history': max_errs
            }
            
            report['summary']['max_errors'].extend(max_errs)
            report['summary']['mean_errors'].extend(mean_errs)
        
        # Overall statistics
        if report['summary']['max_errors']:
            report['summary']['overall_max_error'] = max(
                report['summary']['max_errors']
            )
            report['summary']['overall_mean_error'] = sum(
                report['summary']['mean_errors']
            ) / len(report['summary']['mean_errors'])
        
        return report
    
    def print_stats(self) -> None:
        """Print dispatcher statistics."""
        print("\n" + "="*60)
        print("SYCL Dispatcher Statistics")
        print("="*60)
        print(f"Total hook calls: {self.stats['hook_calls']}")
        print(f"SYCL kernel calls: {self.stats['sycl_calls']}")
        print(f"Validation failures: {self.stats['validation_failures']}")
        print(f"Active hooks: {len(self.hooks)}")
        print(f"Enabled layers: {len(self.enabled_layers)}")
        print(f"Reference device: {self.reference_device}")
        
        if self.stats['errors']:
            print(f"\nErrors ({len(self.stats['errors'])}):")
            for err in self.stats['errors'][-5:]:  # Show last 5
                print(f"  - {err['layer']}: {err['error']}")
        
        if self.test_mode:
            report = self.get_validation_report()
            if 'summary' in report:
                print(f"\nValidation Summary:")
                print(f"  Total validations: {report['summary']['total_validations']}")
                print(f"  Overall max error: {report['summary'].get('overall_max_error', 'N/A'):.2e}")
                print(f"  Overall mean error: {report['summary'].get('overall_mean_error', 'N/A'):.2e}")
        
        print("="*60 + "\n")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup hooks."""
        self.remove_all_hooks()
        return False


class LayerRegistry:
    """
    Registry for layer paths in Wan2.1 model.
    
    Provides convenient access to common layer paths.
    
    Model Structure (Wan2.1 1.3B):
    - 30 blocks (0-29)
    - Each block has:
      - self_attn with norm_q, norm_k (RMSNorm)
      - cross_attn with norm_q, norm_k (RMSNorm)
      - norm3 (LayerNorm)
    - No head.norm (head only has linear projection)
    """
    
    @staticmethod
    def get_block_norm3(block_idx: int) -> str:
        """Get path to block's norm3 (FFN input LayerNorm)."""
        return f"blocks.{block_idx}.norm3"
    
    @staticmethod
    def get_block_self_attn_norm_q(block_idx: int) -> str:
        """Get path to block's self-attention RMSNorm for Q."""
        return f"blocks.{block_idx}.self_attn.norm_q"
    
    @staticmethod
    def get_block_self_attn_norm_k(block_idx: int) -> str:
        """Get path to block's self-attention RMSNorm for K."""
        return f"blocks.{block_idx}.self_attn.norm_k"
    
    @staticmethod
    def get_block_cross_attn_norm_q(block_idx: int) -> str:
        """Get path to block's cross-attention RMSNorm for Q."""
        return f"blocks.{block_idx}.cross_attn.norm_q"
    
    @staticmethod
    def get_block_cross_attn_norm_k(block_idx: int) -> str:
        """Get path to block's cross-attention RMSNorm for K."""
        return f"blocks.{block_idx}.cross_attn.norm_k"
    
    @staticmethod
    def get_all_norm_layers(num_blocks: int = 30) -> List[str]:
        """Get all normalization layer paths."""
        layers = []
        for i in range(num_blocks):
            layers.extend([
                f"blocks.{i}.norm3",
                f"blocks.{i}.self_attn.norm_q",
                f"blocks.{i}.self_attn.norm_k",
                f"blocks.{i}.cross_attn.norm_q",
                f"blocks.{i}.cross_attn.norm_k",
            ])
        return layers
