"""
Fallback Mechanism for SYCL Kernels

Provides automatic fallback to CUDA when SYCL kernels fail or produce
incorrect results.

Author: TurboDiffusion-SYCL Migration Team
Date: 2026-04-01
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Any
from enum import Enum
import warnings
import traceback


class FallbackReason(Enum):
    """Reasons for falling back to CUDA."""
    SYCL_NOT_AVAILABLE = "sycl_not_available"
    SYCL_ERROR = "sycl_error"
    VALIDATION_FAILED = "validation_failed"
    PERFORMANCE_REGRESSION = "performance_regression"
    USER_REQUEST = "user_request"


class FallbackPolicy:
    """
    Policy for handling SYCL failures.
    
    Controls when and how to fall back to CUDA implementations.
    """
    
    def __init__(
        self,
        auto_fallback: bool = True,
        max_error_threshold: float = 1e-3,
        min_cosine_similarity: float = 0.999,
        log_failures: bool = True,
        raise_on_critical: bool = False
    ):
        """
        Initialize fallback policy.
        
        Args:
            auto_fallback: Automatically fallback on SYCL failure
            max_error_threshold: Max allowed error vs CUDA
            min_cosine_similarity: Min cosine similarity with CUDA
            log_failures: Whether to log fallback events
            raise_on_critical: Raise exception on critical failures
        """
        self.auto_fallback = auto_fallback
        self.max_error_threshold = max_error_threshold
        self.min_cosine_similarity = min_cosine_similarity
        self.log_failures = log_failures
        self.raise_on_critical = raise_on_critical
    
    def should_fallback(
        self,
        reason: FallbackReason,
        error: Optional[Exception] = None,
        max_error: Optional[float] = None,
        cosine_sim: Optional[float] = None
    ) -> bool:
        """
        Determine if fallback should occur.
        
        Args:
            reason: Why fallback is being considered
            error: Exception that occurred (if any)
            max_error: Measured max error vs CUDA
            cosine_sim: Measured cosine similarity
            
        Returns:
            True if should fallback to CUDA
        """
        if not self.auto_fallback:
            return False
        
        # Always fallback on SYCL errors
        if reason in [FallbackReason.SYCL_NOT_AVAILABLE, FallbackReason.SYCL_ERROR]:
            return True
        
        # Check numerical accuracy
        if max_error is not None and max_error > self.max_error_threshold:
            return True
        
        if cosine_sim is not None and cosine_sim < self.min_cosine_similarity:
            return True
        
        return False


class FallbackManager:
    """
    Manages fallback state and tracks SYCL/CUDA switching.
    
    This class keeps track of which layers have fallen back to CUDA
    and provides utilities for managing the fallback state.
    """
    
    def __init__(self, policy: Optional[FallbackPolicy] = None):
        """
        Initialize fallback manager.
        
        Args:
            policy: Fallback policy (uses default if None)
        """
        self.policy = policy or FallbackPolicy()
        self.fallback_layers: set = set()  # Layers that have fallen back
        self.failure_history: list = []  # History of failures
        self.success_history: list = []  # History of successes
        
    def register_failure(
        self,
        layer_path: str,
        reason: FallbackReason,
        error: Optional[Exception] = None,
        context: Optional[dict] = None
    ) -> bool:
        """
        Register a SYCL failure and determine if fallback should occur.
        
        Args:
            layer_path: Path to the layer that failed
            reason: Why the failure occurred
            error: Exception details
            context: Additional context (input shape, etc.)
            
        Returns:
            True if fallback was triggered
        """
        # Extract metrics from context
        max_error = context.get('max_error') if context else None
        cosine_sim = context.get('cosine_sim') if context else None
        
        # Check if should fallback
        should_fallback = self.policy.should_fallback(
            reason, error, max_error, cosine_sim
        )
        
        # Record failure
        failure_info = {
            'layer': layer_path,
            'reason': reason.value,
            'fallback_triggered': should_fallback,
            'error': str(error) if error else None,
            'context': context,
            'timestamp': torch.cuda.Event(enable_timing=True)
        }
        self.failure_history.append(failure_info)
        
        if should_fallback:
            self.fallback_layers.add(layer_path)
            
            if self.policy.log_failures:
                self._log_failure(failure_info)
            
            if self.policy.raise_on_critical and reason == FallbackReason.SYCL_ERROR:
                raise RuntimeError(
                    f"Critical SYCL error in {layer_path}: {error}"
                )
        
        return should_fallback
    
    def register_success(
        self,
        layer_path: str,
        metrics: Optional[dict] = None
    ) -> None:
        """
        Register a successful SYCL execution.
        
        Args:
            layer_path: Path to the layer
            metrics: Performance/accuracy metrics
        """
        self.success_history.append({
            'layer': layer_path,
            'metrics': metrics,
            'timestamp': torch.cuda.Event(enable_timing=True)
        })
        
        # Remove from fallback set if previously failed
        if layer_path in self.fallback_layers:
            self.fallback_layers.remove(layer_path)
            if self.policy.log_failures:
                print(f"[Fallback] Layer {layer_path} recovered and using SYCL")
    
    def _log_failure(self, failure_info: dict) -> None:
        """Log a failure event."""
        layer = failure_info['layer']
        reason = failure_info['reason']
        error = failure_info['error']
        
        msg = f"[Fallback] Layer {layer} fell back to CUDA. Reason: {reason}"
        if error:
            msg += f", Error: {error}"
        
        warnings.warn(msg)
    
    def is_fallback(self, layer_path: str) -> bool:
        """
        Check if a layer is currently using fallback.
        
        Args:
            layer_path: Path to the layer
            
        Returns:
            True if layer is using CUDA fallback
        """
        return layer_path in self.fallback_layers
    
    def get_fallback_layers(self) -> set:
        """Get all layers currently using fallback."""
        return self.fallback_layers.copy()
    
    def reset_layer(self, layer_path: str) -> None:
        """
        Reset fallback state for a layer.
        
        Args:
            layer_path: Path to the layer to reset
        """
        self.fallback_layers.discard(layer_path)
        print(f"[Fallback] Reset fallback state for {layer_path}")
    
    def reset_all(self) -> None:
        """Reset all fallback states."""
        self.fallback_layers.clear()
        self.failure_history.clear()
        self.success_history.clear()
        print("[Fallback] All fallback states reset")
    
    def get_statistics(self) -> dict:
        """
        Get fallback statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_failures = len(self.failure_history)
        total_successes = len(self.success_history)
        total = total_failures + total_successes
        
        failure_reasons = {}
        for f in self.failure_history:
            reason = f['reason']
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        return {
            'total_attempts': total,
            'total_failures': total_failures,
            'total_successes': total_successes,
            'fallback_rate': total_failures / total if total > 0 else 0,
            'currently_fallback': len(self.fallback_layers),
            'failure_reasons': failure_reasons,
            'fallback_layers': list(self.fallback_layers)
        }
    
    def print_report(self) -> None:
        """Print a detailed fallback report."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("Fallback Manager Report")
        print("="*60)
        print(f"Total SYCL attempts: {stats['total_attempts']}")
        print(f"Successful: {stats['total_successes']}")
        print(f"Failed: {stats['total_failures']}")
        print(f"Fallback rate: {stats['fallback_rate']*100:.1f}%")
        print(f"Currently using CUDA: {stats['currently_fallback']} layers")
        
        if stats['failure_reasons']:
            print("\nFailure reasons:")
            for reason, count in stats['failure_reasons'].items():
                print(f"  - {reason}: {count}")
        
        if stats['fallback_layers']:
            print("\nLayers using CUDA fallback:")
            for layer in sorted(stats['fallback_layers']):
                print(f"  - {layer}")
        
        print("="*60 + "\n")


class AdaptiveFallback:
    """
    Adaptive fallback that learns which layers work best with SYCL.
    
    Over time, this can automatically enable/disable SYCL for specific
    layers based on their performance and reliability.
    """
    
    def __init__(
        self,
        policy: Optional[FallbackPolicy] = None,
        min_samples: int = 10,
        success_threshold: float = 0.95
    ):
        """
        Initialize adaptive fallback.
        
        Args:
            policy: Base fallback policy
            min_samples: Minimum samples before making decisions
            success_threshold: Success rate threshold for SYCL
        """
        self.policy = policy or FallbackPolicy()
        self.min_samples = min_samples
        self.success_threshold = success_threshold
        self.manager = FallbackManager(policy)
        
        # Track performance per layer
        self.layer_stats: dict = {}
    
    def update_layer_stats(
        self,
        layer_path: str,
        success: bool,
        execution_time: Optional[float] = None,
        error: Optional[float] = None
    ) -> None:
        """
        Update statistics for a layer.
        
        Args:
            layer_path: Path to the layer
            success: Whether SYCL execution succeeded
            execution_time: Time taken (ms)
            error: Numerical error vs CUDA
        """
        if layer_path not in self.layer_stats:
            self.layer_stats[layer_path] = {
                'attempts': 0,
                'successes': 0,
                'failures': 0,
                'times': [],
                'errors': []
            }
        
        stats = self.layer_stats[layer_path]
        stats['attempts'] += 1
        
        if success:
            stats['successes'] += 1
            if execution_time:
                stats['times'].append(execution_time)
            if error:
                stats['errors'].append(error)
        else:
            stats['failures'] += 1
    
    def should_use_sycl(self, layer_path: str) -> bool:
        """
        Determine if SYCL should be used for a layer.
        
        Args:
            layer_path: Path to the layer
            
        Returns:
            True if SYCL should be used
        """
        # If currently in fallback, don't use SYCL
        if self.manager.is_fallback(layer_path):
            return False
        
        # If not enough samples, try SYCL
        if layer_path not in self.layer_stats:
            return True
        
        stats = self.layer_stats[layer_path]
        if stats['attempts'] < self.min_samples:
            return True
        
        # Check success rate
        success_rate = stats['successes'] / stats['attempts']
        return success_rate >= self.success_threshold
    
    def get_recommendations(self) -> dict:
        """
        Get recommendations for which layers to enable/disable.
        
        Returns:
            Dictionary with 'enable' and 'disable' lists
        """
        enable = []
        disable = []
        
        for layer_path, stats in self.layer_stats.items():
            if stats['attempts'] >= self.min_samples:
                success_rate = stats['successes'] / stats['attempts']
                if success_rate >= self.success_threshold:
                    enable.append(layer_path)
                else:
                    disable.append(layer_path)
        
        return {'enable': enable, 'disable': disable}
