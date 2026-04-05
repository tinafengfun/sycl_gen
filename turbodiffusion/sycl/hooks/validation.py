"""
Validation Mechanism for SYCL Kernels

Provides validation and error tracking for SYCL kernel implementations
against PyTorch reference implementations (XPU or CPU).

Author: TurboDiffusion-SYCL Migration Team
Date: 2026-04-01
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Any
from enum import Enum
import warnings
import traceback


class ValidationStatus(Enum):
    """Status of SYCL kernel validation."""
    NOT_TESTED = "not_tested"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class ValidationPolicy:
    """
    Policy for validating SYCL kernel accuracy.
    
    Controls validation thresholds and behavior.
    """
    
    def __init__(
        self,
        max_error_threshold: float = 1e-3,
        min_cosine_similarity: float = 0.999,
        log_validations: bool = True,
        raise_on_critical: bool = False
    ):
        """
        Initialize validation policy.
        
        Args:
            max_error_threshold: Max allowed error vs reference
            min_cosine_similarity: Min cosine similarity with reference
            log_validations: Whether to log validation events
            raise_on_critical: Raise exception on critical errors
        """
        self.max_error_threshold = max_error_threshold
        self.min_cosine_similarity = min_cosine_similarity
        self.log_validations = log_validations
        self.raise_on_critical = raise_on_critical
    
    def validate(
        self,
        max_error: Optional[float] = None,
        cosine_sim: Optional[float] = None
    ) -> bool:
        """
        Determine if validation passes.
        
        Args:
            max_error: Measured max error vs reference
            cosine_sim: Measured cosine similarity
            
        Returns:
            True if validation passes
        """
        # Check numerical accuracy
        if max_error is not None and max_error > self.max_error_threshold:
            return False
        
        if cosine_sim is not None and cosine_sim < self.min_cosine_similarity:
            return False
        
        return True


class ValidationManager:
    """
    Manages validation state and tracks SYCL kernel accuracy.
    
    This class keeps track of which layers have been validated
    and provides utilities for managing the validation state.
    """
    
    def __init__(self, policy: Optional[ValidationPolicy] = None):
        """
        Initialize validation manager.
        
        Args:
            policy: Validation policy (uses default if None)
        """
        self.policy = policy or ValidationPolicy()
        self.failed_layers: set = set()  # Layers that failed validation
        self.validation_history: list = []  # History of validations
        self.success_history: list = []  # History of successes
        
    def register_failure(
        self,
        layer_path: str,
        reason: str,
        error: Optional[Exception] = None,
        context: Optional[dict] = None
    ) -> bool:
        """
        Register a SYCL validation failure.
        
        Args:
            layer_path: Path to the layer that failed
            reason: Why the failure occurred
            error: Exception details
            context: Additional context (input shape, etc.)
            
        Returns:
            True if validation failed
        """
        # Extract metrics from context
        max_error = context.get('max_error') if context else None
        cosine_sim = context.get('cosine_sim') if context else None
        
        # Record failure
        failure_info = {
            'layer': layer_path,
            'reason': reason,
            'error': str(error) if error else None,
            'context': context,
            'timestamp': None  # Removed CUDA event reference
        }
        self.validation_history.append(failure_info)
        
        self.failed_layers.add(layer_path)
        
        if self.policy.log_validations:
            self._log_failure(failure_info)
        
        if self.policy.raise_on_critical and error:
            raise RuntimeError(
                f"Critical SYCL error in {layer_path}: {error}"
            )
        
        return True
    
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
            'timestamp': None  # Removed CUDA event reference
        })
        
        # Remove from failed set if previously failed
        if layer_path in self.failed_layers:
            self.failed_layers.remove(layer_path)
            if self.policy.log_validations:
                print(f"[Validation] Layer {layer_path} recovered")
    
    def _log_failure(self, failure_info: dict) -> None:
        """Log a validation failure event."""
        layer = failure_info['layer']
        reason = failure_info['reason']
        error = failure_info['error']
        
        msg = f"[Validation] Layer {layer} failed validation. Reason: {reason}"
        if error:
            msg += f", Error: {error}"
        
        warnings.warn(msg)
    
    def is_failed(self, layer_path: str) -> bool:
        """
        Check if a layer has failed validation.
        
        Args:
            layer_path: Path to the layer
            
        Returns:
            True if layer failed validation
        """
        return layer_path in self.failed_layers
    
    def get_failed_layers(self) -> set:
        """Get all layers that failed validation."""
        return self.failed_layers.copy()
    
    def reset_layer(self, layer_path: str) -> None:
        """
        Reset validation state for a layer.
        
        Args:
            layer_path: Path to the layer to reset
        """
        self.failed_layers.discard(layer_path)
        print(f"[Validation] Reset validation state for {layer_path}")
    
    def reset_all(self) -> None:
        """Reset all validation states."""
        self.failed_layers.clear()
        self.validation_history.clear()
        self.success_history.clear()
        print("[Validation] All validation states reset")
    
    def get_statistics(self) -> dict:
        """
        Get validation statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_validations = len(self.validation_history)
        total_successes = len(self.success_history)
        total = total_validations + total_successes
        
        failure_reasons = {}
        for f in self.validation_history:
            reason = f['reason']
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        return {
            'total_attempts': total,
            'total_failures': total_validations,
            'total_successes': total_successes,
            'failure_rate': total_validations / total if total > 0 else 0,
            'currently_failed': len(self.failed_layers),
            'failure_reasons': failure_reasons,
            'failed_layers': list(self.failed_layers)
        }
    
    def print_report(self) -> None:
        """Print a detailed validation report."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("Validation Manager Report")
        print("="*60)
        print(f"Total SYCL attempts: {stats['total_attempts']}")
        print(f"Successful: {stats['total_successes']}")
        print(f"Failed: {stats['total_failures']}")
        print(f"Failure rate: {stats['failure_rate']*100:.1f}%")
        print(f"Currently failed: {stats['currently_failed']} layers")
        
        if stats['failure_reasons']:
            print("\nFailure reasons:")
            for reason, count in stats['failure_reasons'].items():
                print(f"  - {reason}: {count}")
        
        if stats['failed_layers']:
            print("\nFailed layers:")
            for layer in sorted(stats['failed_layers']):
                print(f"  - {layer}")
        
        print("="*60 + "\n")


# Backwards compatibility aliases
FallbackReason = ValidationStatus
FallbackPolicy = ValidationPolicy
FallbackManager = ValidationManager
