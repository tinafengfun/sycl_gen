"""
TurboDiffusion SYCL - Python Package

This package provides Python bindings for SYCL kernels optimized for
Intel GPUs (specifically B60/Xe2 architecture).

Modules:
    - attention: Flash Attention and Sparse Attention implementations
    - model_utils: Model modification utilities

Usage:
    import turbodiffusion_sycl as tds
    
    # Flash Attention
    flash_attn = tds.FlashAttentionSYCL(head_dim=64, num_heads=12)
    output = flash_attn(q, k, v)
    
    # Sparse Attention
    sparse_attn = tds.SparseAttentionSYCL(head_dim=64, topk=0.2)
    output = sparse_attn(q, k, v)
    
    # Model modification
    tds.replace_attention_sycl(model, attention_type='sparse', topk=0.2)
"""

__version__ = "0.1.0"
__author__ = "TurboDiffusion-SYCL Team"

# Try to import the compiled extension
SYCL_OPS_AVAILABLE = False
try:
    import turbodiffusion_sycl_ops
    SYCL_OPS_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"SYCL operations module not available: {e}")

# Import PyTorch-based wrappers
from .attention import FlashAttentionSYCL, SparseAttentionSYCL
from .model_utils import replace_attention_sycl, replace_norm_sycl

# Try to import existing norm wrappers
try:
    from .norm import FastRMSNormSYCL, FastLayerNormSYCL
    NORM_AVAILABLE = True
except ImportError:
    NORM_AVAILABLE = False
    FastRMSNormSYCL = None
    FastLayerNormSYCL = None

__all__ = [
    # Attention wrappers
    'FlashAttentionSYCL',
    'SparseAttentionSYCL',
    # Norm wrappers (if available)
    'FastRMSNormSYCL',
    'FastLayerNormSYCL',
    # Model utilities
    'replace_attention_sycl',
    'replace_norm_sycl',
    # Availability flags
    'SYCL_OPS_AVAILABLE',
    'NORM_AVAILABLE',
    # Version info
    '__version__',
]


def is_sycl_available():
    """Check if SYCL operations are available."""
    return SYCL_OPS_AVAILABLE


def get_version():
    """Get package version."""
    return __version__
