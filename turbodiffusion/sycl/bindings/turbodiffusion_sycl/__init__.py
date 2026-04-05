"""
TurboDiffusion SYCL - Python Package

This package provides Python bindings for SYCL kernels optimized for
Intel GPUs (specifically B60/Xe2 architecture).

Modules:
    - rmsnorm: RMSNorm normalization
    - layernorm: Layer normalization
    - quantize: FP32 to INT8 quantization
    - gemm: Matrix multiplication

Usage:
    import turbodiffusion_sycl as tds
    
    # Check device
    info = tds.get_device_info()
    print(f"Using device: {info['name']}")
    
    # Run kernels
    tds.rmsnorm(input_array, weight_array, output_array, eps, m, n)
"""

__version__ = "0.1.0"
__author__ = "TurboDiffusion-SYCL Team"

# Try to import the compiled extension
try:
    from .turbodiffusion_sycl import (
        get_device_info,
        rmsnorm,
        layernorm,
        quantize,
        gemm_int8,
    )
    _HAS_NATIVE = True
except ImportError as e:
    _HAS_NATIVE = False
    _IMPORT_ERROR = str(e)
    
    # Provide fallback implementations
    def _fallback_function(*args, **kwargs):
        raise RuntimeError(
            f"SYCL bindings not available. "
            f"Please compile the bindings first. Error: {_IMPORT_ERROR}"
        )
    
    get_device_info = _fallback_function
    rmsnorm = _fallback_function
    layernorm = _fallback_function
    quantize = _fallback_function
    gemm_int8 = _fallback_function


def is_available():
    """Check if SYCL bindings are available."""
    return _HAS_NATIVE


def get_version():
    """Get package version."""
    return __version__


# Convenience wrappers with better Python integration
import numpy as np


def rmsnorm_numpy(input_arr, weight_arr, eps=1e-6):
    """
    RMSNorm with numpy arrays.
    
    Args:
        input_arr: Input array of shape (m, n)
        weight_arr: Weight array of shape (n,)
        eps: Epsilon value
        
    Returns:
        Normalized array of shape (m, n)
    """
    if not _HAS_NATIVE:
        raise RuntimeError("SYCL bindings not available")
    
    m, n = input_arr.shape
    output_arr = np.empty_like(input_arr)
    
    rmsnorm(input_arr, weight_arr, output_arr, eps, m, n)
    
    return output_arr


def layernorm_numpy(input_arr, gamma_arr, beta_arr, eps=1e-5):
    """
    LayerNorm with numpy arrays.
    
    Args:
        input_arr: Input array of shape (m, n)
        gamma_arr: Gamma array of shape (n,)
        beta_arr: Beta array of shape (n,)
        eps: Epsilon value
        
    Returns:
        Normalized array of shape (m, n)
    """
    if not _HAS_NATIVE:
        raise RuntimeError("SYCL bindings not available")
    
    m, n = input_arr.shape
    output_arr = np.empty_like(input_arr)
    
    layernorm(input_arr, gamma_arr, beta_arr, output_arr, eps, m, n)
    
    return output_arr


def quantize_numpy(input_arr, scale_arr):
    """
    Quantize FP32 to INT8 with numpy arrays.
    
    Args:
        input_arr: Input array of shape (m, n)
        scale_arr: Scale array of shape (m,)
        
    Returns:
        Quantized INT8 array of shape (m, n)
    """
    if not _HAS_NATIVE:
        raise RuntimeError("SYCL bindings not available")
    
    m, n = input_arr.shape
    output_arr = np.empty((m, n), dtype=np.int8)
    
    quantize(input_arr, scale_arr, output_arr, m, n)
    
    return output_arr


def gemm_int8_numpy(A_arr, B_arr, M, N, K):
    """
    GEMM with INT8 inputs and FP32 output.
    
    Args:
        A_arr: Input A array of shape (M, K) with dtype int8
        B_arr: Input B array of shape (K, N) with dtype int8
        M: Rows of A and C
        N: Columns of B and C
        K: Inner dimension
        
    Returns:
        Output array of shape (M, N) with dtype float32
    """
    if not _HAS_NATIVE:
        raise RuntimeError("SYCL bindings not available")
    
    C_arr = np.empty((M, N), dtype=np.float32)
    
    gemm_int8(A_arr, B_arr, C_arr, M, N, K)
    
    return C_arr


__all__ = [
    'get_device_info',
    'rmsnorm',
    'layernorm',
    'quantize',
    'gemm_int8',
    'rmsnorm_numpy',
    'layernorm_numpy',
    'quantize_numpy',
    'gemm_int8_numpy',
    'is_available',
    'get_version',
]
