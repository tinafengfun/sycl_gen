"""
TurboDiffusion SYCL Hooks Package

Provides zero-intrusive SYCL kernel replacement for Wan2.1 models.

Usage:
    from hooks import SyclDispatcher, LayerRegistry
    
    # Create dispatcher
    dispatcher = SyclDispatcher(model, test_mode=True)
    
    # Register hooks on norm layers
    dispatcher.register_norm_hooks()
    
    # Enable SYCL for specific layer
    dispatcher.enable('blocks.0.norm1')
    
    # Run inference
    output = model(input)
"""

from .dispatcher import SyclDispatcher, LayerRegistry
from .validation import ValidationManager, ValidationPolicy, ValidationStatus

__all__ = [
    'SyclDispatcher', 
    'LayerRegistry',
    'ValidationManager',
    'ValidationPolicy', 
    'ValidationStatus'
]
__version__ = '0.1.0'
