"""
Layer Adapters for SYCL Kernel Integration

Provides PyTorch-compatible layer adapters that call SYCL kernels.
These adapters are used by the dispatcher to replace CUDA layers.

Author: TurboDiffusion-SYCL Migration Team
Date: 2026-04-01
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import warnings


class SyclLayerNorm(nn.Module):
    """
    SYCL implementation of LayerNorm.
    
    This is a placeholder adapter that will call the SYCL LayerNorm kernel
    once the Python bindings are available.
    
    Args:
        normalized_shape: Shape of the input to normalize
        eps: Epsilon value for numerical stability
        elementwise_affine: Whether to use learnable parameters
    """
    
    def __init__(
        self, 
        normalized_shape: int or Tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LayerNorm using SYCL kernel.
        
        Args:
            x: Input tensor of shape (..., normalized_shape)
            
        Returns:
            Normalized tensor
        """
        # TODO: Replace with actual SYCL kernel call once bindings are ready
        # For now, use PyTorch implementation as fallback
        return torch.nn.functional.layer_norm(
            x, 
            (self.normalized_shape,), 
            self.weight, 
            self.bias, 
            self.eps
        )
    
    @classmethod
    def from_layernorm(cls, layer: nn.LayerNorm) -> 'SyclLayerNorm':
        """
        Create SyclLayerNorm from an existing PyTorch LayerNorm.
        
        Args:
            layer: PyTorch LayerNorm instance
            
        Returns:
            SyclLayerNorm with copied parameters
        """
        normalized_shape = layer.normalized_shape
        eps = layer.eps
        elementwise_affine = layer.elementwise_affine
        
        sycl_layer = cls(normalized_shape, eps, elementwise_affine)
        
        if elementwise_affine:
            sycl_layer.weight.data = layer.weight.data.clone()
            sycl_layer.bias.data = layer.bias.data.clone()
        
        return sycl_layer


class SyclRMSNorm(nn.Module):
    """
    SYCL implementation of RMSNorm.
    
    RMSNorm normalizes inputs by dividing by the root mean square.
    This is commonly used in transformer models.
    
    Args:
        dim: Dimension to normalize over
        eps: Epsilon value for numerical stability
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm using SYCL kernel.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Normalized tensor
        """
        # TODO: Replace with actual SYCL kernel call once bindings are ready
        # For now, use PyTorch implementation
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
    @classmethod
    def from_rmsnorm(cls, module: nn.Module) -> 'SyclRMSNorm':
        """
        Create SyclRMSNorm from an existing RMSNorm module.
        
        Args:
            module: RMSNorm module (e.g., WanRMSNorm)
            
        Returns:
            SyclRMSNorm with copied parameters
        """
        # Extract dimensions and eps from the original module
        dim = module.weight.shape[0]
        eps = getattr(module, 'eps', 1e-6)
        
        sycl_module = cls(dim, eps)
        sycl_module.weight.data = module.weight.data.clone()
        
        return sycl_module


class SyclInt8Linear(nn.Module):
    """
    SYCL implementation of quantized INT8 Linear layer.
    
    Uses INT8 weights for memory efficiency, dequantizes on-the-fly
    during computation.
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to use bias
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # INT8 weight storage
        self.register_buffer(
            'int8_weight',
            torch.empty((out_features, in_features), dtype=torch.int8)
        )
        
        # Scale factors for dequantization
        self.register_buffer(
            'scale',
            torch.empty((1,), dtype=torch.float32)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation with INT8 weights.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Dequantize weights
        weight = self.int8_weight.float() * self.scale
        
        # TODO: Replace with SYCL GEMM kernel
        output = torch.nn.functional.linear(x, weight, self.bias)
        
        return output
    
    @classmethod
    def from_linear(
        cls, 
        linear: nn.Linear, 
        quantize: bool = True
    ) -> 'SyclInt8Linear':
        """
        Create SyclInt8Linear from an existing Linear layer.
        
        Args:
            linear: PyTorch Linear layer
            quantize: Whether to quantize weights to INT8
            
        Returns:
            SyclInt8Linear with (optionally) quantized weights
        """
        sycl_linear = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None
        )
        
        if quantize:
            # Quantize weights to INT8
            weight = linear.weight.data
            scale = weight.abs().max() / 127.0
            int8_weight = (weight / scale).round().clamp(-128, 127).to(torch.int8)
            
            sycl_linear.int8_weight.data = int8_weight
            sycl_linear.scale.data = scale.unsqueeze(0)
        else:
            # Store as INT8 without quantization (for testing)
            sycl_linear.int8_weight.data = linear.weight.data.to(torch.int8)
            sycl_linear.scale.data = torch.ones(1, dtype=torch.float32)
        
        if linear.bias is not None:
            sycl_linear.bias.data = linear.bias.data.clone()
        
        return sycl_linear


class LayerAdapterFactory:
    """
    Factory for creating appropriate layer adapters.
    
    Provides a unified interface for creating SYCL-compatible layers
    from various PyTorch layer types.
    """
    
    @staticmethod
    def create_adapter(module: nn.Module, layer_type: str) -> nn.Module:
        """
        Create a SYCL adapter for a given module.
        
        Args:
            module: PyTorch module to adapt
            layer_type: Type of layer ('layernorm', 'rmsnorm', 'linear')
            
        Returns:
            SYCL-compatible adapter module
            
        Raises:
            ValueError: If layer_type is not supported
        """
        if layer_type == 'layernorm':
            if isinstance(module, nn.LayerNorm):
                return SyclLayerNorm.from_layernorm(module)
            else:
                raise ValueError(f"Expected LayerNorm, got {type(module)}")
                
        elif layer_type == 'rmsnorm':
            # RMSNorm might be a custom class, check for weight attribute
            if hasattr(module, 'weight') and hasattr(module, 'eps'):
                return SyclRMSNorm.from_rmsnorm(module)
            else:
                raise ValueError(f"Module {type(module)} doesn't look like RMSNorm")
                
        elif layer_type == 'linear':
            if isinstance(module, nn.Linear):
                return SyclInt8Linear.from_linear(module, quantize=False)
            else:
                raise ValueError(f"Expected Linear, got {type(module)}")
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
    @staticmethod
    def can_adapt(module: nn.Module) -> Tuple[bool, Optional[str]]:
        """
        Check if a module can be adapted to SYCL.
        
        Args:
            module: PyTorch module to check
            
        Returns:
            Tuple of (can_adapt, layer_type)
        """
        if isinstance(module, nn.LayerNorm):
            return True, 'layernorm'
        elif hasattr(module, '_norm') or 'RMSNorm' in type(module).__name__:
            return True, 'rmsnorm'
        elif isinstance(module, nn.Linear):
            return True, 'linear'
        else:
            return False, None


def adapt_model_layers(model: nn.Module, verbose: bool = False) -> nn.Module:
    """
    Adapt all compatible layers in a model to use SYCL implementations.
    
    This is an alternative to using hooks - it directly replaces the
    layers in the model.
    
    Args:
        model: PyTorch model to adapt
        verbose: Whether to print adaptation progress
        
    Returns:
        Model with adapted layers
        
    Warning:
        This modifies the model in-place. Use with caution.
    """
    factory = LayerAdapterFactory()
    adapted_count = 0
    
    for name, module in model.named_modules():
        can_adapt, layer_type = factory.can_adapt(module)
        
        if can_adapt:
            try:
                adapter = factory.create_adapter(module, layer_type)
                
                # Navigate to parent and replace
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                
                setattr(parent, parts[-1], adapter)
                adapted_count += 1
                
                if verbose:
                    print(f"[Adapt] {name} -> Sycl{layer_type.capitalize()}")
                    
            except Exception as e:
                if verbose:
                    print(f"[Error] Failed to adapt {name}: {e}")
    
    if verbose:
        print(f"\n[Summary] Adapted {adapted_count} layers to SYCL")
    
    return model
