"""
TurboDiffusion SYCL Model Utilities

Provides functions to modify and replace model components with SYCL versions.
"""

import torch
import torch.nn as nn
from typing import Optional, Union

# Import SYCL wrappers
from .attention import FlashAttentionSYCL, SparseAttentionSYCL

# Try to import existing norm wrappers
try:
    from .norm import FastRMSNormSYCL, FastLayerNormSYCL
    NORM_AVAILABLE = True
except ImportError:
    NORM_AVAILABLE = False
    FastRMSNormSYCL = None
    FastLayerNormSYCL = None


def replace_attention_sycl(
    model: nn.Module,
    attention_type: str = 'sparse',
    topk: float = 0.2,
    BLKQ: int = 64,
    BLKK: int = 64,
    use_bf16: bool = True,
    verbose: bool = True
) -> nn.Module:
    """
    Replace attention modules in Wan model with SYCL versions.
    
    This function mirrors the behavior of modify_model.replace_attention() but
    uses SYCL-based attention implementations instead of CUDA-based ones.
    
    Args:
        model: WanModel instance (WanModel2pt1 or WanModel2pt2)
        attention_type: Type of attention to use ('sparse' or 'flash')
        topk: Top-k ratio for sparse attention (0.0 to 1.0)
        BLKQ: Block size for queries in sparse attention
        BLKK: Block size for keys in sparse attention
        use_bf16: Whether to use bfloat16 for computation
        verbose: Whether to print replacement information
        
    Returns:
        model: Modified model with SYCL attention
        
    Example:
        >>> from turbodiffusion_sycl import replace_attention_sycl
        >>> model = replace_attention_sycl(model, attention_type='sparse', topk=0.2)
    """
    # Try to import model classes
    try:
        from rcm.networks.wan2pt1 import WanSelfAttention as WanSelfAttention2pt1
        from rcm.networks.wan2pt2 import WanSelfAttention as WanSelfAttention2pt2
        WAN_ATTENTION_CLASSES = (WanSelfAttention2pt1, WanSelfAttention2pt2)
    except ImportError:
        # Fallback: try to find attention modules dynamically
        WAN_ATTENTION_CLASSES = None
        if verbose:
            print("Warning: Could not import WanSelfAttention classes. "
                  "Will attempt dynamic detection.")
    
    replaced_count = 0
    
    for name, module in model.named_modules():
        # Check if this is an attention module we should replace
        should_replace = False
        
        if WAN_ATTENTION_CLASSES:
            should_replace = isinstance(module, WAN_ATTENTION_CLASSES)
        else:
            # Dynamic detection: look for modules with attn_op.local_attn
            should_replace = hasattr(module, 'attn_op') and hasattr(module.attn_op, 'local_attn')
        
        if should_replace:
            # Get head dimension from the module
            if hasattr(module, 'dim') and hasattr(module, 'num_heads'):
                head_dim = module.dim // module.num_heads
            else:
                # Try to infer from existing local_attn
                if hasattr(module, 'attn_op') and hasattr(module.attn_op, 'local_attn'):
                    existing = module.attn_op.local_attn
                    if hasattr(existing, 'head_dim'):
                        head_dim = existing.head_dim
                    else:
                        head_dim = 64  # Default fallback
                else:
                    head_dim = 64  # Default fallback
            
            # Create SYCL attention module
            if attention_type == 'sparse':
                new_attn = SparseAttentionSYCL(
                    head_dim=head_dim,
                    topk=topk,
                    BLKQ=BLKQ,
                    BLKK=BLKK,
                    use_bf16=use_bf16
                )
            elif attention_type == 'flash':
                # Try to get num_heads and num_heads_kv from module
                num_heads = getattr(module, 'num_heads', 12)
                num_heads_kv = getattr(module, 'num_heads_kv', num_heads)
                
                new_attn = FlashAttentionSYCL(
                    head_dim=head_dim,
                    num_heads=num_heads,
                    num_heads_kv=num_heads_kv
                )
            else:
                raise ValueError(f"Unknown attention_type: {attention_type}. "
                               "Use 'sparse' or 'flash'.")
            
            # Replace the local_attn module
            if hasattr(module, 'attn_op'):
                module.attn_op.local_attn = new_attn
                replaced_count += 1
                
                if verbose:
                    print(f"Replaced attention in {name}: {attention_type} "
                          f"(head_dim={head_dim}, topk={topk})")
    
    if verbose:
        print(f"Total attention modules replaced: {replaced_count}")
    
    return model


def replace_norm_sycl(
    model: nn.Module,
    replace_rmsnorm: bool = True,
    replace_layernorm: bool = True,
    verbose: bool = True
) -> nn.Module:
    """
    Replace LayerNorm/RMSNorm modules with SYCL versions.
    
    Args:
        model: Model instance
        replace_rmsnorm: Whether to replace RMSNorm modules
        replace_layernorm: Whether to replace LayerNorm modules
        verbose: Whether to print replacement information
        
    Returns:
        model: Modified model with SYCL normalization
    """
    if not NORM_AVAILABLE:
        raise RuntimeError(
            "SYCL norm wrappers not available. "
            "Please check that norm.py exists in turbodiffusion_sycl package."
        )
    
    # Try to import model-specific norm classes
    try:
        from rcm.networks.wan2pt1 import WanLayerNorm as WanLayerNorm2pt1, WanRMSNorm as WanRMSNorm2pt1
        from rcm.networks.wan2pt2 import WanLayerNorm as WanLayerNorm2pt2, WanRMSNorm as WanRMSNorm2pt2
        
        NORM_CLASSES = []
        if replace_rmsnorm:
            NORM_CLASSES.extend([WanRMSNorm2pt1, WanRMSNorm2pt2])
        if replace_layernorm:
            NORM_CLASSES.extend([WanLayerNorm2pt1, WanLayerNorm2pt2])
    except ImportError:
        # Fallback to standard PyTorch norm classes
        NORM_CLASSES = []
        if replace_rmsnorm:
            NORM_CLASSES.append(nn.RMSNorm if hasattr(nn, 'RMSNorm') else None)
        if replace_layernorm:
            NORM_CLASSES.append(nn.LayerNorm)
        NORM_CLASSES = [c for c in NORM_CLASSES if c is not None]
        
        if verbose and not NORM_CLASSES:
            print("Warning: No norm classes to replace.")
    
    replaced_count = 0
    
    for name, module in model.named_modules():
        if NORM_CLASSES and isinstance(module, tuple(NORM_CLASSES)):
            # Determine which SYCL norm to use
            is_rms = 'RMS' in type(module).__name__ or (
                hasattr(module, 'elementwise_affine') and not hasattr(module, 'bias')
            )
            
            if is_rms and replace_rmsnorm:
                # Get normalized shape and eps
                normalized_shape = module.normalized_shape
                eps = getattr(module, 'eps', 1e-6)
                
                # Create SYCL RMSNorm
                new_norm = FastRMSNormSYCL(normalized_shape, eps=eps)
                
                # Copy weights if available
                if hasattr(module, 'weight'):
                    with torch.no_grad():
                        new_norm.weight.copy_(module.weight)
                
            elif not is_rms and replace_layernorm:
                # Get normalized shape, eps, and elementwise_affine
                normalized_shape = module.normalized_shape
                eps = getattr(module, 'eps', 1e-5)
                elementwise_affine = getattr(module, 'elementwise_affine', True)
                
                # Create SYCL LayerNorm
                new_norm = FastLayerNormSYCL(
                    normalized_shape,
                    eps=eps,
                    elementwise_affine=elementwise_affine
                )
                
                # Copy weights if available
                if elementwise_affine:
                    if hasattr(module, 'weight'):
                        with torch.no_grad():
                            new_norm.weight.copy_(module.weight)
                    if hasattr(module, 'bias'):
                        with torch.no_grad():
                            new_norm.bias.copy_(module.bias)
            else:
                continue
            
            # Replace the module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            
            setattr(parent, child_name, new_norm)
            replaced_count += 1
            
            if verbose:
                print(f"Replaced norm in {name}: {type(new_norm).__name__}")
    
    if verbose:
        print(f"Total norm modules replaced: {replaced_count}")
    
    return model


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about attention and norm modules in a model.
    
    Args:
        model: Model to analyze
        
    Returns:
        info: Dictionary with module counts and types
    """
    info = {
        'attention_modules': [],
        'norm_modules': [],
        'total_modules': 0
    }
    
    for name, module in model.named_modules():
        info['total_modules'] += 1
        
        # Check for attention modules
        if hasattr(module, 'attn_op') and hasattr(module.attn_op, 'local_attn'):
            attn_type = type(module.attn_op.local_attn).__name__
            info['attention_modules'].append({
                'name': name,
                'type': attn_type
            })
        
        # Check for norm modules
        if isinstance(module, (nn.LayerNorm,)) or (
            hasattr(nn, 'RMSNorm') and isinstance(module, nn.RMSNorm)
        ):
            info['norm_modules'].append({
                'name': name,
                'type': type(module).__name__
            })
    
    return info
