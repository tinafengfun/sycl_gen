"""
TurboDiffusion SYCL Attention Wrappers

Provides PyTorch-compatible nn.Module implementations for:
    - Flash Attention (via SYCL kernels)
    - Sparse Linear Attention (via SYCL kernels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Try to import SYCL operations
try:
    import turbodiffusion_sycl_ops as sycl_ops
    SYCL_OPS_AVAILABLE = True
except ImportError:
    SYCL_OPS_AVAILABLE = False
    sycl_ops = None


def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64):
    """
    Compute block-wise sparsity pattern using mean pooling.
    
    Args:
        q: Query tensor [B, H, L, D]
        k: Key tensor [B, H, L, D]
        topk_ratio: Ratio of top blocks to keep
        BLKQ: Query block size
        BLKK: Key block size
        
    Returns:
        sparse_map: Binary block map [B, H, M_BLOCKS, N_BLOCKS]
        lut: LUT of selected blocks [B, H, M_BLOCKS, topk]
        topk: Number of top blocks selected
    """
    B, H, L, D = q.shape
    
    # Mean pooling for queries
    M_BLOCKS = (L + BLKQ - 1) // BLKQ
    q_blocks = q.reshape(B, H, M_BLOCKS, BLKQ, D).mean(dim=3)  # [B, H, M_BLOCKS, D]
    
    # Mean pooling for keys
    N_BLOCKS = (L + BLKK - 1) // BLKK
    k_blocks = k.reshape(B, H, N_BLOCKS, BLKK, D).mean(dim=3)  # [B, H, N_BLOCKS, D]
    
    # Compute block-wise similarity
    block_scores = torch.matmul(q_blocks, k_blocks.transpose(-2, -1))  # [B, H, M_BLOCKS, N_BLOCKS]
    
    # Get top-k blocks
    K = block_scores.shape[-1]
    topk = min(K, max(1, int(topk_ratio * K)))
    
    # Get indices of top-k blocks
    _, lut = torch.topk(block_scores, topk, dim=-1, sorted=False)  # [B, H, M_BLOCKS, topk]
    
    # Create binary sparse map
    sparse_map = torch.zeros_like(block_scores, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    
    return sparse_map, lut, topk


class FlashAttentionSYCL(nn.Module):
    """
    Flash Attention using SYCL kernels for Intel GPUs.
    
    Implements Flash Attention v2 algorithm optimized for Xe GPU architecture.
    Supports GQA (Grouped Query Attention) with different number of heads for Q vs K/V.
    
    Args:
        head_dim: Dimension of each attention head
        num_heads: Number of attention heads for Q
        num_heads_kv: Number of attention heads for K/V (defaults to num_heads for standard attention)
        softmax_scale: Scale factor for softmax (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking (default: True for autoregressive models)
    """
    
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_heads_kv: Optional[int] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = True
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        self.softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
        self.causal = causal
        
        if not SYCL_OPS_AVAILABLE:
            raise RuntimeError(
                "SYCL operations not available. "
                "Please compile turbodiffusion_sycl_ops first."
            )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for Flash Attention.
        
        Args:
            query: Query tensor [B, H_q, S_q, D] or [B, S_q, H_q, D]
            key: Key tensor [B, H_kv, S_kv, D] or [B, S_kv, H_kv, D]
            value: Value tensor [B, H_kv, S_kv, D] or [B, S_kv, H_kv, D]
            attn_mask: Optional attention mask (not implemented yet)
            
        Returns:
            output: Attention output [B, H_q, S_q, D]
        """
        # Detect input format and convert to [B, H, S, D]
        query, key, value = self._ensure_format(query, key, value)
        
        # Validate inputs
        self._validate_inputs(query, key, value)
        
        # Convert to BF16 for SYCL kernel
        orig_dtype = query.dtype
        if query.dtype != torch.bfloat16:
            query = query.to(torch.bfloat16)
            key = key.to(torch.bfloat16)
            value = value.to(torch.bfloat16)
        
        # Ensure tensors are on XPU
        if query.device.type != 'xpu':
            query = query.to('xpu')
            key = key.to('xpu')
            value = value.to('xpu')
        
        # Call SYCL kernel
        output = sycl_ops.flash_attention_forward(
            query, key, value,
            attn_mask,
            self.softmax_scale
        )
        
        # Convert back to original dtype
        if orig_dtype != torch.bfloat16:
            output = output.to(orig_dtype)
        
        return output
    
    def _ensure_format(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ensure tensors are in [B, H, S, D] format."""
        # Check if format is [B, S, H, D] (common in some implementations)
        if query.dim() == 4 and query.size(2) == self.num_heads:
            # Likely [B, S, H, D] format, transpose to [B, H, S, D]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        
        return query, key, value
    
    def _validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ):
        """Validate input tensor shapes and types."""
        assert query.dim() == 4, f"Query must be 4D, got {query.dim()}D"
        assert key.dim() == 4, f"Key must be 4D, got {key.dim()}D"
        assert value.dim() == 4, f"Value must be 4D, got {value.dim()}D"
        
        B, H_q, S_q, D_q = query.shape
        B_k, H_kv, S_kv, D_k = key.shape
        B_v, H_kv_v, S_kv_v, D_v = value.shape
        
        assert B == B_k == B_v, f"Batch sizes must match: {B}, {B_k}, {B_v}"
        assert H_q == self.num_heads, f"Query heads {H_q} != {self.num_heads}"
        assert H_kv == H_kv_v == self.num_heads_kv, f"KV heads mismatch"
        assert S_kv == S_kv_v, f"Key/Value sequence lengths must match"
        assert D_q == D_k == self.head_dim, f"Head dimensions must match {self.head_dim}"
        assert self.num_heads % self.num_heads_kv == 0, "num_heads must be divisible by num_heads_kv"


class SparseAttentionSYCL(nn.Module):
    """
    Sparse Linear Attention using SYCL kernels for Intel GPUs.
    
    Combines sparse attention with linear attention for efficient computation.
    Uses block-wise sparsity pattern determined by top-k selection.
    
    Args:
        head_dim: Dimension of each attention head
        topk: Ratio of blocks to select for sparse attention (0.0 to 1.0)
        BLKQ: Block size for queries (default: 64)
        BLKK: Block size for keys (default: 64)
        use_bf16: Whether to use bfloat16 computation (default: True)
        feature_map: Feature map for linear attention ('softmax', 'elu', 'relu')
        tie_feature_map_qk: Whether to use same feature map for Q and K
    """
    
    def __init__(
        self,
        head_dim: int,
        topk: float = 0.2,
        BLKQ: int = 64,
        BLKK: int = 64,
        use_bf16: bool = True,
        feature_map: str = 'softmax',
        tie_feature_map_qk: bool = True
    ):
        super().__init__()
        self.head_dim = head_dim
        self.topk = topk
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.use_bf16 = use_bf16
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        
        # Linear projection for linear attention component
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.float32)
        
        # Setup feature maps
        if feature_map == 'elu':
            def elu_feature_map(x):
                return F.elu(x) + 1
            self.feature_map_q = elu_feature_map
            self.feature_map_k = elu_feature_map
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == 'softmax':
            def softmax_feature_map(x):
                return F.softmax(x, dim=-1)
            self.feature_map_q = softmax_feature_map
            self.feature_map_k = softmax_feature_map
        else:
            raise ValueError(f"Unknown feature_map: {feature_map}")
        
        if tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q
        
        # Initialize weights
        self._init_weights()
        
        if not SYCL_OPS_AVAILABLE:
            raise RuntimeError(
                "SYCL operations not available. "
                "Please compile turbodiffusion_sycl_ops first."
            )
    
    def _init_weights(self):
        """Initialize linear projection weights to zero."""
        with torch.no_grad():
            nn.init.zeros_(self.proj_l.weight)
            nn.init.zeros_(self.proj_l.bias)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_sparsity: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for Sparse Attention.
        
        Args:
            q: Query tensor [B, H, L, D]
            k: Key tensor [B, H, L, D]
            v: Value tensor [B, H, L, D]
            return_sparsity: Whether to return actual sparsity ratio
            
        Returns:
            output: Attention output [B, H, L, D]
            sparsity: (optional) Actual sparsity ratio
        """
        dtype = q.dtype
        device = q.device
        
        # Transpose to [B, L, H, D] for processing
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        # Get block-wise sparsity pattern
        sparse_map, lut, real_topk = get_block_map(
            q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK
        )
        
        # Convert to BF16 for SYCL kernel
        q_bf16 = q.to(self.dtype)
        k_bf16 = k.to(self.dtype)
        v_bf16 = v.to(self.dtype)
        
        # Call SYCL sparse attention kernel
        o_s = self._sparse_attention_forward(q_bf16, k_bf16, v_bf16, lut, real_topk)
        
        # Compute linear attention component
        q_feat = self.feature_map_q(q_bf16).contiguous().to(self.dtype)
        k_feat = self.feature_map_k(k_bf16).contiguous().to(self.dtype)
        o_l = self._compute_linear_attention(q_feat, k_feat, v_bf16)
        
        # Apply linear projection with autocast
        with torch.amp.autocast(device.type, dtype=self.dtype):
            o_l = self.proj_l(o_l)
        
        # Combine and transpose back to [B, H, L, D]
        output = (o_s + o_l).to(dtype).transpose(1, 2)
        
        if return_sparsity:
            sparsity_ratio = real_topk / sparse_map.shape[-1] if sparse_map.shape[-1] > 0 else 0.0
            return output, sparsity_ratio
        else:
            return output
    
    def _sparse_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lut: torch.Tensor,
        topk: int
    ) -> torch.Tensor:
        """
        Call SYCL sparse attention kernel.
        
        Args:
            q: Query tensor [B, L, H, D] - BF16
            k: Key tensor [B, L, H, D] - BF16
            v: Value tensor [B, L, H, D] - BF16
            lut: LUT tensor [B, H, M_BLOCKS, topk]
            topk: Number of top blocks
            
        Returns:
            o_s: Sparse attention output [B, L, H, D]
        """
        B, L, H, D = q.shape
        
        # Reshape for kernel: [B, L, H, D] -> [B*H, L, D]
        q_flat = q.reshape(B * H, L, D)
        k_flat = k.reshape(B * H, L, D)
        v_flat = v.reshape(B * H, L, D)
        
        # Reshape LUT: [B, H, M_BLOCKS, topk] -> [B*H, M_BLOCKS, topk]
        M_BLOCKS = (L + self.BLKQ - 1) // self.BLKQ
        lut_flat = lut.reshape(B * H, M_BLOCKS, topk)
        
        # Check if sparse_attention_forward is available in sycl_ops
        if hasattr(sycl_ops, 'sparse_attention_forward'):
            # Ensure tensors are on XPU
            if q_flat.device.type != 'xpu':
                q_flat = q_flat.to('xpu')
                k_flat = k_flat.to('xpu')
                v_flat = v_flat.to('xpu')
                lut_flat = lut_flat.to('xpu')
            
            o_s_flat = sycl_ops.sparse_attention_forward(
                q_flat, k_flat, v_flat,
                lut_flat, topk,
                self.BLKQ, self.BLKK
            )
        else:
            # Fallback to PyTorch implementation for testing
            o_s_flat = self._sparse_attention_pytorch(q_flat, k_flat, v_flat, lut_flat, topk)
        
        # Reshape back: [B*H, L, D] -> [B, L, H, D]
        o_s = o_s_flat.reshape(B, L, H, D)
        
        return o_s
    
    def _sparse_attention_pytorch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lut: torch.Tensor,
        topk: int
    ) -> torch.Tensor:
        """
        PyTorch fallback for sparse attention (for testing without SYCL).
        
        This is a simplified implementation for testing purposes.
        The full SYCL kernel provides better performance.
        """
        BH, L, D = q.shape
        scale = 1.0 / math.sqrt(D)
        
        # Compute full attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [BH, L, L]
        
        # Apply sparsity mask based on LUT
        M_BLOCKS = (L + self.BLKQ - 1) // self.BLKQ
        N_BLOCKS = (L + self.BLKK - 1) // self.BLKK
        
        # Create attention mask from LUT
        attn_mask = torch.zeros(BH, M_BLOCKS, N_BLOCKS, device=q.device, dtype=torch.bool)
        for b in range(BH):
            for m in range(M_BLOCKS):
                for t in range(min(topk, lut.shape[-1])):
                    n = lut[b, m, t].item()
                    if 0 <= n < N_BLOCKS:
                        attn_mask[b, m, n] = True
        
        # Expand mask to full attention size
        full_mask = torch.zeros(BH, L, L, device=q.device, dtype=torch.bool)
        for m in range(M_BLOCKS):
            m_start = m * self.BLKQ
            m_end = min((m + 1) * self.BLKQ, L)
            for n in range(N_BLOCKS):
                if attn_mask[:, m, n].any():
                    n_start = n * self.BLKK
                    n_end = min((n + 1) * self.BLKK, L)
                    full_mask[:, m_start:m_end, n_start:n_end] = True
        
        # Apply mask and compute softmax
        scores = scores.masked_fill(~full_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        return output
    
    def _compute_linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute linear attention component.
        
        Args:
            q: Feature-mapped queries [B, L, H, D]
            k: Feature-mapped keys [B, L, H, D]
            v: Values [B, L, H, D]
            
        Returns:
            o_l: Linear attention output [B, L, H, D]
        """
        # Compute KV sum: k^T @ v
        kv_sum = torch.matmul(k.transpose(-2, -1), v)  # [B, H, D, D]
        
        # Compute k sum
        k_sum = k.sum(dim=-2, keepdim=True)  # [B, H, 1, D]
        
        # Compute linear attention output
        numerator = torch.matmul(q, kv_sum)  # [B, H, L, D]
        denominator = (q * k_sum).sum(dim=-1, keepdim=True) + 1e-5  # [B, H, L, 1]
        o_l = numerator / denominator
        
        return o_l
