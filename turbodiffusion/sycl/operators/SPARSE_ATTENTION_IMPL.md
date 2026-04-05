# Sparse Attention SYCL Implementation Documentation

## Overview

This document describes the SYCL implementation of Sparse Attention kernels for the TurboDiffusion-SYCL Migration Project. The implementation translates Triton kernels from the TurboDiffusion SLA (Sparse Linear Attention) module to Intel oneAPI DPC++ SYCL.

## Files Created

1. **`sparse_attention_kernels.hpp`** - Header file with kernel declarations
2. **`sparse_attention_sycl.cpp`** - Main implementation with all 4 kernels
3. **`test_sparse_attention.cpp`** - Unit tests for forward and backward kernels

## Kernel Summary

### 1. Forward Pass (`sparse_attn_fwd_kernel`)

**Purpose**: Compute sparse attention output: `O = softmax(Q @ K^T / sqrt(D)) @ V`

**Implementation Details**:
- Uses online softmax algorithm for numerical stability
- Processes one query block per work-group
- Loads Q tile once, iterates over K/V tiles `topk` times based on LUT
- FP32 accumulation for softmax, BF16 for storage
- Configurable block sizes: BLOCK_M (64/128), BLOCK_N (64), D (64/128)

**Key Algorithm**:
```cpp
// Online softmax with log-sum-exp tracking
for each key block from LUT:
    qk = Q @ K^T * scale * LN2_INV
    local_m = max(qk)
    new_m = max(running_m, local_m)
    
    // Update statistics
    alpha = exp2(running_m - new_m)
    o_s = o_s * alpha + exp2(qk - new_m) @ V
    running_l = running_l * alpha + sum(exp2(qk - new_m))
    running_m = new_m
```

### 2. Backward Preprocess (`sparse_attn_bwd_preprocess_kernel`)

**Purpose**: Compute `delta_s = sum(o_s * do_s, dim=-1)`

**Implementation**: Simple element-wise dot product across D dimension

### 3. Backward dQ (`sparse_attn_bwd_dq_kernel`)

**Purpose**: Compute gradient w.r.t. queries: `dQ`

**Algorithm**:
```cpp
for each key block from LUT:
    P = softmax(Q @ K^T)  // Recomputed
    dP = dO @ V^T
    dS = P * (dP - delta)
    dQ += dS @ K
```

### 4. Backward dK/dV (`sparse_attn_bwd_dkdv_kernel`)

**Purpose**: Compute gradients w.r.t. keys and values

**Algorithm**:
```cpp
for each query block:
    if KBID[b,h,m,n] == 1:  // Check if dependency exists
        P^T = softmax(K @ Q^T)
        dV += P^T @ dO
        dP^T = V @ dO^T
        dS^T = P^T * (dP^T - delta)
        dK += dS^T @ Q
```

## Triton-to-SYCL Mapping

| Triton Concept | SYCL Equivalent |
|----------------|-----------------|
| `tl.program_id(0)` | `item.get_group(0)` |
| `tl.program_id(1)` | `item.get_group(1)` |
| `tl.load(ptr, mask=...)` | Conditional loads with bounds checking |
| `tl.dot(a, b)` | Manual dot product (can use XMX via joint_matrix) |
| `tl.max(x, axis)` | `sycl::max()` in loops |
| `tl.exp2(x)` | `sycl::exp2(x)` |
| `tl.log2(x)` | `sycl::log2(x)` |
| Grid `(M, N)` | `sycl::nd_range<2>(range<2>(M, N), range<2>(1, 1))` |

## Memory Layout

**Tensors** (all row-major):
- Q, O: `[B, H, M, D]`
- K, V: `[B, H, N, D]`
- LUT: `[B, H, M_BLOCKS, topk]` - int32 indices
- KBID: `[B, H, M_BLOCKS, N_BLOCKS]` - int8 binary mask
- LSE, Delta: `[B, H, M]` - float32

## Performance Optimizations

1. **Tiling Strategy**:
   - BLOCK_M: 64 or 128 (query tokens per block)
   - BLOCK_N: 64 (key/value tokens per block)
   - D: 64 or 128 (head dimension)

2. **Online Softmax**: Reduces memory bandwidth and improves numerical stability

3. **Loop Unrolling**: `#pragma unroll` for small fixed-size loops

4. **Future XMX Optimization**:
   ```cpp
   // Can use joint_matrix for matrix multiplies
   using tile_a = matrix::joint_matrix<sg, bfloat16, use::a, BLOCK_M, BLOCK_N>;
   using tile_b = matrix::joint_matrix<sg, bfloat16, use::b, BLOCK_N, D>;
   ```

## Assumptions and Limitations

1. **Sequence Length**: M and N should be divisible by BLOCK_M and BLOCK_N respectively for best performance
2. **Block Configuration**: Only BLOCK_M=64/128, BLOCK_N=64, D=64/128 supported
3. **Top-k**: Number of key blocks per query block (topk) is compile-time for templates
4. **Precision**: BF16 for storage, FP32 for accumulation
5. **Work-group Size**: Currently 1 work-group per block (can be optimized)

## Testing Strategy

The test file includes:
1. Forward pass test with PyTorch-like reference comparison
2. Backward preprocess test with exact verification
3. Future: dQ and dK/dV gradient checking (numerical vs analytical)

## Compilation

```bash
# Using Intel oneAPI compiler
icpx -fsycl -O3 sparse_attention_sycl.cpp test_sparse_attention.cpp -o test_sparse_attention

# With XMX optimization (Intel Arc/Battlemage)
icpx -fsycl -O3 -march=alderlake sparse_attention_sycl.cpp test_sparse_attention.cpp -o test_sparse_attention
```

## Integration Notes

To use with PyTorch:
1. Create PyTorch bindings via pybind11
2. Manage tensor lifetime and device placement
3. Implement LUT generation in Python/PyTorch
4. Handle BF16 tensors via PyTorch's bfloat16 dtype

## References

1. Original Triton kernels: `/home/intel/tianfeng/opencode_bench/TurboDiffusion/turbodiffusion/SLA/kernel.py`
2. Technical specification: `/home/intel/tianfeng/opencode_bench/turbodiffusion-sycl/research/sparse_attention_spec.md`
3. FlashAttention paper: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention"
4. Online Softmax: Milakov and Gimelshein, "Online normalizer calculation for softmax"

## Author

Agent 2 - TurboDiffusion-SYCL Migration Project
Date: April 2025
