# Sparse Attention Triton-to-SYCL Migration Technical Specification

## Document Information
- **Project**: TurboDiffusion-SYCL Migration
- **Component**: Sparse Linear Attention (SLA) Kernels
- **Source**: TurboDiffusion/turbodiffusion/SLA/kernel.py
- **Target**: Intel oneAPI DPC++ SYCL
- **Date**: April 2025
- **Version**: 1.0

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [LUT Mechanism Deep Dive](#lut-mechanism-deep-dive)
4. [Tiling Strategy Analysis](#tiling-strategy-analysis)
5. [Parallelism Structure](#parallelism-structure)
6. [Kernel-by-Kernel Analysis](#kernel-by-kernel-analysis)
7. [Triton-to-SYCL Mapping](#triton-to-sycl-mapping)
8. [Memory Layout Specifications](#memory-layout-specifications)
9. [Algorithm Flow Pseudocode](#algorithm-flow-pseudocode)
10. [Implementation Recommendations](#implementation-recommendations)

---

## Executive Summary

This document provides a comprehensive technical specification for migrating the Sparse Linear Attention (SLA) Triton implementation to Intel oneAPI DPC++ SYCL. The SLA mechanism combines sparse attention with linear attention to achieve efficient attention computation in diffusion transformers.

### Key Components
- **4 Triton kernels** requiring SYCL conversion
- **BF16/FP16 compute precision** with FP32 accumulation
- **Block-sparse pattern** using Look-Up Tables (LUT)
- **Online softmax** with numerical stability
- **Forward and backward pass** support for training

### Critical Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| BLOCK_M | 64 or 128 | Query tile size (rows) |
| BLOCK_N | 64 | Key/Value tile size (columns) |
| D | 64 or 128 | Head dimension |
| topk | variable | Number of key blocks per query block |
| qk_scale | 1/sqrt(D) | Attention score scaling |

---

## Architecture Overview

### High-Level Algorithm

The Sparse Attention follows the standard attention formula with sparsity:

```
Attention(Q, K, V) = softmax(Q x K^T / sqrt(d_k)) x V
```

With sparsity, each query block only attends to a subset of key blocks specified by the LUT.

### Two-Stage Attention

The SLA module combines two attention types:

1. **Sparse Softmax Attention (o_s)**: Computed via Triton kernels
   - Uses block-sparse pattern from LUT
   - Full softmax normalization
   - Captures local/explicit dependencies

2. **Linear Attention (o_l)**: Computed via PyTorch operations
   - Feature map transformation (elu, relu, or softmax)
   - Kernel trick: phi(Q) x (phi(K)^T x V)
   - Captures global/implicit dependencies

Final output: `o = o_s + proj_l(o_l)`

---

## LUT Mechanism Deep Dive

### LUT Structure and Purpose

The Look-Up Table (LUT) encodes the sparse attention pattern by specifying which key blocks each query block should attend to.

#### LUT Generation (from utils.py)

```python
def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64):
    # 1. Apply smooth-k technique (subtract mean)
    arg_k = k - torch.mean(k, dim=-2, keepdim=True)
    
    # 2. Mean pool queries and keys to block level
    pooled_qblocks = mean_pool(q, BLKQ)  # Shape: (B, H, L/BLKQ, D)
    pooled_kblocks = mean_pool(arg_k, BLKK)  # Shape: (B, H, L/BLKK, D)
    
    # 3. Compute block-level attention scores
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)
    # Shape: (B, H, M_BLOCKS, N_BLOCKS)
    
    # 4. Select top-k key blocks for each query block
    K = pooled_score.shape[-1]  # Total key blocks (N_BLOCKS)
    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices
    # Shape: (B, H, M_BLOCKS, topk)
    
    # 5. Create binary sparse map
    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    
    return sparse_map, lut, topk
```

### LUT Format Documentation

#### Memory Layout
```
LUT Shape: (B, H, M_BLOCKS, topk)
- B: Batch size
- H: Number of heads
- M_BLOCKS: Number of query blocks = ceil(L / BLOCK_M)
- topk: Number of key blocks each query block attends to
```

#### Per-Query-Block LUT Entry
For query block `m` in batch `b` and head `h`:
```
LUT[b, h, m, :] = [k_block_1, k_block_2, ..., k_block_topk]
```

Each entry is an **integer index** (0 to N_BLOCKS-1) into the key/value blocks.

#### LUT Offset Calculation (from kernel.py line 37)
```python
# In _attn_fwd:
lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
LUT_ptr = LUT + lut_offset

# Access pattern:
for block_idx in range(topk):
    idx_n = LUT_ptr[block_idx]  # Load key block index
    # idx_n is the key block index (0 to N_BLOCKS-1)
```

### KBID (Key Block ID) Structure

The backward pass uses an additional structure called `KBID` (Key Block ID):

```python
# In _attn_bwd_dkdv (line 200):
kbid_offset = idx_bh * M_BLOCKS * N_BLOCKS
KBID_ptr = KBID + kbid_offset + idx_n

# KBID is a flattened (B, H, M_BLOCKS, N_BLOCKS) tensor
# KBID[b, h, m, n] = 1 if query block m attends to key block n, else 0
```

This enables the dK/dV kernel to iterate over query blocks that access a given key block.

---

## Tiling Strategy Analysis

### Tile Dimensions

| Tile | Dimension | Purpose |
|------|-----------|---------|
| BLOCK_M | 64 or 128 | Query tokens per tile (rows) |
| BLOCK_N | 64 | Key/Value tokens per tile (columns) |
| D | 64 or 128 | Head dimension (feature depth) |
| BLOCK_M2 | BLOCK_M // BLOCK_SLICE_FACTOR | Sliced query for dK/dV kernel |

### Recommended Configurations

From core.py lines 186-189:
```python
# For sm90 (H100): Use smaller query blocks, larger key blocks
if arch == "sm90":
    BLKQ, BLKK = 64, 128
else:  # sm80, sm86, sm87 (A100, RTX 3090, etc.)
    BLKQ, BLKK = 128, 64
```

### Memory Access Patterns

#### Forward Pass (_attn_fwd)
```
Q tile: [BLOCK_M, D] - loaded once per query block
K tile: [BLOCK_N, D] - loaded topk times (once per key block)
V tile: [BLOCK_N, D] - loaded topk times (once per key block)

QK computation: [BLOCK_M, D] x [D, BLOCK_N] = [BLOCK_M, BLOCK_N]
Attention: [BLOCK_M, BLOCK_N] x [BLOCK_N, D] = [BLOCK_M, D]
```

#### Backward dQ (_attn_bwd_dq)
```
Q tile: [BLOCK_M, D] - loaded once, stays in SRAM
do_s tile: [BLOCK_M, D] - loaded once, stays in SRAM
K tile: [BLOCK_N, D] - loaded topk times
V tile: [BLOCK_N, D] - loaded topk times
```

#### Backward dK/dV (_attn_bwd_dkdv)
```
Uses BLOCK_SLICE_FACTOR to reduce register pressure:
BLOCK_M2 = BLOCK_M // BLOCK_SLICE_FACTOR (typically 64//1=64 or 128//2=64)

K tile: [BLOCK_N, D] - loaded once, stays in SRAM
V tile: [BLOCK_N, D] - loaded once, stays in SRAM
Q slices: [BLOCK_M2, D] - iterated across L
```

---

## Parallelism Structure

### Triton Grid Organization

#### Forward Pass Grid (line 259)
```python
grid = (M_BLOCKS, B * H)
# Axis 0: idx_m - query block index (0 to M_BLOCKS-1)
# Axis 1: idx_bh - flattened batch*head index (0 to B*H-1)
```

#### Backward dQ Grid (line 298)
```python
grid = (M_BLOCKS, B * H)
# Same organization as forward pass
```

#### Backward dK/dV Grid (line 309)
```python
grid = (N_BLOCKS, B * H)
# Axis 0: idx_n - key block index (0 to N_BLOCKS-1)
# Axis 1: idx_bh - flattened batch*head index
```

#### Preprocessing Grid (line 292)
```python
grid = (M_BLOCKS, B * H)
# Computes delta_s = sum(o_s * do_s, dim=-1)
```

### SYCL Work-Group Mapping

| Triton Concept | SYCL Equivalent | Mapping |
|----------------|-----------------|---------|
| program_id(0) | item.get_group(0) | Query/Key block index |
| program_id(1) | item.get_group(1) | Batch*Head index |
| num_programs(0) | nd_range.get_group_range(0) | Total blocks (M_BLOCKS or N_BLOCKS) |
| num_programs(1) | nd_range.get_group_range(1) | Total batch*head combinations |
| Implicit threads | sycl::nd_item | Work-items within work-group |

### Work-Group Size Determination

From kernel launch configuration (lines 265-266):
```python
num_warps = 4 if D == 64 else 8
# Each warp = 32 threads
# Work-group size = num_warps x 32 = 128 (D=64) or 256 (D=128)
```

In SYCL:
```cpp
// For D=64
constexpr size_t WG_SIZE = 128;
// For D=128  
constexpr size_t WG_SIZE = 256;

sycl::nd_range<2> nd_range(
    sycl::range<2>(M_BLOCKS, B*H),  // Global range
    sycl::range<2>(1, 1)            // Local range (1 work-group per block)
);
```

---

## Kernel-by-Kernel Analysis

### 1. Forward Pass Kernel (_attn_fwd)

#### Location: Lines 21-82 in kernel.py

#### Signature
```python
@triton.jit
def _attn_fwd(
    Q, K, V,                          # Input tensors
    qk_scale: tl.constexpr,           # 1/sqrt(D)
    topk: tl.constexpr,               # Number of key blocks
    LUT, LSE, OS,                     # LUT, log-sum-exp, output
    L: tl.constexpr,                  # Sequence length
    M_BLOCKS: tl.constexpr,           # Number of query blocks
    D: tl.constexpr,                  # Head dimension
    BLOCK_M: tl.constexpr,            # Query tile size
    BLOCK_N: tl.constexpr,            # Key/Value tile size
):
```

#### Algorithm Flow

1. **Compute Offsets** (Lines 33-48)
   ```python
   idx_m = tl.program_id(0).to(tl.int64)    # Query block index
   idx_bh = tl.program_id(1).to(tl.int64)   # Batch*Head index
   
   qkv_offset = idx_bh * L * D              # Base offset for Q/K/V
   lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
   lse_offset = idx_bh * L
   ```

2. **Initialize Accumulators** (Lines 50-52)
   ```python
   m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)  # Running max
   l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # Running sum
   o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)             # Output accumulator
   ```

3. **Load Q Tile** (Line 54)
   ```python
   q = tl.load(Q_ptrs, mask=offs_m[:, None] < L)
   ```

4. **Iterate Over Key Blocks** (Lines 55-76)
   ```python
   for block_idx in tl.range(topk):
       idx_n = tl.load(LUT_ptr + block_idx)  # Get key block index from LUT
       n_mask = offs_n < L - idx_n * BLOCK_N  # Boundary mask
       
       # Load K and V tiles (lines 59, 64)
       k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[None, :])
       v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
       
       # Compute QxK^T with log2 scaling (line 60)
       qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)
       
       # Apply boundary mask (lines 61-62)
       if L - idx_n * BLOCK_N < BLOCK_N:
           qk = tl.where(n_mask[None, :], qk, float("-inf"))
       
       # Online softmax update (lines 65-76)
       local_m = tl.max(qk, 1)           # Max of current block
       new_m = tl.maximum(m_i, local_m)  # Global max update
       qk = qk - new_m[:, None]          # Subtract max for stability
       
       p = tl.math.exp2(qk)              # Exponentiate
       l_ij = tl.sum(p, 1)               # Local sum
       alpha = tl.math.exp2(m_i - new_m) # Scaling factor for previous values
       
       # Update running statistics
       o_s = o_s * alpha[:, None]        # Scale previous output
       o_s += tl.dot(p.to(v.dtype), v)   # Add current contribution
       l_i = l_i * alpha + l_ij          # Update sum
       m_i = new_m                       # Update max
   ```

5. **Normalize and Store** (Lines 78-82)
   ```python
   o_s = o_s / l_i[:, None]  # Normalize by softmax denominator
   tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < L)
   
   m_i += tl.math.log2(l_i)  # Convert to log-sum-exp
   tl.store(LSE_ptrs, m_i, mask=offs_m < L)
   ```

#### Key Observations

- **Online Softmax**: Uses the stabilized algorithm from "Online normalizer calculation for softmax"
- **Log2 Scaling**: Uses `exp2` instead of `exp` for potential performance benefits
- **Tiling**: Q loaded once, K/V loaded `topk` times
- **Precision**: FP32 for softmax, cast to input dtype for matmul

---

### 2. Preprocessing Kernel (_attn_bwd_preprocess)

#### Location: Lines 85-106 in kernel.py

#### Purpose
Computes the `delta_s` term needed for backward pass (line 105):
```
delta_s = sum(o_s * do_s, dim=-1)
```

This is the dot product of output and gradient-output for each token.

#### Algorithm
```python
@triton.jit
def _attn_bwd_preprocess(
    OS, DOS, DELTAS,    # Output, grad_output, delta storage
    L,                  # Sequence length
    D: tl.constexpr,    # Head dimension
    BLOCK_M: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)
    
    # Load o_s and do_s tiles (lines 102-103)
    o_s = tl.load(OS + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L)
    do_s = tl.load(DOS + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L)
    
    # Compute delta_s = sum(o_s * do_s, axis=1)
    delta_s = tl.sum(o_s * do_s, axis=1).to(DELTAS.type.element_ty)
    tl.store(DELTAS + offs_m, delta_s, mask=offs_m < L)
```

---

### 3. Backward dQ Kernel (_attn_bwd_dq)

#### Location: Lines 110-164 in kernel.py

#### Purpose
Computes gradient with respect to queries: `dQ`

#### Mathematical Derivation

The gradient flow for attention:
```
dO = grad from next layer
dP = dO x V^T        # Gradient w.r.t. attention scores
dS = P * (dP - delta_s)  # Account for softmax derivative
dQ = dS x K          # Gradient w.r.t. queries
```

Where `delta_s = sum(O * dO)` is computed by the preprocessing kernel.

#### Algorithm Flow

1. **Load Persistent Data** (Lines 143-146)
   ```python
   q = tl.load(Q_ptrs, mask=offs_m[:, None] < L)      # Q tile
   do_s = tl.load(DOS_ptrs, mask=offs_m[:, None] < L) # dO tile
   delta_s = tl.load(DELTAS_ptrs, mask=offs_m < L)    # delta values
   lse = tl.load(LSE_ptrs, mask=offs_m < L, other=float("inf"))  # LSE
   ```

2. **Initialize Accumulator** (Line 148)
   ```python
   dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
   ```

3. **Iterate Over Key Blocks** (Lines 149-163)
   ```python
   for block_idx in tl.range(topk, num_stages=2):
       idx_n = tl.load(LUT_ptr + block_idx)
       n_mask = offs_n < L - idx_n * BLOCK_N
       
       # Load K and V tiles (lines 153-154)
       k = tl.load(K_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
       v = tl.load(V_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
       
       # Recompute attention scores and probabilities (lines 155-157)
       qk = tl.dot(q, k.T) * (qk_scale * 1.4426950408889634)
       p = tl.math.exp2(qk - lse[:, None])  # P = softmax(QK^T)
       p = tl.where(n_mask[None, :], p, 0.0)
       
       # Compute gradients (lines 160-163)
       dp = tl.dot(do_s, v.T).to(tl.float32)  # dP = dO x V^T
       ds = p * (dp - delta_s[:, None])       # dS = P * (dP - delta)
       dq += tl.dot(ds.to(k.dtype), k)        # Accumulate dQ
   ```

4. **Store Result** (Line 164)
   ```python
   tl.store(DQ_ptrs, dq * qk_scale, mask=offs_m[:, None] < L)
   ```

---

### 4. Backward dK/dV Kernel (_attn_bwd_dkdv)

#### Location: Lines 167-237 in kernel.py

#### Purpose
Computes gradients with respect to keys and values: `dK` and `dV`

#### Mathematical Derivation

```
dV = P^T x dO        # Gradient w.r.t. values
dK = dS^T x Q        # Gradient w.r.t. keys
```

#### Algorithm Flow

1. **Setup Block Slicing** (Lines 179, 185)
   ```python
   BLOCK_M2: tl.constexpr = BLOCK_M // BLOCK_SLICE_FACTOR
   ```

2. **Load Persistent K/V** (Lines 203-204)
   ```python
   k = tl.load(K_ptrs, mask=offs_n[:, None] < L)
   v = tl.load(V_ptrs, mask=offs_n[:, None] < L)
   ```

3. **Initialize Accumulators** (Lines 206-207)
   ```python
   dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
   dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
   ```

4. **Iterate Over Query Blocks** (Lines 208-234)
   ```python
   for idx_m in tl.range(0, L, BLOCK_M2):
       kbid = tl.load(KBID_ptr)  # Check if query uses this key block (line 209)
       
       if kbid == 1:  # Only process if there is a dependency
           # Load Q slice and related data (lines 212-213)
           q = tl.load(Q_ptrs, mask=m_mask[:, None])
           lse = tl.load(LSE_ptrs, mask=m_mask, other=float("inf"))
           
           # Compute P^T (lines 214-216)
           qkT = tl.dot(k, q.T) * (qk_scale * 1.4426950408889634)
           pT = tl.math.exp2(qkT - lse[None, :])
           pT = tl.where(offs_n[:, None] < L, pT, 0.0)
           
           # Load gradient (line 218)
           do = tl.load(DOS_ptrs, mask=m_mask[:, None])
           
           # Compute dV (line 220)
           dv += tl.dot(pT.to(do.dtype), do)
           
           # Compute dK (lines 221-225)
           delta = tl.load(DELTAS_ptrs, mask=m_mask)
           dpT = tl.dot(v, tl.trans(do))
           dsT = pT * (dpT - delta[None, :])
           dk += tl.dot(dsT.to(q.dtype), q)
       
       # Advance pointers (lines 228-233)
       Q_ptrs += BLOCK_M2 * D
       DOS_ptrs += BLOCK_M2 * D
       LSE_ptrs += BLOCK_M2
       DELTAS_ptrs += BLOCK_M2
       if (idx_m + BLOCK_M2) % BLOCK_M == 0:
           KBID_ptr += N_BLOCKS
   ```

5. **Store Results** (Lines 236-237)
   ```python
   tl.store(DK_ptrs, dk * qk_scale, mask=offs_n[:, None] < L)
   tl.store(DV_ptrs, dv, mask=offs_n[:, None] < L)
   ```

---

## Triton-to-SYCL Mapping

### Triton Concepts to SYCL

| Triton | SYCL | Notes |
|--------|------|-------|
| @triton.jit | SYCL kernel function | Mark with [[intel::reqd_sub_group_size(32)]] |
| tl.program_id(0) | item.get_group(0) | Work-group ID in X dimension |
| tl.program_id(1) | item.get_group(1) | Work-group ID in Y dimension |
| tl.arange(0, N) | Manual index calculation | Create thread-local indices |
| tl.load(ptr, mask=...) | Conditional loads | Use if statements or masked ops |
| tl.store(ptr, val, mask=...) | Conditional stores | Check bounds before storing |
| tl.dot(a, b) | joint_matrix or manual MMA | Use Intel XMX if available |
| tl.full([M], val, dtype) | sycl::vec or array init | Initialize local arrays |
| tl.zeros([M], dtype) | sycl::vec or zero init | Initialize to zero |
| tl.maximum(a, b) | sycl::max(a, b) | Element-wise maximum |
| tl.math.exp2(x) | sycl::exp2(x) | Base-2 exponential |
| tl.math.log2(x) | sycl::log2(x) | Base-2 logarithm |
| tl.where(cond, a, b) | cond ? a : b | Ternary operator |
| tl.sum(x, axis) | sycl::reduce or loop | Reduction operation |
| tl.trans(x) | Transpose indices | Swap access pattern |
| num_stages=N | Manual prefetching | Use prefetch API |
| tl.range(N) | for loop | Standard C++ loop |
| a.to(dtype) | static_cast<dtype>(a) | Type conversion |

### Memory Space Mapping

| Triton | SYCL | Purpose |
|--------|------|---------|
| Global pointers | T* (global memory) | Q, K, V, LUT, outputs |
| Registers (implicit) | Local variables | Accumulators, temporaries |
| Shared memory (implicit) | sycl::local_accessor | Tile buffers |

### ND-Range Configuration

```cpp
// Forward Pass and Backward dQ
sycl::nd_range<2> fwd_nd_range(
    sycl::range<2>(M_BLOCKS, B * H),  // Global: (query_blocks, batch_heads)
    sycl::range<2>(1, 1)               // Local: 1 work-group per tile
);

// Backward dK/dV
sycl::nd_range<2> bwd_dkdv_nd_range(
    sycl::range<2>(N_BLOCKS, B * H),  // Global: (key_blocks, batch_heads)
    sycl::range<2>(1, 1)               // Local: 1 work-group per tile
);
```

---

## Memory Layout Specifications

### Input Tensor Layouts

```cpp
// Q, K, V: (B, H, L, D) - Row-major
// Strides:
//   stride_b = H * L * D
//   stride_h = L * D
//   stride_l = D
//   stride_d = 1

// LUT: (B, H, M_BLOCKS, topk) - Row-major
// Strides:
//   stride_b = H * M_BLOCKS * topk
//   stride_h = M_BLOCKS * topk
//   stride_m = topk
//   stride_k = 1

// KBID: (B, H, M_BLOCKS, N_BLOCKS) - Row-major
// Strides:
//   stride_b = H * M_BLOCKS * N_BLOCKS
//   stride_h = M_BLOCKS * N_BLOCKS
//   stride_m = N_BLOCKS
//   stride_n = 1

// LSE, DELTAS: (B, H, L) - Row-major
// Strides:
//   stride_b = H * L
//   stride_h = L
//   stride_l = 1
```

### Offset Calculation Formulas

```cpp
// Q/K/V offset for batch-head
size_t qkv_offset = idx_bh * L * D;

// LUT offset for query block
size_t lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk;

// LSE/DELTAS offset
size_t lse_offset = idx_bh * L;

// KBID offset
size_t kbid_offset = idx_bh * M_BLOCKS * N_BLOCKS;
```

### Pointer Arithmetic

```cpp
// Q pointer for tile (idx_m, idx_d)
T* Q_ptr = Q + qkv_offset + idx_m * BLOCK_M * D + idx_d;

// K pointer for key block idx_n
T* K_ptr = K + qkv_offset + idx_n * BLOCK_N * D + idx_d;

// LUT pointer
int* LUT_ptr = LUT + lut_offset;

// Access: LUT_ptr[block_idx] gives key block index
```

---

## Algorithm Flow Pseudocode

### Forward Pass Pseudocode

```cpp
void sparse_attn_fwd(
    sycl::nd_item<2> item,
    const bfloat16* Q, const bfloat16* K, const bfloat16* V,
    float qk_scale, int topk,
    const int* LUT, float* LSE, bfloat16* OS,
    int L, int M_BLOCKS, int D, int BLOCK_M, int BLOCK_N
) {
    // Grid indices
    int idx_m = item.get_group(0);      // Query block index
    int idx_bh = item.get_group(1);     // Batch*Head index
    
    // Compute offsets
    size_t qkv_offset = idx_bh * L * D;
    size_t lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk;
    size_t lse_offset = idx_bh * L;
    
    // Thread-local indices
    int local_m = item.get_local_id(0);
    int local_d = item.get_local_id(1);
    
    // Global token index
    int offs_m = idx_m * BLOCK_M + local_m;
    
    // Initialize accumulators (FP32)
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_s[D];
    for (int d = 0; d < D; d++) o_s[d] = 0.0f;
    
    // Load Q tile
    float q[D];
    load_q_tile(Q + qkv_offset + offs_m * D, q, D, offs_m < L);
    
    // Iterate over key blocks from LUT
    for (int block_idx = 0; block_idx < topk; block_idx++) {
        int idx_n = LUT[lut_offset + block_idx];
        int offs_n_start = idx_n * BLOCK_N;
        
        // Load K and V tiles
        float k[BLOCK_N][D];
        float v[BLOCK_N][D];
        load_k_tile(K + qkv_offset + offs_n_start * D, k, BLOCK_N, D, L);
        load_v_tile(V + qkv_offset + offs_n_start * D, v, BLOCK_N, D, L);
        
        // Compute QxK^T: [1 x D] x [D x BLOCK_N] = [1 x BLOCK_N]
        float qk[BLOCK_N];
        for (int n = 0; n < BLOCK_N; n++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += q[d] * k[n][d];
            }
            qk[n] = dot * qk_scale * 1.4426950408889634f;
            if (offs_n_start + n >= L) qk[n] = -INFINITY;
        }
        
        // Online softmax update
        float local_m = max_over_n(qk);
        float new_m = sycl::max(m_i, local_m);
        
        float exp_qk[BLOCK_N];
        float l_ij = 0.0f;
        for (int n = 0; n < BLOCK_N; n++) {
            exp_qk[n] = sycl::exp2(qk[n] - new_m);
            if (offs_n_start + n < L) l_ij += exp_qk[n];
        }
        
        float alpha = sycl::exp2(m_i - new_m);
        
        // Update output
        for (int d = 0; d < D; d++) {
            o_s[d] *= alpha;
            for (int n = 0; n < BLOCK_N; n++) {
                o_s[d] += exp_qk[n] * v[n][d];
            }
        }
        
        l_i = l_i * alpha + l_ij;
        m_i = new_m;
    }
    
    // Normalize and store
    for (int d = 0; d < D; d++) {
        o_s[d] /= l_i;
    }
    
    store_o_tile(OS + qkv_offset + offs_m * D, o_s, D, offs_m < L);
    
    if (offs_m < L) {
        LSE[lse_offset + offs_m] = m_i + sycl::log2(l_i);
    }
}
```

### Backward dQ Pseudocode

```cpp
void sparse_attn_bwd_dq(
    sycl::nd_item<2> item,
    const bfloat16* Q, const bfloat16* K, const bfloat16* V,
    const float* LSE, const float* DELTAS,
    const bfloat16* DOS, bfloat16* DQ, const int* LUT,
    float qk_scale, int topk,
    int L, int M_BLOCKS, int D, int BLOCK_M, int BLOCK_N
) {
    int idx_m = item.get_group(0);
    int idx_bh = item.get_group(1);
    
    size_t qkv_offset = idx_bh * L * D;
    size_t lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk;
    size_t lse_offset = idx_bh * L;
    
    int offs_m = idx_m * BLOCK_M + item.get_local_id(0);
    
    // Load persistent data
    float q[D], do_s[D];
    load_tile(Q + qkv_offset + offs_m * D, q, D, offs_m < L);
    load_tile(DOS + qkv_offset + offs_m * D, do_s, D, offs_m < L);
    
    float delta_s = (offs_m < L) ? DELTAS[lse_offset + offs_m] : 0.0f;
    float lse = (offs_m < L) ? LSE[lse_offset + offs_m] : INFINITY;
    
    // Initialize dQ accumulator
    float dq[D];
    for (int d = 0; d < D; d++) dq[d] = 0.0f;
    
    // Iterate over key blocks
    for (int block_idx = 0; block_idx < topk; block_idx++) {
        int idx_n = LUT[lut_offset + block_idx];
        int offs_n_start = idx_n * BLOCK_N;
        
        float k[BLOCK_N][D], v[BLOCK_N][D];
        load_k_tile(K + qkv_offset + offs_n_start * D, k, BLOCK_N, D, L);
        load_v_tile(V + qkv_offset + offs_n_start * D, v, BLOCK_N, D, L);
        
        // Recompute P
        float p[BLOCK_N];
        for (int n = 0; n < BLOCK_N; n++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += q[d] * k[n][d];
            }
            float qk = dot * qk_scale * 1.4426950408889634f;
            p[n] = (offs_n_start + n < L) ? sycl::exp2(qk - lse) : 0.0f;
        }
        
        // Compute dP = dO x V^T
        float dp[BLOCK_N];
        for (int n = 0; n < BLOCK_N; n++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += do_s[d] * v[n][d];
            }
            dp[n] = dot;
        }
        
        // Compute dS = P * (dP - delta)
        float ds[BLOCK_N];
        for (int n = 0; n < BLOCK_N; n++) {
            ds[n] = p[n] * (dp[n] - delta_s);
        }
        
        // Accumulate dQ = dS x K
        for (int d = 0; d < D; d++) {
            for (int n = 0; n < BLOCK_N; n++) {
                dq[d] += ds[n] * k[n][d];
            }
        }
    }
    
    for (int d = 0; d < D; d++) {
        dq[d] *= qk_scale;
    }
    store_tile(DQ + qkv_offset + offs_m * D, dq, D, offs_m < L);
}
```

### Backward dK/dV Pseudocode

```cpp
void sparse_attn_bwd_dkdv(
    sycl::nd_item<2> item,
    const bfloat16* Q, const bfloat16* K, const bfloat16* V,
    const bfloat16* DOS, bfloat16* DK, bfloat16* DV,
    float qk_scale, const int8_t* KBID, const float* LSE, const float* DELTAS,
    int L, int M_BLOCKS, int N_BLOCKS, int D,
    int BLOCK_M, int BLOCK_N, int BLOCK_SLICE_FACTOR
) {
    int BLOCK_M2 = BLOCK_M / BLOCK_SLICE_FACTOR;
    
    int idx_n = item.get_group(0);
    int idx_bh = item.get_group(1);
    
    int offs_n = idx_n * BLOCK_N + item.get_local_id(0);
    
    size_t qkv_offset = idx_bh * L * D;
    size_t kbid_offset = idx_bh * M_BLOCKS * N_BLOCKS;
    size_t lse_offset = idx_bh * L;
    
    // Load persistent K and V
    float k[D], v[D];
    load_tile(K + qkv_offset + offs_n * D, k, D, offs_n < L);
    load_tile(V + qkv_offset + offs_n * D, v, D, offs_n < L);
    
    // Initialize accumulators
    float dk[D], dv[D];
    for (int d = 0; d < D; d++) {
        dk[d] = 0.0f;
        dv[d] = 0.0f;
    }
    
    // Iterate over query blocks
    for (int idx_m = 0; idx_m < L; idx_m += BLOCK_M2) {
        int kbid_idx = (idx_m / BLOCK_M2) / BLOCK_SLICE_FACTOR;
        int8_t kbid = KBID[kbid_offset + kbid_idx * N_BLOCKS + idx_n];
        
        if (kbid == 1) {
            int offs_m_start = idx_m;
            
            float q[BLOCK_M2][D];
            float lse[BLOCK_M2];
            float delta[BLOCK_M2];
            float dos[BLOCK_M2][D];
            
            for (int m = 0; m < BLOCK_M2; m++) {
                int global_m = offs_m_start + m;
                load_tile(Q + qkv_offset + global_m * D, q[m], D, global_m < L);
                lse[m] = (global_m < L) ? LSE[lse_offset + global_m] : INFINITY;
                delta[m] = (global_m < L) ? DELTAS[lse_offset + global_m] : 0.0f;
                load_tile(DOS + qkv_offset + global_m * D, dos[m], D, global_m < L);
            }
            
            // Compute P^T
            float pt[BLOCK_M2];
            for (int m = 0; m < BLOCK_M2; m++) {
                float dot = 0.0f;
                for (int d = 0; d < D; d++) {
                    dot += k[d] * q[m][d];
                }
                float qk = dot * qk_scale * 1.4426950408889634f;
                pt[m] = (offs_n < L) ? sycl::exp2(qk - lse[m]) : 0.0f;
            }
            
            // Compute dV += P^T x dO
            for (int d = 0; d < D; d++) {
                for (int m = 0; m < BLOCK_M2; m++) {
                    dv[d] += pt[m] * dos[m][d];
                }
            }
            
            // Compute dP^T = V x dO^T
            float dpt[BLOCK_M2];
            for (int m = 0; m < BLOCK_M2; m++) {
                float dot = 0.0f;
                for (int d = 0; d < D; d++) {
                    dot += v[d] * dos[m][d];
                }
                dpt[m] = dot;
            }
            
            // Compute dS^T
            float dst[BLOCK_M2];
            for (int m = 0; m < BLOCK_M2; m++) {
                dst[m] = pt[m] * (dpt[m] - delta[m]);
            }
            
            // Compute dK += dS^T x Q
            for (int d = 0; d < D; d++) {
                for (int m = 0; m < BLOCK_M2; m++) {
                    dk[d] += dst[m] * q[m][d];
                }
            }
        }
    }
    
    for (int d = 0; d < D; d++) dk[d] *= qk_scale;
    store_tile(DK + qkv_offset + offs_n * D, dk, D, offs_n < L);
    store_tile(DV + qkv_offset + offs_n * D, dv, D, offs_n < L);
}
```

---

## Implementation Recommendations

### 1. Use Intel XMX (Xe Matrix Extensions)

For optimal performance on Intel GPUs, use the XMX units for matrix multiplication:

```cpp
#include <sycl/ext/oneapi/matrix/matrix.hpp>
namespace matrix = sycl::ext::oneapi::experimental::matrix;

// Define joint matrix types
using tile_a = matrix::joint_matrix<
    sycl::sub_group, bfloat16, matrix::use::a, BLOCK_M, BLOCK_N, matrix::layout::row_major
>;
using tile_b = matrix::joint_matrix<
    sycl::sub_group, bfloat16, matrix::use::b, BLOCK_N, D, matrix::layout::col_major
>;
using tile_c = matrix::joint_matrix<
    sycl::sub_group, float, matrix::use::accumulator, BLOCK_M, D
>;
```

### 2. Sub-Group Size Configuration

```cpp
[[intel::reqd_sub_group_size(32)]]
void kernel(sycl::nd_item<2> item) {
    auto sg = item.get_sub_group();
    // Use sub-group shuffle for efficient communication
}
```

### 3. Memory Prefetching

```cpp
// Prefetch next K/V tiles while computing current
sycl::ext::oneapi::experimental::prefetch(
    item, K_next_ptr, BLOCK_N * D * sizeof(bfloat16)
);
```

### 4. Compiler Optimizations

```cpp
// Unroll loops for small fixed sizes
#pragma unroll
for (int d = 0; d < D; d++) { ... }

// Use restrict for pointer aliasing
void kernel(const bfloat16* __restrict__ Q, ...)
```

### 5. Numerical Precision

```cpp
// Always use FP32 for softmax accumulation
float m_i = -INFINITY;
float l_i = 0.0f;

// Cast to BF16 only for matrix inputs
bfloat16 q_bf16 = sycl::ext::oneapi::bfloat16(q_f32);
```

### 6. Work-Group Local Memory

```cpp
// Allocate adequate local memory for tiles
sycl::local_accessor<bfloat16, 1> k_tile(
    sycl::range<1>(BLOCK_N * D), h
);

// Cooperatively load data
for (int i = item.get_local_linear_id(); i < BLOCK_N * D; 
     i += item.get_local_range().size()) {
    k_tile[i] = global_k[i];
}
item.barrier(sycl::access::fence_space::local_space);
```

### 7. Tuning Parameters

| Parameter | Default | Tuning Range | Notes |
|-----------|---------|--------------|-------|
| BLOCK_M | 64/128 | 32, 64, 128 | Larger = more parallelism |
| BLOCK_N | 64 | 32, 64, 128 | Must balance with BLOCK_M |
| BLOCK_SLICE_FACTOR | 1-2 | 1, 2, 4 | Higher = less register pressure |
| Work-group size | 128/256 | 64, 128, 256 | Match XMX requirements |

---

## Testing Strategy

### Unit Tests

1. **LUT Generation Test**
   - Verify correct top-k selection
   - Check boundary handling

2. **Forward Pass Test**
   - Compare with PyTorch reference
   - Test with various sequence lengths
   - Verify numerical accuracy (BF16 vs FP32)

3. **Backward Pass Test**
   - Gradient checker (finite differences)
   - Compare with autograd reference

4. **Edge Cases**
   - Sequence length not divisible by BLOCK_M/N
   - topk = all blocks (dense attention)
   - topk = 1 block (most sparse)

### Performance Benchmarks

1. **Throughput**: tokens/sec for various (B, H, L, D)
2. **Memory Bandwidth**: GB/s utilization
3. **Scaling**: Strong/weak scaling across GPUs

---

## References

1. **FlashAttention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
2. **Online Softmax**: Milakov and Gimelshein, "Online normalizer calculation for softmax"
3. **Triton Documentation**: https://triton-lang.org/
4. **Intel oneAPI DPC++**: https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html
5. **SLA Paper**: Zhang et al., "SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention"

---

## Appendix: Key Code Sections from Source

### Forward Pass Launch Configuration (kernel.py lines 259-267)
```python
grid = (M_BLOCKS, B * H)
_attn_fwd[grid](
    q, k, v, qk_scale, topk,
    lut, lse, o_s,
    L, M_BLOCKS,
    D, BLOCK_M, BLOCK_N,
    num_warps=4 if q.shape[-1] == 64 else 8,
    num_stages=3
)
```

### Backward Pass Launch Configuration (kernel.py lines 292-318)
```python
# Preprocessing
grid = (M_BLOCKS, B * H)
_attn_bwd_preprocess[grid](o_s, do_s, delta_s, L, D, BLOCK_M)

# dQ computation
grid = (M_BLOCKS, B * H)
_attn_bwd_dq[grid](
    q, k, v, lse, delta_s, do_s, dq, lut,
    ctx.qk_scale, ctx.topk, L, M_BLOCKS,
    D, BLOCK_M, BLOCK_N,
    num_warps=4 if q.shape[-1] == 64 else 8,
    num_stages=4 if q.shape[-1] == 64 else 5
)

# dK/dV computation
grid = (N_BLOCKS, B * H)
_attn_bwd_dkdv[grid](
    q, k, v, do_s, dk, dv,
    ctx.qk_scale, k_block_id, lse, delta_s,
    L, M_BLOCKS, N_BLOCKS,
    D, BLOCK_M, BLOCK_N,
    BLOCK_SLICE_FACTOR=BLOCK_M // 64,
    num_warps=4 if q.shape[-1] == 64 else 8,
    num_stages=4 if q.shape[-1] == 64 else 5
)
```

### Mean Pooling Kernel (utils.py lines 21-41)
```python
@triton.jit
def compress_kernel(
    X, XM,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)

    x_offset = idx_bh * L * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    x = tl.load(X + x_offset + offs_l[:, None] * D + offs_d[None, :], mask=offs_l[:, None] < L)

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d, x_mean.to(XM.dtype.element_ty))
```

---

*End of Technical Specification*
*Document Version: 1.0*
*Generated for TurboDiffusion-SYCL Migration Project*
