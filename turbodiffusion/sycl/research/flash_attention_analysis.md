# Flash Attention v2 SYCL Implementation Analysis
## sycle-tla_internal Deep Dive for TurboDiffusion Integration

**Analysis Date**: April 3, 2026  
**Analyst**: Agent 1 (TurboDiffusion-SYCL Migration Project)  
**Source Repository**: `/home/intel/tianfeng/opencode_bench/sycle-tla_internal`  
**Target Architecture**: Intel Xe (Arc/Battlemage)

---

## 1. Executive Summary

The sycle-tla_internal repository contains a comprehensive Flash Attention v2 implementation optimized for Intel Xe GPUs using SYCL/CUTLASS. It provides three main kernel variants:

1. **Prefill** (`FMHAPrefill`) - Standard forward attention for training/prefill
2. **Decode** (`FMHADecode`) - Optimized for autoregressive generation with KV cache
3. **CachedKV** (`FMHAPrefillCached`) - Prefill with separate KV cache handling

The implementation uses a CUTLASS-based template metaprogramming approach with Xe-specific MMA (Matrix Multiply Accumulate) operations.

---

## 2. Architecture Overview

### 2.1 Directory Structure

```
applications/flash_attention_v2/
├── kernel/
│   ├── xe_fmha_fwd_kernel.hpp          # Main unified kernel (XeFMHAFwdKernel)
│   ├── xe_tile_scheduler.hpp           # Work distribution schedulers
│   └── legacy/
│       ├── xe_flash_attn_prefill.hpp       # Prefill kernel
│       ├── xe_flash_attn_decode.hpp        # Decode kernel
│       └── xe_flash_attn_prefill_cachedKV.hpp  # Prefill with KV cache
├── collective/
│   ├── xe_fmha_fwd_mainloop.hpp        # Mainloop (QK^T + softmax + PV)
│   ├── xe_fmha_fwd_epilogue.hpp        # Output reduction and writeback
│   ├── fmha_fusion.hpp                 # Variable length utilities
│   └── legacy/                         # Legacy collective implementations
```

### 2.2 Key Components

| Component | Purpose | File |
|-----------|---------|------|
| `XeFMHAFwdKernel` | Unified forward kernel with CachedKV, Scale support | `xe_fmha_fwd_kernel.hpp` |
| `FMHAPrefill` | Legacy prefill kernel | `legacy/xe_flash_attn_prefill.hpp` |
| `FMHADecode` | Legacy decode kernel with KV cache support | `legacy/xe_flash_attn_decode.hpp` |
| `FMHAFwdMainloop` | Core attention computation (QK^T, softmax, PV) | `xe_fmha_fwd_mainloop.hpp` |
| `FMHAFwdEpilogue` | Output softmax normalization and writeback | `xe_fmha_fwd_epilogue.hpp` |

---

## 3. API Reference

### 3.1 Data Type Support

**Supported Input Types** (from configuration analysis):
- `cutlass::bfloat16_t` (BF16) - **Recommended for TurboDiffusion**
- `cutlass::half_t` (FP16)
- `cutlass::float_e4m3_t` (FP8) - With scaling
- `cutlass::float_e2m1_t` (FP4) - Experimental

**Accumulator Types**:
- `float` (FP32) - For numerical stability in softmax

**MMA Operations for Xe**:
- BF16: `cute::XE_8x16x16_F32BF16BF16F32_TT` (prefill)
- BF16: `cute::XE_1x16x16_F32BF16BF16F32_TT` (decode)
- FP16: `cute::XE_8x16x16_F32F16F16F32_TT`

### 3.2 Problem Shape Definition

**Unified Kernel (XeFMHAFwdKernel)**:
```cpp
template <bool IsVarLen_ = false>
struct FMHAProblemShape {
  using SeqLenType = cute::conditional_t<IsVarLen_, 
    cutlass::fmha::collective::VariableLength, int>;
  int batch;
  int num_heads_q, num_heads_kv;
  SeqLenType seq_len_qo, seq_len_kv, seq_len_kv_cache;
  int head_size_qk, head_size_vo;
};
```

**Legacy Prefill** (rank-7 tuple):
```cpp
// <batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo>
cute::tuple<int, int, int, int, int, int, int>
```

**Legacy Decode** (rank-8 tuple):
```cpp
// <batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, seq_len_kv_cache, head_size_qk, head_size_vo>
cute::tuple<int, int, int, int, int, int, int, int>
```

### 3.3 Function Signatures

#### Unified Kernel Arguments (XeFMHAFwdKernel)

```cpp
struct KernelArguments {
  ProblemShape shape;
  const ElementQ *Q;           StrideQ dQ;
  const ElementK *K;           StrideK dK;
  const ElementV *V;           StrideV dV;
  ElementO *O;                 StrideO dO;
  
  // Optional scaling (for FP8/FP4)
  const ElementScale *scaleQ = nullptr;  StrideScaleQ dScaleQ{};
  const ElementScale *scaleK = nullptr;  StrideScaleK dScaleK{};
  const ElementScale *scaleV = nullptr;  StrideScaleV dScaleV{};
  float scale_k;
  float scale_v;
  int group_size = 32;
  
  // Optional KV cache
  const ElementK *K_cache;     StrideK dK_cache{};
  const ElementV *V_cache;     StrideV dV_cache{};
};
```

#### Legacy Prefill Arguments

```cpp
struct Arguments {
  gemm::GemmUniversalMode mode;
  ProblemShape problem_shape;
  MainloopArguments mainloop;      // {Q, strideQ, K, strideK, V, strideV}
  SoftmaxArguments softmax;        // {softmax_scale}
  EpilogueArguments epilogue;      // {O, strideO}
  KernelHardwareInfo hw_info;
};
```

#### Legacy Decode Arguments

```cpp
struct Arguments {
  gemm::GemmUniversalMode mode;
  ProblemShape problem_shape;      // Includes seq_len_kv_cache
  MainloopArguments mainloop;      // {Q, K, V, K_cache, V_cache, page_table...}
  SoftmaxArguments softmax;
  EpilogueArguments epilogue;
  KernelHardwareInfo hw_info;
};
```

---

## 4. Memory Layout Requirements

### 4.1 Tensor Layout Specifications

| Tensor | Layout | Stride Requirements |
|--------|--------|---------------------|
| Q (Query) | RowMajor | `[seq_len_qo, head_size_qk, batch * num_heads_q]` |
| K (Key) | ColumnMajor | `[head_size_qk, seq_len_kv, batch * num_heads_kv]` |
| V (Value) | RowMajor | `[seq_len_kv, head_size_vo, batch * num_heads_kv]` |
| O (Output) | RowMajor | `[seq_len_qo, head_size_vo, batch * num_heads_q]` |
| K_cache | ColumnMajor | Same as K |
| V_cache | RowMajor | Same as V |

### 4.2 Alignment Requirements

From `benchmark_runner.hpp`:
```cpp
constexpr int cacheline_bytes = 64;
constexpr int AlignmentQ = cacheline_bytes / sizeof(ElementQ);
constexpr int AlignmentKV = cacheline_bytes / sizeof(ElementK);
constexpr int AlignmentKVCache = 128;  // For paged attention
```

**Key Alignment Rules**:
- Q: 64-byte aligned (seq_len_qo * head_size_qk * sizeof(ElementQ))
- K/V: 64-byte aligned
- KV Cache (paged): 128-byte aligned (page_size must be multiple of 128)

### 4.3 PyTorch Tensor Mapping

```cpp
// PyTorch tensor (B, H, S, D) -> CUTLASS layout
// For Q: [batch, num_heads_q, seq_len_qo, head_size_qk]
// Stride: (num_heads_q * seq_len_qo * head_size_qk, 
//          seq_len_qo * head_size_qk, 
//          head_size_qk, 
//          1)

// For K/V with GQA/MQA: [batch, num_heads_kv, seq_len_kv, head_size]
// Note: num_heads_kv = num_heads_q / num_head_groups
```

---

## 5. Variable Sequence Length Support

### 5.1 VariableLength Structure

```cpp
struct VariableLength {
  int max_length;
  int* cumulative_length = nullptr;  // Size: batch + 1
  int* cumulative_scale_length = nullptr;  // For FP8 scaling
};
```

### 5.2 Usage Pattern

```cpp
// For variable length sequences:
FMHAProblemShape<true> shape;
shape.seq_len_qo = VariableLength{max_seq_len_q, cumsum_q};
shape.seq_len_kv = VariableLength{max_seq_len_kv, cumsum_kv};
shape.seq_len_kv_cache = VariableLength{max_cache_len, cumsum_cache};
```

### 5.3 Cumulative Length Format

```cpp
// Example: batch=3, seq_lens=[10, 20, 15]
// cumsum = [0, 10, 30, 45]
std::vector<int> cumulative_seqlen_q = {0, 10, 30, 45};
```

---

## 6. Configuration Examples

### 6.1 BF16 Prefill Configuration

```cpp
#include "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp"
#include "flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"

// Define types
using ElementQ = cutlass::bfloat16_t;
using ElementK = cutlass::bfloat16_t;
using ElementV = cutlass::bfloat16_t;
using ElementO = cutlass::bfloat16_t;
using ElementAcc = float;

// MMA atom for BF16 on Xe
using MMAOp = cute::XE_8x16x16_F32BF16BF16F32_TT;

// Tile shapes (tunable)
using TileShapeQK = Shape<_64, _64, _64>;   // QK GEMM tile
using TileShapePV = Shape<_64, _32, _64>;   // PV GEMM tile
using TileShapeO  = Shape<_64, _64>;        // Output tile

// Subgroup layout
using SGLayout = Layout<Shape<_2, _2, _1>>;  // 4 subgroups per workgroup

// Collective mainloop
using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
  cutlass::fmha::XeDefault<2>,  // 2-stage pipeline
  true,                          // Causal mask
  false,                         // No scale (use BF16 directly)
  false,                         // F8kvF16mma
  false,                         // CachedKV (set true for unified kernel)
  false,                         // PagedKV
  decltype(cute::make_tiled_mma(MMAOp{}, TileShapeQK{}, SGLayout{})),
  decltype(cute::make_tiled_mma(MMAOp{}, TileShapePV{}, SGLayout{})),
  1,                             // VTiles
  /* Tensor types for Q, K, V, Scale... */
>;
```

### 6.2 Decode Configuration with KV Cache

```cpp
// Decode uses different tile shapes (seq_len_qo = 1 typically)
using TileShapeQK = Shape<_1, _64, _64>;    // Single query token
using TileShapePV = Shape<_1, _32, _64>;
using TileShapeO  = Shape<_1, _64>;

// Enable CachedKV and optionally PagedKV
using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
  cutlass::fmha::XeDefault<2>,
  true,    // Causal
  false,   // UseScale
  false,   // F8kvF16mma
  true,    // CachedKV = true
  false,   // PagedKV (set true for vLLM-style paging)
  /* ... */
>;
```

---

## 7. Integration Code Snippet

### 7.1 Complete PyTorch Integration Pattern

```cpp
// turbodiffusion_sycl_flash_attn.hpp
#pragma once

#include <sycl/sycl.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/bfloat16.h>
#include "flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp"

namespace turbodiffusion {

using Element = cutlass::bfloat16_t;
using ElementAcc = float;

class FlashAttentionOp {
public:
  struct Config {
    int batch_size;
    int num_heads_q;
    int num_heads_kv;
    int seq_len_qo;
    int seq_len_kv;
    int head_size_qk;
    int head_size_vo;
    bool causal = true;
    bool use_cache = false;
  };

  // Main entry point matching PyTorch API
  static void forward(
    sycl::queue& queue,
    const Element* query,      // [B, Hq, Sq, D]
    const Element* key,        // [B, Hkv, Sk, D]  
    const Element* value,      // [B, Hkv, Sk, Dvo]
    Element* output,           // [B, Hq, Sq, Dvo]
    const Config& config,
    const Element* key_cache = nullptr,    // Optional
    const Element* value_cache = nullptr,  // Optional
    float softmax_scale = 0.0f  // If 0, uses 1/sqrt(head_dim)
  );

private:
  using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<false>;
  
  // Predefined kernel configurations
  template<typename TileConfig>
  using KernelType = cutlass::fmha::kernel::XeFMHAFwdKernel<
    ProblemShape,
    /* CollectiveMainloop */,
    /* CollectiveEpilogue */,
    cutlass::fmha::kernel::XeFHMAIndividualTileScheduler
  >;
};

} // namespace turbodiffusion
```

### 7.2 Implementation Skeleton

```cpp
// turbodiffusion_sycl_flash_attn.cpp

void FlashAttentionOp::forward(
    sycl::queue& queue,
    const Element* query,
    const Element* key,
    const Element* value,
    Element* output,
    const Config& config,
    const Element* key_cache,
    const Element* value_cache,
    float softmax_scale) {

  // Set default scale if not provided
  if (softmax_scale == 0.0f) {
    softmax_scale = 1.0f / std::sqrt(static_cast<float>(config.head_size_qk));
  }

  // Define problem shape
  ProblemShape shape;
  shape.batch = config.batch_size;
  shape.num_heads_q = config.num_heads_q;
  shape.num_heads_kv = config.num_heads_kv;
  shape.seq_len_qo = config.seq_len_qo;
  shape.seq_len_kv = config.seq_len_kv;
  shape.seq_len_kv_cache = key_cache ? config.seq_len_kv : 0;
  shape.head_size_qk = config.head_size_qk;
  shape.head_size_vo = config.head_size_vo;

  // Compute strides (PyTorch contiguous layout)
  auto stride_Q = cutlass::make_cute_packed_stride(
    StrideQ{}, 
    make_shape(config.seq_len_qo, config.head_size_qk, 
               config.batch_size * config.num_heads_q)
  );
  
  // ... similar for K, V, O

  // Setup kernel arguments
  using KernelArguments = typename KernelType<>::KernelArguments;
  KernelArguments args{
    shape,
    query, stride_Q,
    key, stride_K,
    value, stride_V,
    output, stride_O,
    nullptr, {}, nullptr, {}, nullptr, {},  // scale tensors (unused for BF16)
    1.0f, 1.0f, 32,  // scale_k, scale_v, group_size
    key_cache, stride_K_cache,
    value_cache, stride_V_cache
  };

  // Create and launch kernel
  typename KernelType<>::Arguments full_args{
    args,  // kernel args
    {softmax_scale, nullptr, 0, nullptr},  // mainloop args
    {},  // epilogue args
    {static_cast<int>(queue.get_device().get_info<sycl::info::device::max_compute_units>())}
  };

  auto params = KernelType<>::to_underlying_arguments(full_args, nullptr);
  
  // Launch via SYCL
  dim3 block = KernelType<>::get_block_shape();
  dim3 grid = KernelType<>::get_grid_shape(params);
  
  queue.parallel_for(
    sycl::nd_range<3>(sycl::range<3>(grid.x*block.x, grid.y*block.y, grid.z*block.z),
                      sycl::range<3>(block.x, block.y, block.z)),
    [=](sycl::nd_item<3> item) {
      KernelType<> kernel;
      kernel(params, nullptr);  // No shared memory for basic case
    }
  ).wait();
}
```

---

## 8. Performance Characteristics

### 8.1 Tile Shape Recommendations

Based on configuration analysis in the codebase:

| Head Size | Prefile TileShapeQK | Prefile TileShapePV | Decode TileShapeQK |
|-----------|---------------------|---------------------|-------------------|
| 64  | Shape<_64, _64, _64> | Shape<_64, _32, _64> | Shape<_1, _64, _64> |
| 96  | Shape<_64, _64, _64> | Shape<_64, _32, _64> | Shape<_1, _64, _64> |
| 128 | Shape<_64, _64, _64> | Shape<_64, _32, _64> | Shape<_1, _64, _64> |
| 192 | Shape<_64, _64, _64> | Shape<_64, _32, _64> | Shape<_1, _64, _64> |

### 8.2 Subgroup Configuration

```cpp
// Standard configuration: 4 subgroups (64 threads) per workgroup
using SubgroupLayout = Layout<Shape<_2, _2, _1>>;  // 2x2x1 = 4 SGS
// Or: Layout<Shape<_4, _1, _1>> for 1D distribution

// For decode: 8 subgroups may be used for larger KV tiles
using DecodeSGLayout = Layout<Shape<_8, _1, _1>>;
```

### 8.3 Memory Bandwidth Considerations

From benchmark analysis:
- **Optimal sequence lengths**: Powers of 2 or multiples of 64
- **Cache-friendly**: seq_len_kv_cache should be multiple of page_size (128)
- **Prefetch stages**: 2-stage pipeline (Stages=2) provides good balance

---

## 9. Limitations and Constraints

### 9.1 Current Limitations

1. **Sequence Length**:
   - `seq_len_qo` must be divisible by tile_shape Q dimension for double buffering
   - Variable length support requires cumulative length arrays

2. **Head Size Constraints**:
   - `head_size_qk` must be divisible by MMA atom K dimension (typically 16 or 32)
   - Common sizes tested: 64, 96, 128, 192

3. **Paged Attention**:
   - Page size must be >= QK_BLK_N and divisible by it
   - Requires separate page table management

4. **Batch Size**:
   - For `XeFMHAFwdDynamicSplitKernel` (decode): 
     `batch * num_heads_q <= hw_info.sm_count` (Xe core count)

### 9.2 GQA/MQA Support

The implementation naturally supports Grouped Query Attention:
```cpp
int head_group_q = num_heads_q / num_heads_kv;
// Each KV head is shared by head_group_q query heads
```

### 9.3 Causal Mask Behavior

For `CausalMask = true`:
- Standard causal masking for self-attention
- Special handling when `seq_len_qo != seq_len_kv` (bottom-up masking)
- Discard sequence coordination for non-square attention

---

## 10. Recommendations for TurboDiffusion

### 10.1 Integration Strategy

1. **Start with Unified Kernel** (`XeFMHAFwdKernel`):
   - Single kernel handles prefill, decode, and cachedKV
   - Better maintained than legacy kernels
   - Supports all features (causal, scale, cache)

2. **Data Type**: Use **BF16**:
   ```cpp
   using Element = cutlass::bfloat16_t;
   ```
   - Native Xe support
   - Better training stability than FP16
   - No scaling needed (unlike FP8)

3. **Tensor Layout**: Match PyTorch defaults:
   - Q/O: `[B, H, S, D]` RowMajor
   - K/V: `[B, H, S, D]` but stored ColumnMajor for K

### 10.2 Recommended Configuration

```cpp
// For Diffusion models (typically head_dim = 64 or 128)
template<int HeadDim>
struct TurboDiffusionConfig {
  static constexpr int kHeadDim = HeadDim;
  static constexpr int kTileQ = 64;
  static constexpr int kTileKV = 64;
  
  using TileShapeQK = Shape<Int<kTileQ>, Int<kTileKV>, Int<kHeadDim>>;
  using TileShapePV = Shape<Int<kTileQ>, _32, Int<kTileKV>>;
  using TileShapeO  = Shape<Int<kTileQ>, Int<kHeadDim>>;
  
  using Element = cutlass::bfloat16_t;
  using ElementAcc = float;
  using MMAOp = cute::XE_8x16x16_F32BF16BF16F32_TT;
};

using Config = TurboDiffusionConfig<128>;  // or 64
```

### 10.3 Build Requirements

Required compiler flags:
```bash
# For Intel oneAPI DPC++
-fsycl
-fsycl-targets=intel_gpu_pvc  # or appropriate target
-Xclang -fsycl-allow-func-ptr

# CUTLASS specific
-DCUTLASS_ENABLE_SYCL
-DCUTLASS_INTEL_XE=1
```

### 10.4 Testing Checklist

- [ ] Unit test with small shapes (B=1, H=1, S=64, D=64)
- [ ] Causal mask correctness verification
- [ ] GQA correctness (num_heads_q > num_heads_kv)
- [ ] Variable length sequence handling
- [ ] Numerical accuracy vs reference (tolerance: 0.05 relative)
- [ ] Performance benchmarking vs naive implementation

---

## 11. References

1. **CUTLASS Documentation**: https://github.com/NVIDIA/cutlass
2. **Intel Xe Architecture**: Intel Xe Matrix Extensions (XMX) specifications
3. **Flash Attention Paper**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
4. **Flash Attention 2 Paper**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"

---

## Appendix A: File Path Reference

| Component | Absolute Path |
|-----------|---------------|
| Main Kernel | `/home/intel/tianfeng/opencode_bench/sycle-tla_internal/applications/flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp` |
| Mainloop | `/home/intel/tianfeng/opencode_bench/sycle-tla_internal/applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp` |
| Epilogue | `/home/intel/tianfeng/opencode_bench/sycle-tla_internal/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp` |
| Prefill (legacy) | `/home/intel/tianfeng/opencode_bench/sycle-tla_internal/applications/flash_attention_v2/kernel/legacy/xe_flash_attn_prefill.hpp` |
| Decode (legacy) | `/home/intel/tianfeng/opencode_bench/sycle-tla_internal/applications/flash_attention_v2/kernel/legacy/xe_flash_attn_decode.hpp` |
| Scheduler | `/home/intel/tianfeng/opencode_bench/sycle-tla_internal/applications/flash_attention_v2/kernel/xe_tile_scheduler.hpp` |
| Prefill Config | `/home/intel/tianfeng/opencode_bench/sycle-tla_internal/benchmarks/flash_attention/legacy/flash_attention_prefill/fmha_prefill_configuration.hpp` |
| Decode Config | `/home/intel/tianfeng/opencode_bench/sycle-tla_internal/benchmarks/flash_attention/legacy/flash_attention_decode/fmha_decode_configuration.hpp` |

