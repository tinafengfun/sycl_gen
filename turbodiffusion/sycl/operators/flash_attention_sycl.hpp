/**
 * TurboDiffusion Flash Attention v2 SYCL Implementation
 * 
 * Integrates sycle-tla_internal's Flash Attention v2 kernel for Intel Xe GPUs.
 * Optimized for BF16 tensors with GQA (Grouped Query Attention) support.
 * 
 * Key features:
 * - BF16 precision for optimal Xe GPU performance
 * - GQA support (num_heads_q >= num_heads_kv)
 * - Variable sequence length support for video generation
 * - USM zero-copy memory access
 * - Causal masking for autoregressive models
 */

#pragma once

#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <cmath>

// CUTLASS includes
#include "cutlass/cutlass.h"
#include "cutlass/bfloat16.h"
#include "cute/tensor.hpp"

// Flash Attention v2 includes
#include "flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp"

namespace turbodiffusion {

using namespace cute;

// Type aliases for clarity
using Element = cutlass::bfloat16_t;
using ElementAcc = float;

/**
 * Flash Attention configuration for diffusion models
 * Template parameters allow compile-time specialization
 */
template<int HeadDim, int TileQ = 64, int TileKV = 64>
struct FlashAttentionConfig {
    static constexpr int kHeadDim = HeadDim;
    static constexpr int kTileQ = TileQ;
    static constexpr int kTileKV = TileKV;
    
    // MMA operation for Xe BF16
    using MMAOp = cute::XE_8x16x16_F32BF16BF16F32_TT;
    
    // Tile shapes
    using TileShapeQK = Shape<Int<TileQ>, Int<TileKV>, Int<HeadDim>>;
    using TileShapePV = Shape<Int<TileQ>, _32, Int<TileKV>>;
    using TileShapeO  = Shape<Int<TileQ>, Int<HeadDim>>;
    
    // Subgroup layout (4 subgroups per workgroup)
    using SGLayout = Layout<Shape<_2, _2, _1>>;
    
    // Tiled MMA
    using TiledMMAQK = decltype(cute::make_tiled_mma(
        MMAOp{}, TileShapeQK{}, SGLayout{}));
    using TiledMMAPV = decltype(cute::make_tiled_mma(
        MMAOp{}, TileShapePV{}, SGLayout{}));
};

/**
 * Main Flash Attention operator class
 */
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
        float softmax_scale = 0.0f;  // 0 means auto-compute 1/sqrt(head_dim)
    };
    
    struct VarlenConfig {
        int batch_size;
        int num_heads_q;
        int num_heads_kv;
        int max_seqlen_qo;
        int max_seqlen_kv;
        int head_size_qk;
        int head_size_vo;
        const int* cu_seqlens_qo;  // [B+1] cumulative sequence lengths
        const int* cu_seqlens_kv;  // [B+1] cumulative sequence lengths
        bool causal = true;
        float softmax_scale = 0.0f;
    };

    /**
     * Forward pass - fixed length sequences
     * 
     * @param queue SYCL queue
     * @param query Query tensor [B, Hq, Sq, D] - BF16
     * @param key Key tensor [B, Hkv, Sk, D] - BF16
     * @param value Value tensor [B, Hkv, Sk, Dvo] - BF16
     * @param output Output tensor [B, Hq, Sq, Dvo] - BF16
     * @param config Attention configuration
     */
    static void forward(
        sycl::queue& queue,
        const Element* query,
        const Element* key,
        const Element* value,
        Element* output,
        const Config& config
    );
    
    /**
     * Forward pass - variable length sequences (for packed video data)
     * 
     * @param queue SYCL queue
     * @param query Query tensor [total_tokens, Hq, D] - BF16
     * @param key Key tensor [total_tokens, Hkv, D] - BF16
     * @param value Value tensor [total_tokens, Hkv, Dvo] - BF16
     * @param output Output tensor [total_tokens, Hq, Dvo] - BF16
     * @param config Variable length configuration
     */
    static void forward_varlen(
        sycl::queue& queue,
        const Element* query,
        const Element* key,
        const Element* value,
        Element* output,
        const VarlenConfig& config
    );

private:
    // Helper to compute default softmax scale
    static float compute_softmax_scale(int head_dim, float scale) {
        if (scale == 0.0f) {
            return 1.0f / std::sqrt(static_cast<float>(head_dim));
        }
        return scale;
    }
};

// ============================================================================
// Template-based kernel wrapper for compile-time optimization
// ============================================================================

template<typename Config>
class FlashAttentionKernel {
public:
    using ProblemShape = cutlass::fmha::kernel::FMHAProblemShape<false>;
    using VarlenProblemShape = cutlass::fmha::kernel::FMHAProblemShape<true>;
    
    // Define tensor types
    using TensorQ = decltype(make_tensor(
        make_gmem_ptr(std::declval<const Element*>()),
        make_layout(make_shape(0, 0, 0, 0), make_stride(0, 0, 0, 0))
    ));
    using TensorK = TensorQ;
    using TensorV = TensorQ;
    using TensorO = decltype(make_tensor(
        make_gmem_ptr(std::declval<Element*>()),
        make_layout(make_shape(0, 0, 0, 0), make_stride(0, 0, 0, 0))
    ));
    using TensorScale = decltype(make_tensor(
        make_gmem_ptr(std::declval<const float*>()),
        make_layout(make_shape(0, 0, 0, 0), make_stride(0, 0, 0, 0))
    ));
    
    // Stride types
    using StrideQ = decltype(stride(TensorQ{}));
    using StrideK = decltype(stride(TensorK{}));
    using StrideV = decltype(stride(TensorV{}));
    using StrideO = decltype(stride(TensorO{}));
    
    // Collective mainloop
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        cutlass::fmha::XeDefault<2>,   // 2-stage pipeline
        true,                           // Causal mask
        false,                          // UseScale (false for BF16)
        false,                          // F8kvF16mma
        false,                          // CachedKV
        false,                          // PagedKV
        typename Config::TiledMMAQK,
        typename Config::TiledMMAPV,
        1,                              // VTiles
        TensorQ, TensorK, TensorV,
        TensorScale, TensorScale, TensorScale,  // Scale tensors (unused)
        TensorK, TensorV,               // Cache tensors (unused)
        void, void, void, void, void   // Default tiled copies
    >;
    
    // Collective epilogue
    using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogue<
        typename Config::TiledMMAPV,
        typename Config::TileShapeO,
        TensorO
    >;
    
    // Kernel type
    using Kernel = cutlass::fmha::kernel::XeFMHAFwdKernel<
        ProblemShape,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::fmha::kernel::XeFHMAIndividualTileScheduler
    >;
    
    // Variable length kernel
    using VarlenKernel = cutlass::fmha::kernel::XeFMHAFwdKernel<
        VarlenProblemShape,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::fmha::kernel::XeFHMAIndividualTileScheduler
    >;
};

// ============================================================================
// Common configurations for diffusion models
// ============================================================================

using FAConfig64 = FlashAttentionConfig<64>;
using FAConfig128 = FlashAttentionConfig<128>;

} // namespace turbodiffusion
