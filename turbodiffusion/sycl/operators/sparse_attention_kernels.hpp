#ifndef SPARSE_ATTENTION_KERNELS_HPP
#define SPARSE_ATTENTION_KERNELS_HPP

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

namespace turbodiffusion {

using bfloat16 = sycl::ext::oneapi::bfloat16;

// Forward pass kernel
void sparse_attn_fwd_kernel(
    sycl::queue& q,
    const bfloat16* Q, const bfloat16* K, const bfloat16* V,
    float softmax_scale,
    const int* LUT,
    bfloat16* O, float* L, float* M,
    int B, int H, int M_len, int N_len, int D,
    int BLOCK_M, int BLOCK_N, int topk);

// Backward preprocess kernel
void sparse_attn_bwd_preprocess_kernel(
    sycl::queue& q,
    const bfloat16* O, const bfloat16* dO,
    float* Delta,
    int B, int H, int L, int D,
    int BLOCK_M);

// Backward dQ kernel
void sparse_attn_bwd_dq_kernel(
    sycl::queue& q,
    const bfloat16* Q, const bfloat16* K, const bfloat16* V,
    const bfloat16* O, const bfloat16* dO,
    const int* LUT,
    const float* Delta, const float* L, const float* M,
    bfloat16* dQ,
    float qk_scale,
    int B, int H, int M_len, int N_len, int D,
    int BLOCK_M, int BLOCK_N, int topk);

// Backward dK/dV kernel
void sparse_attn_bwd_dkdv_kernel(
    sycl::queue& q,
    const bfloat16* Q, const bfloat16* K, const bfloat16* V,
    const bfloat16* dO,
    const int8_t* KBID,
    const float* Delta, const float* L, const float* M,
    bfloat16* dK, bfloat16* dV,
    float qk_scale,
    int B, int H, int M_len, int N_len, int D,
    int BLOCK_M, int BLOCK_N, int BLOCK_SLICE_FACTOR);

} // namespace turbodiffusion

#endif // SPARSE_ATTENTION_KERNELS_HPP