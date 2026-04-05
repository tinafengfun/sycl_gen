/**
 * LayerNorm SYCL Implementation
 * 
 * LayerNorm 公式: y = gamma * (x - mean) / sqrt(var + eps) + beta
 */

#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <cstdint>
#include <cmath>

namespace turbodiffusion {
namespace sycl_backend {

/**
 * Basic FP32 LayerNorm - Collaborative reduction with SLM
 */
class LayerNormFP32 {
public:
  static void launch(sycl::queue& q, const float* input, 
                     const float* gamma, const float* beta,
                     float* output, float eps, int64_t m, int64_t n) {
    const int WG_SIZE = 256;
    
    q.submit([&](sycl::handler& h) {
      sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(WG_SIZE * 2), h);
      
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(m * WG_SIZE), sycl::range<1>(WG_SIZE)),
        [=](sycl::nd_item<1> item) {
          int64_t row = item.get_group(0);
          int lid = item.get_local_id(0);
          
          if (row >= m) return;
          
          float* shared_sum = shared_mem.get_pointer();
          float* shared_sum_sq = shared_mem.get_pointer() + WG_SIZE;
          
          // Step 1: Compute partial sums
          float thread_sum = 0.0f;
          float thread_sum_sq = 0.0f;
          
          for (int64_t i = lid; i < n; i += WG_SIZE) {
            float val = input[row * n + i];
            thread_sum += val;
            thread_sum_sq += val * val;
          }
          
          shared_sum[lid] = thread_sum;
          shared_sum_sq[lid] = thread_sum_sq;
          item.barrier();
          
          // Step 2: Tree reduction for sum
          #pragma unroll
          for (int stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
              shared_sum[lid] += shared_sum[lid + stride];
              shared_sum_sq[lid] += shared_sum_sq[lid + stride];
            }
            item.barrier();
          }
          
          // Step 3: Compute mean and variance
          float mean = shared_sum[0] / n;
          float var = (shared_sum_sq[0] / n) - (mean * mean);
          float inv_std = sycl::rsqrt(var + eps);
          
          // Step 4: Normalize, scale and shift
          for (int64_t i = lid; i < n; i += WG_SIZE) {
            float val = input[row * n + i];
            float normalized = (val - mean) * inv_std;
            output[row * n + i] = gamma[i] * normalized + beta[i];
          }
        }
      );
    });
    q.wait();
  }
};

/**
 * BF16 LayerNorm - 2x memory bandwidth
 * Input/Output: BF16, Gamma/Beta: BF16, Compute: FP32
 */
class LayerNormBF16 {
public:
  using bfloat16 = sycl::ext::oneapi::bfloat16;
  
  static void launch(sycl::queue& q, const bfloat16* input,
                     const bfloat16* gamma, const bfloat16* beta,
                     bfloat16* output, float eps, int64_t m, int64_t n) {
    const int WG_SIZE = 256;
    
    q.submit([&](sycl::handler& h) {
      sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(WG_SIZE * 2), h);
      
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(m * WG_SIZE), sycl::range<1>(WG_SIZE)),
        [=](sycl::nd_item<1> item) {
          int64_t row = item.get_group(0);
          int lid = item.get_local_id(0);
          
          if (row >= m) return;
          
          float* shared_sum = shared_mem.get_pointer();
          float* shared_sum_sq = shared_mem.get_pointer() + WG_SIZE;
          
          // Step 1: Compute partial sums
          float thread_sum = 0.0f;
          float thread_sum_sq = 0.0f;
          
          for (int64_t i = lid; i < n; i += WG_SIZE) {
            float val = static_cast<float>(input[row * n + i]);
            thread_sum += val;
            thread_sum_sq += val * val;
          }
          
          shared_sum[lid] = thread_sum;
          shared_sum_sq[lid] = thread_sum_sq;
          item.barrier();
          
          // Step 2: Tree reduction
          #pragma unroll
          for (int stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
              shared_sum[lid] += shared_sum[lid + stride];
              shared_sum_sq[lid] += shared_sum_sq[lid + stride];
            }
            item.barrier();
          }
          
          // Step 3: Compute statistics
          float mean = shared_sum[0] / n;
          float var = (shared_sum_sq[0] / n) - (mean * mean);
          float inv_std = sycl::rsqrt(var + eps);
          
          // Step 4: Normalize, scale, shift, and write BF16
          for (int64_t i = lid; i < n; i += WG_SIZE) {
            float val = static_cast<float>(input[row * n + i]);
            float g = static_cast<float>(gamma[i]);
            float b = static_cast<float>(beta[i]);
            float normalized = (val - mean) * inv_std;
            output[row * n + i] = static_cast<bfloat16>(g * normalized + b);
          }
        }
      );
    });
    q.wait();
  }
};

/**
 * Work-group tuned LayerNorm - Test different WG sizes
 */
template <int WG_SIZE>
class LayerNormTuned {
public:
  static void launch(sycl::queue& q, const float* input,
                     const float* gamma, const float* beta,
                     float* output, float eps, int64_t m, int64_t n) {
    q.submit([&](sycl::handler& h) {
      sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(WG_SIZE * 2), h);
      
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(m * WG_SIZE), sycl::range<1>(WG_SIZE)),
        [=](sycl::nd_item<1> item) {
          int64_t row = item.get_group(0);
          int lid = item.get_local_id(0);
          
          if (row >= m) return;
          
          float* shared_sum = shared_mem.get_pointer();
          float* shared_sum_sq = shared_mem.get_pointer() + WG_SIZE;
          
          // Compute partial sums
          float thread_sum = 0.0f;
          float thread_sum_sq = 0.0f;
          
          for (int64_t i = lid; i < n; i += WG_SIZE) {
            float val = input[row * n + i];
            thread_sum += val;
            thread_sum_sq += val * val;
          }
          
          shared_sum[lid] = thread_sum;
          shared_sum_sq[lid] = thread_sum_sq;
          item.barrier();
          
          // Tree reduction
          #pragma unroll
          for (int stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
              shared_sum[lid] += shared_sum[lid + stride];
              shared_sum_sq[lid] += shared_sum_sq[lid + stride];
            }
            item.barrier();
          }
          
          // Compute mean and variance
          float mean = shared_sum[0] / n;
          float var = (shared_sum_sq[0] / n) - (mean * mean);
          float inv_std = sycl::rsqrt(var + eps);
          
          // Normalize
          for (int64_t i = lid; i < n; i += WG_SIZE) {
            float val = input[row * n + i];
            float normalized = (val - mean) * inv_std;
            output[row * n + i] = gamma[i] * normalized + beta[i];
          }
        }
      );
    });
    q.wait();
  }
};

/**
 * BF16 Work-group tuned LayerNorm
 */
template <int WG_SIZE>
class LayerNormBF16Tuned {
public:
  using bfloat16 = sycl::ext::oneapi::bfloat16;
  
  static void launch(sycl::queue& q, const bfloat16* input,
                     const bfloat16* gamma, const bfloat16* beta,
                     bfloat16* output, float eps, int64_t m, int64_t n) {
    q.submit([&](sycl::handler& h) {
      sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(WG_SIZE * 2), h);
      
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(m * WG_SIZE), sycl::range<1>(WG_SIZE)),
        [=](sycl::nd_item<1> item) {
          int64_t row = item.get_group(0);
          int lid = item.get_local_id(0);
          
          if (row >= m) return;
          
          float* shared_sum = shared_mem.get_pointer();
          float* shared_sum_sq = shared_mem.get_pointer() + WG_SIZE;
          
          // Compute partial sums
          float thread_sum = 0.0f;
          float thread_sum_sq = 0.0f;
          
          for (int64_t i = lid; i < n; i += WG_SIZE) {
            float val = static_cast<float>(input[row * n + i]);
            thread_sum += val;
            thread_sum_sq += val * val;
          }
          
          shared_sum[lid] = thread_sum;
          shared_sum_sq[lid] = thread_sum_sq;
          item.barrier();
          
          // Tree reduction
          #pragma unroll
          for (int stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
              shared_sum[lid] += shared_sum[lid + stride];
              shared_sum_sq[lid] += shared_sum_sq[lid + stride];
            }
            item.barrier();
          }
          
          // Compute statistics
          float mean = shared_sum[0] / n;
          float var = (shared_sum_sq[0] / n) - (mean * mean);
          float inv_std = sycl::rsqrt(var + eps);
          
          // Normalize and write
          for (int64_t i = lid; i < n; i += WG_SIZE) {
            float val = static_cast<float>(input[row * n + i]);
            float g = static_cast<float>(gamma[i]);
            float b = static_cast<float>(beta[i]);
            float normalized = (val - mean) * inv_std;
            output[row * n + i] = static_cast<bfloat16>(g * normalized + b);
          }
        }
      );
    });
    q.wait();
  }
};

// Convenience interfaces
inline void layernorm_fp32(sycl::queue& q, const float* input,
                           const float* gamma, const float* beta,
                           float* output, float eps, int64_t m, int64_t n) {
  LayerNormFP32::launch(q, input, gamma, beta, output, eps, m, n);
}

inline void layernorm_bf16(sycl::queue& q, 
                           const sycl::ext::oneapi::bfloat16* input,
                           const sycl::ext::oneapi::bfloat16* gamma,
                           const sycl::ext::oneapi::bfloat16* beta,
                           sycl::ext::oneapi::bfloat16* output,
                           float eps, int64_t m, int64_t n) {
  LayerNormBF16::launch(q, input, gamma, beta, output, eps, m, n);
}

template <int WG_SIZE>
inline void layernorm_tuned(sycl::queue& q, const float* input,
                            const float* gamma, const float* beta,
                            float* output, float eps, int64_t m, int64_t n) {
  LayerNormTuned<WG_SIZE>::launch(q, input, gamma, beta, output, eps, m, n);
}

template <int WG_SIZE>
inline void layernorm_bf16_tuned(sycl::queue& q,
                                 const sycl::ext::oneapi::bfloat16* input,
                                 const sycl::ext::oneapi::bfloat16* gamma,
                                 const sycl::ext::oneapi::bfloat16* beta,
                                 sycl::ext::oneapi::bfloat16* output,
                                 float eps, int64_t m, int64_t n) {
  LayerNormBF16Tuned<WG_SIZE>::launch(q, input, gamma, beta, output, eps, m, n);
}

} // namespace sycl_backend
} // namespace turbodiffusion
