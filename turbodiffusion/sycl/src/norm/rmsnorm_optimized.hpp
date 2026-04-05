/**
 * RMSNorm SYCL Optimized Implementation
 * 
 * Optimization strategies (based on XMX Optimizer Skill):
 * 1. Type C: Reduction - Use single-thread-per-output pattern
 * 2. BF16 support: 2x memory bandwidth, FP32 compute for accuracy
 * 3. Work-group tuning: Test 128/256/512
 * 4. Vectorized access via unrolling
 */

#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <cstdint>
#include <cmath>

namespace turbodiffusion {
namespace sycl_backend {

// Type C: Reduction - Single-thread-per-output (OPTIMAL per skill)
// Each thread computes complete RMSNorm for one row

/**
 * FP32 version - High precision reference
 */
class RMSNormFP32 {
public:
  static void launch(sycl::queue& q, const float* input, const float* weight,
                     float* output, float eps, int64_t m, int64_t n) {
    // Type C: 1 work-item per row (not per element)
    q.parallel_for(sycl::range<1>(m), [=](sycl::item<1> item) {
      int64_t row = item.get_id(0);
      if (row >= m) return;
      
      // Single thread computes entire row
      float sum_sq = 0.0f;
      
      // Scalar load with unrolling for optimization
      const int unroll = 4;
      int64_t i = 0;
      
      // Main loop: unrolled scalar processing
      int64_t main_loop_end = (n / unroll) * unroll;
      for (; i < main_loop_end; i += unroll) {
        #pragma unroll
        for (int j = 0; j < unroll; j++) {
          float val = input[row * n + i + j];
          sum_sq += val * val;
        }
      }
      
      // Remaining elements
      for (; i < n; i++) {
        float val = input[row * n + i];
        sum_sq += val * val;
      }
      
      // Compute RMS
      float rms = sycl::sqrt(sum_sq / n + eps);
      
      // Normalize and write back (unrolled)
      i = 0;
      for (; i < main_loop_end; i += unroll) {
        #pragma unroll
        for (int j = 0; j < unroll; j++) {
          float val = input[row * n + i + j];
          float w = weight[i + j];
          output[row * n + i + j] = w * val / rms;
        }
      }
      
      // Remaining elements
      for (; i < n; i++) {
        float val = input[row * n + i];
        float w = weight[i];
        output[row * n + i] = w * val / rms;
      }
    });
    q.wait();
  }
};

/**
 * BF16 version - 2x memory bandwidth
 * Input/Output: BF16, Compute: FP32
 */
class RMSNormBF16 {
public:
  using bfloat16 = sycl::ext::oneapi::bfloat16;
  
  static void launch(sycl::queue& q, const bfloat16* input, const bfloat16* weight,
                     bfloat16* output, float eps, int64_t m, int64_t n) {
    q.parallel_for(sycl::range<1>(m), [=](sycl::item<1> item) {
      int64_t row = item.get_id(0);
      if (row >= m) return;
      
      float sum_sq = 0.0f;
      
      // BF16 scalar load with unrolling
      const int unroll = 8;
      int64_t i = 0;
      
      int64_t main_loop_end = (n / unroll) * unroll;
      for (; i < main_loop_end; i += unroll) {
        #pragma unroll
        for (int j = 0; j < unroll; j++) {
          float val = static_cast<float>(input[row * n + i + j]);
          sum_sq += val * val;
        }
      }
      
      // Remaining elements
      for (; i < n; i++) {
        float val = static_cast<float>(input[row * n + i]);
        sum_sq += val * val;
      }
      
      float rms = sycl::sqrt(sum_sq / n + eps);
      
      // Normalize
      i = 0;
      for (; i < main_loop_end; i += unroll) {
        #pragma unroll
        for (int j = 0; j < unroll; j++) {
          float val = static_cast<float>(input[row * n + i + j]);
          float weight_val = static_cast<float>(weight[i + j]);
          output[row * n + i + j] = static_cast<bfloat16>(weight_val * val / rms);
        }
      }
      
      // Remaining elements
      for (; i < n; i++) {
        float val = static_cast<float>(input[row * n + i]);
        float w = static_cast<float>(weight[i]);
        output[row * n + i] = static_cast<bfloat16>(w * val / rms);
      }
    });
    q.wait();
  }
};

/**
 * Work-group tuning version for FP32 - Test different configurations
 * Traditional mode (collaborative reduction)
 */
template <int WG_SIZE>
class RMSNormTuned {
public:
  static void launch(sycl::queue& q, const float* input, const float* weight,
                     float* output, float eps, int64_t m, int64_t n) {
    q.submit([&](sycl::handler& h) {
      sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(WG_SIZE), h);
      
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(m * WG_SIZE), sycl::range<1>(WG_SIZE)),
        [=](sycl::nd_item<1> item) {
          int64_t row = item.get_group(0);
          int lid = item.get_local_id(0);
          
          if (row >= m) return;
          
          float thread_sum = 0.0f;
          for (int64_t i = lid; i < n; i += WG_SIZE) {
            float val = input[row * n + i];
            thread_sum += val * val;
          }
          shared_mem[lid] = thread_sum;
          
          item.barrier();
          
          // Tree reduction
          #pragma unroll
          for (int stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
              shared_mem[lid] += shared_mem[lid + stride];
            }
            item.barrier();
          }
          
          float rms = sycl::sqrt(shared_mem[0] / n + eps);
          
          for (int64_t i = lid; i < n; i += WG_SIZE) {
            float val = input[row * n + i];
            float w = weight[i];
            output[row * n + i] = w * val / rms;
          }
        }
      );
    });
    q.wait();
  }
};

/**
 * BF16 Work-group tuning version - Collaborative reduction
 * Input/Output: BF16, Compute: FP32, SLM: FP32
 */
template <int WG_SIZE>
class RMSNormBF16Tuned {
public:
  using bfloat16 = sycl::ext::oneapi::bfloat16;
  
  static void launch(sycl::queue& q, const bfloat16* input, const bfloat16* weight,
                     bfloat16* output, float eps, int64_t m, int64_t n) {
    q.submit([&](sycl::handler& h) {
      // Use FP32 for shared memory to maintain precision during reduction
      sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(WG_SIZE), h);
      
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(m * WG_SIZE), sycl::range<1>(WG_SIZE)),
        [=](sycl::nd_item<1> item) {
          int64_t row = item.get_group(0);
          int lid = item.get_local_id(0);
          
          if (row >= m) return;
          
          // Each thread computes partial sum of squares
          float thread_sum = 0.0f;
          for (int64_t i = lid; i < n; i += WG_SIZE) {
            float val = static_cast<float>(input[row * n + i]);
            thread_sum += val * val;
          }
          shared_mem[lid] = thread_sum;
          
          item.barrier();
          
          // Tree reduction in shared memory
          #pragma unroll
          for (int stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
              shared_mem[lid] += shared_mem[lid + stride];
            }
            item.barrier();
          }
          
          // Compute RMS
          float rms = sycl::sqrt(shared_mem[0] / n + eps);
          
          // Normalize and write back as BF16
          for (int64_t i = lid; i < n; i += WG_SIZE) {
            float val = static_cast<float>(input[row * n + i]);
            float w = static_cast<float>(weight[i]);
            output[row * n + i] = static_cast<bfloat16>(w * val / rms);
          }
        }
      );
    });
    q.wait();
  }
};

// Convenience interfaces
inline void rmsnorm_fp32(sycl::queue& q, const float* input, const float* weight,
                         float* output, float eps, int64_t m, int64_t n) {
  RMSNormFP32::launch(q, input, weight, output, eps, m, n);
}

inline void rmsnorm_bf16(sycl::queue& q, 
                         const sycl::ext::oneapi::bfloat16* input,
                         const sycl::ext::oneapi::bfloat16* weight,
                         sycl::ext::oneapi::bfloat16* output,
                         float eps, int64_t m, int64_t n) {
  RMSNormBF16::launch(q, input, weight, output, eps, m, n);
}

template <int WG_SIZE>
inline void rmsnorm_tuned(sycl::queue& q, const float* input, const float* weight,
                          float* output, float eps, int64_t m, int64_t n) {
  RMSNormTuned<WG_SIZE>::launch(q, input, weight, output, eps, m, n);
}

template <int WG_SIZE>
inline void rmsnorm_bf16_tuned(sycl::queue& q, 
                               const sycl::ext::oneapi::bfloat16* input,
                               const sycl::ext::oneapi::bfloat16* weight,
                               sycl::ext::oneapi::bfloat16* output,
                               float eps, int64_t m, int64_t n) {
  RMSNormBF16Tuned<WG_SIZE>::launch(q, input, weight, output, eps, m, n);
}

} // namespace sycl_backend
} // namespace turbodiffusion
