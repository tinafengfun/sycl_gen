/**
 * RMSNorm SYCL Implementation - Working Version
 * 修复SLM分配问题
 */

#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cmath>

namespace turbodiffusion {
namespace sycl_backend {

template <typename InputDtype, typename OutputDtype, typename WeightDtype>
class RMSNormKernel {
public:
  struct Params {
    const InputDtype* input;
    const WeightDtype* weight;
    OutputDtype* output;
    float eps;
    int64_t m;
    int64_t n;
  };

  static void launch(sycl::queue& q, const Params& params) {
    const int64_t M = params.m;
    const int64_t N = params.n;
    const float eps = params.eps;
    const int wg_size = 256;
    
    // 使用sycl::local_accessor正确声明SLM
    q.submit([&](sycl::handler& h) {
      sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(wg_size), h);
      
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(M * wg_size), sycl::range<1>(wg_size)),
        [=](sycl::nd_item<1> item) {
          int64_t row = item.get_group(0);
          int lid = item.get_local_id(0);
          
          if (row >= M) return;
          
          // Step 1: Local sum of squares
          float thread_sum = 0.0f;
          for (int64_t i = lid; i < N; i += wg_size) {
            float val = static_cast<float>(params.input[row * N + i]);
            thread_sum += val * val;
          }
          shared_mem[lid] = thread_sum;
          
          // Step 2: Tree reduction
          item.barrier();
          #pragma unroll
          for (int stride = 128; stride > 0; stride >>= 1) {
            if (lid < stride) {
              shared_mem[lid] += shared_mem[lid + stride];
            }
            item.barrier();
          }
          
          // Step 3: Compute RMS and normalize
          float rms = sycl::sqrt(shared_mem[0] / N + eps);
          
          for (int64_t i = lid; i < N; i += wg_size) {
            float val = static_cast<float>(params.input[row * N + i]);
            float w = static_cast<float>(params.weight[i]);
            params.output[row * N + i] = static_cast<OutputDtype>(w * val / rms);
          }
        }
      );
    });
    q.wait();
  }
};

inline void rmsnorm_sycl(sycl::queue& q, const float* input, const float* weight,
                         float* output, float eps, int64_t m, int64_t n) {
  using Kernel = RMSNormKernel<float, float, float>;
  Kernel::Params params = {input, weight, output, eps, m, n};
  Kernel::launch(q, params);
}

} // namespace sycl_backend
} // namespace turbodiffusion