#include <sycl/sycl.hpp>
#include "include/sycl_standard_header.h"

#include <algorithm>  // for std::max

namespace lczero {
namespace sycldnn_backend {

// Helper function to divide and round up
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// SYCL原子操作实现float最大值（修复作用域和地址空间）
struct AtomicMaxFloat {
  float operator()(sycl::local_accessor<float, 1> local_acc,  // 修正为local_accessor
                   float val) const {
    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                     sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space>
        atomic_max(local_acc[0]);
    float old = atomic_max.load();
    while (val > old && !atomic_max.compare_exchange_strong(old, val))
      ;
    return old;
  }
};

// SYCL原子加法包装器（修复作用域和地址空间）
struct AtomicAddFloat {
  void operator()(sycl::local_accessor<float, 1> local_acc,  // 修正为local_accessor
                  float val) const {
    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                     sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space>
        atomic_sum(local_acc[0]);
    atomic_sum.fetch_add(val);
  }
};

namespace {
constexpr float kTwiceHalfMax = 131008.0f;
}  // namespace

template <typename T>
class SoftmaxKernel;

template <typename T>
void Softmax(int N, int C, T* output, const T* input, const T* input2,
             sycl::queue& queue) {
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> max_acc(sycl::range<1>(1), cgh);
    sycl::local_accessor<float, 1> sum_acc(sycl::range<1>(1), cgh);

    auto global_range = sycl::range<1>(N * C);
    auto local_range = sycl::range<1>(C);

    cgh.parallel_for<SoftmaxKernel<T>>(
        sycl::nd_range<1>(global_range, local_range),
        [=](sycl::nd_item<1> item) {
          const int n = item.get_group(0);
          const int c = item.get_local_id(0);
          const int index = n * C + c;

          float x = static_cast<float>(input[index]);
          if (input2 != nullptr) x += static_cast<float>(input2[index]);
          if (std::is_same<sycl::half, T>::value) {
            x = sycl::clamp(x, -kTwiceHalfMax, kTwiceHalfMax);
          }

          if (c == 0) {
            max_acc[0] = x;
            sum_acc[0] = 0.0f;
          }
          item.barrier(sycl::access::fence_space::local_space);

          auto sg = item.get_sub_group();
          const uint32_t subgroup_size = sg.get_local_range().size();  // 修正为uint32_t
          float warpmax = x;

          // 使用permute_group_by_xor实现子组归约（修正mask类型）
          for (uint32_t mask = subgroup_size / 2; mask > 0; mask >>= 1) {
            float tmp = sycl::permute_group_by_xor(sg, warpmax, mask);
            warpmax = sycl::max(warpmax, tmp);
          }

          if (sg.get_local_id() == 0) {
            AtomicMaxFloat()(max_acc, warpmax);
          }
          item.barrier(sycl::access::fence_space::local_space);

          float ex = sycl::exp(x - max_acc[0]);

          // 使用permute_group_by_xor实现子组求和（修正mask类型）
          float warp_sum = ex;
          for (uint32_t mask = subgroup_size / 2; mask > 0; mask >>= 1) {
            float tmp = sycl::permute_group_by_xor(sg, warp_sum, mask);
            warp_sum += tmp;
          }

          if (sg.get_local_id() == 0) {
            AtomicAddFloat()(sum_acc, warp_sum);
          }
          item.barrier(sycl::access::fence_space::local_space);

          output[index] = static_cast<T>(ex / sum_acc[0]);
        });
  });
  queue.wait_and_throw();
}

// 显式模板实例化
template void Softmax<float>(int N, int C, float* output, const float* input,
                             const float* input2, sycl::queue& queue);
template void Softmax<sycl::half>(int N, int C, sycl::half* output,
                                  const sycl::half* input,
                                  const sycl::half* input2, sycl::queue& queue);

}  // namespace cudnn_backend
}  // namespace lczero