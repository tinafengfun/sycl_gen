// Baseline - global_avg_pool
// Version: V0
// No description

#include <sycl/sycl.hpp>
#include <cmath>

namespace lczero {
namespace sycldnn_backend {


inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void globalAvgPool_kernel(T* output, const T* input, int N, int C,
                          const sycl::nd_item<1> &item) {
    int nc = item.get_global_id(0);
    if (nc >= N * C) return;
    
    // Sum 64 elements (8x8 plane)
    float sum = 0.0f;
    for (int i = 0; i < 64; ++i) {
        sum += (float)input[nc * 64 + i];
    }
    output[nc] = (T)(sum / 64.0f);
}

template <typename T>
void globalAvgPool(T* output, const T* input, int N, int C, sycl::queue &queue) {
    int total = N * C;
    int blocks = DivUp(total, 256);
    
    queue.parallel_for(
        sycl::nd_range<1>(blocks * 256, 256),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            globalAvgPool_kernel(output, input, N, C, item);
        }
    );
}

} // namespace sycldnn_backend
} // namespace lczero
