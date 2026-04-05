// 4-Wide Vectorization - softmax
// Version: V4
// No description

#include <sycl/sycl.hpp>
#include <cmath>

namespace lczero {
namespace sycldnn_backend {


inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void softmax_kernel(T* output, const T* input, int N, int C,
                    const sycl::nd_item<1> &item) {
    int n = item.get_group(0);
    int tid = item.get_local_id(0);
    
    if (n >= N) return;
    
    // Find max for numerical stability
    float max_val = -1e20f;
    for (int c = tid; c < C; c += item.get_local_range(0)) {
        max_val = sycl::max(max_val, (float)input[n * C + c]);
    }
    
    // Sub-group reduction for max
    auto sg = item.get_sub_group();
    for (int offset = 16 / 2; offset > 0; offset /= 2) {
        max_val = sycl::max(max_val, sycl::permute_group_by_xor(sg, max_val, offset));
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int c = tid; c < C; c += item.get_local_range(0)) {
        float val = sycl::exp((float)input[n * C + c] - max_val);
        output[n * C + c] = (T)val;
        sum += val;
    }
    
    // Sub-group reduction for sum
    for (int offset = 16 / 2; offset > 0; offset /= 2) {
        sum += sycl::permute_group_by_xor(sg, sum, offset);
    }
    
    // Normalize
    for (int c = tid; c < C; c += item.get_local_range(0)) {
        output[n * C + c] = (T)((float)output[n * C + c] / sum);
    }
}

template <typename T>
void softmax(T* output, const T* input, int N, int C, sycl::queue &queue) {
    queue.parallel_for(
        sycl::nd_range<1>(N * 512, 512),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            softmax_kernel(output, input, N, C, item);
        }
    );
}

} // namespace sycldnn_backend
} // namespace lczero
