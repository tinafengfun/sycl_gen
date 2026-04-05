// Sub-Group 16 - winograd_input_transform
// Version: V2
// No description

#include <sycl/sycl.hpp>
#include <cmath>

namespace lczero {
namespace sycldnn_backend {


inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// Simplified Winograd input transform (4x4 tile)
template <typename T>
void winogradInputTransform_kernel(T* output, const T* input, 
                                   int N, int C, int H, int W,
                                   const sycl::nd_item<2> &item) {
    int c = item.get_global_id(1);
    int n = item.get_global_id(0);
    
    if (c >= C || n >= N) return;
    
    // Process 8x8 input tile -> 6x6 transformed tile
    // Simplified: just copy for now
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            int in_idx = ((n * C + c) * H + y) * W + x;
            int out_idx = ((n * C + c) * 8 + y) * 8 + x;
            output[out_idx] = input[in_idx];
        }
    }
}

template <typename T>
void winogradInputTransform(T* output, const T* input, int N, int C, int H, int W,
                            sycl::queue &queue) {
    sycl::range<2> global(N, C);
    sycl::range<2> local(1, 512);
    
    queue.parallel_for(
        sycl::nd_range<2>(global, local),
        [=](sycl::nd_item<2> item) {
            winogradInputTransform_kernel(output, input, N, C, H, W, item);
        }
    );
}

} // namespace sycldnn_backend
} // namespace lczero
