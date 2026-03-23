// Fully Optimized - batch_norm
// Version: V5
// No description

#include <sycl/sycl.hpp>
#include <cmath>

namespace lczero {
namespace sycldnn_backend {

enum ActivationFunction {
    ACTIVATION_NONE, ACTIVATION_RELU, ACTIVATION_TANH,
    ACTIVATION_SIGMOID, ACTIVATION_SELU, ACTIVATION_SWISH, ACTIVATION_MISH
};

inline float activate(float val, ActivationFunction act) {
    switch (act) {
        case ACTIVATION_RELU: return val > 0 ? val : 0;
        case ACTIVATION_TANH: return sycl::tanh(val);
        case ACTIVATION_SIGMOID: return 1.0f / (1.0f + sycl::exp(-val));
        default: return val;
    }
}

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void batchNorm_kernel(T* output, const T* input, const T* skipInput,
                      int N, int C, int H, int W,
                      const float* means, const float* varMultipliers,
                      ActivationFunction activation,
                      const sycl::nd_item<1> &item,
                      sycl::local_accessor<float, 1> local_means,
                      sycl::local_accessor<float, 1> local_vars) {
    int tid = item.get_local_id(0);
    int total_elements = N * C * H * W;
    
    // Cache mean/variance in SLM (cooperative load)
    for (int i = tid; i < C; i += 512) {
        local_means[i] = means[i];
        local_vars[i] = varMultipliers[i];
    }
    item.barrier(sycl::access::fence_space::local_space);
    
    int idx = item.get_global_id(0);
    if (idx < total_elements) {
        int wIndex = (sizeof(T) == sizeof(float)) ? 
            ((idx / (H * W)) % C) : (idx % C);
        
        float el = (float)input[idx];
        el -= local_means[wIndex];
        el *= local_vars[wIndex];
        if (skipInput) el += (float)skipInput[idx];
        el = activate(el, activation);
        output[idx] = (T)el;
    }
}

template <typename T>
void batchNorm(T* output, const T* input, const T* skipInput,
               int N, int C, int H, int W,
               const float* means, const float* var_multipliers,
               ActivationFunction activation, sycl::queue &queue) {
    constexpr int kBlockSize = 512;
    int total_elements = N * C * H * W;
    int blocks = DivUp(total_elements, kBlockSize);
    
    queue.parallel_for(
        sycl::nd_range<1>(blocks * kBlockSize, kBlockSize),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            sycl::local_accessor<float, 1> local_means(C, item);
            sycl::local_accessor<float, 1> local_vars(C, item);
            batchNorm_kernel(output, input, skipInput, N, C, H, W,
                           means, var_multipliers, activation,
                           item, local_means, local_vars);
        }
    );
}

} // namespace sycldnn_backend
} // namespace lczero
