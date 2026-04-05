// Sub-Group 16 - batch_norm
// Version: V2
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
                      const sycl::nd_item<1> &item) {
    int idx = item.get_global_id(0);
    int total_elements = N * C * H * W;
    
    if (idx < total_elements) {
        int wIndex = (sizeof(T) == sizeof(float)) ? 
            ((idx / (H * W)) % C) : (idx % C);
        
        float el = (float)input[idx];
        el -= means[wIndex];
        el *= varMultipliers[wIndex];
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
            batchNorm_kernel(output, input, skipInput, N, C, H, W,
                           means, var_multipliers, activation, item);
        }
    );
}

} // namespace sycldnn_backend
} // namespace lczero
