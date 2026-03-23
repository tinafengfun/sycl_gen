// Sub-Group 16 - add_vectors
// Version: V2
// Explicit sub-group size 16

#include <sycl/sycl.hpp>
#include <cmath>

namespace lczero {
namespace sycldnn_backend {

enum ActivationFunction {
    ACTIVATION_NONE,
    ACTIVATION_RELU,
    ACTIVATION_RELU_2,
    ACTIVATION_TANH,
    ACTIVATION_SIGMOID,
    ACTIVATION_SELU,
    ACTIVATION_SWISH,
    ACTIVATION_MISH
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
void addVectors_kernel(T* c, const T* a, const T* b, int size,
                       ActivationFunction activation,
                       const sycl::nd_item<1> &item) {
    int i = item.get_global_id(0);
    if (i < size) {
        float aVal = a ? (float)a[i] : 0.0f;
        float bVal = b ? (float)b[i] : 0.0f;
        c[i] = (T)activate(aVal + bVal, activation);
    }
}

template <typename T>
void addVectors(T* c, T* a, T* b, int size,
                ActivationFunction activation, sycl::queue &queue) {
    constexpr int kBlockSize = 512;
    int blocks = DivUp(size / 1, kBlockSize);
    
    queue.parallel_for(
        sycl::nd_range<1>(blocks * kBlockSize, kBlockSize),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            addVectors_kernel(c, a, b, size, activation, item);
        }
    );
}

} // namespace sycldnn_backend
} // namespace lczero
