#include <sycl/sycl.hpp>
#include "include/sycl_standard_header.h"


namespace lczero {
namespace sycldnn_backend {

// Host wrapper for NHWC fp16 global average pooling
void globalAvgPool_NHWC_fp16(int N, int C, sycl::half* output, const sycl::half* input,
                             const sycl::half* prevLayerBias, sycl::queue& queue) {
  const int kPlaneSize = 64;
  const int inputSize = N * C * kPlaneSize;
  const int outputSize = N * C;

  queue.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(N * C), sycl::range<1>(C)),
    [=](sycl::nd_item<1> item) {
      const int elementsPerThread = 64;

      int blockStart = item.get_group(0) * item.get_local_range(0);
      float S = 0;

      #pragma unroll
      for (int i = 0; i < elementsPerThread; i++) {
        int localIndex = i * item.get_local_range(0) + item.get_local_id(0);
        int inputIndex = blockStart * elementsPerThread + localIndex;
        if (inputIndex < inputSize) S += static_cast<float>(input[inputIndex]);
      }

      float avg = S / elementsPerThread;

      if (prevLayerBias) 
        avg += static_cast<float>(prevLayerBias[item.get_local_id(0)]);

      int opIndex = blockStart + item.get_local_id(0);
      if (opIndex < outputSize) 
        output[opIndex] = static_cast<sycl::half>(avg);
    });
  });
}

}  // namespace sycldnn_backend
}  // namespace lczero