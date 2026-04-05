#include <sycl/sycl.hpp>
#include "include/sycl_standard_header.h"

#include <algorithm>
#include <cassert>

namespace lczero {
namespace sycldnn_backend {

template <typename T>
void addVectorsHNC_NHC(T* a, T* b, int N, int H, int C, sycl::queue& queue) {
  const int kBlockSize = 256;
  int totalElements = N * H * C;
  int blocks = (totalElements + kBlockSize - 1) / kBlockSize;

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize), sycl::range<1>(kBlockSize)),
        [=](sycl::nd_item<1> item) {
          int i = item.get_global_id(0);
          if (i < totalElements) {
            int orig_i = i;
            int c = i % C;
            i /= C;
            int n = i % N;
            i /= N;
            int h = i;
            float aVal = static_cast<float>(a[orig_i]);
            float bVal = static_cast<float>(b[n * H * C + h * C + c]);
            float cVal = aVal + bVal;
            a[orig_i] = static_cast<T>(cVal);
          }
        });
  });
  queue.wait_and_throw();
}

template void addVectorsHNC_NHC<float>(float* a, float* b, int N, int H, int C,
                                       sycl::queue& queue);
template void addVectorsHNC_NHC<sycl::half>(sycl::half* a, sycl::half* b, int N, int H, int C,
                                      sycl::queue& queue);

}  // namespace cudnn_backend
}  // namespace lczero