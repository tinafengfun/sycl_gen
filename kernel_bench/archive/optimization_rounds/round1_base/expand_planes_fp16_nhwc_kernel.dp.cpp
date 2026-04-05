#include <sycl/sycl.hpp>
#include "include/sycl_standard_header.h"


namespace lczero {
namespace sycldnn_backend {

namespace {
constexpr int kInputPlanes = 112;
}

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void expandPlanes_NHWC(T* output, const uint64_t* masks, const T* values, int n,
                       sycl::queue& stream) {
  int threads = n * 8 * 8;
  const int kBlockSize = 256;
  int blocks = DivUp(threads, kBlockSize);

  stream.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize), sycl::range<1>(kBlockSize)),
        [=](sycl::nd_item<1> item) {
          const int index = item.get_global_id(0);
          if (index >= n * 8 * 8) return;

          const int planeIndex = index % kInputPlanes;
          const int boardIndex = index / (kInputPlanes * 8 * 8);
          const int sqIndex = (index / kInputPlanes) & 0x3F;

          uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];

          T op = 0;
          bool set = !!(mask & (1ull << sqIndex));
          if (set) {
            op = values[boardIndex * kInputPlanes + planeIndex];
          }
          output[index] = op;
        });
  });
  stream.wait_and_throw();
}

}  // namespace cudnn_backend
}  // namespace lczero