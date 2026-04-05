#include <sycl/sycl.hpp>
#include "include/sycl_standard_header.h"


namespace lczero {
namespace sycldnn_backend {

template <typename dT, typename sT>
dT readNCHW(const sT* input_tensor, int n, int c, int h, int w,
            int Nin, int Cin, int H, int W) {
  if (n >= Nin || c >= Cin) return 0;

  int index = n;
  index *= Cin;
  index += c;
  index *= H;
  index += h;
  index *= W;
  index += w;

  return static_cast<dT>(input_tensor[index]);
}

template <typename dT, typename sT>
void NCHWtoNHWC_kernel(dT* output_tensor, const sT* input_tensor,
                       int Nin, int Cin, int Nout, int Cout, int H, int W,
                       sycl::nd_item<1> item) {
  int tid = item.get_global_id(0);
  if (tid >= Nout * Cout * H * W) return;

  int index = tid;
  int c = (index % Cout);
  index /= Cout;
  int w = index % W;
  index /= W;
  int h = index % H;
  index /= H;
  int n = index;

  output_tensor[tid] = readNCHW<dT, sT>(input_tensor, n, c, h, w, Nin, Cin, H, W);
}

template <typename DstType, typename SrcType>
void convertNCHWtoNHWC(DstType* output_tensor, const SrcType* input_tensor,
                       int Nin, int Cin, int Nout, int Cout, int H, int W,
                       sycl::queue& queue) {
  size_t numElements = Nout * Cout * H * W;
  const int blockSize = 256;
  int blocks = (numElements + blockSize - 1) / blockSize;
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * blockSize), 
                        sycl::range<1>(blockSize)),
      [=](sycl::nd_item<1> item) {
        NCHWtoNHWC_kernel(output_tensor, input_tensor, 
                         Nin, Cin, Nout, Cout, H, W, item);
      });
}

template void convertNCHWtoNHWC<sycl::half, float>(sycl::half*, const float*, 
                                                  int, int, int, int, int, int,
                                                  sycl::queue&);
template void convertNCHWtoNHWC<float, float>(float*, const float*,
                                             int, int, int, int, int, int,
                                             sycl::queue&);
template void convertNCHWtoNHWC<sycl::half, sycl::half>(sycl::half*, const sycl::half*,
                                                       int, int, int, int, int, int,
                                                       sycl::queue&);

}  // namespace cudnn_backend
}  // namespace lczero