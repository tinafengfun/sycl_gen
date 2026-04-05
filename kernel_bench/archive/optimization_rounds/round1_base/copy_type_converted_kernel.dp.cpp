/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include <sycl/sycl.hpp>
#include "include/sycl_standard_header.h"


namespace lczero {
namespace sycldnn_backend {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename DstType, typename SrcType>
struct copyTypeConverted_kernel {
  DstType* op;
  SrcType* ip;
  int N;

  void operator()(sycl::nd_item<1> item) const {
    int tid = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

    if (tid >= N) return;

    DstType el = static_cast<DstType>(ip[tid]);
    op[tid] = el;
  }
};

template <typename DstType, typename SrcType>
void copyTypeConverted(DstType* op, SrcType* ip, int N, sycl::queue& stream) {
  const int kBlockSize = 256;
  int blocks = DivUp(N, kBlockSize);
  
  stream.submit([&](sycl::handler& cgh) {
    copyTypeConverted_kernel<DstType, SrcType> kernel{op, ip, N};
    cgh.parallel_for(sycl::nd_range<1>(blocks * kBlockSize, kBlockSize), kernel);
  });
}

template void copyTypeConverted<sycl::half, float>(sycl::half* op, float* ip, int N,
                                              sycl::queue& stream);
template void copyTypeConverted<float, sycl::half>(float* op, sycl::half* ip, int N,
                                              sycl::queue& stream);
template void copyTypeConverted<float, float>(float* op, float* ip, int N,
                                               sycl::queue& stream);
template void copyTypeConverted<sycl::half, sycl::half>(sycl::half* op, sycl::half* ip, int N,
                                             sycl::queue& stream);

} // namespace sycldnn_backend
} // namespace lczero