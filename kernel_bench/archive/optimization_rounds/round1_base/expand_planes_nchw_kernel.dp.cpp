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

namespace {
constexpr int kInputPlanes = 112;
}

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
struct expandPlanes_kernel_NCHW {
  T* output;
  const uint64_t* masks;
  const T* values;
  unsigned n;

  expandPlanes_kernel_NCHW(T* output_, const uint64_t* masks_, const T* values_, unsigned n_)
      : output(output_), masks(masks_), values(values_), n(n_) {}

  void operator()(sycl::nd_item<1> item) const {
    unsigned index = item.get_local_id(0) + item.get_local_range(0) * item.get_group(0);

    index *= 2;
    unsigned planeIndex = index >> 6;

    if (planeIndex >= n) return;

    uint64_t mask = masks[planeIndex];

    int sqIndex = index & 0x3F;
    T op[2] = {0, 0};

    bool set = !!(mask & (1ull << sqIndex));
    if (set) {
      op[0] = values[planeIndex];
    }
    sqIndex++;
    set = !!(mask & (1ull << sqIndex));
    if (set) {
      op[1] = values[planeIndex];
    }
    output[index + 0] = op[0];
    output[index + 1] = op[1];
  }
};

template <typename T>
void expandPlanes_NCHW(T* output, const uint64_t* masks, const T* values,
                       int n, sycl::queue& queue) {
  unsigned threads = n * 8 * 8 / 2;  // each thread writes two elements.
  const int blockSize = 256;
  unsigned blocks = DivUp(threads, blockSize);
  
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * blockSize), sycl::range<1>(blockSize)),
                     expandPlanes_kernel_NCHW<T>(output, masks, values, n));
  });
}

}  // namespace sycldnn_backend
}  // namespace lczero