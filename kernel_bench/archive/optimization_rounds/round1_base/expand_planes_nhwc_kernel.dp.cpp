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

void CudaError(int status, const char* file, const int& line);

#define ReportCUDAErrors(status) CudaError(status, __FILE__, __LINE__)

template <typename T>
 void expandPlanes_kernel_NHWC(T* output, const uint64_t* masks,
                                         const T* values, int n) {
  const int index = item.get_local_id(0) + item.get_local_range(0) * item.get_group(0);
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
}

template <typename T>
void expandPlanes_NHWC(T* output, const uint64_t* masks, const T* values, int n,
                       cudaStream_t stream) {
  int threads = n * 8 * 8;  // Each thread writes a single element.
  const int kBlockSize = 256;
  int blocks = DivUp(threads, kBlockSize);
  expandPlanes_kernel_NHWC<<<blocks, kBlockSize, 0, stream>>>(output, masks,
                                                              values, n);
  ReportCUDAErrors(cudaGetLastError());
}

}  // namespace sycldnn_backend
}  // namespace lczero
