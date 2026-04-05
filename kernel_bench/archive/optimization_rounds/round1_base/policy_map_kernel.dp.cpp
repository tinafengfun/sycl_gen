/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2024 The LCZero Authors
  Copyright (C) 2023 Intel Corporation

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
   
  SPDX-License-Identifier:GNU General Public License v3.0 or later
*/

// PolicyMap Kernel - SYCL Implementation
// Extracted from src/neural/backends/sycl/common_kernels.dp.cpp (lines 980-1015, 1817-1823)

#include <sycl/sycl.hpp>
#include "include/sycl_standard_header.h"


namespace lczero {
namespace sycldnn_backend {

// Helper function: divide rounding up
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void policyMap_kernel(T* output, const T* input,
                      const short* indices, int N, int inputSize,
                      int usedSize, int outputSize,
                      const sycl::nd_item<3> &item_ct1) {
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  int n = tid / usedSize;
  int i = tid % usedSize;

  if (n >= N) return;

  int j = indices[i];

  if (j >= 0) {
    output[n * outputSize + j] = input[n * inputSize + i];
  }
}

template <typename T>
void PolicyMap(int N, T* output, const T* input, const short* indices,
               int inputSize, int usedSize, int outputSize, sycl::queue &sycl_queue) {
  // Each thread processes one input element
  // Only some of the threads (with valid mapping) write output
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * usedSize, kBlockSize);

  sycl_queue.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, kBlocks) *
                                             sycl::range<3>(1, 1, kBlockSize),
                                         sycl::range<3>(1, 1, kBlockSize)),
                       [=](sycl::nd_item<3> item_ct1) {
                         policyMap_kernel<T>((T*)output, (T*)input,
                                             (short*)indices, N, inputSize,
                                             usedSize, outputSize, item_ct1);
                       });
}

// Template instantiations
template void PolicyMap<float>(int N, float* output, const float* input,
                               const short* indices, int inputSize,
                               int usedSize, int outputSize, sycl::queue &sycl_queue);

template void PolicyMap<sycl::half>(int N, sycl::half* output, const sycl::half* input,
                                    const short* indices, int inputSize, int usedSize,
                                    int outputSize, sycl::queue &sycl_queue);

}  // namespace sycldnn_backend
}  // namespace lczero
