#include <algorithm>
#include "include/sycl_standard_header.h"

#include <cmath>
#include <sycl/sycl.hpp>


// ActivationFunction enum
// Common constants
static constexpr int kNumOutputPolicy = 1858;
static constexpr int kMaxResBlockFusingChannels = 384;
static constexpr int kMaxResBlockFusingSeKFp16Ampere = 512;
static constexpr int kMaxResBlockFusingSeK = 128;
static constexpr int kInputPlanes = 112;
static constexpr int kOpInpTransformBlockSize = 64;

// Helper functions
inline int DivUp(int a, int b) { return (a + b - 1) / b; }


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


namespace lczero {
namespace sycldnn_backend {

template <typename T, int kWorkPerThread>
 void genOffsetPointers_kernel(T** offsets, int heads, int block_size,
                                         int depth, int d_model, T* k, T* q,
                                         T* b1, T* v, T* b2) {
  const int i = (item.get_group(0) * item.get_local_range(0) + item.get_local_id(0)) * kWorkPerThread;
  if (i >= block_size) return;
  const int h = i % heads;
  const int n = i / heads;
  int w;
  T* res[kWorkPerThread];
  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = k + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = q + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w + block_size] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = b1 + i * 64 * 64 + w * 64 * 64;
    offsets[i + w + 2 * block_size] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = v + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w + 3 * block_size] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = b2 + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w + 4 * block_size] = res[w];
  }
}

template <typename T>
void genOffsetPointers(T** offsets, int heads, int max_batch, int depth,
                       int d_model, T* k, T* q, T* b1, T* v, T* b2,
                       cudaStream_t stream) {
  const int block_size = heads * max_batch;
  // Process two elements per thread to use 128 bit store instructions.
  constexpr int kWorkPerThread = 2;
  constexpr int kWorkGroupSize = 128;
  if (block_size % kWorkPerThread != 0) {
    // Handle odd block sizes.
    int grid = DivUp(block_size, kWorkGroupSize);
    genOffsetPointers_kernel<T, 1><<<grid, kWorkGroupSize, 0, stream>>>(
        offsets, heads, block_size, depth, d_model, k, q, b1, v, b2);
  } else {
    // Handle even block size
    int grid = DivUp(block_size, kWorkGroupSize * kWorkPerThread);
    genOffsetPointers_kernel<T, kWorkPerThread>
        <<<grid, kWorkGroupSize, 0, stream>>>(offsets, heads, block_size, depth,
                                              d_model, k, q, b1, v, b2);
  }
}

}  // namespace sycldnn_backend
}  // namespace lczero
