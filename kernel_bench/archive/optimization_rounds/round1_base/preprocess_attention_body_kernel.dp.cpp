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

template <typename T>
 void preprocess_for_attention_body_kernel(
    T* output, const T* input, const T* encoding, int input_size,
    int encoding_size, bool is_pe_dense_embedding) {
  int n = item.get_group(0);
  int hw = item.get_group(1);
  int c = item.get_local_id(0);

  T op;
  if (c >= input_size) {
    // concatenate from position encoding array
    if (is_pe_dense_embedding) {
      op = (T)(encoding[n * 64 * encoding_size + hw * encoding_size +
                        (c - input_size)]);
    } else {
      op = (T)(encoding[64 * hw + (c - input_size)]);
    }
  } else {
    op = input[n * input_size * 64 + c * 64 + hw];  // nchw
  }

  int outputC = input_size + encoding_size;

  // convert to nhwc
  output[n * 64 * outputC + hw * outputC + c] = op;
}

template <typename T>
void inputPreprocessForAttentionBody(T* output, const T* input,
                                     const T* encoding, int N, int input_size,
                                     int encoding_size,
                                     bool is_pe_dense_embedding,
                                     cudaStream_t stream) {
  // N * 64 blocks
  // (kInputPlanes + kNumPosEncodingChannels) threads
  // Each thread computes a single output element
  dim3 gridSize = dim3(N, 64);
  int blockSize = input_size + encoding_size;
  preprocess_for_attention_body_kernel<T><<<gridSize, blockSize, 0, stream>>>(
      output, input, encoding, input_size, encoding_size,
      is_pe_dense_embedding);
}

}  // namespace sycldnn_backend
}  // namespace lczero
