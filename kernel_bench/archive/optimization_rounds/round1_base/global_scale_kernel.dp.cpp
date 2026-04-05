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

#include <sycl/sycl.hpp>
#include "include/sycl_standard_header.h"

#include <algorithm>
#include <cassert>

namespace lczero {
namespace sycldnn_backend {

// Activation function enum
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

[[gnu::always_inline]]
inline float mishActivate(float el) {
  auto e = sycl::native::exp(el);
  auto n = e * e + 2.0f * e;
  auto d = el / (n + 2.0f);
  if (el <= -0.6f) {
    return n * d;
  } else {
    return el - 2.0f * d;
  }
}

[[gnu::always_inline]]
inline float activate(float cVal, ActivationFunction activation) {
  switch (activation) {
    case ACTIVATION_RELU:
      if (cVal < 0) cVal = 0;
      break;
    case ACTIVATION_RELU_2:
      if (cVal < 0) cVal = 0;
      cVal *= cVal;
      break;
    case ACTIVATION_TANH:
      cVal = sycl::tanh(cVal);
      break;
    case ACTIVATION_SIGMOID:
      cVal = 1.0f / (1.0f + sycl::exp(-cVal));
      break;
    case ACTIVATION_SELU:
      cVal = 1.0507f * (cVal >= 0 ? cVal : 1.67326f * (sycl::exp(cVal) - 1));
      break;
    case ACTIVATION_SWISH:
      cVal = cVal / (1.0f + sycl::exp(-cVal));
      break;
    case ACTIVATION_MISH:
      cVal = mishActivate(cVal);
      break;
    case ACTIVATION_NONE:
    case ACTIVATION_DEFAULT:
      break;
    case ACTIVATION_SOFTMAX:
      break;
  }
  return cVal;
}

// Kernel: globalScale_kernel
// Applies global scaling with sigmoid activation on scale parameters
// Each thread writes one output
template <typename T>
void globalScale_kernel(T* output, const T* input,
                        const T* scaleBias, const T* prevLayerBias,
                        int inputSize, int C,
                        ActivationFunction activation,
                        const sycl::nd_item<3> &item_ct1) {
  const int kPlaneSize = 64;

  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  if (tid > inputSize) return;

  int nc = tid / kPlaneSize;
  int n = nc / C;
  int c = nc % C;

  float val1 = input[tid];   // Output of residual block to be scaled.
  float val2 = output[tid];  // Skip connection to be added directly.

  if (prevLayerBias) {
    val1 += (float)(prevLayerBias[c]);
  }

  int startIdx = n * 2 * C;  // Scale and bias interleaved.

  float s = scaleBias[startIdx + c];
  s = 1.0f / (1.0f + sycl::exp(-s));  // Sigmoid on scale.

  float b = scaleBias[startIdx + c + C];

  float op = val1 * s + val2 + b;
  op = activate(op, activation);
  output[tid] = (T)op;
}

// Host function: globalScale
template <typename T>
void globalScale(int N, int C, T* output, const T* input, const T* scaleBias,
                 const T* prevLayerBias, bool nhwc,
                 ActivationFunction activation, sycl::queue &sycl_queue) {
  const bool fp16 = std::is_same<sycl::half, T>::value;

  // Each thread writes one output.
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * 8 * 8 * C, kBlockSize);

  if (nhwc) {
    assert(fp16);
    // This path uses globalScale_kernel_fp16_nhwc (see separate file)
  } else {
    sycl_queue.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, kBlocks) * sycl::range<3>(1, 1, kBlockSize),
            sycl::range<3>(1, 1, kBlockSize)),
        [=](sycl::nd_item<3> item_ct1) {
          globalScale_kernel(output, input, scaleBias, prevLayerBias,
                             N * C * 8 * 8, C, activation, item_ct1);
        });
  }
}

}  // namespace sycldnn_backend
}  // namespace lczero
