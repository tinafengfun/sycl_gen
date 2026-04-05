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
#include <cmath>

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

// Kernel: batchNorm_kernel
// Applies batch normalization to input tensor
// Supports both NCHW (fp32) and NHWC (fp16) layouts based on data type size
template <typename T>
void batchNorm_kernel(T* output, const T* input, const T* skipInput,
                      int N, int C, int H, int W, const float* means,
                      const float* varMultipliers,
                      ActivationFunction activation,
                      const sycl::nd_item<3> &item_ct1) {
  int index = item_ct1.get_local_id(2) +
              item_ct1.get_local_range(2) * item_ct1.get_group(2);

  int wIndex = 0;
  if (sizeof(T) == sizeof(float))
    wIndex = (index / (H * W)) % C;  // NCHW for fp32.
  else
    wIndex = index % C;  // NHWC for fp16.

  float el = input[index];
  float mean = means[wIndex];
  float varMulti = varMultipliers[wIndex];

  el -= mean;
  el *= varMulti;

  if (skipInput) el += (float)skipInput[index];

  el = activate(el, activation);

  output[index] = (T)el;
}

// Host function: batchNorm
// Every thread processes single element.
template <typename T>
void batchNorm(T* output, const T* input, const T* skipInput, int N, int C,
               int H, int W, const float* means, const float* var_multipliers,
               ActivationFunction activation, sycl::queue &sycl_queue) {
  const int total_elements = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(total_elements, kBlockSize);

  sycl_queue.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
          sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item_ct1) {
        batchNorm_kernel<T>(output, input, skipInput, N, C, H, W, means,
                         var_multipliers, activation, item_ct1);
      });
}

}  // namespace sycldnn_backend
}  // namespace lczero