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
  along with this program.  If not, see <https://www.gnu.org/licenses/licenses/>.
    
  SPDX-License-Identifier:GNU General Public License v3.0 or later
*/

#include <sycl/sycl.hpp>
#include "include/sycl_standard_header.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>

namespace lczero {
namespace sycldnn_backend {

class Exception : public std::runtime_error {
 public:
  explicit Exception(const std::string& msg) : std::runtime_error(msg) {}
};

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
      cVal = 1.0f / (1.0f + sycl::native::exp(-cVal));
      break;
    case ACTIVATION_SELU: {
      float alpha = 1.67326324f, scale = 1.05070098f;
      if (cVal > 0)
        cVal = scale * cVal;
      else
        cVal = scale * alpha * (sycl::native::exp(cVal) - 1.0f);
      break;
    }
    case ACTIVATION_SWISH:
      cVal = cVal / (1.0f + sycl::native::exp(-cVal));
      break;
    case ACTIVATION_MISH: {
      auto e = sycl::native::exp(cVal);
      auto n = e * e + 2.0f * e;
      auto d = cVal / (n + 2.0f);
      if (cVal <= -0.6f) {
        cVal = n * d;
      } else {
        cVal = cVal - 2.0f * d;
      }
      break;
    }
    case ACTIVATION_NONE:
    default:
      break;
  }
  return cVal;
}

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void copyAs(void* dst, const void* src) {
  *reinterpret_cast<T*>(dst) = *reinterpret_cast<const T*>(src);
}

template <typename T, ActivationFunction act>
void addBiasBatched_kernel(T* output, const T* input, const T* bias,
                                       int N, int C,
                                       const sycl::nd_item<3> &item_ct1) {
  int batch = item_ct1.get_group(1);
  int n = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
          item_ct1.get_local_id(1);
  if (n >= N) return;
  int c = item_ct1.get_local_id(2) * 4;

  int biasIndex = batch * C + c;
  int tensorIndex = batch * N * C + n * C + c;

  float val[4];
  float b[4];

  // Load from memory
  const bool fp16 = std::is_same<sycl::half, T>::value;
  if (fp16) {
    sycl::half inp[4];
    copyAs<sycl::uint2>(&inp[0], &input[tensorIndex]);
#pragma unroll
    for (int i = 0; i < 4; i++) val[i] = (float)inp[i];

    copyAs<sycl::uint2>(&inp[0], &bias[biasIndex]);
#pragma unroll
    for (int i = 0; i < 4; i++) b[i] = (float)inp[i];
  } else {
    copyAs<sycl::uint4>(&val[0], &input[tensorIndex]);
    copyAs<sycl::uint4>(&b[0], &bias[biasIndex]);
  }

  // Perform bias add and activation
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float x = val[i] + b[i];
    x = activate(x, act);
    val[i] = x;
  }

  // write to memory
  if (fp16) {
    sycl::half op[4];
#pragma unroll
    for (int i = 0; i < 4; i++) op[i] = (sycl::half)val[i];
    copyAs<sycl::uint2>(&output[tensorIndex], &op[0]);
  } else {
    copyAs<sycl::uint4>(&output[tensorIndex], &val[0]);
  }
}

// Input/output tensors are Batch * N * C
// bias tensor is N * C (i.e, different bias for each Batch dimension)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, ActivationFunction activation, sycl::queue &sycl_queue) {
  // process 4 elements per thread to achieve close to peak memory bandwidth
  if (C % 4 != 0) printf("Error: "unsupported filter size");
  if (C > 2048) printf("Error: "unsupported filter size");

  sycl::range<3> blockDim(1, 1, 1), gridDim(1, 1, 1);
  blockDim[2] = C / 4;
  unsigned int tmp = (512 / blockDim[2]);
  blockDim[1] = sycl::min(sycl::max(tmp, 1u), (unsigned int)N);
  blockDim[0] = 1;
  gridDim[2] = DivUp(N, blockDim[1]);
  gridDim[1] = Batch;
  gridDim[0] = 1;

  switch (activation) {
    case ACTIVATION_NONE:
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                         [=](sycl::nd_item<3> item_ct1) {
                           addBiasBatched_kernel<T, ACTIVATION_NONE>(output, input, bias,
                                                            N, C, item_ct1);
                         });
      break;
    case ACTIVATION_SELU:
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                         [=](sycl::nd_item<3> item_ct1) {
                           addBiasBatched_kernel<T, ACTIVATION_SELU>(output, input, bias,
                                                            N, C, item_ct1);
                         });
      break;
    case ACTIVATION_MISH:
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                         [=](sycl::nd_item<3> item_ct1) {
                           addBiasBatched_kernel<T, ACTIVATION_MISH>(output, input, bias,
                                                            N, C, item_ct1);
                         });
      break;
    case ACTIVATION_RELU:
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                         [=](sycl::nd_item<3> item_ct1) {
                           addBiasBatched_kernel<T, ACTIVATION_RELU>(output, input, bias,
                                                            N, C, item_ct1);
                         });
      break;
    case ACTIVATION_SWISH:
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                         [=](sycl::nd_item<3> item_ct1) {
                           addBiasBatched_kernel<T, ACTIVATION_SWISH>(output, input, bias,
                                                            N, C, item_ct1);
                         });
      break;
    case ACTIVATION_RELU_2:  // square relu
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                         [=](sycl::nd_item<3> item_ct1) {
                           addBiasBatched_kernel<T, ACTIVATION_RELU_2>(output, input, bias,
                                                            N, C, item_ct1);
                         });
      break;
    default:
      printf("Error: 
          "unsupported activation in addBiasBatched. Add in switch-case here");
  }
}

}  // namespace sycldnn_backend
}  // namespace lczero
