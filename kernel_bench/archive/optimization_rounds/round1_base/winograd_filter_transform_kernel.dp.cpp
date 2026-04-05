/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020-2024 The LCZero Authors
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


namespace lczero {
namespace sycldnn_backend {

// Activation function enum (normally from activation_function.h)
// Helper function
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
    case ACTIVATION_MISH:
      cVal = mishActivate(cVal);
      break;
    case ACTIVATION_SWISH:
      cVal /= (1.0f + sycl::native::exp(-cVal));
      break;
  }
  return cVal;
}

template <typename T, int M, int N, int K>
[[gnu::always_inline]]
inline void matrixMul_gpu_serial(T* c, const T* a, const T* b) {
#pragma unroll
  for (int i = 0; i < M; ++i)
#pragma unroll
    for (int j = 0; j < N; ++j) {
      T S = 0;
#pragma unroll
      for (int k = 0; k < K; ++k) S += a[i * K + k] * b[k * N + j];
      c[i * N + j] = S;
    }
}

template <typename T>
[[gnu::always_inline]]
inline void FilterTransform4x4(T* transformed_filter,
                                        const T* filter) {
  // transform applied to filter (of size 3x3)
  T G[6 * 3] = {1.0f / 4,  0,         0,         -1.0f / 6,  -1.0f / 6,
                -1.0f / 6, -1.0f / 6, 1.0f / 6,  -1.0f / 6,  1.0f / 24,
                1.0f / 12, 1.0f / 6,  1.0f / 24, -1.0f / 12, 1.0f / 6,
                0,         0,         1};

  T Gt[3 * 6] = {1.0f / 4, -1.0f / 6, -1.0f / 6, 1.0f / 24, 1.0f / 24,  0,
                 0,        -1.0f / 6, 1.0f / 6,  1.0f / 12, -1.0f / 12, 0,
                 0,        -1.0f / 6, -1.0f / 6, 1.0f / 6,  1.0f / 6,   1};

  T temp_filter[6 * 3];
  matrixMul_gpu_serial<T, 6, 3, 3>(temp_filter, G, filter);
  matrixMul_gpu_serial<T, 6, 6, 3>(transformed_filter, temp_filter, Gt);
}

#define FILTER_IDX_NCHW(k, c, h, w) ((k)*C * S * R + (c)*S * R + (h)*R + w)

template <typename T>
void filterTransform_kernel(int K, int C, int elements,
                                       T* transformed_filter, const T* filter,
                                       const sycl::nd_item<3> &item_ct1) {
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
  if (tid >= elements) return;

  constexpr int S = 3;
  constexpr int R = 3;

  int c = tid % C;
  int k = tid / C;

  T filter_tile[3][3];
  T transformed_tile[6][6];

  // read input from memory
  for (int s = 0; s < S; s++)
    for (int r = 0; r < R; r++) {
      filter_tile[s][r] = filter[FILTER_IDX_NCHW(k, c, s, r)];
    }

  // transform it
  FilterTransform4x4(&(transformed_tile[0][0]), &(filter_tile[0][0]));

  // write to output (output is in HWCK layout)
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 6; j++) {
      transformed_filter[i * 6 * C * K + j * C * K + c * K + k] =
          transformed_tile[i][j];
    }
}

template <typename T>
void FilterTransform(int N, int C, T* transformedFilter, const T* filter, sycl::queue &mqueue) {
  // Each thread processes entire filter block (input 3x3 elements -> output 6x6
  // elements)
  const int kBlockSize = 64;
  const int kBlocks = DivUp(N * C, kBlockSize);

  mqueue.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, kBlocks) * sycl::range<3>(1, 1, kBlockSize),
          sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item_ct1) {
        filterTransform_kernel(N, C, N * C, transformedFilter, filter,
                               item_ct1);
      });
}

}  // namespace sycldnn_backend
}  // namespace lczero
