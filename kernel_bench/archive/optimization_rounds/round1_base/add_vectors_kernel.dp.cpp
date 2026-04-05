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

namespace lczero {
namespace sycldnn_backend {

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
void addVectors_kernel(T* c, T* a, T* b, int size, int asize,
                                   int bsize, ActivationFunction activation,
                                   const sycl::nd_item<3> &item_ct1) {
  int i = item_ct1.get_local_id(2) +
          item_ct1.get_local_range(2) * item_ct1.get_group(2);
  if (i < size) {
    float aVal = 0;
    float bVal = 0;
    if (a) aVal = (float)(a[i % asize]);
    if (b) bVal = (float)(b[i % bsize]);

    float cVal = aVal + bVal;

    cVal = activate(cVal, activation);

    c[i] = (T)cVal;
  }
}

// Adds two vectors (possibly of different sizes), also do optional relu
// activation.
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize,
                ActivationFunction activation, sycl::queue &sycl_queue) {
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  sycl_queue.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
          sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item_ct1) {
        addVectors_kernel(c, a, b, size, asize, bsize, activation, item_ct1);
      });
}

}  // namespace sycldnn_backend
}  // namespace lczero
