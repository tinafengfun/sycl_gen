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

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

SYCL_EXTERNAL inline float mishActivate(float el) {
  auto e = sycl::exp(el);
  auto n = e * e + 2.0f * e;
  auto d = el / (n + 2.0f);
  if (el <= -0.6f) {
    return n * d;
  } else {
    return el - 2.0f * d;
  }
}

SYCL_EXTERNAL inline float activate(float cVal, ActivationFunction activation) {
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
    case ACTIVATION_SELU: {
      float alpha = 1.67326324f, scale = 1.05070098f;
      if (cVal > 0)
        cVal = scale * cVal;
      else
        cVal = scale * alpha * (sycl::exp(cVal) - 1.0f);
      break;
    }
    case ACTIVATION_MISH:
      cVal = mishActivate(cVal);
      break;
    case ACTIVATION_SWISH:
      cVal /= (1.0f + sycl::exp(-cVal));
      break;
    case ACTIVATION_NONE:
      break;
    case ACTIVATION_DEFAULT:
    case ACTIVATION_SOFTMAX:
      sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
      break;
  }
  return cVal;
}

template <typename T>
void addBias_NCHW(T* c, T* a, T* b, int N, int C, int H, int W,
                  ActivationFunction activation, sycl::queue& stream) {
  int size = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  stream.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize), sycl::range<1>(kBlockSize)),
        [=](sycl::nd_item<1> item) {
          int i = item.get_global_id(0);
          if (i < size) {
            float aVal = static_cast<float>(a[i]);
            int biasIndex = (i / (H * W)) % C;
            float bVal = static_cast<float>(b[biasIndex]);
            float cVal = aVal + bVal;
            cVal = activate(cVal, activation);
            c[i] = static_cast<T>(cVal);
          }
        });
  });
  stream.wait_and_throw();
}

template void addBias_NCHW<float>(float* c, float* a, float* b, int N, int C,
                                  int H, int W, ActivationFunction activation,
                                  sycl::queue& stream);

template void addBias_NCHW<sycl::half>(sycl::half* c, sycl::half* a, sycl::half* b, int N, int C,
                                 int H, int W, ActivationFunction activation,
                                 sycl::queue& stream);

}  // namespace cudnn_backend
}  // namespace lczero