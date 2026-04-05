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

// Helper function to divide and round up
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// fast reduction for the warp
inline float warpReduce(float x, sycl::nd_item<1> item) {
  return sycl::reduce_over_group(item.get_sub_group(), x, sycl::plus<>());
}

// fast max reduction for the warp
inline float warpMax(float x, sycl::nd_item<1> item) {
  return sycl::reduce_over_group(item.get_sub_group(), x, sycl::maximum<>());
}

// Helper fuction to do vector loads/stores
template <typename T>
inline void copyAs(void* dst, const void* src) {
  *((T*)(dst)) = *((const T*)(src));
}

inline float clamp(float val, float low, float high) {
  if (sycl::isnan(val)) return val;
  return sycl::clamp(val, low, high);
}

namespace {
constexpr float kTwiceHalfMax = 131008.0f;  // Twice the max finite fp16 value.
}  // namespace

// softmax along C dimension which is assumed to be 64
// each thread processes two elements. Each warp computes a sum (over 64
// elements)
template <typename T>
struct softmax_opt_64_kernel {
  T* output;
  const T* input;
  const T* input2;
  int N;

  void operator()(sycl::nd_item<1> item) const {
    int index = item.get_local_range(0) * item.get_group(0) + item.get_local_id(0);
    if (index >= N) return;

    float x[4];
    float ex[2];

    // Load from memory
    const bool fp16 = std::is_same<sycl::half, T>::value;
    if (fp16) {
      sycl::half inp[2];
      copyAs<int>(&inp[0], &input[index * 2]);
      x[0] = (float)inp[0];
      x[1] = (float)inp[1];
      if (input2 != nullptr) {
        copyAs<int>(&inp[0], &input2[index * 2]);
        x[2] = (float)inp[0];
        x[3] = (float)inp[1];
      }
    } else {
      copyAs<sycl::uint2>(&x[0], &input[index * 2]);
      if (input2 != nullptr) {
        copyAs<sycl::uint2>(&x[2], &input2[index * 2]);
      }
    }

    if (input2 != nullptr) {
      x[0] += x[2];
      x[1] += x[3];
    }
    if (fp16) {
      // Guard against Inf from fp16 overflow.
      x[0] = clamp(x[0], -kTwiceHalfMax, kTwiceHalfMax);
      x[1] = clamp(x[1], -kTwiceHalfMax, kTwiceHalfMax);
    }
    float threadMax = sycl::max(x[0], x[1]);
    float maxval = warpMax(threadMax, item);
    maxval = sycl::group_broadcast(item.get_sub_group(), maxval, 0);

    ex[0] = sycl::exp(x[0] - maxval);
    ex[1] = sycl::exp(x[1] - maxval);

    float threadSum = ex[0] + ex[1];
    float Sum = warpReduce(threadSum, item);
    Sum = sycl::group_broadcast(item.get_sub_group(), Sum, 0);

    ex[0] = ex[0] / Sum;
    ex[1] = ex[1] / Sum;

    // Store to memory
    if (fp16) {
      sycl::half op[2];
      op[0] = (sycl::half)ex[0];
      op[1] = (sycl::half)ex[1];
      copyAs<int>(&output[index * 2], &op[0]);
    } else {
      copyAs<sycl::uint2>(&output[index * 2], &ex[0]);
    }
  }
};

template <typename T>
void Softmax(int N, int C, T* output, const T* input, const T* input2,
             sycl::queue& q) {
  int size = N * 32;  // Total no of threads needed
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);
  
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(blocks * kBlockSize, kBlockSize),
        softmax_opt_64_kernel<T>{output, input, input2, size});
  });
}

// Explicit instantiations
template void Softmax<float>(int N, int C, float* output, const float* input,
                             const float* input2, sycl::queue& q);
template void Softmax<sycl::half>(int N, int C, sycl::half* output, const sycl::half* input,
                            const sycl::half* input2, sycl::queue& q);

}  // namespace sycldnn_backend
}  // namespace lczero