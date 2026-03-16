/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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
#include <sycl/ext/oneapi/experimental/bfloat16.hpp>

namespace lczero {
namespace sycldnn_backend {

// Activation function enum
enum ActivationFunction {
  ACTIVATION_NONE,
  ACTIVATION_RELU,
  ACTIVATION_RELU_2,
  ACTIVATION_TANH,
  ACTIVATION_SIGMOID,
  ACTIVATION_SELU,
  ACTIVATION_MISH,
  ACTIVATION_SWISH,
  ACTIVATION_DEFAULT,
  ACTIVATION_SOFTMAX
};

// Common constants
constexpr int kNumOutputPolicy = 1858;
constexpr int kMaxResBlockFusingChannels = 384;
constexpr int kMaxResBlockFusingSeKFp16Ampere = 512;
constexpr int kMaxResBlockFusingSeK = 128;
constexpr int kInputPlanes = 112;

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T, int M, int N, int K>
inline void matrixMul_gpu_serial(T* c, const T* a, const T* b) {
#ifndef SKIP_FP16_BITS
#pragma unroll
  for (int i = 0; i < M; ++i)
#pragma unroll
    for (int j = 0; j < N; ++j) {
      T S = 0;
#pragma unroll
      for (int k = 0; k < K; ++k) S += a[i * K + k] * b[k * N + j];
      c[i * N + j] = S;
    }
#endif
}

template <typename T>
inline void InputTransform4x4(T* transformedInput, const T* input) {
  // transform applied to input tile (of size 4x4)
  const T Bt[6 * 6] = {4, 0, -5, 0,  1, 0, 0, -4, -4, 1,  1, 0,
                        0, 4, -4, -1, 1, 0, 0, -2, -1, 2,  1, 0,
                        0, 2, -1, -2, 1, 0, 0, 4,  0,  -5, 0, 1};

  const T B[6 * 6] = {4,  0,  0,  0,  0,  0, 0, -4, 4,  -2, 2,  4,
                      -5, -4, -4, -1, -1, 0, 0, 1,  -1, 2,  -2, -5,
                      1,  1,  1,  1,  1,  0, 0, 0,  0,  0,  0,  1};

  T tempIp1[6 * 6];
  matrixMul_gpu_serial<T, 6, 6, 6>(tempIp1, Bt, input);
  matrixMul_gpu_serial<T, 6, 6, 6>(transformedInput, tempIp1, B);
}

// Index conversion functions (replacing macros)
template <int C>
inline int IndexNCHW(int n, int c, int h, int w) {
  return (n) * C * 8 * 8 + (c) * 8 * 8 + (h) * 8 + w;
}

template <int C>
inline int IndexNHCW(int n, int c, int h, int w) {
  return (n) * C * 8 * 8 + (h) * C * 8 + (c) * 8 + w;
}

// index in intermediate/temp tensor
// W, H == 6 here! (6x6 transformed blocks)
// N also includes part of dimension (2x2)
template <int N, int C>
inline int TempIndexHWNC(int h, int w, int n, int c) {
  int GemmN = N * 4;
  return (h) * 6 * GemmN * C + (w) * GemmN * C + (n) * C + c;
}

// 'C' threads per block
// 'N' blocks
// every thread transforms an entire board/plane (8x8 elements)
// - producing 4 x 6x6 elements
template <typename T, bool nhcw>
void InputTransform_kernel(sycl::queue& queue, int N, int C, const T* input, T* output) {
  int GemmN = N * 4;
  
  queue.parallel_for(
    sycl::nd_range<1>(sycl::range<1>(N * C), sycl::range<1>(C)),
    [=](sycl::nd_item<1> item) {
      int c = item.get_local_id(0);
      int n = item.get_group(0);

      T board[8][8];

      const bool fp16 = std::is_same<sycl::half, T>::value;

      // read the board (a row at a time for fp16)
#pragma unroll
      for (int y = 0; y < 8; y++) {
        if (nhcw) {
          *((sycl::uint4*)(&board[y][0])) = *((sycl::uint4*)(&input[IndexNHCW<C>(n, c, y, 0)]));
          if (!fp16)
            *((sycl::uint4*)(&board[y][4])) = *((sycl::uint4*)(&input[IndexNHCW<C>(n, c, y, 4)]));
        } else {
          *((sycl::uint4*)(&board[y][0])) = *((sycl::uint4*)(&input[IndexNCHW<C>(n, c, y, 0)]));
          if (!fp16)
            *((sycl::uint4*)(&board[y][4])) = *((sycl::uint4*)(&input[IndexNCHW<C>(n, c, y, 4)]));
        }
      }

      // top-left
      {
        T inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
        for (int i = 0; i < 5; i++)
#pragma unroll
          for (int j = 0; j < 5; j++) inEl[i + 1][j + 1] = board[i][j];

        InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
        for (int y = 0; y < 6; y++)
#pragma unroll
          for (int x = 0; x < 6; x++)
            output[TempIndexHWNC<N, C>(y, x, n * 4 + 0, c)] = inEl[y][x];
      }

      // top-right
      {
        T inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
        for (int i = 0; i < 5; i++)
#pragma unroll
          for (int j = 0; j < 5; j++) inEl[i + 1][j] = board[i][j + 3];

        InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
        for (int y = 0; y < 6; y++)
#pragma unroll
          for (int x = 0; x < 6; x++)
            output[TempIndexHWNC<N, C>(y, x, n * 4 + 1, c)] = inEl[y][x];
      }

      // bottom-left
      {
        T inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
        for (int i = 0; i < 5; i++)
#pragma unroll
          for (int j = 0; j < 5; j++) inEl[i][j + 1] = board[i + 3][j];

        InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
        for (int y = 0; y < 6; y++)
#pragma unroll
          for (int x = 0; x < 6; x++)
            output[TempIndexHWNC<N, C>(y, x, n * 4 + 2, c)] = inEl[y][x];
      }

      // bottom-right
      {
        T inEl[6][6] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#pragma unroll
        for (int i = 0; i < 5; i++)
#pragma unroll
          for (int j = 0; j < 5; j++) inEl[i][j] = board[i + 3][j + 3];

        InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
        for (int y = 0; y < 6; y++)
#pragma unroll
          for (int x = 0; x < 6; x++)
            output[TempIndexHWNC<N, C>(y, x, n * 4 + 3, c)] = inEl[y][x];
      }
    }
  );
}

// Explicit instantiations
template void InputTransform_kernel<float, true>(sycl::queue& queue, int N, int C, 
                                                const float* input, float* output);
template void InputTransform_kernel<float, false>(sycl::queue& queue, int N, int C,
                                                 const float* input, float* output);
template void InputTransform_kernel<sycl::half, true>(sycl::queue& queue, int N, int C,
                                                     const sycl::half* input, sycl::half* output);
template void InputTransform_kernel<sycl::half, false>(sycl::queue& queue, int N, int C,
                                                      const sycl::half* input, sycl::half* output);

} // namespace sycldnn_backend
} // namespace lczero
