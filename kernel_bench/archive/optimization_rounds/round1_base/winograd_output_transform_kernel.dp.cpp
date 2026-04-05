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
#include "include/sycl_standard_header.h"



namespace lczero {
namespace sycldnn_backend {

// Activation function enum
// Common constants
static constexpr int kNumOutputPolicy = 1858;
static constexpr int kMaxResBlockFusingChannels = 384;
static constexpr int kMaxResBlockFusingSeKFp16Ampere = 512;
static constexpr int kMaxResBlockFusingSeK = 128;
static constexpr int kInputPlanes = 112;

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

 inline float mishActivate(float el) {
  auto e = sycl::exp(el);
  auto n = e * e + 2.0f * e;
  auto d = __fdividef(el, n + 2.0f);
  if (el <= -0.6f) {
    return n * d;
  } else {
    return el - 2.0f * d;
  }
}

 inline float activate(float cVal,
                                           ActivationFunction activation) {
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
      // Trigger an error if we ever get here.
      return;
  }
  return cVal;
}

template <typename T, int M, int N, int K>
 inline void matrixMul_gpu_serial(T* c, const T* a,
                                                      const T* b) {
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
 inline void OutputTransform4x4(T* output,
                                                    const T* transformedOutput) {
  // transform applied to result
  const T At[4 * 6] = {1, 1, 1, 1, 1, 0, 0, 1, -1, 2, -2, 0,
                        0, 1, 1, 4, 4, 0, 0, 1, -1, 8, -8, 1};

  const T A[6 * 4] = {1, 0, 0, 0, 1, 1,  1, 1,  1, -1, 1, -1,
                      1, 2, 4, 8, 1, -2, 4, -8, 0, 0,  0, 1};

  T tempOp[4 * 6];
  matrixMul_gpu_serial<T, 4, 6, 6>(tempOp, At, transformedOutput);
  matrixMul_gpu_serial<T, 4, 4, 6>(output, tempOp, A);
}

#define TEMP_INDEX_HWNC(h, w, n, c) \
  ((h)*6 * GemmN * C + (w)*GemmN * C + (n)*C + c)
#define INDEX_NCHW(n, c, h, w) ((n)*C * 8 * 8 + (c)*8 * 8 + (h)*8 + w)
#define INDEX_NHCW(n, c, h, w) ((n)*C * 8 * 8 + (h)*C * 8 + (c)*8 + w)

#define readw1(row, col) (w1[(row)*se_K + (col)])
#define readw2(row, col) (w2[(row)*2 * C + (col)])

// input is in transformed space (HWNC layout)
// output is NCHW
// 'C' threads per block
// 'N' blocks
// every thread generates an entire board/plane (8x8 elements)
template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip, bool skipInput_nhcw, bool output_nhcw>
 void OutputTransform_kernel(int N, int C, int se_K, T* output,
                                        const T* input, const T* skip,
                                        const T* bias, const T* w1, const T* b1,
                                        const T* w2, const T* b2) {
#ifndef SKIP_FP16_BITS
  const bool fp16 = std::is_same<sycl::half, T>::value;

  int k = item.get_local_id(0);
  int n = item.get_group(0);

  T board[8][8];
  T b = bias[k];

#pragma unroll
  for (int hStart = 0; hStart < 8; hStart += 4)
#pragma unroll
    for (int wStart = 0; wStart < 8; wStart += 4) {
      //  i) read to per thread registers (for doing output transform)
      int shln = n * 4 + (hStart / 4) * 2 + (wStart / 4);
      T outElTransformed[6][6];
#pragma unroll
      for (int y = 0; y < 6; y++)
#pragma unroll
        for (int x = 0; x < 6; x++)
          outElTransformed[y][x] = input[TEMP_INDEX_HWNC(y, x, shln, k)];

      // ii) transform it
      T outEl[4][4];
      OutputTransform4x4(&outEl[0][0], &outElTransformed[0][0]);

#pragma unroll
      for (int y = 0; y < 4; y++)
#pragma unroll
        for (int x = 0; x < 4; x++) board[hStart + y][wStart + x] = outEl[y][x];
    }

  // Add bias, and compute the average for SE.
  float S = 0;
  float B = 0;

#pragma unroll
  for (int y = 0; y < 8; y++)
#pragma unroll
    for (int x = 0; x < 8; x++) {
      if (use_bias) board[y][x] += b;
      if (use_se) S += (float)board[y][x];
    }

  if (use_se) {
    __shared__ float shared_data[1024];
    float avg = S / 64;
    shared_data[k] = avg;
    item.barrier();

    // First fully-connected layer for SE
    if (k < se_K) {
      S = 0;
      for (int i = 0; i < C; i++) {
        S += shared_data[i] * float(readw1(i, k));
      }
      S += (float)b1[k];
      S = activate(S, activation);
    }
    item.barrier();
    if (k < se_K) {
      shared_data[k] = S;
    }
    item.barrier();

    // Second fully-connected layer for SE
    S = 0;
    for (int i = 0; i < se_K; i++) {
      float val = shared_data[i];
      S += val * float(readw2(i, k));
      B += val * float(readw2(i, k + C));
    }
    S += (float)b2[k];
    B += (float)b2[k + C];

    // Sigmoid (only on the scale part).
    S = 1.0f / (1.0f + exp(-S));
  }

  // Scale/bias, add skip connection, perform relu, and write to output.
  for (int h = 0; h < 8; h++) {
    if (use_se)
#pragma unroll
      for (int w = 0; w < 8; w++) board[h][w] = (T)(float(board[h][w]) * S + B);

    // residual add
    if (use_skip) {
      T skipInp[8];
      if (skipInput_nhcw) {
        *((sycl::uint4*)(&skipInp[0])) = *((sycl::uint4*)(&skip[INDEX_NHCW(n, k, h, 0)]));
        if (!fp16)
          *((sycl::uint4*)(&skipInp[4])) = *((sycl::uint4*)(&skip[INDEX_NHCW(n, k, h, 4)]));
      } else {
        *((sycl::uint4*)(&skipInp[0])) = *((sycl::uint4*)(&skip[INDEX_NCHW(n, k, h, 0)]));
        if (!fp16)
          *((sycl::uint4*)(&skipInp[4])) = *((sycl::uint4*)(&skip[INDEX_NCHW(n, k, h, 4)]));
      }
#pragma unroll
      for (int w = 0; w < 8; w++) board[h][w] += skipInp[w];
    }

    // relu
    if (activation != ACTIVATION_NONE) {
#pragma unroll
      for (int w = 0; w < 8; w++)
        board[h][w] = (T)activate((float)board[h][w], activation);
    }

    // Write to output (use 128 bit writes to store one row a time)
    if (output_nhcw) {
      *((sycl::uint4*)(&output[INDEX_NHCW(n, k, h, 0)])) = *((sycl::uint4*)&board[h][0]);
      if (!fp16)
        *((sycl::uint4*)(&output[INDEX_NHCW(n, k, h, 4)])) = *((sycl::uint4*)&board[h][4]);
    } else {
      *((sycl::uint4*)(&output[INDEX_NCHW(n, k, h, 0)])) = *((sycl::uint4*)&board[h][0]);
      if (!fp16)
        *((sycl::uint4*)(&output[INDEX_NCHW(n, k, h, 4)])) = *((sycl::uint4*)&board[h][4]);
    }
  }
#endif
}

template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip, bool skipInput_nhcw, bool output_nhcw>
void OutputTransform(int N, int C, int se_K, T* output, const T* input,
                      const T* skip, const T* bias, const T* w1, const T* b1,
                      const T* w2, const T* b2, cudaStream_t stream) {
  // Each thread processes entire chess board
  OutputTransform_kernel<T, use_se, activation, use_bias, use_skip,
                          skipInput_nhcw, output_nhcw><<<N, C, 0, stream>>>(
      N, C, se_K, output, input, skip, bias, w1, b1, w2, b2);
  ReportCUDAErrors(cudaGetLastError());
}

}  // namespace sycldnn_backend
}  // namespace lczero
