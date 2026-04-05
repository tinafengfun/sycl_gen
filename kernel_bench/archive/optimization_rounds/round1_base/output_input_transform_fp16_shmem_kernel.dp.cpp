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
static constexpr int kOpInpTransformBlockSize = 64;

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
 inline void InputTransform4x4(T* transformedInput,
                                                   const T* input) {
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
#define INDEX_NHCW(n, c, h, w) ((n)*C * 8 * 8 + (h)*C * 8 + (c)*8 + w)

#define readw1(row, col) (w1[(row)*se_K + (col)])
#define readw2(row, col) (w2[(row)*2 * C + (col)])

// Helper fuction to do vector loads/stores
template <typename T>
 inline void copyAs(void* dst, const void* src) {
  *((T*)(dst)) = *((const T*)(src));
}

// fast reduction for the warp
 inline float warpReduce(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    x += x;

  return x;
}

// Combined Output Transform, SE, ReLU, and Input Transform kernel using FP16
// and shared memory for optimization
// input is in transformed space (HWNC layout) --- output of GEMM
// output is also in transformed space (HWNC layout) --- input to GEMM (for next
// layer)
// 'C' threads per block
// 'N' blocks
// every thread generates an entire board/plane (8x8 elements)
template <typename T, ActivationFunction activation, bool use_bias,
          bool use_skip>
 __launch_bounds__(
    kMaxResBlockFusingChannels,
    1) void OutputTransform_SE_relu_InputTransform_fp16_shmem_kernel(int N, int C,
                                                           int se_K, T* output,
                                                           const T* input,
                                                           T* skip,
                                                           const T* bias,
                                                           const T* w1,
                                                           const T* b1,
                                                           const T* w2,
                                                           const T* b2) {
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
      S += (float)board[y][x];
    }

  {
    __shared__ float shared_data[kMaxResBlockFusingChannels];
    float avg = S / 64;
    shared_data[k] = avg;

    int lane = k & 0x1F;
    int warp = k >> 5;
    item.barrier();

    // First fully-connected layer for SE

    // As se_K << C, we want to loop over se_K instead of C
    // even if it means taking the sum across threads

    __shared__ float shared_sums[kMaxResBlockFusingChannels / 32]
                                [kMaxResBlockFusingSeK];  // per-warp sums

    for (int i = 0; i < se_K; i++) {
      float val = shared_data[k] * float(readw1(k, i));
      val = warpReduce(val);
      if (lane == 0) shared_sums[warp][i] = val;
    }
    item.barrier();
    if (k < se_K) {
      S = 0;
      for (int i = 0; i < C / 32; i++) S += shared_sums[i][k];

      S += (float)b1[k];
      S = activate(S, activation);
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
#pragma unroll
    for (int w = 0; w < 8; w++) board[h][w] = (T)(float(board[h][w]) * S + B);

    // residual add
    if (use_skip) {
      T skipInp[8];
      copyAs<sycl::uint4>(&skipInp[0], &skip[INDEX_NHCW(n, k, h, 0)]);
      if (!fp16) copyAs<sycl::uint4>(&skipInp[4], &skip[INDEX_NHCW(n, k, h, 4)]);
#pragma unroll
      for (int w = 0; w < 8; w++) board[h][w] += skipInp[w];
    }

    // relu
    if (activation != ACTIVATION_NONE) {
#pragma unroll
      for (int w = 0; w < 8; w++)
        board[h][w] = (T)activate((float)board[h][w], activation);
    }

    // write un-transformed output to 'skip' if required
    if (use_skip) {
      // Write to skip (use 128 bit writes to store one row a time)
      copyAs<sycl::uint4>(&skip[INDEX_NHCW(n, k, h, 0)], &board[h][0]);
      if (!fp16) copyAs<sycl::uint4>(&skip[INDEX_NHCW(n, k, h, 4)], &board[h][4]);
    }
  }

  // perform input transform

  int c = k;
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
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 0, c)] = inEl[y][x];
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
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 1, c)] = inEl[y][x];
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
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 2, c)] = inEl[y][x];
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
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 3, c)] = inEl[y][x];
  }
#endif
}

// Alternative kernel with relu but without SE, also using shared memory optimizations
template <typename T, ActivationFunction activation, bool use_bias,
          bool use_skip>
 __launch_bounds__(
    kOpInpTransformBlockSize,
    4) void OutputTransform_relu_InputTransform_fp16_shmem_kernel(int N, int C, T* output,
                                                        const T* input, T* skip,
                                                        const T* bias) {
#ifndef SKIP_FP16_BITS
  const bool fp16 = std::is_same<sycl::half, T>::value;

  int k = item.get_local_id(0) + item.get_group(0) * kOpInpTransformBlockSize;
  if (k >= C) return;  // wasted threads (for non-multiple of 64 channel counts)
  int n = item.get_group(1);

  T board[8][8];
  T b = bias[k];

  T skipInp[8][8];
#pragma unroll
  for (int h = 0; h < 8; h++) {
    copyAs<sycl::uint4>(&skipInp[h][0], &skip[INDEX_NHCW(n, k, h, 0)]);
    if (!fp16) copyAs<sycl::uint4>(&skipInp[h][4], &skip[INDEX_NHCW(n, k, h, 4)]);
  }

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

    // Add bias
#pragma unroll
  for (int y = 0; y < 8; y++)
#pragma unroll
    for (int x = 0; x < 8; x++)
      if (use_bias) board[y][x] += b;

  // Add skip connection, perform relu, and write to output.
  for (int h = 0; h < 8; h++) {
    // residual add
    if (use_skip) {
#pragma unroll
      for (int w = 0; w < 8; w++) board[h][w] += skipInp[h][w];
    }

    // activation
    if (activation != ACTIVATION_NONE) {
#pragma unroll
      for (int w = 0; w < 8; w++)
        board[h][w] = (T)activate((float)board[h][w], activation);
    }

    // write un-transformed output to 'skip' if required
    if (use_skip) {
      // Write to skip (use 128 bit writes to store one row a time)
      copyAs<sycl::uint4>(&skip[INDEX_NHCW(n, k, h, 0)], &board[h][0]);
      if (!fp16) copyAs<sycl::uint4>(&skip[INDEX_NHCW(n, k, h, 4)], &board[h][4]);
    }
  }

  // perform input transform

  int c = k;
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
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 0, c)] = inEl[y][x];
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
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 1, c)] = inEl[y][x];
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
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 2, c)] = inEl[y][x];
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
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 3, c)] = inEl[y][x];
  }
#endif
}

}  // namespace sycldnn_backend
}  // namespace lczero
