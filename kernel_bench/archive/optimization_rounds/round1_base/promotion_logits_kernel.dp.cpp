#include <algorithm>
#include "include/sycl_standard_header.h"

#include <cmath>
#include <sycl/sycl.hpp>


// ActivationFunction enum
// Common constants
static constexpr int kNumOutputPolicy = 1858;
static constexpr int kMaxResBlockFusingChannels = 384;
static constexpr int kMaxResBlockFusingSeKFp16Ampere = 512;
static constexpr int kMaxResBlockFusingSeK = 128;
static constexpr int kInputPlanes = 112;
static constexpr int kOpInpTransformBlockSize = 64;

// Helper functions
inline int DivUp(int a, int b) { return (a + b - 1) / b; }


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


namespace lczero {
namespace sycldnn_backend {

// Compute promotion logits in a single kernel
// keys matrix is of N * 64 * C (but we use only last 8 from the 'rows'
// dimension, so N * 8 * C)
// ppo matrix is 4 * C (weights for dense layer / matrix multiplication)
// policy_attn_logits matrix is N * 64 * 64, but we use only 8x8 part of it
// from each batch dimension (so, N * 8 * 8)
// output matrix (promotion logits) is of N * 8 * 24 size
template <typename T>
 void promotion_logits_kernel(int C, T* output, const T* keys,
                                        const T* ppo,
                                        const T* policy_attn_logits) {
  constexpr int output_stride = 64 * 64 + 8 * 24;
  int n = item.get_group(0);   // [0..N)
  int y = item.get_local_id(1);  // [0..8)
  int x = item.get_local_id(0);  // [0..24)     // Can split into 8 * 3

  int threadInGroup = item.get_local_id(1) * 24 + item.get_local_id(0);

  // phase 1 : compute promotion_offsets by multiplying keys and ppo matrices
  const T* keys_start =
      keys + n * 64 * C + C * 56;  // we are interested only in last 8 out of 64
                                   // 'rows' of keys matrix
  __shared__ float promotion_offsets[4][8];

  // only 32 threads out of 192 in the group are active in this phase, and each
  // thread computes one element of the promotion_offsets matrix
  // TODO: opt idea1, can use more threads to reduce the length of the loop for
  // the matrix multiply (do parallel reduction of partial sums later)
  //       opt idea2, the below loop for matrix mul has very poor memory access
  //       pattern, can do the loop over 32, and do parallel reductions
  if (threadInGroup < 32) {
    int x = threadInGroup % 4;
    int y = threadInGroup / 4;

    float S = 0;
    for (int i = 0; i < C;
         i++) {  // TODO: modify to loop over 32 instead of C (doing parallel
                 // reductions for the 32 sums)
      float a = (float)keys_start[y * C + i];
      float b =
          (float)ppo[x * C + i];  // weight matrix is transposed (col major)
      S += a * b;
    }

    // write the product (promotion_offsets) in shared memory
    promotion_offsets[x][y] = S;
  }

  item.barrier();

  // phase 2: add the last "row" to the other 3
  // #knight offset is added to the other three
  // promotion_offsets = promotion_offsets[:, :3, :] + promotion_offsets[:, 3:4,
  // :]
  // Only 24 threads in the group are active in this phase
  if (threadInGroup < 32) {
    int x = threadInGroup % 4;
    int y = threadInGroup / 4;
    if (x < 3) {
      promotion_offsets[x][y] += promotion_offsets[3][y];
    }
  }

  item.barrier();

  // phase 3: add 8x8 chunk of policy_attn_logits matrix to promotion offsets
  //          the output is 3x8x8 (written as 8 * 24)
  // All threads are active in this phase and they compute one element each
  int w = x / 3;
  int c = x % 3;

  // n_promo_logits = matmul_qk[:, -16:-8, -8:]  # default traversals from rank
  // 7 to rank 8
  float n_promo_logit =
      (float)policy_attn_logits[n * output_stride + (48 + y) * 64 + (56 + w)];
  float promo_offset = promotion_offsets[c][w];

  float op = n_promo_logit + promo_offset;

  output[n * output_stride + threadInGroup] = (T)op;
}

template <typename T>
void ComputePromotionLogits(int N, int C, T* output, const T* keys,
                            const T* ppo, const T* policy_attn_logits,
                            cudaStream_t stream) {
  // N blocks
  // 8 * 24 threads
  // Each thread computes a single output element
  dim3 blockDim(24, 8, 1);
  promotion_logits_kernel<T>
      <<<N, blockDim, 0, stream>>>(C, output, keys, ppo, policy_attn_logits);
}

}  // namespace sycldnn_backend
}  // namespace lczero
