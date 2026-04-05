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
  along with this program.  If not, see <https://www.gnu.org/licenses/licenses.html>.
   
  SPDX-License-Identifier:GNU General Public License v3.0 or later
*/

#include <sycl/sycl.hpp>
#include "include/sycl_standard_header.h"


#if defined(__HIP_PLATFORM_AMD__) && (defined(__GFX9__) || defined(__GFX8__))
#define SYCL_SUB_GROUP_SIZE 64
#else
#define SYCL_SUB_GROUP_SIZE 32
#endif

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// Each thread reads 2 inputs (8x8/32), and each warp writes a single output.
template <typename T>
void globalAvgPool_kernel(T* output, const T* input,
                                     const T* prevLayerBias, int inputSize,
                                     int outputSize, int C,
                                     const sycl::nd_item<3> &item_ct1) {
  const int elementsPerWarp = 64;
  const int elementsPerThread = 2;

  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  int laneId = item_ct1.get_local_id(2) & 0x1F;
  int laneStartIndex = (tid - laneId) * elementsPerThread;

  // Compute per-thread sum for elementsPerThread elements.
  float S = 0;

#pragma unroll
  for (int i = 0; i < elementsPerWarp; i += 32) {
    int index = laneStartIndex + laneId + i;
    if (index < inputSize) S += (float)(input[index]);
  }

// Compute warp wide sum (for entire plane - elementsPerWarp elements).
#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2) {
    /*
    DPCT1023:10: The SYCL sub-group does not support mask options for
    dpct::shift_sub_group_left. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_down_sync.
    */
    S += sycl::shift_group_left(item_ct1.get_sub_group(), S, offset);
  }

  float avg = S / elementsPerWarp;
  int opIndex = tid >> 5;

  // First thread in warp has the sum, write it in output.
  if (laneId == 0) {
    if (opIndex < outputSize) {
      if (prevLayerBias) avg += (float)prevLayerBias[opIndex % C];
      output[opIndex] = (T)avg;
    }
  }
}

template <typename T>
void globalAvgPool(int N, int C, T* output, const T* input,
                   const T* prevLayerBias, bool nhwc, sycl::queue &sycl_queue) {
  const int kPlaneSize = 64;

  const bool fp16 = std::is_same<sycl::half, T>::value;
  if (nhwc) {
    // NHWC not handled here - see global_avg_pool_nhwc_fp16_kernel.dp.cpp
    assert(fp16);
  } else {
    // For NCHW layout (used with fp32),
    // each warp processes a full plane (64 elements), and writes a single
    // average N*C warps are launched.

    const int kTotalWarps = N * C;
    const int kWarpsPerBlock = 8;
    const int kBlockSize = kWarpsPerBlock * 32;

    int blocks = DivUp(kTotalWarps, kWarpsPerBlock);
    sycl_queue.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
            sycl::range<3>(1, 1, kBlockSize)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(SYCL_SUB_GROUP_SIZE)]] {
          globalAvgPool_kernel(output, input, prevLayerBias, N * C * kPlaneSize,
                               N * C, C, item_ct1);
        });
  }
}
