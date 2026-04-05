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
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
   
  SPDX-License-Identifier:GNU General Public License v3.0 or later
*/

#include <sycl/sycl.hpp>
#include "include/sycl_standard_header.h"


namespace lczero {
namespace sycldnn_backend {

// Activation function enum (normally from activation.h)
// Helper function
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

/////////////////////////////////////////////////////////////////////////////
//          SE layer kernel for FP16 NHWC format                          //
/////////////////////////////////////////////////////////////////////////////

// N blocks.
// C threads per block.
// 'HWC' input data processed by thread block.
// Each thread processes 8x8 elements.
// K is the no. of outputs of first fully connected layer (same as no. of inputs
// for second fully connected layer).
// The kernel assumes K <= C.

template <int C, int K>
/*
DPCT1110:20: The total declared local variable size in device function
SE_Layer_NHWC exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void SE_Layer_NHWC(sycl::half* output, const sycl::half* skip,
                   const sycl::half* input, const sycl::half* w1,
                   const sycl::half* b1, const sycl::half* w2,
                   const sycl::half* b2, const sycl::half* bPrev,
                   ActivationFunction activation,
                   const sycl::nd_item<3>& item_ct1, sycl::half* sharedData) {
#if DPCT_COMPATIBILITY_TEMP >= 530
  const int elementsPerThread = 64;  // 8x8 board
  const int se_K = K;

  int n = item_ct1.get_group(2);
  int c = item_ct1.get_local_id(2);

  sycl::half2 localData[elementsPerThread];

  sycl::half S = 0;

  sycl::half bias = 0;
  if (bPrev) bias = bPrev[c];

// 1. Global avg (1 avg per thread).
#pragma unroll
  for (int i = 0; i < elementsPerThread; i++) {
    int localIndex = i * C + c;
    int inputIndex = n * C * elementsPerThread + localIndex;
    localData[i].x() = input[inputIndex] + bias;
    localData[i].y() = skip[inputIndex];
    S += localData[i].x();
  }

  sycl::half avg = S / (sycl::half)elementsPerThread;
  sharedData[c] = avg;

  item_ct1.barrier(sycl::access::fence_space::local_space);

  // 2. First fully connected layer.
  if (c < K) {
    S = 0;

#pragma unroll
    for (int i = 0; i < C; i++) {
      S += sharedData[i] * readw1(i, c);
    }

    S += b1[c];

    S = activate(S, activation);

    sharedData[c] = S;
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // 3. Second fully connected layer.
  S = 0;
  sycl::half B = 0;
#pragma unroll
  for (int i = 0; i < K; i++) {
    sycl::half val = sharedData[i];
    S += val * readw2(i, c);
    B += val * readw2(i, c + C);
  }
  S += b2[c];
  B += b2[c + C];

  // Sigmoid (only on the scale part).
  S = (sycl::half)(1.0f / (1.0f + sycl::exp(-(float)(S))));

// 4. Scale, and add skip connection, perform relu, and write to output.
#pragma unroll
  for (int i = 0; i < elementsPerThread; i++) {
    int localIndex = i * C + c;
    int inputIndex = n * C * elementsPerThread + localIndex;
    sycl::half val = localData[i].y() + localData[i].x() * S + B;

    // Relu activation function.
    val = (sycl::half)activate((float)val, activation);

    output[inputIndex] = val;
  }
#endif
}

bool Se_Fp16_NHWC(int N, int C, int numFc1Out, sycl::half* output,
                  const sycl::half* skip, const sycl::half* input,
                  const sycl::half* w1, const sycl::half* b1,
                  const sycl::half* w2, const sycl::half* b2,
                  const sycl::half* bPrev, ActivationFunction activation, sycl::queue &sycl_queue) {
  // TODO: Think of more elegant way to avoid this hardcoding :-/
  if (numFc1Out == 16) {
    if (C == 64) {
      /*
      DPCT1049:21: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      
      sycl_queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> sharedData_acc_ct1(
            sycl::range<1>(64), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                              sycl::range<3>(1, 1, C)),
            [=](sycl::nd_item<3> item_ct1) {
              SE_Layer_NHWC<64, 16>(output, skip, input, w1, b1, w2, b2, bPrev,
                                    activation, item_ct1,
                                    sharedData_acc_ct1.get_pointer());
            });
      });
    } else {
      // TODO: support other channel counts.
      printf("Error: channel count unsupported by SE layer\n");
      return;
    }
  } else if (numFc1Out == 32) {
    if (C == 64) {
      /*
      DPCT1049:22: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      
      sycl_queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> sharedData_acc_ct1(
            sycl::range<1>(64), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                              sycl::range<3>(1, 1, C)),
            [=](sycl::nd_item<3> item_ct1) {
              SE_Layer_NHWC<64, 32>(output, skip, input, w1, b1, w2, b2, bPrev,
                                    activation, item_ct1,
                                    sharedData_acc_ct1.get_pointer());
            });
      });
    } else if (C == 128) {
      /*
      DPCT1049:23: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      
      sycl_queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> sharedData_acc_ct1(
            sycl::range<1>(128), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                              sycl::range<3>(1, 1, C)),
            [=](sycl::nd_item<3> item_ct1) {
              SE_Layer_NHWC<128, 32>(output, skip, input, w1, b1, w2, b2, bPrev,
                                     activation, item_ct1,
                                     sharedData_acc_ct1.get_pointer());
            });
      });
    } else if (C == 192) {
      /*
      DPCT1049:24: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      
      sycl_queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> sharedData_acc_ct1(
            sycl::range<1>(192), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                              sycl::range<3>(1, 1, C)),
            [=](sycl::nd_item<3> item_ct1) {
              SE_Layer_NHWC<192, 32>(output, skip, input, w1, b1, w2, b2, bPrev,
                                     activation, item_ct1,
                                     sharedData_acc_ct1.get_pointer());
            });
      });
    } else if (C == 256) {
      /*
      DPCT1049:25: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      
      sycl_queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> sharedData_acc_ct1(
            sycl::range<1>(256), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                              sycl::range<3>(1, 1, C)),
            [=](sycl::nd_item<3> item_ct1) {
              SE_Layer_NHWC<256, 32>(output, skip, input, w1, b1, w2, b2, bPrev,
                                     activation, item_ct1,
                                     sharedData_acc_ct1.get_pointer());
            });
      });
    } else if (C == 320) {
      /*
      DPCT1049:26: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      
      sycl_queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> sharedData_acc_ct1(
            sycl::range<1>(320), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                              sycl::range<3>(1, 1, C)),
            [=](sycl::nd_item<3> item_ct1) {
              SE_Layer_NHWC<320, 32>(output, skip, input, w1, b1, w2, b2, bPrev,
                                     activation, item_ct1,
                                     sharedData_acc_ct1.get_pointer());
            });
      });
    } else if (C == 352) {
      /*
      DPCT1049:27: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      
      sycl_queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> sharedData_acc_ct1(
            sycl::range<1>(352), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                              sycl::range<3>(1, 1, C)),
            [=](sycl::nd_item<3> item_ct1) {
              SE_Layer_NHWC<352, 32>(output, skip, input, w1, b1, w2, b2, bPrev,
                                     activation, item_ct1,
                                     sharedData_acc_ct1.get_pointer());
            });
      });
    } else if (C == 384) {
      /*
      DPCT1049:28: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      
      sycl_queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> sharedData_acc_ct1(
            sycl::range<1>(384), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                              sycl::range<3>(1, 1, C)),
            [=](sycl::nd_item<3> item_ct1) {
              SE_Layer_NHWC<384, 32>(output, skip, input, w1, b1, w2, b2, bPrev,
                                     activation, item_ct1,
                                     sharedData_acc_ct1.get_pointer());
            });
      });
    } else {
      // TODO: support other channel counts.
      return false;
    }
  } else if (numFc1Out == 64) {
    if (C == 64) {
      /*
      DPCT1049:29: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      
      sycl_queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> sharedData_acc_ct1(
            sycl::range<1>(64), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                              sycl::range<3>(1, 1, C)),
            [=](sycl::nd_item<3> item_ct1) {
              SE_Layer_NHWC<64, 64>(output, skip, input, w1, b1, w2, b2, bPrev,
                                    activation, item_ct1,
                                    sharedData_acc_ct1.get_pointer());
            });
      });
    } else if (C == 128) {
      /*
      DPCT1049:30: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      
      sycl_queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> sharedData_acc_ct1(
            sycl::range<1>(128), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                              sycl::range<3>(1, 1, C)),
            [=](sycl::nd_item<3> item_ct1) {
              SE_Layer_NHWC<128, 64>(output, skip, input, w1, b1, w2, b2, bPrev,
                                     activation, item_ct1,
                                     sharedData_acc_ct1.get_pointer());
            });
      });
    } else if (C == 192) {
      /*
      DPCT1049:31: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      
      sycl_queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> sharedData_acc_ct1(
            sycl::range<1>(192), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                              sycl::range<3>(1, 1, C)),
            [=](sycl::nd_item<3> item_ct1) {
              SE_Layer_NHWC<192, 64>(output, skip, input, w1, b1, w2, b2, bPrev,
                                     activation, item_ct1,
                                     sharedData_acc_ct1.get_pointer());
            });
      });
    } else if (C == 256) {
      /*
      DPCT1049:32: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      
      sycl_queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> sharedData_acc_ct1(
            sycl::range<1>(256), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                              sycl::range<3>(1, 1, C)),
            [=](sycl::nd_item<3> item_ct1) {
              SE_Layer_NHWC<256, 64>(output, skip, input, w1, b1, w2, b2, bPrev,
                                     activation, item_ct1,
                                     sharedData_acc_ct1.get_pointer());
            });
      });
    } else if (C == 320) {
      /*
      DPCT1049:33: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      
      sycl_queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> sharedData_acc_ct1(
            sycl::range<1>(320), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                              sycl::range<3>(1, 1, C)),
            [=](sycl::nd_item<3> item_ct1) {
              SE_Layer_NHWC<320, 64>(output, skip, input, w1, b1, w2, b2, bPrev,
                                     activation, item_ct1,
                                     sharedData_acc_ct1.get_pointer());
            });
      });
    } else if (C == 384) {
      /*
      DPCT1049:34: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      
      sycl_queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<sycl::half, 1> sharedData_acc_ct1(
            sycl::range<1>(384), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                              sycl::range<3>(1, 1, C)),
            [=](sycl::nd_item<3> item_ct1) {
              SE_Layer_NHWC<384, 64>(output, skip, input, w1, b1, w2, b2, bPrev,
                                     activation, item_ct1,
                                     sharedData_acc_ct1.get_pointer());
            });
      });
    } else {
      // TODO: support other channel counts.
      return false;
    }
  } else {
    // TODO: support other sizes.
    return false;
  }
  return true;
}

}  // namespace sycldnn_backend
}  // namespace lczero
