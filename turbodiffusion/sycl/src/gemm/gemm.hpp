/**
 * GEMM SYCL Implementation
 * 
 * Matrix Multiplication: C[M,N] = A[M,K] × B[K,N]
 * 
 * Kernel Type: Type D-Small (Matrix < 256x256)
 * Strategy: Single-thread-per-row for small M
 */

#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cmath>

namespace turbodiffusion {
namespace sycl_backend {

/**
 * Basic FP32 GEMM - Single-thread-per-row
 * Optimal for small M (Type D-Small per XMX skill)
 */
class GemmFP32 {
public:
  static void launch(sycl::queue& q, const float* A, const float* B, float* C,
                     int M, int N, int K) {
    // Each thread computes one row of C (N elements)
    q.parallel_for(sycl::range<1>(M), [=](sycl::item<1> item) {
      int m = item.get_id(0);
      if (m >= M) return;
      
      // Compute row m of C
      for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < K; k++) {
          sum += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = sum;
      }
    });
    q.wait();
  }
};

/**
 * INT8 Input GEMM - Dequantize on-the-fly
 * Input: INT8, Output: FP32
 */
class GemmINT8 {
public:
  static void launch(sycl::queue& q, const int8_t* A, const int8_t* B, float* C,
                     int M, int N, int K) {
    q.parallel_for(sycl::range<1>(M), [=](sycl::item<1> item) {
      int m = item.get_id(0);
      if (m >= M) return;
      
      for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < K; k++) {
          float a = static_cast<float>(A[m * K + k]);
          float b = static_cast<float>(B[k * N + n]);
          sum += a * b;
        }
        C[m * N + n] = sum;
      }
    });
    q.wait();
  }
};

/**
 * Work-group tuned GEMM - Collaborative computation
 * Type D-Small with configurable WG size
 */
template <int WG_SIZE>
class GemmTuned {
public:
  static void launch(sycl::queue& q, const float* A, const float* B, float* C,
                     int M, int N, int K) {
    q.submit([&](sycl::handler& h) {
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(M * WG_SIZE), sycl::range<1>(WG_SIZE)),
        [=](sycl::nd_item<1> item) {
          int m = item.get_group(0);
          int lid = item.get_local_id(0);
          
          if (m >= M) return;
          
          // Each thread computes partial sum for subset of N
          for (int n = lid; n < N; n += WG_SIZE) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
              sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
          }
        }
      );
    });
    q.wait();
  }
};

/**
 * INT8 Work-group tuned GEMM
 */
template <int WG_SIZE>
class GemmINT8Tuned {
public:
  static void launch(sycl::queue& q, const int8_t* A, const int8_t* B, float* C,
                     int M, int N, int K) {
    q.submit([&](sycl::handler& h) {
      h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(M * WG_SIZE), sycl::range<1>(WG_SIZE)),
        [=](sycl::nd_item<1> item) {
          int m = item.get_group(0);
          int lid = item.get_local_id(0);
          
          if (m >= M) return;
          
          for (int n = lid; n < N; n += WG_SIZE) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
              float a = static_cast<float>(A[m * K + k]);
              float b = static_cast<float>(B[k * N + n]);
              sum += a * b;
            }
            C[m * N + n] = sum;
          }
        }
      );
    });
    q.wait();
  }
};

// Convenience interfaces
inline void gemm_fp32(sycl::queue& q, const float* A, const float* B, float* C,
                      int M, int N, int K) {
  GemmFP32::launch(q, A, B, C, M, N, K);
}

inline void gemm_int8(sycl::queue& q, const int8_t* A, const int8_t* B, float* C,
                      int M, int N, int K) {
  GemmINT8::launch(q, A, B, C, M, N, K);
}

template <int WG_SIZE>
inline void gemm_tuned(sycl::queue& q, const float* A, const float* B, float* C,
                       int M, int N, int K) {
  GemmTuned<WG_SIZE>::launch(q, A, B, C, M, N, K);
}

template <int WG_SIZE>
inline void gemm_int8_tuned(sycl::queue& q, const int8_t* A, const int8_t* B, float* C,
                            int M, int N, int K) {
  GemmINT8Tuned<WG_SIZE>::launch(q, A, B, C, M, N, K);
}

} // namespace sycl_backend
} // namespace turbodiffusion
