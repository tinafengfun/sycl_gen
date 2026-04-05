/**
 * Quantization SYCL Implementation
 * 
 * Converts FP32 to INT8 with per-row scaling
 * Formula: int8_output = clamp(input / scale, -127, 127)
 * 
 * Kernel Type: Type A (Element-wise)
 * Optimization: Vectorized loads, BF16 input support
 */

#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <cstdint>
#include <cmath>

namespace turbodiffusion {
namespace sycl_backend {

/**
 * FP32 Quantization - Basic version
 */
class QuantizeFP32 {
public:
  static void launch(sycl::queue& q, const float* input, const float* scale,
                     int8_t* output, int64_t m, int64_t n) {
    q.parallel_for(sycl::range<1>(m * n), [=](sycl::item<1> item) {
      int64_t idx = item.get_id(0);
      int64_t row = idx / n;
      
      float val = input[idx];
      float scaled = val / scale[row];
      
      // Clamp to int8 range
      scaled = sycl::clamp(scaled, -127.0f, 127.0f);
      output[idx] = static_cast<int8_t>(sycl::rint(scaled));
    });
    q.wait();
  }
};

/**
 * BF16 Quantization - 2x memory bandwidth for input
 * Input: BF16, Scale: FP32, Output: INT8
 */
class QuantizeBF16Input {
public:
  using bfloat16 = sycl::ext::oneapi::bfloat16;
  
  static void launch(sycl::queue& q, const bfloat16* input, const float* scale,
                     int8_t* output, int64_t m, int64_t n) {
    q.parallel_for(sycl::range<1>(m * n), [=](sycl::item<1> item) {
      int64_t idx = item.get_id(0);
      int64_t row = idx / n;
      
      float val = static_cast<float>(input[idx]);
      float scaled = val / scale[row];
      
      // Clamp to int8 range
      scaled = sycl::clamp(scaled, -127.0f, 127.0f);
      output[idx] = static_cast<int8_t>(sycl::rint(scaled));
    });
    q.wait();
  }
};

/**
 * Vectorized FP32 Quantization - Type A optimization
 * Process 4 elements per thread for better memory bandwidth
 */
class QuantizeFP32Vec4 {
public:
  static void launch(sycl::queue& q, const float* input, const float* scale,
                     int8_t* output, int64_t m, int64_t n) {
    const int vec_size = 4;
    int64_t total_elements = m * n;
    int64_t num_vectors = (total_elements + vec_size - 1) / vec_size;
    
    q.parallel_for(sycl::range<1>(num_vectors), [=](sycl::item<1> item) {
      int64_t vec_idx = item.get_id(0);
      int64_t base_idx = vec_idx * vec_size;
      
      #pragma unroll
      for (int i = 0; i < vec_size; i++) {
        int64_t idx = base_idx + i;
        if (idx < total_elements) {
          int64_t row = idx / n;
          
          float val = input[idx];
          float scaled = val / scale[row];
          scaled = sycl::clamp(scaled, -127.0f, 127.0f);
          output[idx] = static_cast<int8_t>(sycl::rint(scaled));
        }
      }
    });
    q.wait();
  }
};

/**
 * Vectorized BF16 Quantization
 * Best of both: BF16 input bandwidth + vectorized processing
 */
class QuantizeBF16Vec4 {
public:
  using bfloat16 = sycl::ext::oneapi::bfloat16;
  
  static void launch(sycl::queue& q, const bfloat16* input, const float* scale,
                     int8_t* output, int64_t m, int64_t n) {
    const int vec_size = 4;
    int64_t total_elements = m * n;
    int64_t num_vectors = (total_elements + vec_size - 1) / vec_size;
    
    q.parallel_for(sycl::range<1>(num_vectors), [=](sycl::item<1> item) {
      int64_t vec_idx = item.get_id(0);
      int64_t base_idx = vec_idx * vec_size;
      
      #pragma unroll
      for (int i = 0; i < vec_size; i++) {
        int64_t idx = base_idx + i;
        if (idx < total_elements) {
          int64_t row = idx / n;
          
          float val = static_cast<float>(input[idx]);
          float scaled = val / scale[row];
          scaled = sycl::clamp(scaled, -127.0f, 127.0f);
          output[idx] = static_cast<int8_t>(sycl::rint(scaled));
        }
      }
    });
    q.wait();
  }
};

/**
 * Work-group tuned quantization
 * Type A kernel with configurable WG size
 */
template <int WG_SIZE>
class QuantizeTuned {
public:
  static void launch(sycl::queue& q, const float* input, const float* scale,
                     int8_t* output, int64_t m, int64_t n) {
    int64_t total = m * n;
    int num_wg = (total + WG_SIZE - 1) / WG_SIZE;
    
    q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(num_wg * WG_SIZE), 
                        sycl::range<1>(WG_SIZE)),
      [=](sycl::nd_item<1> item) {
        int64_t idx = item.get_global_id(0);
        if (idx >= total) return;
        
        int64_t row = idx / n;
        
        float val = input[idx];
        float scaled = val / scale[row];
        scaled = sycl::clamp(scaled, -127.0f, 127.0f);
        output[idx] = static_cast<int8_t>(sycl::rint(scaled));
      }
    );
    q.wait();
  }
};

/**
 * BF16 Work-group tuned quantization
 */
template <int WG_SIZE>
class QuantizeBF16Tuned {
public:
  using bfloat16 = sycl::ext::oneapi::bfloat16;
  
  static void launch(sycl::queue& q, const bfloat16* input, const float* scale,
                     int8_t* output, int64_t m, int64_t n) {
    int64_t total = m * n;
    int num_wg = (total + WG_SIZE - 1) / WG_SIZE;
    
    q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(num_wg * WG_SIZE), 
                        sycl::range<1>(WG_SIZE)),
      [=](sycl::nd_item<1> item) {
        int64_t idx = item.get_global_id(0);
        if (idx >= total) return;
        
        int64_t row = idx / n;
        
        float val = static_cast<float>(input[idx]);
        float scaled = val / scale[row];
        scaled = sycl::clamp(scaled, -127.0f, 127.0f);
        output[idx] = static_cast<int8_t>(sycl::rint(scaled));
      }
    );
    q.wait();
  }
};

// Convenience interfaces
inline void quantize_fp32(sycl::queue& q, const float* input, const float* scale,
                          int8_t* output, int64_t m, int64_t n) {
  QuantizeFP32::launch(q, input, scale, output, m, n);
}

inline void quantize_bf16(sycl::queue& q, 
                          const sycl::ext::oneapi::bfloat16* input,
                          const float* scale,
                          int8_t* output, int64_t m, int64_t n) {
  QuantizeBF16Input::launch(q, input, scale, output, m, n);
}

inline void quantize_fp32_vec4(sycl::queue& q, const float* input, const float* scale,
                               int8_t* output, int64_t m, int64_t n) {
  QuantizeFP32Vec4::launch(q, input, scale, output, m, n);
}

inline void quantize_bf16_vec4(sycl::queue& q,
                               const sycl::ext::oneapi::bfloat16* input,
                               const float* scale,
                               int8_t* output, int64_t m, int64_t n) {
  QuantizeBF16Vec4::launch(q, input, scale, output, m, n);
}

template <int WG_SIZE>
inline void quantize_tuned(sycl::queue& q, const float* input, const float* scale,
                           int8_t* output, int64_t m, int64_t n) {
  QuantizeTuned<WG_SIZE>::launch(q, input, scale, output, m, n);
}

template <int WG_SIZE>
inline void quantize_bf16_tuned(sycl::queue& q,
                                const sycl::ext::oneapi::bfloat16* input,
                                const float* scale,
                                int8_t* output, int64_t m, int64_t n) {
  QuantizeBF16Tuned<WG_SIZE>::launch(q, input, scale, output, m, n);
}

} // namespace sycl_backend
} // namespace turbodiffusion
