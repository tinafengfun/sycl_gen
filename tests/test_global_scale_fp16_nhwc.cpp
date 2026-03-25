#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

namespace lczero {
namespace sycldnn_backend {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

enum ActivationFunction {
  ACTIVATION_NONE = 0,
  ACTIVATION_RELU = 1,
  ACTIVATION_TANH = 2,
  ACTIVATION_SIGMOID = 3,
  ACTIVATION_SELU = 4,
  ACTIVATION_SWISH = 5,
  ACTIVATION_MISH = 6,
  ACTIVATION_RELU_2 = 7,
  ACTIVATION_SOFTMAX = 8
};

[[gnu::always_inline]]
inline float activate(float cVal, ActivationFunction activation) {
  if (activation == ACTIVATION_RELU) {
    if (cVal < 0) cVal = 0;
  }
  return cVal;
}

// V0: Baseline FP16 WG=256
void globalScale_fp16_nhwc_V0(sycl::half* output, const sycl::half* input,
                               const sycl::half* scaleBias, int N, int C,
                               sycl::queue& queue) {
  const int kBlockSize = 256;
  int totalElements = N * 8 * 8 * C;
  int blocks = DivUp(totalElements, kBlockSize);
  int HWC = 8 * 8 * C;
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid >= totalElements) return;
        
        int c = tid % C;
        int n = tid / HWC;
        
        float val1 = static_cast<float>(input[tid]);
        float val2 = static_cast<float>(output[tid]);
        
        int startIdx = n * 2 * C;
        float s = static_cast<float>(scaleBias[startIdx + c]);
        s = 1.0f / (1.0f + sycl::exp(-s));
        float b = static_cast<float>(scaleBias[startIdx + c + C]);
        
        float op = val1 * s + val2 + b;
        op = activate(op, ACTIVATION_RELU);
        output[tid] = static_cast<sycl::half>(op);
      });
  queue.wait_and_throw();
}

// V1: FP16 WG=128
void globalScale_fp16_nhwc_V1(sycl::half* output, const sycl::half* input,
                               const sycl::half* scaleBias, int N, int C,
                               sycl::queue& queue) {
  const int kBlockSize = 128;
  int totalElements = N * 8 * 8 * C;
  int blocks = DivUp(totalElements, kBlockSize);
  int HWC = 8 * 8 * C;
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid >= totalElements) return;
        
        int c = tid % C;
        int n = tid / HWC;
        
        float val1 = static_cast<float>(input[tid]);
        float val2 = static_cast<float>(output[tid]);
        
        int startIdx = n * 2 * C;
        float s = static_cast<float>(scaleBias[startIdx + c]);
        s = 1.0f / (1.0f + sycl::exp(-s));
        float b = static_cast<float>(scaleBias[startIdx + c + C]);
        
        float op = val1 * s + val2 + b;
        op = activate(op, ACTIVATION_RELU);
        output[tid] = static_cast<sycl::half>(op);
      });
  queue.wait_and_throw();
}

// V2: Grid-stride
void globalScale_fp16_nhwc_V2(sycl::half* output, const sycl::half* input,
                               const sycl::half* scaleBias, int N, int C,
                               sycl::queue& queue) {
  const int kBlockSize = 128;
  int totalElements = N * 8 * 8 * C;
  int blocks = DivUp(totalElements, kBlockSize);
  int HWC = 8 * 8 * C;
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        int gridSize = item.get_global_range(0);
        
        #pragma unroll 4
        for (int idx = tid; idx < totalElements; idx += gridSize) {
          int c = idx % C;
          int n = idx / HWC;
          
          float val1 = static_cast<float>(input[idx]);
          float val2 = static_cast<float>(output[idx]);
          
          int startIdx = n * 2 * C;
          float s = static_cast<float>(scaleBias[startIdx + c]);
          s = 1.0f / (1.0f + sycl::exp(-s));
          float b = static_cast<float>(scaleBias[startIdx + c + C]);
          
          float op = val1 * s + val2 + b;
          op = activate(op, ACTIVATION_RELU);
          output[idx] = static_cast<sycl::half>(op);
        }
      });
  queue.wait_and_throw();
}

}  // namespace sycldnn_backend
}  // namespace lczero

void testKernel(const char* name,
                void (*kernel)(sycl::half*, const sycl::half*, const sycl::half*, int, int, sycl::queue&),
                sycl::half* d_output, sycl::half* d_input, sycl::half* d_scaleBias,
                int N, int C, int totalElements, sycl::queue& queue) {
  // Warmup
  for (int i = 0; i < 5; i++) {
    kernel(d_output, d_input, d_scaleBias, N, C, queue);
  }
  
  auto start = std::chrono::high_resolution_clock::now();
  const int iterations = 50;
  
  for (int i = 0; i < iterations; i++) {
    kernel(d_output, d_input, d_scaleBias, N, C, queue);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  double timePerKernel = duration.count() / iterations;
  
  // Bandwidth calculation (read input/output/scaleBias, write output)
  double bandwidth = ((3.0 * totalElements + N * 2 * C) * sizeof(sycl::half)) / (timePerKernel * 1e-3) / 1e9;
  
  std::cout << name << ",N=" << N << ",C=" << C << ",Time=" << timePerKernel
            << " ms,Bandwidth=" << bandwidth << " GB/s" << std::endl;
}

int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  
  std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Version,N,C,Time_ms,Bandwidth_GB/s" << std::endl;
  
  std::vector<std::pair<int, int>> testSizes = {
    {4, 64},
    {8, 128},
    {16, 256},
    {32, 512}
  };
  
  for (const auto& [N, C] : testSizes) {
    int totalElements = N * 8 * 8 * C;
    int scaleBiasSize = N * 2 * C;
    
    sycl::half* d_output = sycl::malloc_device<sycl::half>(totalElements, queue);
    sycl::half* d_input = sycl::malloc_device<sycl::half>(totalElements, queue);
    sycl::half* d_scaleBias = sycl::malloc_device<sycl::half>(scaleBiasSize, queue);
    
    // Initialize
    std::vector<sycl::half> h_input(totalElements, sycl::half(0.5f));
    std::vector<sycl::half> h_scaleBias(scaleBiasSize, sycl::half(0.1f));
    queue.memcpy(d_input, h_input.data(), totalElements * sizeof(sycl::half)).wait();
    queue.memcpy(d_scaleBias, h_scaleBias.data(), scaleBiasSize * sizeof(sycl::half)).wait();
    
    testKernel("V0", lczero::sycldnn_backend::globalScale_fp16_nhwc_V0,
               d_output, d_input, d_scaleBias, N, C, totalElements, queue);
    testKernel("V1", lczero::sycldnn_backend::globalScale_fp16_nhwc_V1,
               d_output, d_input, d_scaleBias, N, C, totalElements, queue);
    testKernel("V2", lczero::sycldnn_backend::globalScale_fp16_nhwc_V2,
               d_output, d_input, d_scaleBias, N, C, totalElements, queue);
    
    sycl::free(d_output, queue);
    sycl::free(d_input, queue);
    sycl::free(d_scaleBias, queue);
  }
  
  return 0;
}
