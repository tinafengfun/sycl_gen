#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

namespace lczero {
namespace sycldnn_backend {

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

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

[[gnu::always_inline]]
inline float activate(float cVal, ActivationFunction activation) {
  switch (activation) {
    case ACTIVATION_RELU:
      if (cVal < 0) cVal = 0;
      break;
    case ACTIVATION_TANH:
      cVal = sycl::tanh(cVal);
      break;
    case ACTIVATION_SIGMOID:
      cVal = 1.0f / (1.0f + sycl::exp(-cVal));
      break;
    case ACTIVATION_MISH:
      cVal = cVal * sycl::tanh(sycl::log(1.0f + sycl::exp(cVal)));
      break;
    case ACTIVATION_SWISH:
      cVal = cVal / (1.0f + sycl::exp(-cVal));
      break;
    default:
      break;
  }
  return cVal;
}

// V0: Baseline WG=256
void globalScale_V0(float* output, const float* input, const float* scaleBias,
                    int N, int C, sycl::queue& queue) {
  const int kBlockSize = 256;
  int totalElements = N * C * 8 * 8;
  int blocks = DivUp(totalElements, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize), 
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid >= totalElements) return;
        
        int nc = tid / 64;
        int n = nc / C;
        int c = nc % C;
        
        float val = input[tid];
        int startIdx = n * 2 * C;
        float s = scaleBias[startIdx + c];
        s = 1.0f / (1.0f + sycl::exp(-s));
        float b = scaleBias[startIdx + c + C];
        
        float op = val * s + b;
        op = activate(op, ACTIVATION_RELU);
        output[tid] = op;
      });
  queue.wait_and_throw();
}

// V1: WG=128
void globalScale_V1(float* output, const float* input, const float* scaleBias,
                    int N, int C, sycl::queue& queue) {
  const int kBlockSize = 128;
  int totalElements = N * C * 8 * 8;
  int blocks = DivUp(totalElements, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize), 
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid >= totalElements) return;
        
        int nc = tid / 64;
        int n = nc / C;
        int c = nc % C;
        
        float val = input[tid];
        int startIdx = n * 2 * C;
        float s = scaleBias[startIdx + c];
        s = 1.0f / (1.0f + sycl::exp(-s));
        float b = scaleBias[startIdx + c + C];
        
        float op = val * s + b;
        op = activate(op, ACTIVATION_RELU);
        output[tid] = op;
      });
  queue.wait_and_throw();
}

// V2: Grid-stride with loop unrolling
void globalScale_V2(float* output, const float* input, const float* scaleBias,
                    int N, int C, sycl::queue& queue) {
  const int kBlockSize = 128;
  int totalElements = N * C * 8 * 8;
  int blocks = DivUp(totalElements, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize), 
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        int gridSize = item.get_global_range(0);
        
        #pragma unroll 4
        for (int idx = tid; idx < totalElements; idx += gridSize) {
          int nc = idx / 64;
          int n = nc / C;
          int c = nc % C;
          
          float val = input[idx];
          int startIdx = n * 2 * C;
          float s = scaleBias[startIdx + c];
          s = 1.0f / (1.0f + sycl::exp(-s));
          float b = scaleBias[startIdx + c + C];
          
          float op = val * s + b;
          op = activate(op, ACTIVATION_RELU);
          output[idx] = op;
        }
      });
  queue.wait_and_throw();
}

}  // namespace sycldnn_backend
}  // namespace lczero

void testKernel(const char* name, 
                void (*kernel)(float*, const float*, const float*, int, int, sycl::queue&),
                float* d_output, float* d_input, float* d_scaleBias,
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
  
  double gflops = (totalElements * 8.0) / (timePerKernel * 1e-3) / 1e9;  // 8 ops per element
  double bandwidth = (3.0 * totalElements * sizeof(float)) / (timePerKernel * 1e-3) / 1e9;
  
  std::cout << name << ",N=" << N << ",C=" << C << ",Time=" << timePerKernel 
            << " ms,GFLOPS=" << gflops << ",Bandwidth=" << bandwidth << " GB/s" << std::endl;
}

int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  
  std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Version,N,C,Time_ms,GFLOPS,Bandwidth_GB/s" << std::endl;
  
  std::vector<std::pair<int, int>> testSizes = {
    {4, 64},
    {8, 128},
    {16, 256},
    {32, 512}
  };
  
  for (const auto& [N, C] : testSizes) {
    int totalElements = N * C * 8 * 8;
    int scaleBiasSize = N * 2 * C;
    
    float* d_output = sycl::malloc_device<float>(totalElements, queue);
    float* d_input = sycl::malloc_device<float>(totalElements, queue);
    float* d_scaleBias = sycl::malloc_device<float>(scaleBiasSize, queue);
    
    // Initialize data
    std::vector<float> h_input(totalElements, 0.5f);
    std::vector<float> h_scaleBias(scaleBiasSize, 0.1f);
    queue.memcpy(d_input, h_input.data(), totalElements * sizeof(float)).wait();
    queue.memcpy(d_scaleBias, h_scaleBias.data(), scaleBiasSize * sizeof(float)).wait();
    
    testKernel("V0", lczero::sycldnn_backend::globalScale_V0, d_output, d_input, d_scaleBias,
               N, C, totalElements, queue);
    testKernel("V1", lczero::sycldnn_backend::globalScale_V1, d_output, d_input, d_scaleBias,
               N, C, totalElements, queue);
    testKernel("V2", lczero::sycldnn_backend::globalScale_V2, d_output, d_input, d_scaleBias,
               N, C, totalElements, queue);
    
    sycl::free(d_output, queue);
    sycl::free(d_input, queue);
    sycl::free(d_scaleBias, queue);
  }
  
  return 0;
}
