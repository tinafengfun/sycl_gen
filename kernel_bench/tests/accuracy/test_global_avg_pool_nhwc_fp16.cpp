#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

namespace lczero {
namespace sycldnn_backend {

// V0: Baseline FP16 reduction
void globalAvgPool_NHWC_fp16_V0(sycl::half* output, const sycl::half* input,
                                 int N, int C, sycl::queue& queue) {
  const int kPlaneSize = 64;
  int inputSize = N * C * kPlaneSize;
  int outputSize = N * C;
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(N * C), sycl::range<1>(C)),
      [=](sycl::nd_item<1> item) {
        int blockStart = item.get_group(0) * item.get_local_range(0);
        float S = 0;
        
        #pragma unroll
        for (int i = 0; i < kPlaneSize; i++) {
          int localIndex = i * item.get_local_range(0) + item.get_local_id(0);
          int inputIndex = blockStart * kPlaneSize + localIndex;
          if (inputIndex < inputSize) {
            S += static_cast<float>(input[inputIndex]);
          }
        }
        
        float avg = S / kPlaneSize;
        int opIndex = blockStart + item.get_local_id(0);
        if (opIndex < outputSize) {
          output[opIndex] = static_cast<sycl::half>(avg);
        }
      });
  queue.wait_and_throw();
}

// V1: WG=128
void globalAvgPool_NHWC_fp16_V1(sycl::half* output, const sycl::half* input,
                                 int N, int C, sycl::queue& queue) {
  const int kPlaneSize = 64;
  int inputSize = N * C * kPlaneSize;
  int outputSize = N * C;
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(N * C), sycl::range<1>(128)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid >= outputSize) return;
        
        float S = 0;
        int inputStart = tid * kPlaneSize;
        
        #pragma unroll 8
        for (int i = 0; i < kPlaneSize; i++) {
          int inputIndex = inputStart + i;
          if (inputIndex < inputSize) {
            S += static_cast<float>(input[inputIndex]);
          }
        }
        
        output[tid] = static_cast<sycl::half>(S / kPlaneSize);
      });
  queue.wait_and_throw();
}

// V2: Single-thread per output
void globalAvgPool_NHWC_fp16_V2(sycl::half* output, const sycl::half* input,
                                 int N, int C, sycl::queue& queue) {
  const int kPlaneSize = 64;
  int outputSize = N * C;
  
  queue.parallel_for(
      sycl::range<1>(outputSize),
      [=](sycl::item<1> item) {
        int idx = item.get_id(0);
        float S = 0;
        int inputStart = idx * kPlaneSize;
        
        #pragma unroll
        for (int i = 0; i < kPlaneSize; i++) {
          S += static_cast<float>(input[inputStart + i]);
        }
        
        output[idx] = static_cast<sycl::half>(S / kPlaneSize);
      });
  queue.wait_and_throw();
}

}  // namespace sycldnn_backend
}  // namespace lczero

void testKernel(const char* name,
                void (*kernel)(sycl::half*, const sycl::half*, int, int, sycl::queue&),
                sycl::half* d_output, sycl::half* d_input,
                int N, int C, sycl::queue& queue) {
  int inputSize = N * C * 64;
  int outputSize = N * C;
  
  // Warmup
  for (int i = 0; i < 5; i++) {
    kernel(d_output, d_input, N, C, queue);
  }
  
  auto start = std::chrono::high_resolution_clock::now();
  const int iterations = 50;
  
  for (int i = 0; i < iterations; i++) {
    kernel(d_output, d_input, N, C, queue);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  double timePerKernel = duration.count() / iterations;
  
  double gflops = (outputSize * 64.0) / (timePerKernel * 1e-3) / 1e9;
  double bandwidth = ((inputSize + outputSize) * sizeof(sycl::half)) / (timePerKernel * 1e-3) / 1e9;
  
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
    int inputSize = N * C * 64;
    int outputSize = N * C;
    
    sycl::half* d_output = sycl::malloc_device<sycl::half>(outputSize, queue);
    sycl::half* d_input = sycl::malloc_device<sycl::half>(inputSize, queue);
    
    // Initialize
    std::vector<sycl::half> h_input(inputSize, sycl::half(0.5f));
    queue.memcpy(d_input, h_input.data(), inputSize * sizeof(sycl::half)).wait();
    
    testKernel("V0", lczero::sycldnn_backend::globalAvgPool_NHWC_fp16_V0,
               d_output, d_input, N, C, queue);
    testKernel("V1", lczero::sycldnn_backend::globalAvgPool_NHWC_fp16_V1,
               d_output, d_input, N, C, queue);
    testKernel("V2", lczero::sycldnn_backend::globalAvgPool_NHWC_fp16_V2,
               d_output, d_input, N, C, queue);
    
    sycl::free(d_output, queue);
    sycl::free(d_input, queue);
  }
  
  return 0;
}
