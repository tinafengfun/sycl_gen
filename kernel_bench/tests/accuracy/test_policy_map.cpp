#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

namespace lczero {
namespace sycldnn_backend {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// V0: Baseline WG=256
void policyMap_V0(float* output, const float* input, const short* indices,
                  int N, int inputSize, int usedSize, int outputSize,
                  sycl::queue& queue) {
  const int kBlockSize = 256;
  int blocks = DivUp(N * usedSize, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        int n = tid / usedSize;
        int i = tid % usedSize;
        
        if (n >= N) return;
        
        int j = indices[i];
        if (j >= 0) {
          output[n * outputSize + j] = input[n * inputSize + i];
        }
      });
  queue.wait_and_throw();
}

// V1: WG=128
void policyMap_V1(float* output, const float* input, const short* indices,
                  int N, int inputSize, int usedSize, int outputSize,
                  sycl::queue& queue) {
  const int kBlockSize = 128;
  int blocks = DivUp(N * usedSize, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        int n = tid / usedSize;
        int i = tid % usedSize;
        
        if (n >= N) return;
        
        int j = indices[i];
        if (j >= 0) {
          output[n * outputSize + j] = input[n * inputSize + i];
        }
      });
  queue.wait_and_throw();
}

// V2: Grid-stride
void policyMap_V2(float* output, const float* input, const short* indices,
                  int N, int inputSize, int usedSize, int outputSize,
                  sycl::queue& queue) {
  const int kBlockSize = 128;
  int blocks = DivUp(N * usedSize, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        int gridSize = item.get_global_range(0);
        
        #pragma unroll 4
        for (int idx = tid; idx < N * usedSize; idx += gridSize) {
          int n = idx / usedSize;
          int i = idx % usedSize;
          
          if (n >= N) continue;
          
          int j = indices[i];
          if (j >= 0) {
            output[n * outputSize + j] = input[n * inputSize + i];
          }
        }
      });
  queue.wait_and_throw();
}

}  // namespace sycldnn_backend
}  // namespace lczero

void testKernel(const char* name,
                void (*kernel)(float*, const float*, const short*, int, int, int, int, sycl::queue&),
                float* d_output, float* d_input, short* d_indices,
                int N, int inputSize, int usedSize, int outputSize,
                sycl::queue& queue) {
  // Clear output
  queue.memset(d_output, 0, N * outputSize * sizeof(float)).wait();
  
  // Warmup
  for (int i = 0; i < 5; i++) {
    kernel(d_output, d_input, d_indices, N, inputSize, usedSize, outputSize, queue);
  }
  
  auto start = std::chrono::high_resolution_clock::now();
  const int iterations = 50;
  
  for (int i = 0; i < iterations; i++) {
    queue.memset(d_output, 0, N * outputSize * sizeof(float)).wait();
    kernel(d_output, d_input, d_indices, N, inputSize, usedSize, outputSize, queue);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  double timePerKernel = duration.count() / iterations;
  
  // Bandwidth: read input + indices, write output
  double bandwidth = (N * inputSize * sizeof(float) + usedSize * sizeof(short) + 
                      N * outputSize * sizeof(float)) / (timePerKernel * 1e-3) / 1e9;
  
  std::cout << name << ",N=" << N << ",Time=" << timePerKernel
            << " ms,Bandwidth=" << bandwidth << " GB/s" << std::endl;
}

int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  
  std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Version,N,Time_ms,Bandwidth_GB/s" << std::endl;
  
  // Policy map typical sizes for LCZero
  int inputSize = 1858;   // Input policy size
  int outputSize = 1858;  // Output policy size  
  int usedSize = 1858;    // Used indices
  
  std::vector<int> testSizes = {4, 8, 16, 32, 64};
  
  for (int N : testSizes) {
    float* d_output = sycl::malloc_device<float>(N * outputSize, queue);
    float* d_input = sycl::malloc_device<float>(N * inputSize, queue);
    short* d_indices = sycl::malloc_device<short>(usedSize, queue);
    
    // Initialize
    std::vector<float> h_input(N * inputSize, 0.5f);
    std::vector<short> h_indices(usedSize);
    for (int i = 0; i < usedSize; i++) {
      h_indices[i] = i;  // Identity mapping
    }
    queue.memcpy(d_input, h_input.data(), N * inputSize * sizeof(float)).wait();
    queue.memcpy(d_indices, h_indices.data(), usedSize * sizeof(short)).wait();
    
    testKernel("V0", lczero::sycldnn_backend::policyMap_V0,
               d_output, d_input, d_indices, N, inputSize, usedSize, outputSize, queue);
    testKernel("V1", lczero::sycldnn_backend::policyMap_V1,
               d_output, d_input, d_indices, N, inputSize, usedSize, outputSize, queue);
    testKernel("V2", lczero::sycldnn_backend::policyMap_V2,
               d_output, d_input, d_indices, N, inputSize, usedSize, outputSize, queue);
    
    sycl::free(d_output, queue);
    sycl::free(d_input, queue);
    sycl::free(d_indices, queue);
  }
  
  return 0;
}
