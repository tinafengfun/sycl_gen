#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

namespace lczero {
namespace sycldnn_backend {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// V0: Baseline WG=256
void copyTypeConverted_V0(float* op, float* ip, int N, sycl::queue& stream) {
  const int kBlockSize = 256;
  int blocks = DivUp(N, kBlockSize);
  
  stream.parallel_for(
      sycl::nd_range<1>(blocks * kBlockSize, kBlockSize),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid >= N) return;
        op[tid] = ip[tid];
      });
  stream.wait_and_throw();
}

// V1: WG=128
void copyTypeConverted_V1(float* op, float* ip, int N, sycl::queue& stream) {
  const int kBlockSize = 128;
  int blocks = DivUp(N, kBlockSize);
  
  stream.parallel_for(
      sycl::nd_range<1>(blocks * kBlockSize, kBlockSize),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid >= N) return;
        op[tid] = ip[tid];
      });
  stream.wait_and_throw();
}

// V2: Grid-stride with unroll
void copyTypeConverted_V2(float* op, float* ip, int N, sycl::queue& stream) {
  const int kBlockSize = 128;
  int blocks = DivUp(N, kBlockSize);
  
  stream.parallel_for(
      sycl::nd_range<1>(blocks * kBlockSize, kBlockSize),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        int gridSize = item.get_global_range(0);
        
        #pragma unroll 4
        for (int idx = tid; idx < N; idx += gridSize) {
          op[idx] = ip[idx];
        }
      });
  stream.wait_and_throw();
}

} // namespace sycldnn_backend
} // namespace lczero

void testKernel(const char* name,
                void (*kernel)(float*, float*, int, sycl::queue&),
                float* d_output, float* d_input, int N, sycl::queue& queue) {
  // Warmup
  for (int i = 0; i < 5; i++) {
    kernel(d_output, d_input, N, queue);
  }
  
  auto start = std::chrono::high_resolution_clock::now();
  const int iterations = 50;
  
  for (int i = 0; i < iterations; i++) {
    kernel(d_output, d_input, N, queue);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  double timePerKernel = duration.count() / iterations;
  
  // Bandwidth = read + write = 2 * N * sizeof(float)
  double bandwidth = (2.0 * N * sizeof(float)) / (timePerKernel * 1e-3) / 1e9;
  
  std::cout << name << ",N=" << N << ",Time=" << timePerKernel 
            << " ms,Bandwidth=" << bandwidth << " GB/s" << std::endl;
}

int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  
  std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Version,N,Time_ms,Bandwidth_GB/s" << std::endl;
  
  std::vector<int> testSizes = {1024, 4096, 16384, 65536, 262144, 1048576};
  
  for (int N : testSizes) {
    float* d_output = sycl::malloc_device<float>(N, queue);
    float* d_input = sycl::malloc_device<float>(N, queue);
    
    // Initialize
    std::vector<float> h_input(N, 1.0f);
    queue.memcpy(d_input, h_input.data(), N * sizeof(float)).wait();
    
    testKernel("V0", lczero::sycldnn_backend::copyTypeConverted_V0,
               d_output, d_input, N, queue);
    testKernel("V1", lczero::sycldnn_backend::copyTypeConverted_V1,
               d_output, d_input, N, queue);
    testKernel("V2", lczero::sycldnn_backend::copyTypeConverted_V2,
               d_output, d_input, N, queue);
    
    sycl::free(d_output, queue);
    sycl::free(d_input, queue);
  }
  
  return 0;
}
