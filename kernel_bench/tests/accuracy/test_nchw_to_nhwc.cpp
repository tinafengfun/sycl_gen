#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

namespace lczero {
namespace sycldnn_backend {

// V0: Baseline
void nchwToNhwc_V0(float* output, float* input, int N, int C, int H, int W, sycl::queue& queue) {
  int totalElements = N * C * H * W;
  const int blockSize = 256;
  int blocks = (totalElements + blockSize - 1) / blockSize;
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * blockSize), sycl::range<1>(blockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid >= totalElements) return;
        
        int idx = tid;
        int c = idx % C;
        idx /= C;
        int w = idx % W;
        idx /= W;
        int h = idx % H;
        int n = idx / H;
        
        int srcIdx = ((n * C + c) * H + h) * W + w;
        output[tid] = input[srcIdx];
      });
  queue.wait_and_throw();
}

// V1: WG=128
void nchwToNhwc_V1(float* output, float* input, int N, int C, int H, int W, sycl::queue& queue) {
  int totalElements = N * C * H * W;
  const int blockSize = 128;
  int blocks = (totalElements + blockSize - 1) / blockSize;
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * blockSize), sycl::range<1>(blockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid >= totalElements) return;
        
        int idx = tid;
        int c = idx % C;
        idx /= C;
        int w = idx % W;
        idx /= W;
        int h = idx % H;
        int n = idx / H;
        
        int srcIdx = ((n * C + c) * H + h) * W + w;
        output[tid] = input[srcIdx];
      });
  queue.wait_and_throw();
}

// V2: Grid-stride with unroll
void nchwToNhwc_V2(float* output, float* input, int N, int C, int H, int W, sycl::queue& queue) {
  int totalElements = N * C * H * W;
  const int blockSize = 128;
  int blocks = (totalElements + blockSize - 1) / blockSize;
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * blockSize), sycl::range<1>(blockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        int gridSize = item.get_global_range(0);
        
        #pragma unroll 4
        for (int idx = tid; idx < totalElements; idx += gridSize) {
          int tmp = idx;
          int c = tmp % C;
          tmp /= C;
          int w = tmp % W;
          tmp /= W;
          int h = tmp % H;
          int n = tmp / H;
          
          int srcIdx = ((n * C + c) * H + h) * W + w;
          output[idx] = input[srcIdx];
        }
      });
  queue.wait_and_throw();
}

}  // namespace sycldnn_backend
}  // namespace lczero

void testKernel(const char* name, 
                void (*kernel)(float*, float*, int, int, int, int, sycl::queue&),
                std::vector<float>& input, std::vector<float>& output,
                int N, int C, int H, int W, sycl::queue& queue) {
  int totalElements = N * C * H * W;
  float* d_input = sycl::malloc_device<float>(totalElements, queue);
  float* d_output = sycl::malloc_device<float>(totalElements, queue);
  
  queue.memcpy(d_input, input.data(), totalElements * sizeof(float)).wait();
  
  // Warmup
  for (int i = 0; i < 5; i++) {
    kernel(d_output, d_input, N, C, H, W, queue);
  }
  
  auto start = std::chrono::high_resolution_clock::now();
  const int iterations = 50;
  
  for (int i = 0; i < iterations; i++) {
    kernel(d_output, d_input, N, C, H, W, queue);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  double timePerKernel = duration.count() / iterations;
  
  // Bandwidth = 2 * totalElements * sizeof(float) / time / 1e9 GB/s
  double bandwidth = (2.0 * totalElements * sizeof(float)) / (timePerKernel * 1e-3) / 1e9;
  
  std::cout << name << ",N=" << N << ",C=" << C << ",H=" << H << ",W=" << W
            << ",Time=" << timePerKernel << " ms,Bandwidth=" << bandwidth << " GB/s" << std::endl;
  
  sycl::free(d_input, queue);
  sycl::free(d_output, queue);
}

int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  
  std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Version,N,C,H,W,Time_ms,Bandwidth_GB/s" << std::endl;
  
  std::vector<std::tuple<int, int, int, int>> testSizes = {
    {4, 64, 8, 8},
    {8, 128, 16, 16},
    {16, 256, 32, 32},
    {32, 512, 64, 64}
  };
  
  for (const auto& [N, C, H, W] : testSizes) {
    int totalElements = N * C * H * W;
    std::vector<float> input(totalElements);
    std::vector<float> output(totalElements);
    
    for (int i = 0; i < totalElements; i++) {
      input[i] = static_cast<float>(i % 1000) / 1000.0f;
    }
    
    testKernel("V0", lczero::sycldnn_backend::nchwToNhwc_V0, input, output, N, C, H, W, queue);
    testKernel("V1", lczero::sycldnn_backend::nchwToNhwc_V1, input, output, N, C, H, W, queue);
    testKernel("V2", lczero::sycldnn_backend::nchwToNhwc_V2, input, output, N, C, H, W, queue);
  }
  
  return 0;
}
