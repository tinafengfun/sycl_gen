#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

namespace lczero {
namespace sycldnn_backend {

// V0: Baseline implementation
void addVectorsHNC_NHC_V0(float* a, float* b, int N, int H, int C, sycl::queue& queue) {
  const int kBlockSize = 256;
  int totalElements = N * H * C;
  int blocks = (totalElements + kBlockSize - 1) / kBlockSize;

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize), sycl::range<1>(kBlockSize)),
        [=](sycl::nd_item<1> item) {
          int i = item.get_global_id(0);
          if (i < totalElements) {
            int orig_i = i;
            int c = i % C;
            i /= C;
            int n = i % N;
            i /= N;
            int h = i;
            float aVal = static_cast<float>(a[orig_i]);
            float bVal = static_cast<float>(b[n * H * C + h * C + c]);
            float cVal = aVal + bVal;
            a[orig_i] = static_cast<float>(cVal);
          }
        });
  });
  queue.wait_and_throw();
}

// V1: Optimized - WG=128
void addVectorsHNC_NHC_V1(float* a, float* b, int N, int H, int C, sycl::queue& queue) {
  const int kBlockSize = 128;
  int totalElements = N * H * C;
  int blocks = (totalElements + kBlockSize - 1) / kBlockSize;

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize), sycl::range<1>(kBlockSize)),
        [=](sycl::nd_item<1> item) {
          int i = item.get_global_id(0);
          if (i < totalElements) {
            int orig_i = i;
            int c = i % C;
            i /= C;
            int n = i % N;
            i /= N;
            int h = i;
            float aVal = static_cast<float>(a[orig_i]);
            float bVal = static_cast<float>(b[n * H * C + h * C + c]);
            float cVal = aVal + bVal;
            a[orig_i] = static_cast<float>(cVal);
          }
        });
  });
  queue.wait_and_throw();
}

// V2: Grid-stride with WG=128
void addVectorsHNC_NHC_V2(float* a, float* b, int N, int H, int C, sycl::queue& queue) {
  const int kBlockSize = 128;
  int totalElements = N * H * C;
  int blocks = (totalElements + kBlockSize - 1) / kBlockSize;

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize), sycl::range<1>(kBlockSize)),
        [=](sycl::nd_item<1> item) {
          int tid = item.get_global_id(0);
          int gridSize = item.get_global_range(0);
          
          #pragma unroll 4
          for (int i = tid; i < totalElements; i += gridSize) {
            int orig_i = i;
            int c = i % C;
            int tmp = i / C;
            int n = tmp % N;
            int h = tmp / N;
            float aVal = static_cast<float>(a[orig_i]);
            float bVal = static_cast<float>(b[n * H * C + h * C + c]);
            float cVal = aVal + bVal;
            a[orig_i] = static_cast<float>(cVal);
          }
        });
  });
  queue.wait_and_throw();
}

}  // namespace sycldnn_backend
}  // namespace lczero

void testKernel(const char* name, 
                void (*kernel)(float*, float*, int, int, int, sycl::queue&),
                std::vector<float>& a, std::vector<float>& b, 
                int N, int H, int C, sycl::queue& queue) {
  int totalElements = N * H * C;
  float* d_a = sycl::malloc_device<float>(totalElements, queue);
  float* d_b = sycl::malloc_device<float>(totalElements, queue);
  
  queue.memcpy(d_a, a.data(), totalElements * sizeof(float)).wait();
  queue.memcpy(d_b, b.data(), totalElements * sizeof(float)).wait();
  
  // Warmup
  for (int i = 0; i < 10; i++) {
    queue.memcpy(d_a, a.data(), totalElements * sizeof(float)).wait();
    kernel(d_a, d_b, N, H, C, queue);
  }
  
  auto start = std::chrono::high_resolution_clock::now();
  const int iterations = 100;
  
  for (int i = 0; i < iterations; i++) {
    queue.memcpy(d_a, a.data(), totalElements * sizeof(float)).wait();
    kernel(d_a, d_b, N, H, C, queue);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  double timePerKernel = duration.count() / iterations;
  
  // GFLOPS = totalElements * 1 operation (add) / time (seconds) / 1e9
  double gflops = (totalElements * 1.0) / (timePerKernel * 1e-3) / 1e9;
  // Bandwidth = 2 * totalElements * sizeof(float) / time / 1e9 GB/s
  double bandwidth = (2.0 * totalElements * sizeof(float)) / (timePerKernel * 1e-3) / 1e9;
  
  std::cout << name << ",N=" << N << ",H=" << H << ",C=" << C << ",Time=" << timePerKernel 
            << " ms,GFLOPS=" << gflops << ",Bandwidth=" << bandwidth << " GB/s" << std::endl;
  
  sycl::free(d_a, queue);
  sycl::free(d_b, queue);
}

int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  
  std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Version,N,H,C,Time_ms,GFLOPS,Bandwidth_GB/s" << std::endl;
  
  std::vector<std::tuple<int, int, int>> testSizes = {
    {4, 4, 64},
    {8, 8, 64},
    {16, 16, 64},
    {32, 32, 64},
    {64, 64, 64}
  };
  
  for (const auto& [N, H, C] : testSizes) {
    int totalElements = N * H * C;
    std::vector<float> a(totalElements), b(totalElements);
    
    for (int i = 0; i < totalElements; i++) {
      a[i] = static_cast<float>(i % 100) / 100.0f;
      b[i] = static_cast<float>((i * 2) % 100) / 100.0f;
    }
    
    testKernel("V0", lczero::sycldnn_backend::addVectorsHNC_NHC_V0, a, b, N, H, C, queue);
    testKernel("V1", lczero::sycldnn_backend::addVectorsHNC_NHC_V1, a, b, N, H, C, queue);
    testKernel("V2", lczero::sycldnn_backend::addVectorsHNC_NHC_V2, a, b, N, H, C, queue);
  }
  
  return 0;
}
