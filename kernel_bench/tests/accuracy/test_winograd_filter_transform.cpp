#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

namespace lczero {
namespace sycldnn_backend {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// V0: Baseline
void winogradFilterTransform_V0(float* transformed_filter, const float* filter,
                                int C, int K, sycl::queue& queue) {
  const int kBlockSize = 256;
  // Each filter tile is 3x3, transformed to 6x6
  int totalTiles = C * K;
  int blocks = DivUp(totalTiles, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid >= totalTiles) return;
        
        int c = tid / K;
        int k = tid % K;
        
        // Load 3x3 filter
        float f[3][3];
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            f[i][j] = filter[((c * 3 + i) * 3 + j) * K + k];
          }
        }
        
        // Winograd transform (simplified 4x4 for demo)
        float t[4][4];
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 4; j++) {
            t[i][j] = 0.0f;
            for (int di = 0; di < 3; di++) {
              for (int dj = 0; dj < 3; dj++) {
                t[i][j] += f[di][dj];
              }
            }
          }
        }
        
        // Store transformed filter
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 4; j++) {
            int idx = ((c * 4 + i) * 4 + j) * K + k;
            transformed_filter[idx] = t[i][j];
          }
        }
      });
  queue.wait_and_throw();
}

// V1: WG=128 with unroll
void winogradFilterTransform_V1(float* transformed_filter, const float* filter,
                                int C, int K, sycl::queue& queue) {
  const int kBlockSize = 128;
  int totalTiles = C * K;
  int blocks = DivUp(totalTiles, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid >= totalTiles) return;
        
        int c = tid / K;
        int k = tid % K;
        
        // Load 3x3 filter with unroll
        float f[3][3];
        #pragma unroll
        for (int i = 0; i < 3; i++) {
          #pragma unroll
          for (int j = 0; j < 3; j++) {
            f[i][j] = filter[((c * 3 + i) * 3 + j) * K + k];
          }
        }
        
        // Winograd transform with unroll
        float t[4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
          #pragma unroll
          for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            #pragma unroll
            for (int di = 0; di < 3; di++) {
              #pragma unroll
              for (int dj = 0; dj < 3; dj++) {
                sum += f[di][dj];
              }
            }
            t[i][j] = sum;
          }
        }
        
        // Store with unroll
        #pragma unroll
        for (int i = 0; i < 4; i++) {
          #pragma unroll
          for (int j = 0; j < 4; j++) {
            int idx = ((c * 4 + i) * 4 + j) * K + k;
            transformed_filter[idx] = t[i][j];
          }
        }
      });
  queue.wait_and_throw();
}

// V2: 2D work-group with better locality
void winogradFilterTransform_V2(float* transformed_filter, const float* filter,
                                int C, int K, sycl::queue& queue) {
  // Use 2D work-group: 16x8 = 128
  sycl::range<2> local(16, 8);
  sycl::range<2> global(DivUp(C, 16) * 16, DivUp(K, 8) * 8);
  
  queue.parallel_for(
      sycl::nd_range<2>(global, local),
      [=](sycl::nd_item<2> item) {
        int c = item.get_global_id(0);
        int k = item.get_global_id(1);
        
        if (c >= C || k >= K) return;
        
        // Load 3x3 filter
        float f[3][3];
        #pragma unroll
        for (int i = 0; i < 3; i++) {
          #pragma unroll
          for (int j = 0; j < 3; j++) {
            f[i][j] = filter[((c * 3 + i) * 3 + j) * K + k];
          }
        }
        
        // Transform
        float t[4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
          #pragma unroll
          for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            #pragma unroll
            for (int di = 0; di < 3; di++) {
              #pragma unroll
              for (int dj = 0; dj < 3; dj++) {
                sum += f[di][dj];
              }
            }
            t[i][j] = sum;
          }
        }
        
        // Store
        #pragma unroll
        for (int i = 0; i < 4; i++) {
          #pragma unroll
          for (int j = 0; j < 4; j++) {
            int idx = ((c * 4 + i) * 4 + j) * K + k;
            transformed_filter[idx] = t[i][j];
          }
        }
      });
  queue.wait_and_throw();
}

}  // namespace sycldnn_backend
}  // namespace lczero

void testKernel(const char* name,
                void (*kernel)(float*, const float*, int, int, sycl::queue&),
                float* d_output, float* d_input,
                int C, int K, int totalTiles, sycl::queue& queue) {
  // Warmup
  for (int i = 0; i < 5; i++) {
    kernel(d_output, d_input, C, K, queue);
  }
  
  auto start = std::chrono::high_resolution_clock::now();
  const int iterations = 50;
  
  for (int i = 0; i < iterations; i++) {
    kernel(d_output, d_input, C, K, queue);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  double timePerKernel = duration.count() / iterations;
  
  // Approximate FLOPS: 3*3*4*4 = 36 mul-adds per tile
  double gflops = (totalTiles * 36.0 * 2.0) / (timePerKernel * 1e-3) / 1e9;
  double bandwidth = ((totalTiles * 9 + totalTiles * 16) * sizeof(float)) / (timePerKernel * 1e-3) / 1e9;
  
  std::cout << name << ",C=" << C << ",K=" << K << ",Time=" << timePerKernel
            << " ms,GFLOPS=" << gflops << ",Bandwidth=" << bandwidth << " GB/s" << std::endl;
}

int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  
  std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Version,C,K,Time_ms,GFLOPS,Bandwidth_GB/s" << std::endl;
  
  std::vector<std::pair<int, int>> testSizes = {
    {64, 64},
    {128, 128},
    {256, 256},
    {512, 512}
  };
  
  for (const auto& [C, K] : testSizes) {
    int totalTiles = C * K;
    int filterSize = C * 3 * 3 * K;
    int transformedSize = C * 4 * 4 * K;
    
    float* d_output = sycl::malloc_device<float>(transformedSize, queue);
    float* d_input = sycl::malloc_device<float>(filterSize, queue);
    
    // Initialize
    std::vector<float> h_input(filterSize, 0.1f);
    queue.memcpy(d_input, h_input.data(), filterSize * sizeof(float)).wait();
    
    testKernel("V0", lczero::sycldnn_backend::winogradFilterTransform_V0,
               d_output, d_input, C, K, totalTiles, queue);
    testKernel("V1", lczero::sycldnn_backend::winogradFilterTransform_V1,
               d_output, d_input, C, K, totalTiles, queue);
    testKernel("V2", lczero::sycldnn_backend::winogradFilterTransform_V2,
               d_output, d_input, C, K, totalTiles, queue);
    
    sycl::free(d_output, queue);
    sycl::free(d_input, queue);
  }
  
  return 0;
}
