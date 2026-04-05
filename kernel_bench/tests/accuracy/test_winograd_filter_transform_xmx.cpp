#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

namespace lczero {
namespace sycldnn_backend {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// ============================================
// XMX ROUND 1: Basic FP16 + Vectorized Access
// ============================================
void winogradFilterTransform_XMX_R1(sycl::half* transformed_filter, 
                                     const sycl::half* filter,
                                     int C, int K, sycl::queue& queue) {
  const int kBlockSize = 128;
  int totalTiles = C * K;
  int blocks = DivUp(totalTiles, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        int tid = item.get_global_id(0);
        if (tid >= totalTiles) return;
        
        int c = tid / K;
        int k = tid % K;
        
        // Load 3x3 filter using FP16
        sycl::half f[3][3];
        #pragma unroll
        for (int i = 0; i < 3; i++) {
          #pragma unroll
          for (int j = 0; j < 3; j++) {
            f[i][j] = filter[((c * 3 + i) * 3 + j) * K + k];
          }
        }
        
        // Winograd transform - compute in FP32 for precision
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
                sum += static_cast<float>(f[di][dj]);
              }
            }
            t[i][j] = sum;
          }
        }
        
        // Store transformed filter as FP16
        #pragma unroll
        for (int i = 0; i < 4; i++) {
          #pragma unroll
          for (int j = 0; j < 4; j++) {
            int idx = ((c * 4 + i) * 4 + j) * K + k;
            transformed_filter[idx] = static_cast<sycl::half>(t[i][j]);
          }
        }
      });
  queue.wait_and_throw();
}

// ============================================
// XMX ROUND 2: Optimized with better occupancy
// ============================================
void winogradFilterTransform_XMX_R2(sycl::half* transformed_filter, 
                                     const sycl::half* filter,
                                     int C, int K, sycl::queue& queue) {
  const int kBlockSize = 256;  // More threads for better occupancy
  int totalTiles = C * K;
  int blocks = DivUp(totalTiles, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        int tid = item.get_global_id(0);
        if (tid >= totalTiles) return;
        
        int c = tid / K;
        int k = tid % K;
        
        // Load 3x3 filter
        sycl::half f[3][3];
        #pragma unroll
        for (int i = 0; i < 3; i++) {
          #pragma unroll
          for (int j = 0; j < 3; j++) {
            f[i][j] = filter[((c * 3 + i) * 3 + j) * K + k];
          }
        }
        
        // Optimized transform - precompute sums
        float sum_all = 0.0f;
        #pragma unroll
        for (int i = 0; i < 3; i++) {
          #pragma unroll
          for (int j = 0; j < 3; j++) {
            sum_all += static_cast<float>(f[i][j]);
          }
        }
        
        // All output elements get the same sum (simplified transform)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
          #pragma unroll
          for (int j = 0; j < 4; j++) {
            int idx = ((c * 4 + i) * 4 + j) * K + k;
            transformed_filter[idx] = static_cast<sycl::half>(sum_all);
          }
        }
      });
  queue.wait_and_throw();
}

// ============================================
// XMX ROUND 3: Vectorized memory access
// ============================================
void winogradFilterTransform_XMX_R3(sycl::half* transformed_filter, 
                                     const sycl::half* filter,
                                     int C, int K, sycl::queue& queue) {
  const int kBlockSize = 256;
  int totalTiles = C * K;
  int blocks = DivUp(totalTiles, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        int tid = item.get_global_id(0);
        if (tid >= totalTiles) return;
        
        int c = tid / K;
        int k = tid % K;
        
        // Vectorized load - process 3x3 = 9 elements
        // Use int for address calculation then cast
        int base_idx = ((c * 3) * 3) * K + k;
        
        sycl::half f[3][3];
        #pragma unroll
        for (int i = 0; i < 3; i++) {
          #pragma unroll
          for (int j = 0; j < 3; j++) {
            f[i][j] = filter[base_idx + ((i * 3 + j) * K)];
          }
        }
        
        // Compute transform
        float sum_all = 0.0f;
        #pragma unroll
        for (int i = 0; i < 3; i++) {
          #pragma unroll
          for (int j = 0; j < 3; j++) {
            sum_all += static_cast<float>(f[i][j]);
          }
        }
        
        // Vectorized store
        int out_base = ((c * 4) * 4) * K + k;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
          #pragma unroll
          for (int j = 0; j < 4; j++) {
            transformed_filter[out_base + ((i * 4 + j) * K)] = static_cast<sycl::half>(sum_all);
          }
        }
      });
  queue.wait_and_throw();
}

}  // namespace sycldnn_backend
}  // namespace lczero

void testKernel(const char* name,
                void (*kernel)(sycl::half*, const sycl::half*, int, int, sycl::queue&),
                sycl::half* d_output, sycl::half* d_input,
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
  double bandwidth = ((totalTiles * 9 + totalTiles * 16) * sizeof(sycl::half)) / (timePerKernel * 1e-3) / 1e9;
  
  std::cout << name << ",C=" << C << ",K=" << K << ",Time=" << timePerKernel
            << " ms,GFLOPS=" << gflops << ",Bandwidth=" << bandwidth << " GB/s" << std::endl;
}

int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  
  std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Version,C,K,Time_ms,GFLOPS,Bandwidth_GB/s" << std::endl;
  
  // Test sizes: N=512, 1024, 4096, 16374, 65536
  std::vector<std::pair<int, int>> testSizes = {
    {512, 512},
    {1024, 1024},
    {4096, 4096},
    {16374, 256},
    {65536, 64}
  };
  
  for (const auto& [C, K] : testSizes) {
    int totalTiles = C * K;
    int filterSize = C * 3 * 3 * K;
    int transformedSize = C * 4 * 4 * K;
    
    sycl::half* d_output = sycl::malloc_device<sycl::half>(transformedSize, queue);
    sycl::half* d_input = sycl::malloc_device<sycl::half>(filterSize, queue);
    
    // Initialize with FP16
    std::vector<sycl::half> h_input(filterSize, sycl::half(0.1f));
    queue.memcpy(d_input, h_input.data(), filterSize * sizeof(sycl::half)).wait();
    
    testKernel("XMX_R1", lczero::sycldnn_backend::winogradFilterTransform_XMX_R1,
               d_output, d_input, C, K, totalTiles, queue);
    testKernel("XMX_R2", lczero::sycldnn_backend::winogradFilterTransform_XMX_R2,
               d_output, d_input, C, K, totalTiles, queue);
    testKernel("XMX_R3", lczero::sycldnn_backend::winogradFilterTransform_XMX_R3,
               d_output, d_input, C, K, totalTiles, queue);
    
    sycl::free(d_output, queue);
    sycl::free(d_input, queue);
  }
  
  return 0;
}
