#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>

namespace lczero {
namespace sycldnn_backend {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// V0: Baseline WG=256 - process 2 elements per thread
void expandPlanes_NCHW_V0(float* output, const uint64_t* masks, const float* values, 
                          int n, sycl::queue& queue) {
  int totalElements = n * 64;  // n * 64 squares
  const int kBlockSize = 256;
  int blocks = DivUp(totalElements / 2, kBlockSize);  // Each thread handles 2 elements
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        int index = tid * 2;
        int planeIndex = index >> 6;
        
        if (planeIndex >= n) return;
        
        uint64_t mask = masks[planeIndex];
        
        int sqIndex = index & 0x3F;
        float op[2] = {0, 0};
        
        bool set = !!(mask & (1ull << sqIndex));
        if (set) {
          op[0] = values[planeIndex];
        }
        sqIndex++;
        set = !!(mask & (1ull << sqIndex));
        if (set) {
          op[1] = values[planeIndex];
        }
        output[index + 0] = op[0];
        output[index + 1] = op[1];
      });
  queue.wait_and_throw();
}

// V1: WG=128
void expandPlanes_NCHW_V1(float* output, const uint64_t* masks, const float* values, 
                          int n, sycl::queue& queue) {
  int totalElements = n * 64;
  const int kBlockSize = 128;
  int blocks = DivUp(totalElements / 2, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        int index = tid * 2;
        int planeIndex = index >> 6;
        
        if (planeIndex >= n) return;
        
        uint64_t mask = masks[planeIndex];
        
        int sqIndex = index & 0x3F;
        float op[2] = {0, 0};
        
        bool set = !!(mask & (1ull << sqIndex));
        if (set) {
          op[0] = values[planeIndex];
        }
        sqIndex++;
        set = !!(mask & (1ull << sqIndex));
        if (set) {
          op[1] = values[planeIndex];
        }
        output[index + 0] = op[0];
        output[index + 1] = op[1];
      });
  queue.wait_and_throw();
}

// V2: Process 4 elements per thread with unroll
void expandPlanes_NCHW_V2(float* output, const uint64_t* masks, const float* values, 
                          int n, sycl::queue& queue) {
  int totalElements = n * 64;
  const int kBlockSize = 128;
  int blocks = DivUp(totalElements / 4, kBlockSize);  // Each thread handles 4 elements
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        int index = tid * 4;
        int planeIndex = index >> 6;
        
        if (planeIndex >= n) return;
        
        uint64_t mask = masks[planeIndex];
        float val = values[planeIndex];
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
          int sqIndex = (index + i) & 0x3F;
          bool set = !!(mask & (1ull << sqIndex));
          output[index + i] = set ? val : 0.0f;
        }
      });
  queue.wait_and_throw();
}

}  // namespace sycldnn_backend
}  // namespace lczero

void testKernel(const char* name,
                void (*kernel)(float*, const uint64_t*, const float*, int, sycl::queue&),
                float* d_output, uint64_t* d_masks, float* d_values,
                int n, int totalElements, sycl::queue& queue) {
  // Warmup
  for (int i = 0; i < 5; i++) {
    kernel(d_output, d_masks, d_values, n, queue);
  }
  
  auto start = std::chrono::high_resolution_clock::now();
  const int iterations = 50;
  
  for (int i = 0; i < iterations; i++) {
    kernel(d_output, d_masks, d_values, n, queue);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  double timePerKernel = duration.count() / iterations;
  
  double bandwidth = (n * sizeof(uint64_t) + n * sizeof(float) + totalElements * sizeof(float)) 
                     / (timePerKernel * 1e-3) / 1e9;
  
  std::cout << name << ",n=" << n << ",Time=" << timePerKernel 
            << " ms,Bandwidth=" << bandwidth << " GB/s" << std::endl;
}

int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  
  std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Version,n,Time_ms,Bandwidth_GB/s" << std::endl;
  
  std::vector<int> testSizes = {64, 128, 256, 512, 1024};
  
  for (int n : testSizes) {
    int totalElements = n * 64;
    
    float* d_output = sycl::malloc_device<float>(totalElements, queue);
    uint64_t* d_masks = sycl::malloc_device<uint64_t>(n, queue);
    float* d_values = sycl::malloc_device<float>(n, queue);
    
    // Initialize
    std::vector<uint64_t> h_masks(n, 0xFFFFFFFFFFFFFFFF);
    std::vector<float> h_values(n, 1.0f);
    queue.memcpy(d_masks, h_masks.data(), n * sizeof(uint64_t)).wait();
    queue.memcpy(d_values, h_values.data(), n * sizeof(float)).wait();
    
    testKernel("V0", lczero::sycldnn_backend::expandPlanes_NCHW_V0,
               d_output, d_masks, d_values, n, totalElements, queue);
    testKernel("V1", lczero::sycldnn_backend::expandPlanes_NCHW_V1,
               d_output, d_masks, d_values, n, totalElements, queue);
    testKernel("V2", lczero::sycldnn_backend::expandPlanes_NCHW_V2,
               d_output, d_masks, d_values, n, totalElements, queue);
    
    sycl::free(d_output, queue);
    sycl::free(d_masks, queue);
    sycl::free(d_values, queue);
  }
  
  return 0;
}
