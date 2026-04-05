#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>

namespace lczero {
namespace sycldnn_backend {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }
constexpr int kInputPlanes = 112;

// V0: Baseline WG=256
void expandPlanes_NHWC_V0(float* output, const uint64_t* masks, const float* values, 
                          int n, sycl::queue& queue) {
  int totalElements = n * kInputPlanes * 8 * 8;
  const int kBlockSize = 256;
  int blocks = DivUp(totalElements, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int index = item.get_global_id(0);
        if (index >= totalElements) return;
        
        int planeIndex = index % kInputPlanes;
        int boardIndex = index / (kInputPlanes * 8 * 8);
        int sqIndex = (index / kInputPlanes) & 0x3F;
        
        uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];
        
        float op = 0;
        bool set = !!(mask & (1ull << sqIndex));
        if (set) {
          op = values[boardIndex * kInputPlanes + planeIndex];
        }
        output[index] = op;
      });
  queue.wait_and_throw();
}

// V1: WG=128
void expandPlanes_NHWC_V1(float* output, const uint64_t* masks, const float* values, 
                          int n, sycl::queue& queue) {
  int totalElements = n * kInputPlanes * 8 * 8;
  const int kBlockSize = 128;
  int blocks = DivUp(totalElements, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int index = item.get_global_id(0);
        if (index >= totalElements) return;
        
        int planeIndex = index % kInputPlanes;
        int boardIndex = index / (kInputPlanes * 8 * 8);
        int sqIndex = (index / kInputPlanes) & 0x3F;
        
        uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];
        
        float op = 0;
        bool set = !!(mask & (1ull << sqIndex));
        if (set) {
          op = values[boardIndex * kInputPlanes + planeIndex];
        }
        output[index] = op;
      });
  queue.wait_and_throw();
}

// V2: Grid-stride with unroll
void expandPlanes_NHWC_V2(float* output, const uint64_t* masks, const float* values, 
                          int n, sycl::queue& queue) {
  int totalElements = n * kInputPlanes * 8 * 8;
  const int kBlockSize = 128;
  int blocks = DivUp(totalElements, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        int gridSize = item.get_global_range(0);
        
        #pragma unroll 4
        for (int index = tid; index < totalElements; index += gridSize) {
          int planeIndex = index % kInputPlanes;
          int boardIndex = index / (kInputPlanes * 8 * 8);
          int sqIndex = (index / kInputPlanes) & 0x3F;
          
          uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];
          
          float op = 0;
          bool set = !!(mask & (1ull << sqIndex));
          if (set) {
            op = values[boardIndex * kInputPlanes + planeIndex];
          }
          output[index] = op;
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
  
  // Bandwidth = read masks (uint64) + values (float) + write output (float)
  double bandwidth = ((totalElements / 64) * sizeof(uint64_t) + totalElements * 2 * sizeof(float)) 
                     / (timePerKernel * 1e-3) / 1e9;
  
  std::cout << name << ",n=" << n << ",Time=" << timePerKernel 
            << " ms,Bandwidth=" << bandwidth << " GB/s" << std::endl;
}

int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  
  std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Version,n,Time_ms,Bandwidth_GB/s" << std::endl;
  
  std::vector<int> testSizes = {4, 8, 16, 32, 64};
  
  for (int n : testSizes) {
    int totalElements = n * 112 * 8 * 8;  // n * kInputPlanes * 8 * 8
    int maskCount = n * 112;
    
    float* d_output = sycl::malloc_device<float>(totalElements, queue);
    uint64_t* d_masks = sycl::malloc_device<uint64_t>(maskCount, queue);
    float* d_values = sycl::malloc_device<float>(maskCount, queue);
    
    // Initialize
    std::vector<uint64_t> h_masks(maskCount, 0xFFFFFFFFFFFFFFFF);
    std::vector<float> h_values(maskCount, 1.0f);
    queue.memcpy(d_masks, h_masks.data(), maskCount * sizeof(uint64_t)).wait();
    queue.memcpy(d_values, h_values.data(), maskCount * sizeof(float)).wait();
    
    testKernel("V0", lczero::sycldnn_backend::expandPlanes_NHWC_V0,
               d_output, d_masks, d_values, n, totalElements, queue);
    testKernel("V1", lczero::sycldnn_backend::expandPlanes_NHWC_V1,
               d_output, d_masks, d_values, n, totalElements, queue);
    testKernel("V2", lczero::sycldnn_backend::expandPlanes_NHWC_V2,
               d_output, d_masks, d_values, n, totalElements, queue);
    
    sycl::free(d_output, queue);
    sycl::free(d_masks, queue);
    sycl::free(d_values, queue);
  }
  
  return 0;
}
