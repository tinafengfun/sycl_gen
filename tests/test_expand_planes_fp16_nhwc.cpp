#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>

namespace lczero {
namespace sycldnn_backend {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }
constexpr int kInputPlanes = 112;

// V0: Baseline FP16 WG=256
void expandPlanes_fp16_nhwc_V0(sycl::half* output, const uint64_t* masks, 
                                const sycl::half* values, int n, sycl::queue& queue) {
  int threads = n * 8 * 8;
  const int kBlockSize = 256;
  int blocks = DivUp(threads, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int index = item.get_global_id(0);
        if (index >= n * 8 * 8) return;
        
        int planeIndex = index % kInputPlanes;
        int boardIndex = index / (kInputPlanes * 8 * 8);
        int sqIndex = (index / kInputPlanes) & 0x3F;
        
        uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];
        
        sycl::half op = sycl::half(0.0f);
        bool set = !!(mask & (1ull << sqIndex));
        if (set) {
          op = values[boardIndex * kInputPlanes + planeIndex];
        }
        output[index] = op;
      });
  queue.wait_and_throw();
}

// V1: WG=128
void expandPlanes_fp16_nhwc_V1(sycl::half* output, const uint64_t* masks, 
                                const sycl::half* values, int n, sycl::queue& queue) {
  int threads = n * 8 * 8;
  const int kBlockSize = 128;
  int blocks = DivUp(threads, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int index = item.get_global_id(0);
        if (index >= n * 8 * 8) return;
        
        int planeIndex = index % kInputPlanes;
        int boardIndex = index / (kInputPlanes * 8 * 8);
        int sqIndex = (index / kInputPlanes) & 0x3F;
        
        uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];
        
        sycl::half op = sycl::half(0.0f);
        bool set = !!(mask & (1ull << sqIndex));
        if (set) {
          op = values[boardIndex * kInputPlanes + planeIndex];
        }
        output[index] = op;
      });
  queue.wait_and_throw();
}

// V2: Grid-stride
void expandPlanes_fp16_nhwc_V2(sycl::half* output, const uint64_t* masks, 
                                const sycl::half* values, int n, sycl::queue& queue) {
  int threads = n * 8 * 8;
  const int kBlockSize = 128;
  int blocks = DivUp(threads, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        int gridSize = item.get_global_range(0);
        
        #pragma unroll 4
        for (int index = tid; index < n * 8 * 8; index += gridSize) {
          int planeIndex = index % kInputPlanes;
          int boardIndex = index / (kInputPlanes * 8 * 8);
          int sqIndex = (index / kInputPlanes) & 0x3F;
          
          uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];
          
          sycl::half op = sycl::half(0.0f);
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
                void (*kernel)(sycl::half*, const uint64_t*, const sycl::half*, int, sycl::queue&),
                sycl::half* d_output, uint64_t* d_masks, sycl::half* d_values,
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
  
  // Bandwidth calculation
  int maskCount = n * kInputPlanes;
  double bandwidth = ((maskCount * sizeof(uint64_t) + maskCount * sizeof(sycl::half) + 
                      totalElements * sizeof(sycl::half)) / (timePerKernel * 1e-3) / 1e9);
  
  std::cout << name << ",n=" << n << ",Time=" << timePerKernel 
            << " ms,Bandwidth=" << bandwidth << " GB/s" << std::endl;
}

int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  
  std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Version,n,Time_ms,Bandwidth_GB/s" << std::endl;
  
  std::vector<int> testSizes = {4, 8, 16, 32, 64};
  
  for (int n : testSizes) {
    int totalElements = n * 8 * 8;
    int maskCount = n * kInputPlanes;
    
    sycl::half* d_output = sycl::malloc_device<sycl::half>(totalElements, queue);
    uint64_t* d_masks = sycl::malloc_device<uint64_t>(maskCount, queue);
    sycl::half* d_values = sycl::malloc_device<sycl::half>(maskCount, queue);
    
    // Initialize
    std::vector<uint64_t> h_masks(maskCount, 0xFFFFFFFFFFFFFFFF);
    std::vector<sycl::half> h_values(maskCount, sycl::half(1.0f));
    queue.memcpy(d_masks, h_masks.data(), maskCount * sizeof(uint64_t)).wait();
    queue.memcpy(d_values, h_values.data(), maskCount * sizeof(sycl::half)).wait();
    
    testKernel("V0", lczero::sycldnn_backend::expandPlanes_fp16_nhwc_V0,
               d_output, d_masks, d_values, n, totalElements, queue);
    testKernel("V1", lczero::sycldnn_backend::expandPlanes_fp16_nhwc_V1,
               d_output, d_masks, d_values, n, totalElements, queue);
    testKernel("V2", lczero::sycldnn_backend::expandPlanes_fp16_nhwc_V2,
               d_output, d_masks, d_values, n, totalElements, queue);
    
    sycl::free(d_output, queue);
    sycl::free(d_masks, queue);
    sycl::free(d_values, queue);
  }
  
  return 0;
}
