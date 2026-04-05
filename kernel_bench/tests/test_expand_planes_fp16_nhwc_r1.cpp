#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

using namespace std;

namespace lczero {
namespace sycldnn_backend {

constexpr int kInputPlanes = 112;

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void expandPlanes_NHWC(T* output, const uint64_t* masks, const T* values, int n,
                       sycl::queue& stream) {
  int threads = n * 8 * 8;
  const int kBlockSize = 256;
  int blocks = DivUp(threads, kBlockSize);

  stream.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize), sycl::range<1>(kBlockSize)),
        [=](sycl::nd_item<1> item) {
          const int index = item.get_global_id(0);
          if (index >= n * 8 * 8) return;

          const int planeIndex = index % kInputPlanes;
          const int boardIndex = index / (kInputPlanes * 8 * 8);
          const int sqIndex = (index / kInputPlanes) & 0x3F;

          uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];

          T op = 0;
          bool set = !!(mask & (1ull << sqIndex));
          if (set) {
            op = values[boardIndex * kInputPlanes + planeIndex];
          }
          output[index] = op;
        });
  });
  stream.wait_and_throw();
}

} // namespace sycldnn_backend
} // namespace lczero

using namespace lczero::sycldnn_backend;

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== expand_planes_fp16_nhwc - Round 1 (Real Kernel) ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    
    // Test configurations: n boards
    struct TestConfig {
      int n;
      int totalElements;
    };
    
    vector<TestConfig> configs = {
      {1, 1 * 112 * 8 * 8},
      {4, 4 * 112 * 8 * 8},
      {16, 16 * 112 * 8 * 8},
      {64, 64 * 112 * 8 * 8},
      {256, 256 * 112 * 8 * 8},
    };
    
    cout << setw(12) << "Boards" 
         << setw(15) << "Total"
         << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS"
         << setw(18) << "GB/s" << endl;
    cout << string(75, '-') << endl;
    
    for (const auto& cfg : configs) {
      const int n = cfg.n;
      const int totalElements = cfg.totalElements;
      const int maskElements = n * kInputPlanes;
      
      // Allocate device memory
      sycl::half* d_output = sycl::malloc_device<sycl::half>(totalElements, q);
      uint64_t* d_masks = sycl::malloc_device<uint64_t>(maskElements, q);
      sycl::half* d_values = sycl::malloc_device<sycl::half>(maskElements, q);
      
      // Initialize: random masks with some bits set, random values
      vector<uint64_t> h_masks(maskElements, 0);
      vector<sycl::half> h_values(maskElements);
      
      srand(42);
      for (int i = 0; i < maskElements; i++) {
        // Random 64-bit mask with ~50% bits set
        h_masks[i] = ((uint64_t)rand() << 32) | (uint64_t)rand();
        h_values[i] = sycl::half((float)(rand() % 100) / 100.0f);
      }
      
      q.memcpy(d_masks, h_masks.data(), maskElements * sizeof(uint64_t)).wait();
      q.memcpy(d_values, h_values.data(), maskElements * sizeof(sycl::half)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        expandPlanes_NHWC(d_output, d_masks, d_values, n, q);
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        
        expandPlanes_NHWC(d_output, d_masks, d_values, n, q);
        
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        times.push_back(duration.count());
      }
      
      // Stats
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      // Metrics: bit test + conditional write ~ 5 FLOPs per element
      double flops = 5.0 * totalElements;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      
      // Memory: read masks + read values + write output
      double bytes = maskElements * sizeof(uint64_t) + maskElements * sizeof(sycl::half) + totalElements * sizeof(sycl::half);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(12) << n
           << setw(15) << totalElements
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_output, q);
      sycl::free(d_masks, q);
      sycl::free(d_values, q);
    }
    
    cout << endl << "Test completed!" << endl;
    return 0;
  } catch (sycl::exception const &e) {
    cerr << "SYCL Exception: " << e.what() << endl;
    return 1;
  } catch (exception const &e) {
    cerr << "Exception: " << e.what() << endl;
    return 1;
  }
}
