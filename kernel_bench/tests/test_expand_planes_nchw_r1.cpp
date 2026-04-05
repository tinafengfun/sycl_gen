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
struct expandPlanes_kernel_NCHW {
  T* output;
  const uint64_t* masks;
  const T* values;
  unsigned n;

  expandPlanes_kernel_NCHW(T* output_, const uint64_t* masks_, const T* values_, unsigned n_)
      : output(output_), masks(masks_), values(values_), n(n_) {}

  void operator()(sycl::nd_item<1> item) const {
    unsigned index = item.get_local_id(0) + item.get_local_range(0) * item.get_group(0);

    index *= 2;
    unsigned planeIndex = index >> 6;

    if (planeIndex >= n) return;

    uint64_t mask = masks[planeIndex];

    int sqIndex = index & 0x3F;
    T op[2] = {0, 0};

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
  }
};

template <typename T>
void expandPlanes_NCHW(T* output, const uint64_t* masks, const T* values,
                       int n, sycl::queue& queue) {
  unsigned threads = n * 8 * 8 / 2;  // each thread writes two elements.
  const int blockSize = 256;
  unsigned blocks = DivUp(threads, blockSize);
  
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * blockSize), sycl::range<1>(blockSize)),
                     expandPlanes_kernel_NCHW<T>(output, masks, values, n));
  });
}

} // namespace sycldnn_backend
} // namespace lczero

using namespace lczero::sycldnn_backend;

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== expand_planes_nchw - Round 1 (Real Kernel) ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    
    // Test configurations: n boards (n = kInputPlanes for valid data)
    struct TestConfig {
      int n;
      int totalElements;
    };
    
    vector<TestConfig> configs = {
      {112, 112 * 8 * 8},        // 1 board worth of planes
      {448, 448 * 8 * 8},        // 4 boards
      {1792, 1792 * 8 * 8},      // 16 boards
      {7168, 7168 * 8 * 8},      // 64 boards
      {28672, 28672 * 8 * 8},    // 256 boards
    };
    
    cout << setw(12) << "Planes" 
         << setw(15) << "Total"
         << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS"
         << setw(18) << "GB/s" << endl;
    cout << string(75, '-') << endl;
    
    for (const auto& cfg : configs) {
      const int n = cfg.n;
      const int totalElements = cfg.totalElements;
      
      // Allocate device memory
      sycl::half* d_output = sycl::malloc_device<sycl::half>(totalElements, q);
      uint64_t* d_masks = sycl::malloc_device<uint64_t>(n, q);
      sycl::half* d_values = sycl::malloc_device<sycl::half>(n, q);
      
      // Initialize
      vector<uint64_t> h_masks(n);
      vector<sycl::half> h_values(n);
      
      srand(42);
      for (int i = 0; i < n; i++) {
        h_masks[i] = ((uint64_t)rand() << 32) | (uint64_t)rand();
        h_values[i] = sycl::half((float)(rand() % 100) / 100.0f);
      }
      
      q.memcpy(d_masks, h_masks.data(), n * sizeof(uint64_t)).wait();
      q.memcpy(d_values, h_values.data(), n * sizeof(sycl::half)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        expandPlanes_NCHW(d_output, d_masks, d_values, n, q);
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        
        expandPlanes_NCHW(d_output, d_masks, d_values, n, q);
        q.wait();
        
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
      double bytes = n * sizeof(uint64_t) + n * sizeof(sycl::half) + totalElements * sizeof(sycl::half);
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
