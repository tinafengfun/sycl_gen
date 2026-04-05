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

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T, int kWorkPerThread>
void genOffsetPointers_kernel(T** offsets, int heads, int block_size,
                              int depth, int d_model, T* k, T* q,
                              T* b1, T* v, T* b2, sycl::nd_item<1> item) {
  const int i = (item.get_group(0) * item.get_local_range(0) + item.get_local_id(0)) * kWorkPerThread;
  if (i >= block_size) return;
  const int h = i % heads;
  const int n = i / heads;
  int w;
  T* res[kWorkPerThread];
  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = k + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = q + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w + block_size] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = b1 + i * 64 * 64 + w * 64 * 64;
    offsets[i + w + 2 * block_size] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = v + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w + 3 * block_size] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = b2 + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w + 4 * block_size] = res[w];
  }
}

template <typename T>
void genOffsetPointers(T** offsets, int heads, int max_batch, int depth,
                       int d_model, T* k, T* q, T* b1, T* v, T* b2,
                       sycl::queue& stream) {
  const int block_size = heads * max_batch;
  constexpr int kWorkPerThread = 2;
  constexpr int kWorkGroupSize = 128;
  
  if (block_size % kWorkPerThread != 0) {
    int grid = DivUp(block_size, kWorkGroupSize);
    stream.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(grid * kWorkGroupSize), sycl::range<1>(kWorkGroupSize)),
        [=](sycl::nd_item<1> item) {
          genOffsetPointers_kernel<T, 1>(offsets, heads, block_size, depth, d_model, k, q, b1, v, b2, item);
        });
    });
  } else {
    int grid = DivUp(block_size, kWorkGroupSize * kWorkPerThread);
    stream.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(grid * kWorkGroupSize), sycl::range<1>(kWorkGroupSize)),
        [=](sycl::nd_item<1> item) {
          genOffsetPointers_kernel<T, kWorkPerThread>(offsets, heads, block_size, depth, d_model, k, q, b1, v, b2, item);
        });
    });
  }
  stream.wait();
}

} // namespace sycldnn_backend
} // namespace lczero

using namespace lczero::sycldnn_backend;

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== gen_offset_pointers - Round 1 (Real Kernel) ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    
    // Test configurations
    struct TestConfig {
      int heads;
      int max_batch;
      int depth;
      int d_model;
      int block_size;
    };
    
    vector<TestConfig> configs = {
      {8, 64, 64, 512, 8 * 64},
      {16, 128, 64, 512, 16 * 128},
      {32, 256, 64, 512, 32 * 256},
    };
    
    cout << setw(12) << "Heads" 
         << setw(12) << "Batch"
         << setw(12) << "BlockSize"
         << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS"
         << setw(18) << "GB/s" << endl;
    cout << string(84, '-') << endl;
    
    for (const auto& cfg : configs) {
      const int heads = cfg.heads;
      const int max_batch = cfg.max_batch;
      const int depth = cfg.depth;
      const int d_model = cfg.d_model;
      const int block_size = cfg.block_size;
      const int num_offsets = block_size * 5;
      
      // Allocate device memory
      sycl::half** d_offsets = sycl::malloc_device<sycl::half*>(num_offsets, q);
      sycl::half* d_k = sycl::malloc_device<sycl::half>(heads * depth * 64 * max_batch, q);
      sycl::half* d_q = sycl::malloc_device<sycl::half>(heads * depth * 64 * max_batch, q);
      sycl::half* d_b1 = sycl::malloc_device<sycl::half>(block_size * 64 * 64, q);
      sycl::half* d_v = sycl::malloc_device<sycl::half>(heads * depth * 64 * max_batch, q);
      sycl::half* d_b2 = sycl::malloc_device<sycl::half>(heads * depth * 64 * max_batch, q);
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        genOffsetPointers(d_offsets, heads, max_batch, depth, d_model, d_k, d_q, d_b1, d_v, d_b2, q);
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        
        genOffsetPointers(d_offsets, heads, max_batch, depth, d_model, d_k, d_q, d_b1, d_v, d_b2, q);
        
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        times.push_back(duration.count());
      }
      
      // Stats
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      // Metrics: pointer arithmetic ~ 10 FLOPs per offset
      double flops = 10.0 * num_offsets;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      
      // Memory: write offsets array
      double bytes = num_offsets * sizeof(void*);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(12) << heads
           << setw(12) << max_batch
           << setw(12) << block_size
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_offsets, q);
      sycl::free(d_k, q);
      sycl::free(d_q, q);
      sycl::free(d_b1, q);
      sycl::free(d_v, q);
      sycl::free(d_b2, q);
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
