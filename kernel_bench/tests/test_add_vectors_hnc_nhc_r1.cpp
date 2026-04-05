#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

using namespace std;

// Kernel function declaration (from add_vectors_hnc_nhc_kernel.dp.cpp)
template <typename T>
void addVectorsHNC_NHC(T* a, T* b, int N, int H, int C, sycl::queue& queue) {
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
            a[orig_i] = static_cast<T>(cVal);
          }
        });
  });
  queue.wait_and_throw();
}

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== add_vectors_hnc_nhc - Round 1 (Real Kernel) ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    
    // Test configurations: (N, H, C) - total elements = N*H*C
    struct TestConfig {
      int N, H, C;
      int totalElements;
    };
    
    vector<TestConfig> configs = {
      {1, 8, 64},      // 512 elements
      {2, 16, 128},    // 4096 elements
      {4, 32, 256},    // 32768 elements
      {8, 64, 512},    // 262144 elements
      {16, 64, 1024},  // 1048576 elements (~1M)
    };
    
    // Calculate total elements
    for (auto& cfg : configs) {
      cfg.totalElements = cfg.N * cfg.H * cfg.C;
    }
    
    cout << setw(15) << "Config (NxHxC)" 
         << setw(15) << "Total"
         << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS"
         << setw(18) << "GB/s" << endl;
    cout << string(78, '-') << endl;
    
    for (const auto& cfg : configs) {
      const int N = cfg.N;
      const int H = cfg.H;
      const int C = cfg.C;
      const int total = cfg.totalElements;
      
      // Allocate device memory
      // a is HNC format: H * N * C
      sycl::half* d_a = sycl::malloc_device<sycl::half>(total, q);
      // b is NHC format: N * H * C
      sycl::half* d_b = sycl::malloc_device<sycl::half>(total, q);
      
      // Initialize a (HNC format)
      vector<sycl::half> h_a(total);
      for (int h = 0; h < H; h++) {
        for (int n = 0; n < N; n++) {
          for (int c = 0; c < C; c++) {
            int idx_hnc = h * N * C + n * C + c;
            h_a[idx_hnc] = sycl::half(1.0f);
          }
        }
      }
      q.memcpy(d_a, h_a.data(), total * sizeof(sycl::half)).wait();
      
      // Initialize b (NHC format)
      vector<sycl::half> h_b(total);
      for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
          for (int c = 0; c < C; c++) {
            int idx_nhc = n * H * C + h * C + c;
            h_b[idx_nhc] = sycl::half(0.5f);
          }
        }
      }
      q.memcpy(d_b, h_b.data(), total * sizeof(sycl::half)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        addVectorsHNC_NHC(d_a, d_b, N, H, C, q);
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        // Reset a for each iteration
        q.memcpy(d_a, h_a.data(), total * sizeof(sycl::half)).wait();
        
        auto start = chrono::high_resolution_clock::now();
        
        addVectorsHNC_NHC(d_a, d_b, N, H, C, q);
        
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        times.push_back(duration.count());
      }
      
      // Stats
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      // Calculate metrics: 1 add per element = 1 FLOP
      double flops = 1.0 * total;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      
      // Memory: read a + read b + write a = 3 * total * 2 bytes
      double bytes = 3.0 * total * sizeof(sycl::half);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      // Verify result (check first few elements)
      vector<sycl::half> h_result(total);
      q.memcpy(h_result.data(), d_a, total * sizeof(sycl::half)).wait();
      
      bool correct = true;
      for (int i = 0; i < min(10, total); i++) {
        float expected = 1.0f + 0.5f;  // a + b
        if (abs(static_cast<float>(h_result[i]) - expected) > 0.01f) {
          correct = false;
          break;
        }
      }
      
      string status = correct ? "" : " [FAIL]";
      
      cout << setw(15) << (to_string(N) + "x" + to_string(H) + "x" + to_string(C))
           << setw(15) << total
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth
           << status << endl;
      
      sycl::free(d_a, q);
      sycl::free(d_b, q);
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
