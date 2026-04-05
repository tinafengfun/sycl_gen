#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

using namespace std;

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename DstType, typename SrcType>
struct copyTypeConverted_kernel {
  DstType* op;
  SrcType* ip;
  int N;

  void operator()(sycl::nd_item<1> item) const {
    int tid = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

    if (tid >= N) return;

    DstType el = static_cast<DstType>(ip[tid]);
    op[tid] = el;
  }
};

template <typename DstType, typename SrcType>
void copyTypeConverted(DstType* op, SrcType* ip, int N, sycl::queue& stream) {
  const int kBlockSize = 256;
  int blocks = DivUp(N, kBlockSize);
  
  stream.submit([&](sycl::handler& cgh) {
    copyTypeConverted_kernel<DstType, SrcType> kernel{op, ip, N};
    cgh.parallel_for(sycl::nd_range<1>(blocks * kBlockSize, kBlockSize), kernel);
  });
}

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== copy_type_converted - Round 1 (Real Kernel) ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    
    // Test sizes
    vector<int> sizes = {4096, 65536, 1048576, 4194304};
    
    cout << setw(12) << "Size" 
         << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS"
         << setw(18) << "GB/s" << endl;
    cout << string(60, '-') << endl;
    
    for (int size : sizes) {
      // Allocate device memory (float to half conversion)
      float* d_input = sycl::malloc_device<float>(size, q);
      sycl::half* d_output = sycl::malloc_device<sycl::half>(size, q);
      
      // Initialize input
      vector<float> h_input(size, 1.0f);
      q.memcpy(d_input, h_input.data(), size * sizeof(float)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        copyTypeConverted(d_output, d_input, size, q);
        q.wait();
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        
        copyTypeConverted(d_output, d_input, size, q);
        q.wait();
        
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        times.push_back(duration.count());
      }
      
      // Stats
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      // Metrics: type conversion is 1 FLOP per element
      double flops = 1.0 * size;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      
      // Memory: read float + write half = size*4 + size*2 bytes
      double bytes = size * sizeof(float) + size * sizeof(sycl::half);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(12) << size
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_input, q);
      sycl::free(d_output, q);
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
