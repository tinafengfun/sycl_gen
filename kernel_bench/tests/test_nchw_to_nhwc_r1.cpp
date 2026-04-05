#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

using namespace std;

template <typename dT, typename sT>
dT readNCHW(const sT* input_tensor, int n, int c, int h, int w,
            int Nin, int Cin, int H, int W) {
  if (n >= Nin || c >= Cin) return 0;

  int index = n;
  index *= Cin;
  index += c;
  index *= H;
  index += h;
  index *= W;
  index += w;

  return static_cast<dT>(input_tensor[index]);
}

template <typename dT, typename sT>
void NCHWtoNHWC_kernel(dT* output_tensor, const sT* input_tensor,
                       int Nin, int Cin, int Nout, int Cout, int H, int W,
                       sycl::nd_item<1> item) {
  int tid = item.get_global_id(0);
  if (tid >= Nout * Cout * H * W) return;

  int index = tid;
  int c = (index % Cout);
  index /= Cout;
  int w = index % W;
  index /= W;
  int h = index % H;
  index /= H;
  int n = index;

  output_tensor[tid] = readNCHW<dT, sT>(input_tensor, n, c, h, w, Nin, Cin, H, W);
}

template <typename DstType, typename SrcType>
void convertNCHWtoNHWC(DstType* output_tensor, const SrcType* input_tensor,
                       int Nin, int Cin, int Nout, int Cout, int H, int W,
                       sycl::queue& queue) {
  size_t numElements = Nout * Cout * H * W;
  const int blockSize = 256;
  int blocks = (numElements + blockSize - 1) / blockSize;
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * blockSize), 
                        sycl::range<1>(blockSize)),
      [=](sycl::nd_item<1> item) {
        NCHWtoNHWC_kernel(output_tensor, input_tensor, 
                         Nin, Cin, Nout, Cout, H, W, item);
      });
}

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== nchw_to_nhwc - Round 1 (Real Kernel) ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    
    // Test configurations: (N, C, H, W)
    struct TestConfig {
      int N, C, H, W;
      int totalElements;
    };
    
    vector<TestConfig> configs = {
      {1, 64, 8, 8},      // 4096 elements
      {2, 128, 16, 16},   // 65536 elements
      {4, 256, 32, 32},   // 1048576 elements (1M)
      {8, 512, 32, 32},   // 4194304 elements (4M)
    };
    
    for (auto& cfg : configs) {
      cfg.totalElements = cfg.N * cfg.C * cfg.H * cfg.W;
    }
    
    cout << setw(20) << "Config (NxCxHxW)" 
         << setw(12) << "Total"
         << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS"
         << setw(18) << "GB/s" << endl;
    cout << string(80, '-') << endl;
    
    for (const auto& cfg : configs) {
      const int N = cfg.N;
      const int C = cfg.C;
      const int H = cfg.H;
      const int W = cfg.W;
      const int total = cfg.totalElements;
      
      // Allocate device memory (FP32 input, FP16 output)
      float* d_input = sycl::malloc_device<float>(total, q);
      sycl::half* d_output = sycl::malloc_device<sycl::half>(total, q);
      
      // Initialize input (NCHW format, float)
      vector<float> h_input(total, 1.0f);
      q.memcpy(d_input, h_input.data(), total * sizeof(float)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        convertNCHWtoNHWC(d_output, d_input, N, C, N, C, H, W, q);
        q.wait();
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        
        convertNCHWtoNHWC(d_output, d_input, N, C, N, C, H, W, q);
        q.wait();
        
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        times.push_back(duration.count());
      }
      
      // Stats
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      // Metrics: This is memory bound, very few FLOPs (just index calculation)
      double flops = 8.0 * total;  // Rough estimate: 8 index operations per element
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      
      // Memory: read float input + write half output = total*4 + total*2 bytes
      double bytes = total * sizeof(float) + total * sizeof(sycl::half);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(20) << (to_string(N) + "x" + to_string(C) + "x" + to_string(H) + "x" + to_string(W))
           << setw(12) << total
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
