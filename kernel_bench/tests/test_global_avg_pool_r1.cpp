#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

using namespace std;

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// Each thread reads 2 inputs (8x8/32), and each warp writes a single output.
template <typename T>
void globalAvgPool_kernel(T* output, const T* input,
                          const T* prevLayerBias, int inputSize,
                          int outputSize, int C,
                          const sycl::nd_item<3> &item_ct1) {
  const int elementsPerWarp = 64;
  const int elementsPerThread = 2;

  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  int laneId = item_ct1.get_local_id(2) & 0x1F;
  int laneStartIndex = (tid - laneId) * elementsPerThread;

  float S = 0;

  for (int i = 0; i < elementsPerWarp; i += 32) {
    int index = laneStartIndex + laneId + i;
    if (index < inputSize) S += (float)(input[index]);
  }

  for (int offset = 1; offset < 32; offset *= 2) {
    S += sycl::shift_group_left(item_ct1.get_sub_group(), S, offset);
  }

  float avg = S / elementsPerWarp;
  int opIndex = tid >> 5;

  if (laneId == 0) {
    if (opIndex < outputSize) {
      if (prevLayerBias) avg += (float)prevLayerBias[opIndex % C];
      output[opIndex] = (T)avg;
    }
  }
}

template <typename T>
void globalAvgPool(int N, int C, T* output, const T* input,
                   const T* prevLayerBias, bool nhwc, sycl::queue &sycl_queue) {
  const int kPlaneSize = 64;

  const bool fp16 = std::is_same<sycl::half, T>::value;
  if (nhwc) {
    assert(fp16);
  } else {
    const int kTotalWarps = N * C;
    const int kWarpsPerBlock = 8;
    const int kBlockSize = kWarpsPerBlock * 32;

    int blocks = DivUp(kTotalWarps, kWarpsPerBlock);
    sycl_queue.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
            sycl::range<3>(1, 1, kBlockSize)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          globalAvgPool_kernel(output, input, prevLayerBias, N * C * kPlaneSize,
                               N * C, C, item_ct1);
        });
  }
  sycl_queue.wait();
}

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== global_avg_pool - Round 1 (Real Kernel) ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    
    struct TestConfig {
      int N, C;
      int inputSize;
      int outputSize;
    };
    
    vector<TestConfig> configs = {
      {16, 64, 16 * 64 * 64, 16 * 64},
      {64, 128, 64 * 128 * 64, 64 * 128},
      {256, 256, 256 * 256 * 64, 256 * 256},
    };
    
    cout << setw(10) << "N" 
         << setw(10) << "C"
         << setw(15) << "InputSize"
         << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS"
         << setw(18) << "GB/s" << endl;
    cout << string(83, '-') << endl;
    
    for (const auto& cfg : configs) {
      // Allocate device memory (fp32)
      float* d_input = sycl::malloc_device<float>(cfg.inputSize, q);
      float* d_output = sycl::malloc_device<float>(cfg.outputSize, q);
      float* d_bias = sycl::malloc_device<float>(cfg.C, q);
      
      // Initialize
      vector<float> h_input(cfg.inputSize, 1.0f);
      vector<float> h_bias(cfg.C, 0.5f);
      
      q.memcpy(d_input, h_input.data(), cfg.inputSize * sizeof(float)).wait();
      q.memcpy(d_bias, h_bias.data(), cfg.C * sizeof(float)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        globalAvgPool(cfg.N, cfg.C, d_output, d_input, d_bias, false, q);
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        globalAvgPool(cfg.N, cfg.C, d_output, d_input, d_bias, false, q);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        times.push_back(duration.count());
      }
      
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      // Metrics: sum 64 elements + divide = ~65 FLOPs per output
      double flops = 65.0 * cfg.outputSize;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      
      // Memory: read input + read bias + write output
      double bytes = cfg.inputSize * sizeof(float) + cfg.C * sizeof(float) + cfg.outputSize * sizeof(float);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(10) << cfg.N
           << setw(10) << cfg.C
           << setw(15) << cfg.inputSize
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_input, q);
      sycl::free(d_output, q);
      sycl::free(d_bias, q);
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
