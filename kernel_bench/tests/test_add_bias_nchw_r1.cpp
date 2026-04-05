#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

using namespace std;

// Activation function enum
enum ActivationFunction {
  ACTIVATION_NONE,
  ACTIVATION_RELU,
  ACTIVATION_RELU_2,
  ACTIVATION_TANH,
  ACTIVATION_SIGMOID,
  ACTIVATION_SELU,
  ACTIVATION_SWISH,
  ACTIVATION_MISH,
  ACTIVATION_DEFAULT,
  ACTIVATION_SOFTMAX
};

inline float mishActivate(float el) {
  auto e = sycl::exp(el);
  auto n = e * e + 2.0f * e;
  auto d = el / (n + 2.0f);
  if (el <= -0.6f) {
    return n * d;
  } else {
    return el - 2.0f * d;
  }
}

inline float activate(float cVal, ActivationFunction activation) {
  switch (activation) {
    case ACTIVATION_RELU:
      if (cVal < 0) cVal = 0;
      break;
    case ACTIVATION_RELU_2:
      if (cVal < 0) cVal = 0;
      cVal *= cVal;
      break;
    case ACTIVATION_TANH:
      cVal = sycl::tanh(cVal);
      break;
    case ACTIVATION_SIGMOID:
      cVal = 1.0f / (1.0f + sycl::exp(-cVal));
      break;
    case ACTIVATION_SELU: {
      float alpha = 1.67326324f, scale = 1.05070098f;
      if (cVal > 0)
        cVal = scale * cVal;
      else
        cVal = scale * alpha * (sycl::exp(cVal) - 1.0f);
      break;
    }
    case ACTIVATION_MISH:
      cVal = mishActivate(cVal);
      break;
    case ACTIVATION_SWISH:
      cVal /= (1.0f + sycl::exp(-cVal));
      break;
    case ACTIVATION_NONE:
    default:
      break;
  }
  return cVal;
}

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void addBias_NCHW(T* c, T* a, T* b, int N, int C, int H, int W,
                  ActivationFunction activation, sycl::queue& stream) {
  int size = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  stream.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize), sycl::range<1>(kBlockSize)),
        [=](sycl::nd_item<1> item) {
          int i = item.get_global_id(0);
          if (i < size) {
            float aVal = static_cast<float>(a[i]);
            int biasIndex = (i / (H * W)) % C;
            float bVal = static_cast<float>(b[biasIndex]);
            float cVal = aVal + bVal;
            cVal = activate(cVal, activation);
            c[i] = static_cast<T>(cVal);
          }
        });
  });
  stream.wait_and_throw();
}

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== add_bias_nchw - Round 1 (Real Kernel) ===" << endl;
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
      
      // Allocate device memory
      sycl::half* d_a = sycl::malloc_device<sycl::half>(total, q);
      sycl::half* d_b = sycl::malloc_device<sycl::half>(C, q);  // bias is per-channel
      sycl::half* d_c = sycl::malloc_device<sycl::half>(total, q);
      
      // Initialize input (NCHW format)
      vector<sycl::half> h_a(total, sycl::half(1.0f));
      q.memcpy(d_a, h_a.data(), total * sizeof(sycl::half)).wait();
      
      // Initialize bias (C elements)
      vector<sycl::half> h_b(C, sycl::half(0.5f));
      q.memcpy(d_b, h_b.data(), C * sizeof(sycl::half)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        addBias_NCHW(d_c, d_a, d_b, N, C, H, W, ACTIVATION_RELU, q);
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        
        addBias_NCHW(d_c, d_a, d_b, N, C, H, W, ACTIVATION_RELU, q);
        
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        times.push_back(duration.count());
      }
      
      // Stats
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      // Metrics: 1 add + 1 ReLU = 2 FLOPs per element
      double flops = 2.0 * total;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      
      // Memory: read a + read b + write c = 3 * total * 2 bytes (b is small, ignore for bandwidth)
      double bytes = (2.0 * total + C) * sizeof(sycl::half);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(20) << (to_string(N) + "x" + to_string(C) + "x" + to_string(H) + "x" + to_string(W))
           << setw(12) << total
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_a, q);
      sycl::free(d_b, q);
      sycl::free(d_c, q);
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
