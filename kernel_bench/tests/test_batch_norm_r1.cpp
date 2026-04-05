#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

using namespace std;

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
  auto e = sycl::native::exp(el);
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
    case ACTIVATION_SELU:
      cVal = 1.0507f * (cVal >= 0 ? cVal : 1.67326f * (sycl::exp(cVal) - 1));
      break;
    case ACTIVATION_SWISH:
      cVal = cVal / (1.0f + sycl::exp(-cVal));
      break;
    case ACTIVATION_MISH:
      cVal = mishActivate(cVal);
      break;
    case ACTIVATION_NONE:
    case ACTIVATION_DEFAULT:
      break;
    case ACTIVATION_SOFTMAX:
      break;
  }
  return cVal;
}

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void batchNorm_kernel(T* output, const T* input, const T* skipInput,
                      int N, int C, int H, int W, const float* means,
                      const float* varMultipliers,
                      ActivationFunction activation,
                      const sycl::nd_item<3> &item_ct1) {
  int index = item_ct1.get_local_id(2) +
              item_ct1.get_local_range(2) * item_ct1.get_group(2);

  int wIndex = 0;
  if (sizeof(T) == sizeof(float))
    wIndex = (index / (H * W)) % C;  // NCHW for fp32.
  else
    wIndex = index % C;  // NHWC for fp16.

  float el = input[index];
  float mean = means[wIndex];
  float varMulti = varMultipliers[wIndex];

  el -= mean;
  el *= varMulti;

  if (skipInput) el += (float)skipInput[index];

  el = activate(el, activation);

  output[index] = (T)el;
}

template <typename T>
void batchNorm(T* output, const T* input, const T* skipInput, int N, int C,
               int H, int W, const float* means, const float* var_multipliers,
               ActivationFunction activation, sycl::queue &sycl_queue) {
  const int total_elements = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(total_elements, kBlockSize);

  sycl_queue.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
          sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item_ct1) {
        batchNorm_kernel<T>(output, input, skipInput, N, C, H, W, means,
                         var_multipliers, activation, item_ct1);
      });
}

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== batch_norm - Round 1 (Real Kernel) ===" << endl;
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
      sycl::half* d_input = sycl::malloc_device<sycl::half>(total, q);
      sycl::half* d_output = sycl::malloc_device<sycl::half>(total, q);
      float* d_means = sycl::malloc_device<float>(C, q);
      float* d_varMultipliers = sycl::malloc_device<float>(C, q);
      
      // Initialize input
      vector<sycl::half> h_input(total, sycl::half(1.0f));
      q.memcpy(d_input, h_input.data(), total * sizeof(sycl::half)).wait();
      
      // Initialize means and varMultipliers (per channel)
      vector<float> h_means(C, 0.5f);
      vector<float> h_varMultipliers(C, 1.0f);
      q.memcpy(d_means, h_means.data(), C * sizeof(float)).wait();
      q.memcpy(d_varMultipliers, h_varMultipliers.data(), C * sizeof(float)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        batchNorm(d_output, d_input, (const sycl::half*)nullptr, N, C, H, W, d_means, d_varMultipliers, ACTIVATION_RELU, q);
        q.wait();
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        
        batchNorm(d_output, d_input, (const sycl::half*)nullptr, N, C, H, W, d_means, d_varMultipliers, ACTIVATION_RELU, q);
        q.wait();
        
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        times.push_back(duration.count());
      }
      
      // Stats
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      // Metrics: 1 sub + 1 mul + 1 ReLU = 3 FLOPs per element
      double flops = 3.0 * total;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      
      // Memory: read input + read means + read var + write output
      // = total*2 + C*4 + C*4 + total*2 bytes
      double bytes = 2.0 * total * sizeof(sycl::half) + 2.0 * C * sizeof(float);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(20) << (to_string(N) + "x" + to_string(C) + "x" + to_string(H) + "x" + to_string(W))
           << setw(12) << total
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_input, q);
      sycl::free(d_output, q);
      sycl::free(d_means, q);
      sycl::free(d_varMultipliers, q);
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
