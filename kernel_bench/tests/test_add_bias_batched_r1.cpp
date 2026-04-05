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
  ACTIVATION_MISH
};

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
      cVal = 1.0f / (1.0f + sycl::native::exp(-cVal));
      break;
    case ACTIVATION_SELU: {
      float alpha = 1.67326324f, scale = 1.05070098f;
      if (cVal > 0)
        cVal = scale * cVal;
      else
        cVal = scale * alpha * (sycl::native::exp(cVal) - 1.0f);
      break;
    }
    case ACTIVATION_SWISH:
      cVal = cVal / (1.0f + sycl::native::exp(-cVal));
      break;
    case ACTIVATION_MISH: {
      auto e = sycl::native::exp(cVal);
      auto n = e * e + 2.0f * e;
      auto d = cVal / (n + 2.0f);
      if (cVal <= -0.6f) {
        cVal = n * d;
      } else {
        cVal = cVal - 2.0f * d;
      }
      break;
    }
    case ACTIVATION_NONE:
    default:
      break;
  }
  return cVal;
}

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void copyAs(void* dst, const void* src) {
  *reinterpret_cast<T*>(dst) = *reinterpret_cast<const T*>(src);
}

template <typename T, ActivationFunction act>
void addBiasBatched_kernel(T* output, const T* input, const T* bias,
                           int N, int C,
                           const sycl::nd_item<3> &item_ct1) {
  int batch = item_ct1.get_group(1);
  int n = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
          item_ct1.get_local_id(1);
  if (n >= N) return;
  int c = item_ct1.get_local_id(2) * 4;

  int biasIndex = batch * C + c;
  int tensorIndex = batch * N * C + n * C + c;

  float val[4];
  float b[4];

  // Load from memory
  const bool fp16 = std::is_same<sycl::half, T>::value;
  if (fp16) {
    sycl::half inp[4];
    copyAs<sycl::uint2>(&inp[0], &input[tensorIndex]);
    #pragma unroll
    for (int i = 0; i < 4; i++) val[i] = (float)inp[i];

    copyAs<sycl::uint2>(&inp[0], &bias[biasIndex]);
    #pragma unroll
    for (int i = 0; i < 4; i++) b[i] = (float)inp[i];
  } else {
    copyAs<sycl::uint4>(&val[0], &input[tensorIndex]);
    copyAs<sycl::uint4>(&b[0], &bias[biasIndex]);
  }

  // Perform bias add and activation
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    float x = val[i] + b[i];
    x = activate(x, act);
    val[i] = x;
  }

  // write to memory
  if (fp16) {
    sycl::half op[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) op[i] = (sycl::half)val[i];
    copyAs<sycl::uint2>(&output[tensorIndex], &op[0]);
  } else {
    copyAs<sycl::uint4>(&output[tensorIndex], &val[0]);
  }
}

template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, ActivationFunction activation, sycl::queue &sycl_queue) {
  // process 4 elements per thread to achieve close to peak memory bandwidth
  if (C % 4 != 0) printf("Error: unsupported filter size");
  if (C > 2048) printf("Error: unsupported filter size");

  sycl::range<3> blockDim(1, 1, 1), gridDim(1, 1, 1);
  blockDim[2] = C / 4;
  unsigned int tmp = (512 / blockDim[2]);
  blockDim[1] = sycl::min(sycl::max(tmp, 1u), (unsigned int)N);
  blockDim[0] = 1;
  gridDim[2] = DivUp(N, blockDim[1]);
  gridDim[1] = Batch;
  gridDim[0] = 1;

  switch (activation) {
    case ACTIVATION_NONE:
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                         [=](sycl::nd_item<3> item_ct1) {
                           addBiasBatched_kernel<T, ACTIVATION_NONE>(output, input, bias,
                                                            N, C, item_ct1);
                         });
      break;
    case ACTIVATION_RELU:
      sycl_queue.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                         [=](sycl::nd_item<3> item_ct1) {
                           addBiasBatched_kernel<T, ACTIVATION_RELU>(output, input, bias,
                                                            N, C, item_ct1);
                         });
      break;
    default:
      printf("Error: unsupported activation");
  }
}

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== add_bias_batched - Round 1 (Real Kernel) ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    
    // Test configurations: (Batch, N, C) - C must be multiple of 4
    struct TestConfig {
      int Batch, N, C;
      int totalElements;
    };
    
    vector<TestConfig> configs = {
      {1, 8, 64},       // 512 elements
      {2, 16, 128},     // 4096 elements
      {4, 32, 256},     // 32768 elements
      {8, 64, 512},     // 262144 elements
      {16, 64, 1024},   // 1048576 elements (~1M)
    };
    
    for (auto& cfg : configs) {
      cfg.totalElements = cfg.Batch * cfg.N * cfg.C;
    }
    
    cout << setw(18) << "Config (BxNxC)" 
         << setw(12) << "Total"
         << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS"
         << setw(18) << "GB/s" << endl;
    cout << string(78, '-') << endl;
    
    for (const auto& cfg : configs) {
      const int Batch = cfg.Batch;
      const int N = cfg.N;
      const int C = cfg.C;
      const int total = cfg.totalElements;
      
      // Allocate device memory
      sycl::half* d_input = sycl::malloc_device<sycl::half>(total, q);
      sycl::half* d_bias = sycl::malloc_device<sycl::half>(Batch * C, q);
      sycl::half* d_output = sycl::malloc_device<sycl::half>(total, q);
      
      // Initialize input (Batch x N x C)
      vector<sycl::half> h_input(total);
      for (int b = 0; b < Batch; b++) {
        for (int n = 0; n < N; n++) {
          for (int c = 0; c < C; c++) {
            int idx = b * N * C + n * C + c;
            h_input[idx] = sycl::half(1.0f);
          }
        }
      }
      q.memcpy(d_input, h_input.data(), total * sizeof(sycl::half)).wait();
      
      // Initialize bias (Batch x C)
      vector<sycl::half> h_bias(Batch * C);
      for (int b = 0; b < Batch; b++) {
        for (int c = 0; c < C; c++) {
          h_bias[b * C + c] = sycl::half(0.5f);
        }
      }
      q.memcpy(d_bias, h_bias.data(), Batch * C * sizeof(sycl::half)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        addBiasBatched(d_output, d_input, d_bias, Batch, N, C, ACTIVATION_RELU, q);
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        
        addBiasBatched(d_output, d_input, d_bias, Batch, N, C, ACTIVATION_RELU, q);
        
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
      
      // Memory: read input + read bias + write output = 3 * total * 2 bytes
      double bytes = 3.0 * total * sizeof(sycl::half);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(18) << (to_string(Batch) + "x" + to_string(N) + "x" + to_string(C))
           << setw(12) << total
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_input, q);
      sycl::free(d_bias, q);
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
