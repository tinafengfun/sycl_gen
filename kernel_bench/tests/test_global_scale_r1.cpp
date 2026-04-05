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

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void globalScale_kernel(T* output, const T* input,
                        const T* scaleBias, const T* prevLayerBias,
                        int inputSize, int C,
                        ActivationFunction activation,
                        const sycl::nd_item<3> &item_ct1) {
  const int kPlaneSize = 64;

  int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

  if (tid > inputSize) return;

  int nc = tid / kPlaneSize;
  int n = nc / C;
  int c = nc % C;

  float val1 = input[tid];
  float val2 = output[tid];

  if (prevLayerBias) {
    val1 += (float)(prevLayerBias[c]);
  }

  int startIdx = n * 2 * C;

  float s = scaleBias[startIdx + c];
  s = 1.0f / (1.0f + sycl::exp(-s));

  float b = scaleBias[startIdx + c + C];

  float op = val1 * s + val2 + b;
  
  switch (activation) {
    case ACTIVATION_RELU:
      if (op < 0) op = 0;
      break;
    case ACTIVATION_TANH:
      op = sycl::tanh(op);
      break;
    case ACTIVATION_SIGMOID:
      op = 1.0f / (1.0f + sycl::exp(-op));
      break;
    default:
      break;
  }
  
  output[tid] = (T)op;
}

template <typename T>
void globalScale(int N, int C, T* output, const T* input, const T* scaleBias,
                 const T* prevLayerBias, bool nhwc,
                 ActivationFunction activation, sycl::queue &sycl_queue) {
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * 8 * 8 * C, kBlockSize);

  sycl_queue.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, kBlocks) * sycl::range<3>(1, 1, kBlockSize),
          sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item_ct1) {
        globalScale_kernel(output, input, scaleBias, prevLayerBias,
                           N * C * 8 * 8, C, activation, item_ct1);
      });
  sycl_queue.wait();
}

} // namespace sycldnn_backend
} // namespace lczero

using namespace lczero::sycldnn_backend;

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== global_scale - Round 1 (Real Kernel) ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    
    struct TestConfig {
      int N, C;
      int inputSize;
    };
    
    vector<TestConfig> configs = {
      {16, 64, 16 * 64 * 64},
      {64, 128, 64 * 128 * 64},
      {256, 256, 256 * 256 * 64},
    };
    
    cout << setw(10) << "N" 
         << setw(10) << "C"
         << setw(15) << "InputSize"
         << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS"
         << setw(18) << "GB/s" << endl;
    cout << string(83, '-') << endl;
    
    for (const auto& cfg : configs) {
      sycl::half* d_input = sycl::malloc_device<sycl::half>(cfg.inputSize, q);
      sycl::half* d_output = sycl::malloc_device<sycl::half>(cfg.inputSize, q);
      sycl::half* d_scaleBias = sycl::malloc_device<sycl::half>(cfg.N * 2 * cfg.C, q);
      sycl::half* d_bias = sycl::malloc_device<sycl::half>(cfg.C, q);
      
      vector<sycl::half> h_input(cfg.inputSize, sycl::half(1.0f));
      vector<sycl::half> h_output(cfg.inputSize, sycl::half(0.5f));
      vector<sycl::half> h_scaleBias(cfg.N * 2 * cfg.C, sycl::half(0.0f));
      vector<sycl::half> h_bias(cfg.C, sycl::half(0.5f));
      
      q.memcpy(d_input, h_input.data(), cfg.inputSize * sizeof(sycl::half)).wait();
      q.memcpy(d_output, h_output.data(), cfg.inputSize * sizeof(sycl::half)).wait();
      q.memcpy(d_scaleBias, h_scaleBias.data(), cfg.N * 2 * cfg.C * sizeof(sycl::half)).wait();
      q.memcpy(d_bias, h_bias.data(), cfg.C * sizeof(sycl::half)).wait();
      
      for (int i = 0; i < 3; i++) {
        globalScale(cfg.N, cfg.C, d_output, d_input, d_scaleBias, d_bias, false, ACTIVATION_RELU, q);
      }
      
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        globalScale(cfg.N, cfg.C, d_output, d_input, d_scaleBias, d_bias, false, ACTIVATION_RELU, q);
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(end - start).count());
      }
      
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      double flops = 15.0 * cfg.inputSize;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      
      double bytes = cfg.inputSize * 2 * sizeof(sycl::half) + cfg.N * 2 * cfg.C * sizeof(sycl::half) + cfg.C * sizeof(sycl::half);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(10) << cfg.N
           << setw(10) << cfg.C
           << setw(15) << cfg.inputSize
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_input, q);
      sycl::free(d_output, q);
      sycl::free(d_scaleBias, q);
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
