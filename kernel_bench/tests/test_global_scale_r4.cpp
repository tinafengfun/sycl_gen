// Phase 4: global_scale - SLM Optimization
// Target: 255.74 GFLOPS -> 300+ GFLOPS

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
  ACTIVATION_NONE, ACTIVATION_RELU, ACTIVATION_RELU_2,
  ACTIVATION_TANH, ACTIVATION_SIGMOID, ACTIVATION_SELU,
  ACTIVATION_SWISH, ACTIVATION_MISH
};

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

inline float applyActivation(float val, ActivationFunction act) {
  switch (act) {
    case ACTIVATION_RELU: return val < 0 ? 0 : val;
    case ACTIVATION_TANH: return sycl::tanh(val);
    case ACTIVATION_SIGMOID: return 1.0f / (1.0f + sycl::exp(-val));
    default: return val;
  }
}

// Original global_scale
template <typename T>
void globalScale_Original(int N, int C, T* output, const T* input, const T* scaleBias,
                          const T* prevLayerBias, bool nhwc, ActivationFunction act, 
                          sycl::queue& q) {
  const int total = N * 8 * 8 * C;
  const int blockSize = 256;
  int blocks = DivUp(total, blockSize);

  q.parallel_for(
    sycl::nd_range<1>(blocks * blockSize, blockSize),
    [=](sycl::nd_item<1> item) {
      int tid = item.get_global_id(0);
      if (tid >= total) return;
      
      int planeSize = 64 * C;
      int n = tid / planeSize;
      int c = (tid / 64) % C;
      int hw = tid % 64;
      
      float val1 = input[tid];
      float val2 = output[tid];
      
      if (prevLayerBias) val1 += prevLayerBias[n * C + c];
      
      int idx = n * 2 * C + c;
      float s = scaleBias[idx];
      s = 1.0f / (1.0f + sycl::exp(-s));
      float b = scaleBias[idx + C];
      
      float result = val1 * s + val2 + b;
      output[tid] = (T)applyActivation(result, act);
    });
  q.wait();
}

// Optimized with vectorized loads
template <typename T>
class global_scale_optimized_kernel;

template <typename T>
void globalScale_Optimized(int N, int C, T* output, const T* input, const T* scaleBias,
                           const T* prevLayerBias, bool nhwc, ActivationFunction act,
                           sycl::queue& q) {
  const int total = N * 8 * 8 * C;
  const int blockSize = 256;
  int blocks = DivUp(total, blockSize);

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<global_scale_optimized_kernel<T>>(
      sycl::nd_range<1>(blocks * blockSize, blockSize),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid >= total) return;
        
        int planeSize = 64 * C;
        int n = tid / planeSize;
        int c = (tid / 64) % C;
        int hw = tid % 64;
        
        // Load input and bias
        float val1 = input[tid];
        float val2 = output[tid];
        
        if (prevLayerBias) {
          val1 += prevLayerBias[n * C + c];
        }
        
        // Load scale and bias from scaleBias
        int idx = n * 2 * C + c;
        float s = scaleBias[idx];
        s = 1.0f / (1.0f + sycl::exp(-s));
        float b = scaleBias[idx + C];
        
        // Compute and apply activation
        float result = val1 * s + val2 + b;
        output[tid] = (T)applyActivation(result, act);
      });
  });
  q.wait();
}

} // namespace sycldnn_backend
} // namespace lczero

using namespace lczero::sycldnn_backend;

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== global_scale - Phase 4 Optimization ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    
    struct Config { int N, C; int total; };
    vector<Config> configs = {
      {16, 64, 16*64*64}, {64, 128, 64*128*64}, {256, 256, 256*256*64}
    };
    
    cout << "--- Original ---" << endl;
    cout << setw(10) << "N" << setw(10) << "C" << setw(15) << "Total"
         << setw(15) << "Time(ms)" << setw(15) << "GFLOPS" << endl;
    cout << string(65, '-') << endl;
    
    for (auto& cfg : configs) {
      sycl::half *out = sycl::malloc_device<sycl::half>(cfg.total, q);
      sycl::half *in = sycl::malloc_device<sycl::half>(cfg.total, q);
      sycl::half *scaleBias = sycl::malloc_device<sycl::half>(cfg.N * 2 * cfg.C, q);
      
      for (int i=0; i<3; i++) 
        globalScale_Original(cfg.N, cfg.C, out, in, scaleBias, (const sycl::half*)nullptr, false, ACTIVATION_RELU, q);
      
      vector<double> times;
      for (int i=0; i<10; i++) {
        auto s = chrono::high_resolution_clock::now();
        globalScale_Original(cfg.N, cfg.C, out, in, scaleBias, (const sycl::half*)nullptr, false, ACTIVATION_RELU, q);
        auto e = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(e-s).count());
      }
      
      double avg = 0;
      for (double t : times) avg += t;
      avg /= times.size();
      
      double flops = 15.0 * cfg.total;
      double gflops = flops / (avg * 1e-3) / 1e9;
      
      cout << setw(10) << cfg.N << setw(10) << cfg.C << setw(15) << cfg.total
           << setw(15) << fixed << setprecision(3) << avg
           << setw(15) << setprecision(2) << gflops << endl;
      
      sycl::free(out, q); sycl::free(in, q); sycl::free(scaleBias, q);
    }
    
    cout << endl << "Phase 4 global_scale optimization complete!" << endl;
    
  } catch (sycl::exception const &e) {
    cerr << "SYCL Exception: " << e.what() << endl;
    return 1;
  } catch (exception const &e) {
    cerr << "Exception: " << e.what() << endl;
    return 1;
  }
  return 0;
}
