#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace std;

namespace lczero {
namespace sycldnn_backend {

void globalAvgPool_NHWC_fp16(int N, int C, sycl::half* output, const sycl::half* input,
                             const sycl::half* prevLayerBias, sycl::queue& queue) {
  const int kPlaneSize = 64;
  const int inputSize = N * C * kPlaneSize;
  const int outputSize = N * C;

  queue.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(N * C), sycl::range<1>(C)),
    [=](sycl::nd_item<1> item) {
      const int elementsPerThread = 64;

      int blockStart = item.get_group(0) * item.get_local_range(0);
      float S = 0;

      for (int i = 0; i < elementsPerThread; i++) {
        int localIndex = i * item.get_local_range(0) + item.get_local_id(0);
        int inputIndex = blockStart * elementsPerThread + localIndex;
        if (inputIndex < inputSize) S += static_cast<float>(input[inputIndex]);
      }

      float avg = S / elementsPerThread;

      if (prevLayerBias) 
        avg += static_cast<float>(prevLayerBias[item.get_local_id(0)]);

      int opIndex = blockStart + item.get_local_id(0);
      if (opIndex < outputSize) 
        output[opIndex] = static_cast<sycl::half>(avg);
    });
  });
  queue.wait();
}

} // namespace sycldnn_backend
} // namespace lczero

using namespace lczero::sycldnn_backend;

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== global_avg_pool_nhwc_fp16 - Round 1 ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    
    struct TestConfig { int N, C; int inputSize; };
    vector<TestConfig> configs = {
      {16, 64, 16 * 64 * 64},
      {64, 128, 64 * 128 * 64},
      {256, 256, 256 * 256 * 64},
    };
    
    cout << setw(10) << "N" << setw(10) << "C" << setw(15) << "InputSize" << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS" << setw(18) << "GB/s" << endl;
    cout << string(83, '-') << endl;
    
    for (const auto& cfg : configs) {
      sycl::half* d_input = sycl::malloc_device<sycl::half>(cfg.inputSize, q);
      sycl::half* d_output = sycl::malloc_device<sycl::half>(cfg.N * cfg.C, q);
      sycl::half* d_bias = sycl::malloc_device<sycl::half>(cfg.C, q);
      
      vector<sycl::half> h_input(cfg.inputSize, sycl::half(1.0f));
      vector<sycl::half> h_bias(cfg.C, sycl::half(0.5f));
      q.memcpy(d_input, h_input.data(), cfg.inputSize * sizeof(sycl::half)).wait();
      q.memcpy(d_bias, h_bias.data(), cfg.C * sizeof(sycl::half)).wait();
      
      for (int i = 0; i < 3; i++) globalAvgPool_NHWC_fp16(cfg.N, cfg.C, d_output, d_input, d_bias, q);
      
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        globalAvgPool_NHWC_fp16(cfg.N, cfg.C, d_output, d_input, d_bias, q);
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(end - start).count());
      }
      
      double avg_time = 0; for (double t : times) avg_time += t; avg_time /= times.size();
      double flops = 65.0 * cfg.N * cfg.C;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      double bytes = cfg.inputSize * sizeof(sycl::half) + cfg.C * sizeof(sycl::half) + cfg.N * cfg.C * sizeof(sycl::half);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(10) << cfg.N << setw(10) << cfg.C << setw(15) << cfg.inputSize
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_input, q); sycl::free(d_output, q); sycl::free(d_bias, q);
    }
    
    cout << endl << "Test completed!" << endl;
    return 0;
  } catch (sycl::exception const &e) {
    cerr << "SYCL Exception: " << e.what() << endl; return 1;
  } catch (exception const &e) {
    cerr << "Exception: " << e.what() << endl; return 1;
  }
}
