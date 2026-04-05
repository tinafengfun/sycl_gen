#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>

using namespace std;

namespace lczero {
namespace sycldnn_backend {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void PolicyMap(int N, T* output, const T* input, const short* indices,
               int inputSize, int usedSize, int outputSize, sycl::queue &sycl_queue) {
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * usedSize, kBlockSize);

  sycl_queue.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, kBlocks) *
                                             sycl::range<3>(1, 1, kBlockSize),
                                         sycl::range<3>(1, 1, kBlockSize)),
                       [=](sycl::nd_item<3> item_ct1) {
                         int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                                   item_ct1.get_local_id(2);

                         int n = tid / usedSize;
                         int i = tid % usedSize;

                         if (n >= N) return;

                         int j = indices[i];

                         if (j >= 0) {
                           output[n * outputSize + j] = input[n * inputSize + i];
                         }
                       });
  sycl_queue.wait();
}

} // namespace sycldnn_backend
} // namespace lczero

using namespace lczero::sycldnn_backend;

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== policy_map - Round 1 ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    
    struct TestConfig { 
      int N, inputSize, usedSize, outputSize; 
      int totalInput;
    };
    
    // Policy mapping: input (NCHW) -> output (policy indices)
    vector<TestConfig> configs = {
      {64, 64 * 8 * 8, 64 * 8 * 8, 1858, 64 * 64 * 8 * 8},
      {256, 128 * 8 * 8, 128 * 8 * 8, 1858, 256 * 128 * 8 * 8},
      {1024, 256 * 8 * 8, 256 * 8 * 8, 1858, 1024 * 256 * 8 * 8},
    };
    
    cout << setw(10) << "N" << setw(15) << "InputSize" << setw(15) << "UsedSize"
         << setw(15) << "OutputSize" << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS" << setw(18) << "GB/s" << endl;
    cout << string(103, '-') << endl;
    
    for (const auto& cfg : configs) {
      sycl::half* d_output = sycl::malloc_device<sycl::half>(cfg.N * cfg.outputSize, q);
      sycl::half* d_input = sycl::malloc_device<sycl::half>(cfg.totalInput, q);
      short* d_indices = sycl::malloc_device<short>(cfg.usedSize, q);
      
      vector<sycl::half> h_input(cfg.totalInput);
      vector<short> h_indices(cfg.usedSize);
      
      srand(42);
      for (int i = 0; i < cfg.totalInput; i++) h_input[i] = sycl::half((float)(rand() % 100) / 100.0f);
      for (int i = 0; i < cfg.usedSize; i++) h_indices[i] = (short)(rand() % cfg.outputSize);
      
      q.memcpy(d_input, h_input.data(), cfg.totalInput * sizeof(sycl::half)).wait();
      q.memcpy(d_indices, h_indices.data(), cfg.usedSize * sizeof(short)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        PolicyMap(cfg.N, d_output, d_input, d_indices, cfg.inputSize, cfg.usedSize, cfg.outputSize, q);
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        PolicyMap(cfg.N, d_output, d_input, d_indices, cfg.inputSize, cfg.usedSize, cfg.outputSize, q);
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(end - start).count());
      }
      
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      double flops = 2.0 * cfg.N * cfg.usedSize;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      
      double bytes = cfg.totalInput * sizeof(sycl::half) + cfg.N * cfg.usedSize * sizeof(short) + cfg.N * cfg.outputSize * sizeof(sycl::half);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(10) << cfg.N << setw(15) << cfg.inputSize << setw(15) << cfg.usedSize
           << setw(15) << cfg.outputSize << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_output, q);
      sycl::free(d_input, q);
      sycl::free(d_indices, q);
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
