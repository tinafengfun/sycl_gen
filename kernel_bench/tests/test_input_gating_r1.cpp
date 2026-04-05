#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace std;

namespace lczero {
namespace sycldnn_backend {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void applyInputGating(T* output, const T* input, const T* mult, const T* add,
                      int N, int HW, int C, sycl::queue& stream) {
  sycl::range<3> blockSize(DivUp(1024, HW), HW, 1);
  sycl::range<3> gridSize(DivUp(C, blockSize[0]), 1, N);
  
  stream.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<3>(gridSize * blockSize, blockSize),
      [=](sycl::nd_item<3> item) {
        int n_offset = item.get_group(2) * HW * C;
        int idx = item.get_local_id(1) * C + item.get_group(0) * item.get_local_range(0) +
                  item.get_local_id(0);
        int idxT = (item.get_group(0) * item.get_local_range(0) + item.get_local_id(0)) * HW +
                   item.get_local_id(1);

        if (idx < HW * C) {
          float op = (float)input[n_offset + idx] * (float)mult[idxT] + (float)add[idxT];
          output[n_offset + idx] = (T)op;
        }
      });
  });
  stream.wait();
}

} // namespace sycldnn_backend
} // namespace lczero

using namespace lczero::sycldnn_backend;

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== input_gating - Round 1 ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    
    struct TestConfig { int N, HW, C; int totalSize; };
    vector<TestConfig> configs = {
      {16, 64, 64, 16 * 64 * 64},
      {64, 64, 128, 64 * 64 * 128},
      {256, 64, 256, 256 * 64 * 256},
    };
    
    cout << setw(10) << "N" << setw(10) << "HW" << setw(10) << "C" << setw(15) << "TotalSize"
         << setw(15) << "Time(ms)" << setw(15) << "GFLOPS" << setw(18) << "GB/s" << endl;
    cout << string(93, '-') << endl;
    
    for (const auto& cfg : configs) {
      int multAddSize = cfg.HW * cfg.C;
      sycl::half* d_output = sycl::malloc_device<sycl::half>(cfg.totalSize, q);
      sycl::half* d_input = sycl::malloc_device<sycl::half>(cfg.totalSize, q);
      sycl::half* d_mult = sycl::malloc_device<sycl::half>(multAddSize, q);
      sycl::half* d_add = sycl::malloc_device<sycl::half>(multAddSize, q);
      
      vector<sycl::half> h_input(cfg.totalSize, sycl::half(1.0f));
      vector<sycl::half> h_mult(multAddSize, sycl::half(0.5f));
      vector<sycl::half> h_add(multAddSize, sycl::half(0.1f));
      
      q.memcpy(d_input, h_input.data(), cfg.totalSize * sizeof(sycl::half)).wait();
      q.memcpy(d_mult, h_mult.data(), multAddSize * sizeof(sycl::half)).wait();
      q.memcpy(d_add, h_add.data(), multAddSize * sizeof(sycl::half)).wait();
      
      for (int i = 0; i < 3; i++) applyInputGating(d_output, d_input, d_mult, d_add, cfg.N, cfg.HW, cfg.C, q);
      
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        applyInputGating(d_output, d_input, d_mult, d_add, cfg.N, cfg.HW, cfg.C, q);
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(end - start).count());
      }
      
      double avg_time = 0; for (double t : times) avg_time += t; avg_time /= times.size();
      double flops = 3.0 * cfg.totalSize;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      double bytes = (cfg.totalSize * 2 + multAddSize * 2) * sizeof(sycl::half);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(10) << cfg.N << setw(10) << cfg.HW << setw(10) << cfg.C << setw(15) << cfg.totalSize
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_output, q); sycl::free(d_input, q); sycl::free(d_mult, q); sycl::free(d_add, q);
    }
    
    cout << endl << "Test completed!" << endl;
    return 0;
  } catch (sycl::exception const &e) {
    cerr << "SYCL Exception: " << e.what() << endl; return 1;
  } catch (exception const &e) {
    cerr << "Exception: " << e.what() << endl; return 1;
  }
}
