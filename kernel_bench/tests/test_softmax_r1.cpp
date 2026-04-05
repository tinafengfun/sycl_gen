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

struct AtomicMaxFloat {
  float operator()(sycl::local_accessor<float, 1> local_acc, float val) const {
    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                     sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space>
        atomic_max(local_acc[0]);
    float old = atomic_max.load();
    while (val > old && !atomic_max.compare_exchange_strong(old, val));
    return old;
  }
};

struct AtomicAddFloat {
  void operator()(sycl::local_accessor<float, 1> local_acc, float val) const {
    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                     sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space>
        atomic_sum(local_acc[0]);
    atomic_sum.fetch_add(val);
  }
};

template <typename T>
void Softmax(int N, int C, T* output, const T* input, const T* input2,
             sycl::queue& queue) {
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> max_acc(sycl::range<1>(1), cgh);
    sycl::local_accessor<float, 1> sum_acc(sycl::range<1>(1), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(N * C), sycl::range<1>(C)),
        [=](sycl::nd_item<1> item) {
          const int n = item.get_group(0);
          const int c = item.get_local_id(0);
          const int index = n * C + c;

          float x = static_cast<float>(input[index]);
          if (input2 != nullptr) x += static_cast<float>(input2[index]);

          if (c == 0) {
            max_acc[0] = x;
            sum_acc[0] = 0.0f;
          }
          item.barrier(sycl::access::fence_space::local_space);

          auto sg = item.get_sub_group();
          const uint32_t subgroup_size = sg.get_local_range().size();
          float warpmax = x;

          for (uint32_t mask = subgroup_size / 2; mask > 0; mask >>= 1) {
            float tmp = sycl::permute_group_by_xor(sg, warpmax, mask);
            warpmax = sycl::max(warpmax, tmp);
          }

          if (sg.get_local_id() == 0) {
            AtomicMaxFloat()(max_acc, warpmax);
          }
          item.barrier(sycl::access::fence_space::local_space);

          float ex = sycl::exp(x - max_acc[0]);

          float warp_sum = ex;
          for (uint32_t mask = subgroup_size / 2; mask > 0; mask >>= 1) {
            float tmp = sycl::permute_group_by_xor(sg, warp_sum, mask);
            warp_sum += tmp;
          }

          if (sg.get_local_id() == 0) {
            AtomicAddFloat()(sum_acc, warp_sum);
          }
          item.barrier(sycl::access::fence_space::local_space);

          output[index] = static_cast<T>(ex / sum_acc[0]);
        });
  });
  queue.wait_and_throw();
}

} // namespace sycldnn_backend
} // namespace lczero

using namespace lczero::sycldnn_backend;

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== softmax - Round 1 ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    
    struct TestConfig { int N, C; int totalSize; };
    vector<TestConfig> configs = {
      {64, 64, 64 * 64},
      {256, 256, 256 * 256},
      {1024, 1024, 1024 * 1024},
    };
    
    cout << setw(10) << "N" << setw(10) << "C" << setw(15) << "TotalSize"
         << setw(15) << "Time(ms)" << setw(15) << "GFLOPS" << setw(18) << "GB/s" << endl;
    cout << string(83, '-') << endl;
    
    for (const auto& cfg : configs) {
      sycl::half* d_output = sycl::malloc_device<sycl::half>(cfg.totalSize, q);
      sycl::half* d_input = sycl::malloc_device<sycl::half>(cfg.totalSize, q);
      
      vector<sycl::half> h_input(cfg.totalSize);
      srand(42);
      for (int i = 0; i < cfg.totalSize; i++) h_input[i] = sycl::half((float)(rand() % 100) / 100.0f);
      q.memcpy(d_input, h_input.data(), cfg.totalSize * sizeof(sycl::half)).wait();
      
      for (int i = 0; i < 3; i++) Softmax(cfg.N, cfg.C, d_output, d_input, (const sycl::half*)nullptr, q);
      
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        Softmax(cfg.N, cfg.C, d_output, d_input, (const sycl::half*)nullptr, q);
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(end - start).count());
      }
      
      double avg_time = 0; for (double t : times) avg_time += t; avg_time /= times.size();
      double flops = 10.0 * cfg.totalSize;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      double bytes = cfg.totalSize * 2 * sizeof(sycl::half);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(10) << cfg.N << setw(10) << cfg.C << setw(15) << cfg.totalSize
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_output, q); sycl::free(d_input, q);
    }
    
    cout << endl << "Test completed!" << endl;
    return 0;
  } catch (sycl::exception const &e) {
    cerr << "SYCL Exception: " << e.what() << endl; return 1;
  } catch (exception const &e) {
    cerr << "Exception: " << e.what() << endl; return 1;
  }
}
