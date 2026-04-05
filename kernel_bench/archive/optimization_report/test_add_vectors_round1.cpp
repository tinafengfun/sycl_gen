/*
  add_vectors_kernel - Round 1 Optimization
  Type A: Element-wise operation
  Strategy: FP16 + Vectorized loads (half2), WG=128
*/

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>

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

[[gnu::always_inline]]
inline float activate(float cVal, ActivationFunction activation) {
  switch (activation) {
    case ACTIVATION_RELU:
      if (cVal < 0) cVal = 0;
      break;
    case ACTIVATION_SIGMOID:
      cVal = 1.0f / (1.0f + sycl::native::exp(-cVal));
      break;
    default:
      break;
  }
  return cVal;
}

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// BASELINE: Original kernel
template <typename T>
void addVectors_baseline_kernel(T* c, T* a, T* b, int size, int asize,
                                int bsize, ActivationFunction activation,
                                const sycl::nd_item<3> &item_ct1) {
  int i = item_ct1.get_local_id(2) +
          item_ct1.get_local_range(2) * item_ct1.get_group(2);
  if (i < size) {
    float aVal = a ? (float)(a[i % asize]) : 0;
    float bVal = b ? (float)(b[i % bsize]) : 0;
    float cVal = aVal + bVal;
    cVal = activate(cVal, activation);
    c[i] = (T)cVal;
  }
}

// ROUND 1: FP16 + Vectorized loads (half2), WG=128
template <typename T>
void addVectors_round1_kernel(T* c, T* a, T* b, int size, int asize,
                              int bsize, ActivationFunction activation,
                              const sycl::nd_item<1> &item_ct1) {
  const int vec_size = 2;
  int tid = item_ct1.get_global_id(0);
  int start_idx = tid * vec_size;
  
  if (start_idx < size) {
    #pragma unroll
    for (int i = 0; i < vec_size && (start_idx + i) < size; i++) {
      float aVal = a ? (float)(a[(start_idx + i) % asize]) : 0;
      float bVal = b ? (float)(b[(start_idx + i) % bsize]) : 0;
      float cVal = aVal + bVal;
      cVal = activate(cVal, activation);
      c[start_idx + i] = (T)cVal;
    }
  }
}

template <typename T>
void addVectors_baseline(T* c, T* a, T* b, int size, int asize, int bsize,
                         ActivationFunction activation, sycl::queue &sycl_queue) {
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);
  
  sycl_queue.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
          sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item_ct1) {
        addVectors_baseline_kernel(c, a, b, size, asize, bsize, activation, item_ct1);
      });
}

template <typename T>
void addVectors_round1(T* c, T* a, T* b, int size, int asize, int bsize,
                       ActivationFunction activation, sycl::queue &sycl_queue) {
  const int wg_size = 128;
  const int vec_size = 2;
  int total_threads = DivUp(size, vec_size);
  int num_wgs = DivUp(total_threads, wg_size);
  
  sycl_queue.parallel_for(
      sycl::nd_range<1>(num_wgs * wg_size, wg_size),
      [=](sycl::nd_item<1> item_ct1) {
        addVectors_round1_kernel(c, a, b, size, asize, bsize, activation, item_ct1);
      });
}

} // namespace sycldnn_backend
} // namespace lczero

using namespace lczero::sycldnn_backend;

struct TestResult {
  size_t size;
  double avg_time_ms;
  double gflops;
  double bandwidth_gbps;
};

std::vector<TestResult> run_tests(sycl::queue &q, const std::vector<size_t> &sizes, 
                                   int iterations, bool use_round1) {
  std::vector<TestResult> results;
  
  for (size_t size : sizes) {
    sycl::half *d_a = sycl::malloc_device<sycl::half>(size, q);
    sycl::half *d_b = sycl::malloc_device<sycl::half>(size, q);
    sycl::half *d_c = sycl::malloc_device<sycl::half>(size, q);
    
    std::vector<sycl::half> h_a(size, sycl::half(1.0f));
    std::vector<sycl::half> h_b(size, sycl::half(2.0f));
    q.memcpy(d_a, h_a.data(), size * sizeof(sycl::half)).wait();
    q.memcpy(d_b, h_b.data(), size * sizeof(sycl::half)).wait();
    
    // Warmup
    for (int i = 0; i < 3; i++) {
      if (use_round1) {
        addVectors_round1(d_c, d_a, d_b, size, size, size, ACTIVATION_RELU, q);
      } else {
        addVectors_baseline(d_c, d_a, d_b, size, size, size, ACTIVATION_RELU, q);
      }
    }
    q.wait();
    
    // Benchmark
    std::vector<double> times;
    for (int i = 0; i < iterations; i++) {
      auto start = std::chrono::high_resolution_clock::now();
      
      if (use_round1) {
        addVectors_round1(d_c, d_a, d_b, size, size, size, ACTIVATION_RELU, q);
      } else {
        addVectors_baseline(d_c, d_a, d_b, size, size, size, ACTIVATION_RELU, q);
      }
      q.wait();
      
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> duration = end - start;
      times.push_back(duration.count());
    }
    
    double avg_time = 0;
    for (double t : times) avg_time += t;
    avg_time /= times.size();
    
    double flops = 2.0 * size;
    double gflops = flops / (avg_time * 1e-3) / 1e9;
    
    double bytes = 3.0 * size * sizeof(sycl::half);
    double bandwidth_gbps = bytes / (avg_time * 1e-3) / 1e9;
    
    results.push_back({size, avg_time, gflops, bandwidth_gbps});
    
    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);
  }
  
  return results;
}

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    
    std::vector<size_t> sizes = {64, 512, 1024, 4096, 16384, 65536};
    const int iterations = 10;
    
    auto baseline_results = run_tests(q, sizes, iterations, false);
    auto round1_results = run_tests(q, sizes, iterations, true);
    
    std::cout << "\n=== add_vectors - Round 1 Results ===" << std::endl;
    std::cout << std::setw(10) << "Size" 
              << std::setw(15) << "Baseline(ms)" 
              << std::setw(15) << "Round1(ms)"
              << std::setw(12) << "Speedup"
              << std::setw(15) << "GFLOPS"
              << std::setw(18) << "GB/s" << std::endl;
    std::cout << std::string(85, '-') << std::endl;
    
    for (size_t i = 0; i < sizes.size(); i++) {
      double speedup = baseline_results[i].avg_time_ms / round1_results[i].avg_time_ms;
      std::cout << std::setw(10) << sizes[i]
                << std::setw(15) << std::fixed << std::setprecision(3) << baseline_results[i].avg_time_ms
                << std::setw(15) << round1_results[i].avg_time_ms
                << std::setw(12) << std::setprecision(2) << speedup
                << std::setw(15) << std::setprecision(2) << round1_results[i].gflops
                << std::setw(18) << std::setprecision(2) << round1_results[i].bandwidth_gbps << std::endl;
    }
    
    return 0;
  } catch (sycl::exception const &e) {
    std::cerr << "SYCL Exception: " << e.what() << std::endl;
    return 1;
  }
}
