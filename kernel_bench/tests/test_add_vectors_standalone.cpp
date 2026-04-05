#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>

using namespace std;

// Standalone test for add_vectors performance
// This test calls the kernel through queue submission

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
    case ACTIVATION_SIGMOID:
      cVal = 1.0f / (1.0f + sycl::native::exp(-cVal));
      break;
    default:
      break;
  }
  return cVal;
}

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void addVectors_kernel(T* c, T* a, T* b, int size, int asize,
                       int bsize, ActivationFunction activation,
                       const sycl::nd_item<3> &item_ct1) {
  int i = item_ct1.get_local_id(2) +
          item_ct1.get_local_range(2) * item_ct1.get_group(2);
  if (i < size) {
    float aVal = 0;
    float bVal = 0;
    if (a) aVal = (float)(a[i % asize]);
    if (b) bVal = (float)(b[i % bsize]);
    float cVal = aVal + bVal;
    cVal = activate(cVal, activation);
    c[i] = (T)cVal;
  }
}

template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize,
                ActivationFunction activation, sycl::queue &sycl_queue) {
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  sycl_queue.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
          sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item_ct1) {
        addVectors_kernel(c, a, b, size, asize, bsize, activation, item_ct1);
      });
}

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    std::cout << "=== add_vectors - Round 1 (Real Kernel) ===" << std::endl;
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl << std::endl;
    
    std::vector<size_t> sizes = {4096, 32768, 262144, 1048576};
    
    std::cout << std::setw(12) << "Size" 
              << std::setw(15) << "Time(ms)" 
              << std::setw(15) << "GFLOPS"
              << std::setw(18) << "GB/s" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (size_t size : sizes) {
      // Allocate device memory
      sycl::half *d_a = sycl::malloc_device<sycl::half>(size, q);
      sycl::half *d_b = sycl::malloc_device<sycl::half>(size, q);
      sycl::half *d_c = sycl::malloc_device<sycl::half>(size, q);
      
      // Initialize
      std::vector<sycl::half> h_data(size, sycl::half(0.5f));
      q.memcpy(d_a, h_data.data(), size * sizeof(sycl::half)).wait();
      q.memcpy(d_b, h_data.data(), size * sizeof(sycl::half)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        addVectors(d_c, d_a, d_b, size, size, size, ACTIVATION_RELU, q);
      }
      q.wait();
      
      // Benchmark
      std::vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        addVectors(d_c, d_a, d_b, size, size, size, ACTIVATION_RELU, q);
        q.wait();
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        times.push_back(duration.count());
      }
      
      // Stats
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      // Metrics: 1 add + 1 ReLU = 2 FLOPs per element
      double flops = 2.0 * size;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      
      // Memory: read a + read b + write c = 3 * size * 2 bytes
      double bytes = 3.0 * size * sizeof(sycl::half);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      std::cout << std::setw(12) << size
                << std::setw(15) << std::fixed << std::setprecision(3) << avg_time
                << std::setw(15) << std::setprecision(2) << gflops
                << std::setw(18) << std::setprecision(2) << bandwidth << std::endl;
      
      sycl::free(d_a, q);
      sycl::free(d_b, q);
      sycl::free(d_c, q);
    }
    
    std::cout << std::endl << "Test completed successfully!" << std::endl;
    return 0;
  } catch (sycl::exception const &e) {
    std::cerr << "SYCL Exception: " << e.what() << std::endl;
    return 1;
  }
}
