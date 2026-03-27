// Type A: Element-wise Kernel Template
// Usage: Point-wise operations (add, multiply, bias_add, etc.)
// Expected improvement: <15%
// Rounds: 1 only (stop if minimal gain)

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

// Configuration
constexpr int WORK_GROUP_SIZE = 128;
constexpr int VECTOR_SIZE = 4;  // Process 4 elements per thread

// V1: Vectorized element-wise kernel
void elementwise_kernel_v1(float* output, const float* input_a, const float* input_b,
                           int N, sycl::nd_item<1> item) {
  int tid = item.get_global_id(0);
  int start_idx = tid * VECTOR_SIZE;
  
  if (start_idx < N) {
    // Load vectorized
    float4 vec_a, vec_b;
    #pragma unroll
    for (int i = 0; i < VECTOR_SIZE && (start_idx + i) < N; i++) {
      vec_a[i] = input_a[start_idx + i];
      vec_b[i] = input_b[start_idx + i];
    }
    
    // Element-wise operation (example: add)
    float4 vec_out;
    #pragma unroll
    for (int i = 0; i < VECTOR_SIZE; i++) {
      vec_out[i] = vec_a[i] + vec_b[i];  // Change operation here
    }
    
    // Store vectorized
    #pragma unroll
    for (int i = 0; i < VECTOR_SIZE && (start_idx + i) < N; i++) {
      output[start_idx + i] = vec_out[i];
    }
  }
}

// V2: Optimized with better memory access (if V1 insufficient)
void elementwise_kernel_v2(float* output, const float* input_a, const float* input_b,
                           int N, sycl::nd_item<1> item) {
  // Add SLM caching if memory bound
  // Usually not needed for element-wise (already memory bound)
  elementwise_kernel_v1(output, input_a, input_b, N, item);
}

int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  std::cout << "GPU: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  
  // Test sizes
  std::vector<int> sizes = {256, 1024, 4096, 16384};
  int iterations = 100;
  
  for (int N : sizes) {
    // Allocate and initialize
    std::vector<float> h_a(N), h_b(N), h_out(N);
    for (int i = 0; i < N; i++) {
      h_a[i] = static_cast<float>(i) * 0.01f;
      h_b[i] = static_cast<float>(i) * 0.02f;
    }
    
    float *d_a = sycl::malloc_device<float>(N, queue);
    float *d_b = sycl::malloc_device<float>(N, queue);
    float *d_out = sycl::malloc_device<float>(N, queue);
    
    queue.memcpy(d_a, h_a.data(), N * sizeof(float));
    queue.memcpy(d_b, h_b.data(), N * sizeof(float));
    queue.wait();
    
    // Calculate ops and bytes
    double total_ops = static_cast<double>(N);  // 1 op per element
    double total_bytes = 3.0 * N * sizeof(float);  // read a, read b, write out
    
    // Benchmark V1
    auto run_benchmark = [&](const char* name, auto kernel_func) {
      // Warmup
      for (int i = 0; i < 3; i++) kernel_func();
      queue.wait();
      
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < iterations; i++) kernel_func();
      queue.wait();
      auto end = std::chrono::high_resolution_clock::now();
      
      double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
      double gflops = (total_ops / (time_ms * 1e-3)) / 1e9;
      double bw = (total_bytes / (time_ms * 1e-3)) / 1e9;
      
      std::cout << name << "\tN=" << N << "\tTime: " << time_ms << " ms\t"
                << "GFLOPS: " << gflops << "\tBW: " << bw << " GB/s" << std::endl;
    };
    
    std::cout << "=== N=" << N << " ===" << std::endl;
    
    run_benchmark("V1", [&]() {
      queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>((N + VECTOR_SIZE - 1) / VECTOR_SIZE),
                          sycl::range<1>(WORK_GROUP_SIZE)),
        [=](sycl::nd_item<1> item) {
          elementwise_kernel_v1(d_out, d_a, d_b, N, item);
        }
      );
    });
    
    // Cleanup
    sycl::free(d_a, queue);
    sycl::free(d_b, queue);
    sycl::free(d_out, queue);
  }
  
  return 0;
}

/* Compilation:
icpx -fsycl -O3 -std=c++17 \
  -fsycl-targets=spir64_gen \
  -Xsycl-target-backend "-device bmg -options -ze-opt-large-register-file" \
  -o type_a_example type_a_elementwise.cpp
*/
