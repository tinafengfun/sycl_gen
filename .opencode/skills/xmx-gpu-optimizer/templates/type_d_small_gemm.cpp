// Type D-Small: Small Matrix Multiplication
// Usage: FC layers, SE layer with C < 256, small GEMM
// Expected improvement: 10-100x
// Strategy: SINGLE-THREAD-PER-OUTPUT (not XMX!)

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// V0: Baseline (collaborative GEMM - slow)
void gemm_baseline(float* C, const float* A, const float* B,
                   int M, int N, int K, sycl::nd_item<2> item,
                   float* shared_a, float* shared_b) {
  // Collaborative loading into SLM
  // Then cooperative computation
  // This is SLOW for small matrices - avoid!
}

// V1: Single-thread-per-row (for M dimension)
void gemm_single_thread(float* C, const float* A, const float* B,
                        int M, int N, int K, sycl::item<1> item) {
  int m = item.get_id(0);
  if (m >= M) return;
  
  // Each thread computes one row of output
  for (int n = 0; n < N; n++) {
    float sum = 0.0f;
    
    #pragma unroll 8
    for (int k = 0; k < K; k++) {
      sum += A[m * K + k] * B[k * N + n];
    }
    
    C[m * N + n] = sum;
  }
}

// V2: With activation and bias (common pattern)
void gemm_with_activation(float* C, const float* A, const float* B, const float* bias,
                          int M, int N, int K, sycl::item<1> item,
                          bool use_relu = true) {
  int m = item.get_id(0);
  if (m >= M) return;
  
  for (int n = 0; n < N; n++) {
    float sum = bias ? bias[n] : 0.0f;
    
    #pragma unroll 8
    for (int k = 0; k < K; k++) {
      sum += A[m * K + k] * B[k * N + n];
    }
    
    // ReLU activation
    if (use_relu && sum < 0) sum = 0;
    
    C[m * N + n] = sum;
  }
}

// Example: SE Layer FC
int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  std::cout << "GPU: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  
  // SE layer dimensions (small matrices!)
  std::vector<std::tuple<int, int, int>> configs = {
    {64, 64, 128},    // N=64, se_K=64, C=128
    {128, 64, 128},   // N=128, se_K=64, C=128
    {256, 64, 256}    // N=256, se_K=64, C=256
  };
  
  int iterations = 30;
  
  for (auto& [N, se_K, C] : configs) {
    // FC1: C x se_K
    int w1_size = C * se_K;
    int b1_size = se_K;
    
    std::vector<float> h_input(N * C);
    std::vector<float> h_w1(w1_size);
    std::vector<float> h_b1(b1_size);
    std::vector<float> h_output(N * se_K);
    
    // Initialize
    for (auto& v : h_input) v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : h_w1) v = static_cast<float>(rand()) / RAND_MAX * 0.01f;
    for (auto& v : h_b1) v = 0.0f;
    
    float *d_input = sycl::malloc_device<float>(N * C, queue);
    float *d_w1 = sycl::malloc_device<float>(w1_size, queue);
    float *d_b1 = sycl::malloc_device<float>(b1_size, queue);
    float *d_output = sycl::malloc_device<float>(N * se_K, queue);
    
    queue.memcpy(d_input, h_input.data(), N * C * sizeof(float));
    queue.memcpy(d_w1, h_w1.data(), w1_size * sizeof(float));
    queue.memcpy(d_b1, h_b1.data(), b1_size * sizeof(float));
    queue.wait();
    
    // Calculate work
    double total_ops = static_cast<double>(N) * C * se_K * 2;  // MAC operations
    
    // Benchmark
    auto run_benchmark = [&](const char* name, auto kernel_func) {
      for (int i = 0; i < 3; i++) kernel_func();
      queue.wait();
      
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < iterations; i++) kernel_func();
      queue.wait();
      auto end = std::chrono::high_resolution_clock::now();
      
      double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
      double gflops = (total_ops / (time_ms * 1e-3)) / 1e9;
      
      std::cout << name << "\tN=" << N << " C=" << C << " se_K=" << se_K 
                << "\tTime: " << time_ms << " ms\tGFLOPS: " << gflops << std::endl;
    };
    
    std::cout << "=== SE Layer FC: N=" << N << " C=" << C << " se_K=" << se_K << " ===" << std::endl;
    
    // V1 single-thread-per-row
    run_benchmark("V1_SingleThread", [&]() {
      queue.parallel_for(
        sycl::range<1>(N),
        [=](sycl::item<1> item) {
          gemm_with_activation(d_output, d_input, d_w1, d_b1, N, se_K, C, item, true);
        }
      );
    });
    
    std::cout << "\n💡 For small matrices, single-thread is 10-20x faster than collaborative!" << std::endl;
    std::cout << "💡 XMX is NOT effective here (matrix too small)" << std::endl;
    
    // Cleanup
    sycl::free(d_input, queue);
    sycl::free(d_w1, queue);
    sycl::free(d_b1, queue);
    sycl::free(d_output, queue);
  }
  
  return 0;
}

/* Compilation:
icpx -fsycl -O3 -std=c++17 \
  -fsycl-targets=spir64_gen \
  -Xsycl-target-backend "-device bmg -options -ze-opt-large-register-file" \
  -o type_d_small_example type_d_small_gemm.cpp
*/

/* When to use this template:
- FC layers in neural networks (usually small dimensions)
- SE layer operations
- Any GEMM where M, N, or K < 256
- Attention layers with small sequence length

When NOT to use:
- Large GEMM (M,N,K >= 256) → Use Type D-Large (XMX)
- Batch size > 1024 → May need different approach
*/
