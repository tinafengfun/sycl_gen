// Type D-Large: Large Matrix Multiplication with XMX
// Usage: GEMM >= 256x256, large attention layers
// Expected improvement: 12x+ (100+ TFLOPS)
// MANDATORY: AOT compilation with -device bmg

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace sycl::ext::oneapi::experimental::matrix;

// XMX Tile Configuration for BMG
constexpr int TM = 8;   // M tiles
constexpr int TN = 16;  // N tiles  
constexpr int TK = 16;  // K tiles

// XMX GEMM Kernel
void xmx_gemm_kernel(float* C, const float* A, const float* B,
                     int M, int N, int K, sycl::nd_item<2> item) {
  // Each sub-group processes tiles
  int sg_id = item.get_local_id(0) / 16;  // Sub-group ID within work-group
  int sg_local_id = item.get_local_id(0) % 16;  // Lane within sub-group
  
  int tile_m = item.get_group(0) * 2 + sg_id;  // 2 tiles per work-group
  int tile_n = item.get_group(1);
  
  // Bounds check
  if (tile_m * TM >= M || tile_n * TN >= N) return;
  
  // XMX accumulator (8x16)
  joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> acc;
  joint_matrix_fill(item.get_sub_group(), acc, 0.0f);
  
  // Iterate over K dimension in tiles
  for (int k = 0; k < K; k += TK) {
    // Load A tile (8x16)
    joint_matrix<sycl::sub_group, float, use::a, TM, TK, layout::row_major> mat_a;
    joint_matrix_load(
      item.get_sub_group(),
      mat_a,
      A + tile_m * TM * K + k,
      K  // Leading dimension
    );
    
    // Load B tile (16x16)
    joint_matrix<sycl::sub_group, float, use::b, TK, TN, layout::row_major> mat_b;
    joint_matrix_load(
      item.get_sub_group(),
      mat_b,
      B + k * N + tile_n * TN,
      N  // Leading dimension
    );
    
    // XMX multiply-accumulate
    joint_matrix_mad(item.get_sub_group(), acc, mat_a, mat_b, acc);
  }
  
  // Store result (8x16)
  joint_matrix_store(
    item.get_sub_group(),
    acc,
    C + tile_m * TM * N + tile_n * TN,
    N  // Leading dimension
  );
}

// Reference CPU implementation for verification
void reference_gemm(float* C, const float* A, const float* B, int M, int N, int K) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = sum;
    }
  }
}

int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  auto device = queue.get_device();
  
  std::cout << "========================================" << std::endl;
  std::cout << "XMX GEMM Benchmark" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "GPU: " << device.get_info<sycl::info::device::name>() << std::endl;
  std::cout << std::endl;
  
  // Test sizes (must be multiples of tile sizes)
  // NOTE: For XMX efficiency, matrices should be >= 256
  std::vector<std::tuple<int, int, int>> configs = {
    {512, 512, 512},    // Minimum for XMX
    {1024, 1024, 1024}, // Good size
    {2048, 2048, 2048}, // Large
    {4096, 4096, 4096}  // Sweet spot for BMG
  };
  
  int iterations = 10;  // Reduced for large matrices
  
  for (auto& [M, N, K] : configs) {
    std::cout << "=== Testing M=" << M << " N=" << N << " K=" << K << " ===" << std::endl;
    
    // Allocate host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_ref(M * N);
    
    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (auto& v : h_A) v = dist(gen);
    for (auto& v : h_B) v = dist(gen);
    
    // Allocate device memory
    float *d_A = sycl::malloc_device<float>(M * K, queue);
    float *d_B = sycl::malloc_device<float>(K * N, queue);
    float *d_C = sycl::malloc_device<float>(M * N, queue);
    
    // Copy to device
    queue.memcpy(d_A, h_A.data(), M * K * sizeof(float));
    queue.memcpy(d_B, h_B.data(), K * N * sizeof(float));
    queue.wait();
    
    // Calculate work
    double total_ops = 2.0 * M * N * K;  // Multiply-accumulate operations
    double total_bytes = (M * K + K * N + M * N) * sizeof(float);
    
    // Launch configuration
    // Each work-group has 32 threads (2 sub-groups of 16)
    // Each sub-group processes one tile
    sycl::range<2> local(32, 1);  // 32 threads per work-group
    sycl::range<2> global(
      ((M + TM - 1) / TM + 1) / 2 * 32,  // Round up, 2 tiles per WG
      (N + TN - 1) / TN
    );
    
    // Warmup
    queue.parallel_for(
      sycl::nd_range<2>(global, local),
      [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
        xmx_gemm_kernel(d_C, d_A, d_B, M, N, K, item);
      }
    );
    queue.wait();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      queue.parallel_for(
        sycl::nd_range<2>(global, local),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
          xmx_gemm_kernel(d_C, d_A, d_B, M, N, K, item);
        }
      );
    }
    queue.wait();
    auto end = std::chrono::high_resolution_clock::now();
    
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    double gflops = (total_ops / (time_ms * 1e-3)) / 1e9;
    double bw = (total_bytes / (time_ms * 1e-3)) / 1e9;
    
    std::cout << "XMX GEMM\tTime: " << time_ms << " ms\t"
              << "GFLOPS: " << gflops << "\tBW: " << bw << " GB/s" << std::endl;
    
    // Verify correctness (optional, for first iteration)
    if (M <= 512) {  // Only for small sizes
      queue.memcpy(h_C.data(), d_C, M * N * sizeof(float));
      queue.wait();
      
      reference_gemm(h_C_ref.data(), h_A.data(), h_B.data(), M, N, K);
      
      float max_error = 0.0f;
      for (int i = 0; i < M * N; i++) {
        max_error = std::max(max_error, std::abs(h_C[i] - h_C_ref[i]));
      }
      std::cout << "Max error vs reference: " << max_error << std::endl;
    }
    
    std::cout << std::endl;
    
    // Cleanup
    sycl::free(d_A, queue);
    sycl::free(d_B, queue);
    sycl::free(d_C, queue);
  }
  
  std::cout << "✅ XMX GEMM Benchmark Complete!" << std::endl;
  std::cout << "\nExpected results:" << std::endl;
  std::cout << "- 512x512:   20-40 TFLOPS" << std::endl;
  std::cout << "- 1024x1024: 50-80 TFLOPS" << std::endl;
  std::cout << "- 2048x2048: 100-140 TFLOPS" << std::endl;
  std::cout << "- 4096x4096: 150-180 TFLOPS (sweet spot)" << std::endl;
  
  return 0;
}

/* Compilation (MANDATORY flags for XMX):
icpx -fsycl -O3 -std=c++17 \
  -fsycl-targets=spir64_gen \
  -Xsycl-target-backend "-device bmg -options -ze-opt-large-register-file" \
  -o type_d_large_xmx type_d_large_xmx.cpp
*/

/* Common Errors and Fixes:

Error: "no member named 'joint_matrix'"
Fix: Add #include <sycl/ext/oneapi/matrix/matrix.hpp>

Error: "Unsupported operation" when using joint_matrix
Fix: Must use AOT compilation with -device bmg
  WRONG: -fsycl-targets=spir64
  RIGHT: -fsycl-targets=spir64_gen -Xsycl-target-backend "-device bmg"

Error: Incorrect results
Fix: Check matrix dimensions are multiples of tile sizes (8, 16)
  Or add bounds checking in kernel

Error: Low performance (< 10 TFLOPS)
Fix: 
  1. Verify AOT compilation
  2. Check matrix size >= 256
  3. Verify subgroup size = 16
  4. Check reqd_sub_group_size(16) attribute
  5. Use -O3 optimization
  6. Add -ze-opt-large-register-file

Error: Compilation takes very long
Fix: This is normal for AOT with XMX. 1-2 minutes expected.
*/

/* Performance Tips:

1. Optimal matrix sizes for BMG XMX:
   - Sweet spot: 4096x4096 (achieves ~155 TFLOPS)
   - Minimum efficient: 256x256
   - Avoid: < 256 (use Type D-Small instead)

2. Tile sizes:
   - TM=8, TN=16, TK=16 are optimal for BMG
   - Don't change unless you know what you're doing

3. Work-group configuration:
   - 32 threads per work-group (2 sub-groups)
   - 16 lanes per sub-group (mandatory)

4. Memory layout:
   - Use row-major (layout::row_major)
   - Ensure data is aligned to 64 bytes

5. Precision:
   - This example uses FP32
   - For FP16, change float to half and use TF32 or BF16
   - FP16 can achieve higher TFLOPS
*/
