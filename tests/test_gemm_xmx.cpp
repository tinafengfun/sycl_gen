#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// ============================================
// XMX-Optimized GEMM Kernel
// Based on Intel XMX/DPAS specifications from XMX.md
// Tile size: 8x16x16 (M x N x K)
// Subgroup size: 16
// ============================================

namespace xmx {

using namespace sycl::ext::oneapi::experimental::matrix;

// XMX tile dimensions for FP16
constexpr int M_TILE = 8;   // M dimension per DPAS tile
constexpr int N_TILE = 16;  // N dimension per DPAS tile
constexpr int K_TILE = 16;  // K dimension per DPAS tile

// ============================================
// Real XMX GEMM using joint_matrix API
// This maps directly to DPAS instructions
// ============================================
void gemm_xmx_real(sycl::half* C, const sycl::half* A, const sycl::half* B,
                   int M, int N, int K, sycl::queue& queue) {
    
    // Each work-group processes one 8x16 output tile
    // Using 16-wide subgroups as required by XMX
    sycl::range<2> wg(M_TILE, 1);
    sycl::range<2> global(
        ((M + M_TILE - 1) / M_TILE) * M_TILE,
        ((N + N_TILE - 1) / N_TILE) * N_TILE
    );
    
    queue.parallel_for(
        sycl::nd_range<2>(global, wg),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
            int m = item.get_global_id(0);
            int n = item.get_global_id(1);
            
            if (m >= M || n >= N) return;
            
            // Get sub-group for XMX operations
            sycl::sub_group sg = item.get_sub_group();
            
            // XMX joint_matrix for accumulator (output)
            // Template: joint_matrix<Group, T, Use, Rows, Cols>
            joint_matrix<sycl::sub_group, sycl::half, use::accumulator, M_TILE, N_TILE> acc;
            
            // Initialize to zero
            joint_matrix_fill(sg, acc, sycl::half(0.0f));
            
            // Tile over K dimension in 16-element chunks
            for (int k = 0; k < K; k += K_TILE) {
                // XMX joint_matrix for A tile (8x16)
                joint_matrix<sycl::sub_group, sycl::half, use::a, M_TILE, K_TILE> mat_a;
                // XMX joint_matrix for B tile (16x16)
                joint_matrix<sycl::sub_group, sycl::half, use::b, K_TILE, N_TILE> mat_b;
                
                // Load A tile from global memory using multi_ptr
                auto ptr_a = sycl::address_space_cast<
                    sycl::access::address_space::global_space,
                    sycl::access::decorated::no>(const_cast<sycl::half*>(A + m * K + k));
                joint_matrix_load(
                    sg, mat_a,
                    ptr_a,
                    K
                );
                
                // Load B tile from global memory using multi_ptr
                auto ptr_b = sycl::address_space_cast<
                    sycl::access::address_space::global_space,
                    sycl::access::decorated::no>(const_cast<sycl::half*>(B + k * N + n));
                joint_matrix_load(
                    sg, mat_b,
                    ptr_b,
                    N
                );
                
                // XMX DPAS: multiply-accumulate
                joint_matrix_mad(sg, acc, mat_a, mat_b, acc);
            }
            
            // Store result to global memory using multi_ptr
            auto ptr_c = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::no>(C + m * N + n);
            joint_matrix_store(
                sg, acc,
                ptr_c,
                N,
                sycl::ext::oneapi::experimental::matrix::layout::row_major
            );
        }
    );
    queue.wait_and_throw();
}

// ============================================
// Fallback GEMM (non-XMX for comparison)
// ============================================
void gemm_fallback(sycl::half* C, const sycl::half* A, const sycl::half* B,
                   int M, int N, int K, sycl::queue& queue) {
    sycl::range<2> wg(8, 16);
    sycl::range<2> global((M + 7) / 8 * 8, (N + 15) / 16 * 16);
    
    queue.parallel_for(
        sycl::nd_range<2>(global, wg),
        [=](sycl::nd_item<2> item) {
            int m = item.get_global_id(0);
            int n = item.get_global_id(1);
            
            if (m >= M || n >= N) return;
            
            float acc = 0.0f;
            #pragma unroll 8
            for (int k = 0; k < K; k++) {
                acc += static_cast<float>(A[m * K + k]) * 
                       static_cast<float>(B[k * N + n]);
            }
            C[m * N + n] = static_cast<sycl::half>(acc);
        }
    );
    queue.wait_and_throw();
}

}  // namespace xmx

// ============================================
// Test function with extended warmup
// ============================================
void testGemm(const char* name,
              void (*kernel)(sycl::half*, const sycl::half*, const sycl::half*, 
                            int, int, int, sycl::queue&),
              sycl::half* d_C, sycl::half* d_A, sycl::half* d_B,
              int M, int N, int K, sycl::queue& queue) {
    // Extended warmup - 100 iterations
    for (int i = 0; i < 100; i++) {
        kernel(d_C, d_A, d_B, M, N, K, queue);
    }
    queue.wait_and_throw();
    
    // Timing run
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 100;
    
    for (int i = 0; i < iterations; i++) {
        kernel(d_C, d_A, d_B, M, N, K, queue);
    }
    
    queue.wait_and_throw();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end - start;
    double timePerKernel = duration.count() / iterations;
    
    double ops = 2.0 * M * N * K;
    double gflops = ops / (timePerKernel * 1e-3) / 1e9;
    double bandwidth = ((M * K + K * N + M * N) * sizeof(sycl::half)) / 
                       (timePerKernel * 1e-3) / 1e9;
    double theoretical_peak = 160000.0;  // 160 TFLOPS
    double utilization = (gflops / theoretical_peak) * 100.0;
    
    std::cout << name << ",M=" << M << ",N=" << N << ",K=" << K
              << ",Time=" << timePerKernel << " ms"
              << ",GFLOPS=" << gflops 
              << ",Bandwidth=" << bandwidth << " GB/s"
              << ",Utilization=" << utilization << "%" << std::endl;
}

int main() {
    sycl::queue queue(sycl::gpu_selector_v);
    auto device = queue.get_device();
    
    std::cout << "========================================" << std::endl;
    std::cout << "XMX GEMM Test (Real DPAS Instructions)" << std::endl;
    std::cout << "Device: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Compute Units: " 
              << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "Target: 160 TFLOPS (B60 FP16)" << std::endl;
    std::cout << "XMX Tile: 8x16x16" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Check for matrix extension support
    bool has_matrix = device.has(sycl::aspect::ext_intel_matrix);
    std::cout << "Matrix Extension Support: " << (has_matrix ? "YES" : "NO") << std::endl;
    std::cout << std::endl;
    
    std::cout << "Version,M,N,K,Time_ms,GFLOPS,Bandwidth_GB/s,Utilization_%" << std::endl;
    
    // Test sizes - must be multiples of tile dimensions
    std::vector<int> testSizes = {2048, 4096};
    
    for (int size : testSizes) {
        int M = size, N = size, K = size;
        
        size_t size_A = M * K * sizeof(sycl::half);
        size_t size_B = K * N * sizeof(sycl::half);
        size_t size_C = M * N * sizeof(sycl::half);
        
        sycl::half* d_A = sycl::malloc_device<sycl::half>(M * K, queue);
        sycl::half* d_B = sycl::malloc_device<sycl::half>(K * N, queue);
        sycl::half* d_C = sycl::malloc_device<sycl::half>(M * N, queue);
        
        // Initialize
        std::vector<sycl::half> h_A(M * K);
        std::vector<sycl::half> h_B(K * N);
        for (int i = 0; i < M * K; i++) h_A[i] = sycl::half(0.01f);
        for (int i = 0; i < K * N; i++) h_B[i] = sycl::half(0.01f);
        
        queue.memcpy(d_A, h_A.data(), size_A).wait();
        queue.memcpy(d_B, h_B.data(), size_B).wait();
        
        // Test fallback first
        testGemm("Fallback", xmx::gemm_fallback, d_C, d_A, d_B, M, N, K, queue);
        
        // Test XMX if supported
        if (has_matrix) {
            testGemm("XMX_Real", xmx::gemm_xmx_real, d_C, d_A, d_B, M, N, K, queue);
        }
        
        sycl::free(d_A, queue);
        sycl::free(d_B, queue);
        sycl::free(d_C, queue);
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "Expected: 80-160 TFLOPS with XMX DPAS" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
