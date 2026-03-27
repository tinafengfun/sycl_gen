#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <iostream>
#include <vector>
#include <chrono>

// ============================================
// XMX Matrix Multiplication using joint_matrix
// Based on Intel oneAPI GPU Optimization Guide
// ============================================

using namespace sycl::ext::oneapi::experimental::matrix;

// XMX tile configuration for FP16
// Intel XMX supports 8x16x16 tiles for FP16/BF16
constexpr int TM = 8;   // M dimension per tile
constexpr int TN = 16;  // N dimension per tile  
constexpr int TK = 16;  // K dimension per tile

void gemm_xmx_joint_matrix(sycl::half* C, const sycl::half* A, const sycl::half* B,
                           int M, int N, int K, sycl::queue& queue) {
    
    // Each sub-group processes one tile
    sycl::range<2> global((M + TM - 1) / TM, (N + TN - 1) / TN);
    
    queue.parallel_for(
        sycl::nd_range<2>(global, sycl::range<2>(1, 1)),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
            sycl::sub_group sg = item.get_sub_group();
            
            const auto global_idx = item.get_global_id(0);
            const auto global_idy = item.get_global_id(1);
            
            // Cast pointers to multi_ptr for joint_matrix API
            auto pA = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::no>(const_cast<sycl::half*>(A));
            auto pB = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::no>(const_cast<sycl::half*>(B));
            auto pC = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::no>(C);
            
            // Declare joint_matrix tiles
            joint_matrix<sycl::sub_group, sycl::half, use::a, TM, TK, layout::row_major> sub_a;
            joint_matrix<sycl::sub_group, sycl::half, use::b, TK, TN, layout::row_major> sub_b;
            joint_matrix<sycl::sub_group, sycl::half, use::accumulator, TM, TN> sub_c;
            
            // Initialize accumulator
            joint_matrix_fill(sg, sub_c, sycl::half(0.0f));
            
            // Loop over K dimension in tiles
            for (int k = 0; k < K / TK; k++) {
                // Load A tile (TM x TK)
                joint_matrix_load(sg, sub_a, 
                    pA + (global_idx * TM) * K + k * TK, 
                    K);
                
                // Load B tile (TK x TN)
                joint_matrix_load(sg, sub_b, 
                    pB + (k * TK) * N + global_idy * TN, 
                    N);
                
                // DPAS: multiply-accumulate
                joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
            }
            
            // Store result
            joint_matrix_store(sg, sub_c, 
                pC + (global_idx * TM) * N + global_idy * TN, 
                N, layout::row_major);
        }
    );
    queue.wait_and_throw();
}

// ============================================
// Test function
// ============================================
void testGemm(sycl::half* d_C, sycl::half* d_A, sycl::half* d_B,
              int M, int N, int K, sycl::queue& queue) {
    // Warmup
    for (int i = 0; i < 10; i++) {
        gemm_xmx_joint_matrix(d_C, d_A, d_B, M, N, K, queue);
    }
    queue.wait_and_throw();
    
    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 50;
    
    for (int i = 0; i < iterations; i++) {
        gemm_xmx_joint_matrix(d_C, d_A, d_B, M, N, K, queue);
    }
    
    queue.wait_and_throw();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end - start;
    double time_ms = duration.count() / iterations;
    
    double ops = 2.0 * M * N * K;
    double tflops = ops / (time_ms * 1e-3) / 1e12;
    
    // B60: 160 EUs, XMX provides 2 TFLOPS/EU for FP16
    double theoretical_peak = 320.0;  // 320 TFLOPS
    double utilization = (tflops / theoretical_peak) * 100.0;
    
    std::cout << "XMX_joint_matrix,M=" << M << ",N=" << N << ",K=" << K
              << ",Time=" << time_ms << " ms"
              << ",TFLOPS=" << tflops 
              << ",Utilization=" << utilization << "%" << std::endl;
}

int main() {
    sycl::queue queue(sycl::gpu_selector_v);
    auto device = queue.get_device();
    
    std::cout << "========================================" << std::endl;
    std::cout << "XMX GEMM using joint_matrix (Intel Guide)" << std::endl;
    std::cout << "Device: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "CUs: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "XMX Tile: " << TM << "x" << TN << "x" << TK << std::endl;
    std::cout << "Target: 320 TFLOPS (160 EU x 2 TFLOPS/EU XMX)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Test sizes - must be multiples of tile dimensions
    std::vector<int> sizes = {2048, 4096, 8192};
    
    for (int size : sizes) {
        int M = size, N = size, K = size;
        
        size_t bytes = M * K * sizeof(sycl::half);
        
        sycl::half* d_A = sycl::malloc_device<sycl::half>(M * K, queue);
        sycl::half* d_B = sycl::malloc_device<sycl::half>(K * N, queue);
        sycl::half* d_C = sycl::malloc_device<sycl::half>(M * N, queue);
        
        std::vector<sycl::half> h_A(M * K, sycl::half(0.01f));
        std::vector<sycl::half> h_B(K * N, sycl::half(0.01f));
        queue.memcpy(d_A, h_A.data(), bytes).wait();
        queue.memcpy(d_B, h_B.data(), bytes).wait();
        
        try {
            testGemm(d_C, d_A, d_B, M, N, K, queue);
        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
        
        sycl::free(d_A, queue);
        sycl::free(d_B, queue);
        sycl::free(d_C, queue);
    }
    
    return 0;
}
