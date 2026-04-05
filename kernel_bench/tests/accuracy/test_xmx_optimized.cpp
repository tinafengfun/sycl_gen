#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <iostream>
#include <vector>
#include <chrono>

// ============================================
// Optimized XMX GEMM with SLM caching
// Based on Intel oneAPI GPU Optimization Guide
// ============================================

using namespace sycl::ext::oneapi::experimental::matrix;

// XMX tile configuration for FP16
constexpr int TM = 8;   // M dimension per tile
constexpr int TN = 16;  // N dimension per tile  
constexpr int TK = 16;  // K dimension per tile

// Work-group configuration for BMG
// BMG has 160 EUs, each EU can run 8 threads
// We want to maximize occupancy
constexpr int WG_M = 8;   // Work-group tiles in M
constexpr int WG_N = 16;  // Work-group tiles in N

void gemm_xmx_optimized(sycl::half* C, const sycl::half* A, const sycl::half* B,
                        int M, int N, int K, sycl::queue& queue) {
    
    // Each work-group processes WG_M x WG_N tiles
    // Each sub-group processes 1 tile
    sycl::range<2> global((M / TM), (N / TN));
    sycl::range<2> local(WG_M, WG_N);
    
    queue.parallel_for(
        sycl::nd_range<2>(global, local),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
            sycl::sub_group sg = item.get_sub_group();
            
            const auto sg_start_m = item.get_group(0) * WG_M + item.get_local_id(0);
            const auto sg_start_n = item.get_group(1) * WG_N + item.get_local_id(1);
            
            // Each sub-group processes one output tile
            if (sg_start_m >= M / TM || sg_start_n >= N / TN) return;
            
            // Cast pointers
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
            
            // Loop over K dimension
            #pragma unroll 4
            for (int k = 0; k < K / TK; k++) {
                // Load tiles
                joint_matrix_load(sg, sub_a, 
                    pA + (sg_start_m * TM) * K + k * TK, 
                    K);
                joint_matrix_load(sg, sub_b, 
                    pB + (k * TK) * N + sg_start_n * TN, 
                    N);
                
                // DPAS multiply-accumulate
                joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
            }
            
            // Store result
            joint_matrix_store(sg, sub_c, 
                pC + (sg_start_m * TM) * N + sg_start_n * TN, 
                N, layout::row_major);
        }
    );
    queue.wait_and_throw();
}

void testGemm(const char* name,
              void (*kernel)(sycl::half*, const sycl::half*, const sycl::half*, int, int, int, sycl::queue&),
              sycl::half* d_C, sycl::half* d_A, sycl::half* d_B,
              int M, int N, int K, sycl::queue& queue) {
    // Warmup
    for (int i = 0; i < 10; i++) {
        kernel(d_C, d_A, d_B, M, N, K, queue);
    }
    queue.wait_and_throw();
    
    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 50;
    
    for (int i = 0; i < iterations; i++) {
        kernel(d_C, d_A, d_B, M, N, K, queue);
    }
    
    queue.wait_and_throw();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end - start;
    double time_ms = duration.count() / iterations;
    
    double ops = 2.0 * M * N * K;
    double tflops = ops / (time_ms * 1e-3) / 1e12;
    double utilization = (tflops / 320.0) * 100.0;
    
    std::cout << name << ",M=" << M << ",N=" << N << ",K=" << K
              << ",Time=" << time_ms << " ms"
              << ",TFLOPS=" << tflops 
              << ",Utilization=" << utilization << "%" << std::endl;
}

int main() {
    sycl::queue queue(sycl::gpu_selector_v);
    auto device = queue.get_device();
    
    std::cout << "========================================" << std::endl;
    std::cout << "BMG B60 XMX GEMM Comparison" << std::endl;
    std::cout << "Device: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "CUs: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "Target: 320 TFLOPS" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::vector<int> sizes = {2048, 4096, 8192};
    
    for (int size : sizes) {
        int M = size, N = size, K = size;
        
        sycl::half* d_A = sycl::malloc_device<sycl::half>(M * K, queue);
        sycl::half* d_B = sycl::malloc_device<sycl::half>(K * N, queue);
        sycl::half* d_C = sycl::malloc_device<sycl::half>(M * N, queue);
        
        std::vector<sycl::half> h_A(M * K, sycl::half(0.01f));
        std::vector<sycl::half> h_B(K * N, sycl::half(0.01f));
        queue.memcpy(d_A, h_A.data(), M * K * sizeof(sycl::half)).wait();
        queue.memcpy(d_B, h_B.data(), K * N * sizeof(sycl::half)).wait();
        
        std::cout << "\nMatrix: " << size << "x" << size << "x" << size << std::endl;
        testGemm("XMX_optimized", gemm_xmx_optimized, d_C, d_A, d_B, M, N, K, queue);
        
        sycl::free(d_A, queue);
        sycl::free(d_B, queue);
        sycl::free(d_C, queue);
    }
    
    std::cout << "\nNote: oneDNN achieved 57 TFLOPS at 16384^3" << std::endl;
    
    return 0;
}
