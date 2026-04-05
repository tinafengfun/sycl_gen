#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// ============================================
// XMX-Optimized GEMM using oneDNN
// oneDNN automatically uses XMX/DPAS on Intel GPUs
// ============================================

namespace xmx {

// ============================================
// GEMM using oneDNN matmul primitive
// This is the recommended way to get peak XMX performance
// ============================================
void gemm_onednn(sycl::half* C, const sycl::half* A, const sycl::half* B,
                 int M, int N, int K, sycl::queue& queue) {
    // Create oneDNN engine
    dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    dnnl::stream stream(engine);
    
    // Memory descriptors for A, B, C using dnnl::dim type
    std::vector<dnnl::dim> a_dims = {M, K};
    std::vector<dnnl::dim> b_dims = {K, N};
    std::vector<dnnl::dim> c_dims = {M, N};
    
    dnnl::memory::desc a_desc(a_dims, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab);
    dnnl::memory::desc b_desc(b_dims, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab);
    dnnl::memory::desc c_desc(c_dims, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab);
    
    // Create memory objects
    dnnl::memory a_mem(a_desc, engine, const_cast<sycl::half*>(A));
    dnnl::memory b_mem(b_desc, engine, const_cast<sycl::half*>(B));
    dnnl::memory c_mem(c_desc, engine, C);
    
    // Create matmul primitive
    dnnl::matmul::primitive_desc matmul_pd(engine, a_desc, b_desc, c_desc);
    dnnl::matmul matmul(matmul_pd);
    
    matmul.execute(stream, {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem}
    });
    
    stream.wait();
}

// ============================================
// Fallback GEMM (manual implementation for comparison)
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
    std::cout << "XMX GEMM Test using oneDNN" << std::endl;
    std::cout << "Device: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Compute Units: " 
              << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "Target: 160 TFLOPS (B60 FP16)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Version,M,N,K,Time_ms,GFLOPS,Bandwidth_GB/s,Utilization_%" << std::endl;
    
    // Test sizes - including 16384 for large matrix test
    std::vector<int> testSizes = {2048, 4096, 8192, 16384};
    
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
        
        // Test fallback
        testGemm("Fallback", xmx::gemm_fallback, d_C, d_A, d_B, M, N, K, queue);
        
        // Test oneDNN
        try {
            testGemm("oneDNN", xmx::gemm_onednn, d_C, d_A, d_B, M, N, K, queue);
        } catch (const std::exception& e) {
            std::cout << "oneDNN Error: " << e.what() << std::endl;
        }
        
        sycl::free(d_A, queue);
        sycl::free(d_B, queue);
        sycl::free(d_C, queue);
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "Expected: 80-160 TFLOPS with oneDNN XMX" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
