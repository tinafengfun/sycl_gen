#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

namespace xmx {

void gemm_onednn(sycl::half* C, const sycl::half* A, const sycl::half* B,
                 int M, int N, int K, sycl::queue& queue) {
    dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    dnnl::stream stream(engine);
    
    std::vector<dnnl::memory::dim> a_dims = {M, K};
    std::vector<dnnl::memory::dim> b_dims = {K, N};
    std::vector<dnnl::memory::dim> c_dims = {M, N};
    
    dnnl::memory::desc a_desc(a_dims, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab);
    dnnl::memory::desc b_desc(b_dims, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab);
    dnnl::memory::desc c_desc(c_dims, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab);
    
    dnnl::memory a_mem(a_desc, engine, const_cast<sycl::half*>(A));
    dnnl::memory b_mem(b_desc, engine, const_cast<sycl::half*>(B));
    dnnl::memory c_mem(c_desc, engine, C);
    
    dnnl::matmul::primitive_desc matmul_pd(engine, a_desc, b_desc, c_desc);
    dnnl::matmul matmul(matmul_pd);
    
    matmul.execute(stream, {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem}
    });
    
    stream.wait();
}

}  // namespace xmx

int main() {
    sycl::queue queue(sycl::gpu_selector_v);
    auto device = queue.get_device();
    
    std::cout << "========================================" << std::endl;
    std::cout << "BMG B60 XMX GEMM - Large Matrix Test" << std::endl;
    std::cout << "Compile Options: -ze-opt-large-register-file" << std::endl;
    std::cout << "Device: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "CUs: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Test different sizes
    std::vector<int> sizes = {8192, 12288, 16384};
    
    for (int size : sizes) {
        int M = size, N = size, K = size;
        
        std::cout << "\nTesting GEMM " << size << "x" << size << "x" << size << " (FP16):" << std::endl;
        
        size_t size_A = M * K * sizeof(sycl::half);
        size_t size_B = K * N * sizeof(sycl::half);
        size_t size_C = M * N * sizeof(sycl::half);
        
        std::cout << "  Memory: A=" << (size_A/1e9) << "GB, B=" << (size_B/1e9) << "GB, C=" << (size_C/1e9) << "GB" << std::endl;
        
        sycl::half* d_A = sycl::malloc_device<sycl::half>(M * K, queue);
        sycl::half* d_B = sycl::malloc_device<sycl::half>(K * N, queue);
        sycl::half* d_C = sycl::malloc_device<sycl::half>(M * N, queue);
        
        // Initialize
        std::vector<sycl::half> h_A(M * K, sycl::half(0.01f));
        std::vector<sycl::half> h_B(K * N, sycl::half(0.01f));
        queue.memcpy(d_A, h_A.data(), size_A).wait();
        queue.memcpy(d_B, h_B.data(), size_B).wait();
        
        // Warmup
        for (int i = 0; i < 5; i++) {
            xmx::gemm_onednn(d_C, d_A, d_B, M, N, K, queue);
        }
        queue.wait_and_throw();
        
        // Timing
        auto start = std::chrono::high_resolution_clock::now();
        const int iterations = 20;
        
        for (int i = 0; i < iterations; i++) {
            xmx::gemm_onednn(d_C, d_A, d_B, M, N, K, queue);
        }
        
        queue.wait_and_throw();
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> duration = end - start;
        double time_ms = duration.count() / iterations;
        
        double ops = 2.0 * M * N * K;
        double tflops = ops / (time_ms * 1e-3) / 1e12;
        
        // Memory bandwidth
        double bytes = (M * K + K * N + M * N) * sizeof(sycl::half);
        double bandwidth_gbps = bytes / (time_ms * 1e-3) / 1e9;
        
        // Calculate efficiency
        double theoretical_peak = 320.0;  // 320 TFLOPS
        double efficiency = (tflops / theoretical_peak) * 100.0;
        
        std::cout << "  Time: " << time_ms << " ms" << std::endl;
        std::cout << "  Performance: " << tflops << " TFLOPS" << std::endl;
        std::cout << "  Bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;
        std::cout << "  Efficiency: " << efficiency << "% (vs 320 TFLOPS peak)" << std::endl;
        
        sycl::free(d_A, queue);
        sycl::free(d_B, queue);
        sycl::free(d_C, queue);
    }
    
    std::cout << "\n========================================" << std::endl;
    
    return 0;
}
