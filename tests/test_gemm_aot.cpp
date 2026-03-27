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

void testGemm(sycl::half* d_C, sycl::half* d_A, sycl::half* d_B,
              int M, int N, int K, sycl::queue& queue) {
    // Warmup
    for (int i = 0; i < 10; i++) {
        xmx::gemm_onednn(d_C, d_A, d_B, M, N, K, queue);
    }
    queue.wait_and_throw();
    
    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 50;
    
    for (int i = 0; i < iterations; i++) {
        xmx::gemm_onednn(d_C, d_A, d_B, M, N, K, queue);
    }
    
    queue.wait_and_throw();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> duration = end - start;
    double timePerKernel = duration.count() / iterations;
    
    double ops = 2.0 * M * N * K;
    double gflops = ops / (timePerKernel * 1e-3) / 1e9;
    double tflops = gflops / 1000.0;
    
    // B60: 160 EUs * 2 TFLOPS/EU = 320 TFLOPS theoretical
    double theoretical_peak = 320000.0;
    double utilization = (gflops / theoretical_peak) * 100.0;
    
    std::cout << "M=" << M << ",N=" << N << ",K=" << K
              << ",Time=" << timePerKernel << " ms"
              << ",TFLOPS=" << tflops 
              << ",Utilization=" << utilization << "%" << std::endl;
}

int main() {
    sycl::queue queue(sycl::gpu_selector_v);
    auto device = queue.get_device();
    
    std::cout << "========================================" << std::endl;
    std::cout << "BMG B60 XMX GEMM (AOT Compiled)" << std::endl;
    std::cout << "Device: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "CUs: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "Theoretical Peak: 320 TFLOPS (160 EU x 2 TFLOPS/EU)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Test 16384 first
    std::vector<int> sizes = {8192, 16384};
    
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
        
        std::cout << "Testing " << size << "x" << size << "x" << size << ":" << std::endl;
        testGemm(d_C, d_A, d_B, M, N, K, queue);
        std::cout << std::endl;
        
        sycl::free(d_A, queue);
        sycl::free(d_B, queue);
        sycl::free(d_C, queue);
    }
    
    return 0;
}
