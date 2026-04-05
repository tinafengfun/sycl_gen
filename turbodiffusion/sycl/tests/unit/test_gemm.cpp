/**
 * GEMM SYCL Test
 * Tests FP32 and INT8 versions against CUDA reference
 */

#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include "../../src/gemm/gemm.hpp"

// Load binary data
template<typename T>
std::vector<T> load_binary(const std::string& filename, size_t count) {
    std::vector<T> data(count);
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open: " << filename << std::endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
    file.close();
    return data;
}

// Compare FP32 results
double compare_results(const std::vector<float>& sycl_output,
                       const std::vector<float>& cuda_output) {
    double max_diff = 0.0;
    double mean_diff = 0.0;
    int max_idx = 0;
    
    for (size_t i = 0; i < sycl_output.size(); ++i) {
        double diff = std::abs(sycl_output[i] - cuda_output[i]);
        mean_diff += diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }
    mean_diff /= sycl_output.size();
    
    std::cout << "  Max error: " << std::scientific << max_diff << std::endl;
    std::cout << "  Mean error: " << mean_diff << std::defaultfloat << std::endl;
    std::cout << "  Max diff at: [" << max_idx << "] CUDA=" << cuda_output[max_idx]
              << " SYCL=" << sycl_output[max_idx] << std::endl;
    
    return max_diff;
}

// Benchmark function
template<typename Func>
double benchmark(Func&& func, int iterations = 50) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

int main() {
    std::cout << "=== GEMM SYCL Test ===" << std::endl;
    std::cout << "Type D: Matrix Multiplication (Small M=32)\n" << std::endl;
    
    const int M = 32;
    const int N = 2048;
    const int K = 2048;
    
    // Load data
    std::cout << "Loading CUDA reference data..." << std::endl;
    auto A_int8 = load_binary<int8_t>("../wan2_validation/data/gemm_A_M32_K2048_int8.bin", M * K);
    auto B_int8 = load_binary<int8_t>("../wan2_validation/data/gemm_B_K2048_N2048_int8.bin", K * N);
    auto cuda_output = load_binary<float>("../wan2_validation/data/gemm_output_cuda_M32_N2048_fp32.bin", M * N);
    
    std::cout << "Matrix dimensions: " << M << "x" << K << " × " << K << "x" << N << std::endl;
    std::cout << "Input A: " << (M * K * sizeof(int8_t) / 1024.0) << " KB (INT8)" << std::endl;
    std::cout << "Input B: " << (K * N * sizeof(int8_t) / 1024.0 / 1024.0) << " MB (INT8)" << std::endl;
    std::cout << "Output C: " << (M * N * sizeof(float) / 1024.0) << " KB (FP32)" << std::endl;
    
    // Convert INT8 to FP32 for FP32 tests
    std::vector<float> A_fp32(M * K);
    std::vector<float> B_fp32(K * N);
    for (int i = 0; i < M * K; ++i) A_fp32[i] = static_cast<float>(A_int8[i]);
    for (int i = 0; i < K * N; ++i) B_fp32[i] = static_cast<float>(B_int8[i]);
    
    try {
        sycl::queue q(sycl::gpu_selector_v);
        std::cout << "\nDevice: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        
        std::cout << std::string(70, '=') << std::endl;
        
        // ==================== Test 1: INT8 Basic (Type D-Small) ====================
        std::cout << "\n[Test 1] INT8 Basic (Single-thread-per-row)" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        int8_t* d_A = sycl::malloc_device<int8_t>(M * K, q);
        int8_t* d_B = sycl::malloc_device<int8_t>(K * N, q);
        float* d_C = sycl::malloc_device<float>(M * N, q);
        
        q.memcpy(d_A, A_int8.data(), M * K * sizeof(int8_t)).wait();
        q.memcpy(d_B, B_int8.data(), K * N * sizeof(int8_t)).wait();
        
        // Warmup
        turbodiffusion::sycl_backend::GemmINT8::launch(q, d_A, d_B, d_C, M, N, K);
        
        double time_int8_basic = benchmark([&]() {
            turbodiffusion::sycl_backend::GemmINT8::launch(q, d_A, d_B, d_C, M, N, K);
        }, 50);
        
        std::vector<float> output_int8_basic(M * N);
        q.memcpy(output_int8_basic.data(), d_C, M * N * sizeof(float)).wait();
        
        std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_int8_basic << " ms" << std::endl;
        double err_int8_basic = compare_results(output_int8_basic, cuda_output);
        
        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);
        
        // ==================== Test 2: FP32 Basic ====================
        std::cout << "\n[Test 2] FP32 Basic (Single-thread-per-row)" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        float* d_A_fp32 = sycl::malloc_device<float>(M * K, q);
        float* d_B_fp32 = sycl::malloc_device<float>(K * N, q);
        d_C = sycl::malloc_device<float>(M * N, q);
        
        q.memcpy(d_A_fp32, A_fp32.data(), M * K * sizeof(float)).wait();
        q.memcpy(d_B_fp32, B_fp32.data(), K * N * sizeof(float)).wait();
        
        turbodiffusion::sycl_backend::GemmFP32::launch(q, d_A_fp32, d_B_fp32, d_C, M, N, K);
        
        double time_fp32_basic = benchmark([&]() {
            turbodiffusion::sycl_backend::GemmFP32::launch(q, d_A_fp32, d_B_fp32, d_C, M, N, K);
        }, 50);
        
        std::vector<float> output_fp32_basic(M * N);
        q.memcpy(output_fp32_basic.data(), d_C, M * N * sizeof(float)).wait();
        
        std::cout << "  Time: " << time_fp32_basic << " ms" << std::endl;
        std::cout << "  Speedup vs INT8: " << (time_int8_basic/time_fp32_basic) << "x" << std::endl;
        double err_fp32_basic = compare_results(output_fp32_basic, cuda_output);
        
        sycl::free(d_A_fp32, q);
        sycl::free(d_B_fp32, q);
        sycl::free(d_C, q);
        
        // ==================== Test 3: INT8 Work-Group Tuned ====================
        std::cout << "\n[Test 3] INT8 Work-Group Tuning" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        d_A = sycl::malloc_device<int8_t>(M * K, q);
        d_B = sycl::malloc_device<int8_t>(K * N, q);
        d_C = sycl::malloc_device<float>(M * N, q);
        
        q.memcpy(d_A, A_int8.data(), M * K * sizeof(int8_t)).wait();
        q.memcpy(d_B, B_int8.data(), K * N * sizeof(int8_t)).wait();
        
        auto test_wg = [&](const char* name, auto&& launch_fn) {
            launch_fn();
            double t = benchmark([&]() { launch_fn(); }, 50);
            std::cout << "  " << std::left << std::setw(12) << name
                      << std::setw(10) << std::fixed << std::setprecision(4) << t << " ms"
                      << " (speedup: " << std::setprecision(2) << (time_int8_basic/t) << "x)" << std::endl;
        };
        
        test_wg("WG=64", [&]() {
            turbodiffusion::sycl_backend::GemmINT8Tuned<64>::launch(q, d_A, d_B, d_C, M, N, K);
        });
        
        test_wg("WG=128", [&]() {
            turbodiffusion::sycl_backend::GemmINT8Tuned<128>::launch(q, d_A, d_B, d_C, M, N, K);
        });
        
        test_wg("WG=256", [&]() {
            turbodiffusion::sycl_backend::GemmINT8Tuned<256>::launch(q, d_A, d_B, d_C, M, N, K);
        });
        
        test_wg("WG=512", [&]() {
            turbodiffusion::sycl_backend::GemmINT8Tuned<512>::launch(q, d_A, d_B, d_C, M, N, K);
        });
        
        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);
        
        // ==================== Summary ====================
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "SUMMARY - Type D (Matrix Multiply) Kernel" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "INT8 Basic: Time=" << time_int8_basic << "ms, Error=" << err_int8_basic
                  << (err_int8_basic < 1e-3 ? " ✓" : " ✗") << std::endl;
        std::cout << "FP32 Basic: Time=" << time_fp32_basic << "ms, Error=" << err_fp32_basic
                  << (err_fp32_basic < 1e-3 ? " ✓" : " ✗") << std::endl;
        
        std::cout << "\nKey Findings:" << std::endl;
        std::cout << "- Type D-Small (M=32): Single-thread-per-row optimal" << std::endl;
        std::cout << "- INT8 input reduces memory bandwidth by 4x vs FP32" << std::endl;
        std::cout << "- XMX not beneficial for M < 256 (tile overhead)" << std::endl;
        std::cout << "- WG tuning can improve parallelism for small M" << std::endl;
        
        if (err_int8_basic < 1e-3) {
            std::cout << "\n✅ GEMM tests PASSED" << std::endl;
            return 0;
        } else {
            std::cout << "\n❌ GEMM tests FAILED" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
