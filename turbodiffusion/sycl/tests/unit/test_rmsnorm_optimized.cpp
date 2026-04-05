/**
 * RMSNorm SYCL Optimized Unit Test
 * Tests FP32 and BF16 versions against CUDA reference
 */

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <chrono>
#include "../../src/norm/rmsnorm_optimized.hpp"

// Load binary data
std::vector<float> load_binary(const std::string& filename, size_t count) {
    std::vector<float> data(count);
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open: " << filename << std::endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));
    file.close();
    return data;
}

// Compare results and return max error
double compare_results(const std::vector<float>& sycl_output, 
                       const std::vector<float>& cuda_output,
                       const std::string& label) {
    double max_diff = 0.0;
    double mean_diff = 0.0;
    int max_diff_idx = 0;
    
    for (size_t i = 0; i < sycl_output.size(); ++i) {
        double diff = std::abs(sycl_output[i] - cuda_output[i]);
        mean_diff += diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    mean_diff /= sycl_output.size();
    
    std::cout << "\n" << label << ":" << std::endl;
    std::cout << "  Max absolute error: " << max_diff << std::endl;
    std::cout << "  Mean absolute error: " << mean_diff << std::endl;
    std::cout << "  Max diff at index: " << max_diff_idx << std::endl;
    std::cout << "    CUDA value: " << cuda_output[max_diff_idx] << std::endl;
    std::cout << "    SYCL value: " << sycl_output[max_diff_idx] << std::endl;
    
    return max_diff;
}

// Verify RMS is approximately 1.0
void verify_rms(const std::vector<float>& output, int M, int N, const std::string& label) {
    std::cout << "\n" << label << " RMS Verification:" << std::endl;
    for (int row = 0; row < std::min(3, M); ++row) {
        double sum_sq = 0;
        for (int i = 0; i < N; ++i) {
            sum_sq += output[row * N + i] * output[row * N + i];
        }
        float rms = std::sqrt(sum_sq / N);
        std::cout << "  Row " << row << " RMS: " << rms;
        if (std::abs(rms - 1.0f) < 0.01f) {
            std::cout << " ✓" << std::endl;
        } else {
            std::cout << " ✗" << std::endl;
        }
    }
}

// Benchmark function
template<typename Func>
double benchmark(Func&& func, int iterations = 100) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

int main() {
    std::cout << "=== RMSNorm SYCL Optimized Unit Test ===" << std::endl;
    
    const int M = 32;
    const int N = 2048;
    const float EPS = 1e-6;
    
    std::cout << "\nLoading CUDA reference data..." << std::endl;
    
    // Load input data
    auto input = load_binary("../wan2_validation/data/rmsnorm_input_M32_N2048_fp32.bin", M * N);
    auto weight = load_binary("../wan2_validation/data/rmsnorm_weight_N2048_fp32.bin", N);
    auto cuda_output = load_binary("../wan2_validation/data/rmsnorm_output_cuda_M32_N2048_fp32.bin", M * N);
    
    std::cout << "Input: " << input.size() << " elements" << std::endl;
    std::cout << "Weight: " << weight.size() << " elements" << std::endl;
    std::cout << "CUDA output: " << cuda_output.size() << " elements" << std::endl;
    
    try {
        // Create SYCL queue
        sycl::queue q(sycl::gpu_selector_v);
        std::cout << "\nDevice: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        
        // ==================== Test 1: FP32 Optimized ====================
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Test 1: FP32 Optimized (Type C Pattern)" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // Allocate device memory
        float* d_input = sycl::malloc_device<float>(M * N, q);
        float* d_weight = sycl::malloc_device<float>(N, q);
        float* d_output_fp32 = sycl::malloc_device<float>(M * N, q);
        
        // Copy input data to device
        q.memcpy(d_input, input.data(), M * N * sizeof(float)).wait();
        q.memcpy(d_weight, weight.data(), N * sizeof(float)).wait();
        
        std::cout << "\nRunning SYCL RMSNorm FP32 Optimized..." << std::endl;
        
        // Warmup
        turbodiffusion::sycl_backend::RMSNormFP32::launch(
            q, d_input, d_weight, d_output_fp32, EPS, M, N
        );
        
        // Benchmark
        double time_fp32 = benchmark([&]() {
            turbodiffusion::sycl_backend::RMSNormFP32::launch(
                q, d_input, d_weight, d_output_fp32, EPS, M, N
            );
        }, 100);
        
        std::cout << "  Average time: " << time_fp32 << " ms" << std::endl;
        
        // Copy results back
        std::vector<float> sycl_output_fp32(M * N);
        q.memcpy(sycl_output_fp32.data(), d_output_fp32, M * N * sizeof(float)).wait();
        
        // Compare
        double max_err_fp32 = compare_results(sycl_output_fp32, cuda_output, "FP32 Optimized");
        verify_rms(sycl_output_fp32, M, N, "FP32");
        
        // ==================== Test 2: BF16 ====================
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Test 2: BF16 (2x Memory Bandwidth)" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        using bfloat16 = sycl::ext::oneapi::bfloat16;
        
        // Convert to BF16
        std::vector<bfloat16> input_bf16(M * N);
        std::vector<bfloat16> weight_bf16(N);
        for (int i = 0; i < M * N; ++i) input_bf16[i] = bfloat16(input[i]);
        for (int i = 0; i < N; ++i) weight_bf16[i] = bfloat16(weight[i]);
        
        // Allocate BF16 device memory
        bfloat16* d_input_bf16 = sycl::malloc_device<bfloat16>(M * N, q);
        bfloat16* d_weight_bf16 = sycl::malloc_device<bfloat16>(N, q);
        bfloat16* d_output_bf16 = sycl::malloc_device<bfloat16>(M * N, q);
        
        q.memcpy(d_input_bf16, input_bf16.data(), M * N * sizeof(bfloat16)).wait();
        q.memcpy(d_weight_bf16, weight_bf16.data(), N * sizeof(bfloat16)).wait();
        
        std::cout << "\nRunning SYCL RMSNorm BF16..." << std::endl;
        
        // Warmup
        turbodiffusion::sycl_backend::RMSNormBF16::launch(
            q, d_input_bf16, d_weight_bf16, d_output_bf16, EPS, M, N
        );
        
        // Benchmark
        double time_bf16 = benchmark([&]() {
            turbodiffusion::sycl_backend::RMSNormBF16::launch(
                q, d_input_bf16, d_weight_bf16, d_output_bf16, EPS, M, N
            );
        }, 100);
        
        std::cout << "  Average time: " << time_bf16 << " ms" << std::endl;
        
        // Copy results back and convert to FP32 for comparison
        std::vector<bfloat16> sycl_output_bf16(M * N);
        q.memcpy(sycl_output_bf16.data(), d_output_bf16, M * N * sizeof(bfloat16)).wait();
        
        std::vector<float> sycl_output_bf16_fp32(M * N);
        for (int i = 0; i < M * N; ++i) {
            sycl_output_bf16_fp32[i] = float(sycl_output_bf16[i]);
        }
        
        // Compare
        double max_err_bf16 = compare_results(sycl_output_bf16_fp32, cuda_output, "BF16");
        verify_rms(sycl_output_bf16_fp32, M, N, "BF16");
        
        // ==================== Test 3: Work-group Tuning ====================
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Test 3: Work-group Size Tuning (Traditional Mode)" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        float* d_output_tuned = sycl::malloc_device<float>(M * N, q);
        
        // Test WG=128
        std::cout << "\nTesting WG_SIZE=128..." << std::endl;
        turbodiffusion::sycl_backend::RMSNormTuned<128>::launch(
            q, d_input, d_weight, d_output_tuned, EPS, M, N
        );
        
        std::vector<float> sycl_output_wg128(M * N);
        q.memcpy(sycl_output_wg128.data(), d_output_tuned, M * N * sizeof(float)).wait();
        
        double time_wg128 = benchmark([&]() {
            turbodiffusion::sycl_backend::RMSNormTuned<128>::launch(
                q, d_input, d_weight, d_output_tuned, EPS, M, N
            );
        }, 100);
        
        std::cout << "  Average time: " << time_wg128 << " ms" << std::endl;
        compare_results(sycl_output_wg128, cuda_output, "WG_SIZE=128");
        
        // Test WG=256
        std::cout << "\nTesting WG_SIZE=256..." << std::endl;
        turbodiffusion::sycl_backend::RMSNormTuned<256>::launch(
            q, d_input, d_weight, d_output_tuned, EPS, M, N
        );
        
        std::vector<float> sycl_output_wg256(M * N);
        q.memcpy(sycl_output_wg256.data(), d_output_tuned, M * N * sizeof(float)).wait();
        
        double time_wg256 = benchmark([&]() {
            turbodiffusion::sycl_backend::RMSNormTuned<256>::launch(
                q, d_input, d_weight, d_output_tuned, EPS, M, N
            );
        }, 100);
        
        std::cout << "  Average time: " << time_wg256 << " ms" << std::endl;
        compare_results(sycl_output_wg256, cuda_output, "WG_SIZE=256");
        
        // ==================== Summary ====================
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Performance Summary" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "FP32 Optimized (Type C): " << time_fp32 << " ms" << std::endl;
        std::cout << "BF16:                     " << time_bf16 << " ms (" 
                  << (time_fp32/time_bf16) << "x vs FP32)" << std::endl;
        std::cout << "WG=128 (Traditional):     " << time_wg128 << " ms" << std::endl;
        std::cout << "WG=256 (Traditional):     " << time_wg256 << " ms" << std::endl;
        
        // ==================== Test Results ====================
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Test Results" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        bool pass_fp32 = max_err_fp32 < 1e-4;
        bool pass_bf16 = max_err_bf16 < 1e-4;
        
        std::cout << "FP32 Optimized: " << (pass_fp32 ? "✅ PASS" : "❌ FAIL") 
                  << " (max error: " << max_err_fp32 << ")" << std::endl;
        std::cout << "BF16:           " << (pass_bf16 ? "✅ PASS" : "❌ FAIL") 
                  << " (max error: " << max_err_bf16 << ")" << std::endl;
        
        // Cleanup
        sycl::free(d_input, q);
        sycl::free(d_weight, q);
        sycl::free(d_output_fp32, q);
        sycl::free(d_input_bf16, q);
        sycl::free(d_weight_bf16, q);
        sycl::free(d_output_bf16, q);
        sycl::free(d_output_tuned, q);
        
        if (pass_fp32 && pass_bf16) {
            std::cout << "\n✅ All tests PASSED" << std::endl;
            return 0;
        } else {
            std::cout << "\n❌ Some tests FAILED" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "SYCL Error: " << e.what() << std::endl;
        return 1;
    }
}
