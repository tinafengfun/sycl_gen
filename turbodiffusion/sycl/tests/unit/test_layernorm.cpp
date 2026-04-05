/**
 * LayerNorm SYCL Test
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
#include <iomanip>
#include "../../src/norm/layernorm.hpp"

using bfloat16 = sycl::ext::oneapi::bfloat16;

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

// Compare results
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
double benchmark(Func&& func, int iterations = 100) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

// Verify mean ≈ 0 and variance ≈ 1 after normalization (before scale/shift)
void verify_stats(const std::vector<float>& input, const std::vector<float>& output,
                  int M, int N, const std::string& label) {
    std::cout << "\n  " << label << " Statistics Check:" << std::endl;
    for (int row = 0; row < std::min(3, M); ++row) {
        double sum = 0, sum_sq = 0;
        for (int i = 0; i < N; ++i) {
            float val = output[row * N + i];
            sum += val;
            sum_sq += val * val;
        }
        float mean = sum / N;
        float var = sum_sq / N - mean * mean;
        std::cout << "    Row " << row << ": mean=" << mean << " var=" << var << std::endl;
    }
}

int main() {
    std::cout << "=== LayerNorm SYCL Test ===" << std::endl;
    
    const int M = 32;
    const int N = 2048;
    const float EPS = 1e-6;
    
    // Load data
    std::cout << "\nLoading CUDA reference data..." << std::endl;
    auto input = load_binary("../wan2_validation/data/layernorm_input_M32_N2048_fp32.bin", M * N);
    auto gamma = load_binary("../wan2_validation/data/layernorm_gamma_N2048_fp32.bin", N);
    auto beta = load_binary("../wan2_validation/data/layernorm_beta_N2048_fp32.bin", N);
    auto cuda_output = load_binary("../wan2_validation/data/layernorm_output_cuda_M32_N2048_fp32.bin", M * N);
    
    std::cout << "Data loaded: " << M << "x" << N << std::endl;
    
    try {
        sycl::queue q(sycl::gpu_selector_v);
        std::cout << "\nDevice: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        
        std::cout << std::string(70, '=') << std::endl;
        
        // ==================== Test 1: FP32 ====================
        std::cout << "\n[Test 1] FP32 LayerNorm" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        float* d_input = sycl::malloc_device<float>(M * N, q);
        float* d_gamma = sycl::malloc_device<float>(N, q);
        float* d_beta = sycl::malloc_device<float>(N, q);
        float* d_output = sycl::malloc_device<float>(M * N, q);
        
        q.memcpy(d_input, input.data(), M * N * sizeof(float)).wait();
        q.memcpy(d_gamma, gamma.data(), N * sizeof(float)).wait();
        q.memcpy(d_beta, beta.data(), N * sizeof(float)).wait();
        
        // Warmup
        turbodiffusion::sycl_backend::LayerNormFP32::launch(
            q, d_input, d_gamma, d_beta, d_output, EPS, M, N
        );
        
        double time_fp32 = benchmark([&]() {
            turbodiffusion::sycl_backend::LayerNormFP32::launch(
                q, d_input, d_gamma, d_beta, d_output, EPS, M, N
            );
        }, 200);
        
        std::vector<float> output_fp32(M * N);
        q.memcpy(output_fp32.data(), d_output, M * N * sizeof(float)).wait();
        
        std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_fp32 << " ms" << std::endl;
        double err_fp32 = compare_results(output_fp32, cuda_output);
        verify_stats(input, output_fp32, M, N, "FP32");
        
        sycl::free(d_input, q);
        sycl::free(d_gamma, q);
        sycl::free(d_beta, q);
        sycl::free(d_output, q);
        
        // ==================== Test 2: BF16 ====================
        std::cout << "\n[Test 2] BF16 LayerNorm" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        // Convert to BF16
        std::vector<bfloat16> input_bf16(M * N);
        std::vector<bfloat16> gamma_bf16(N);
        std::vector<bfloat16> beta_bf16(N);
        for (int i = 0; i < M * N; ++i) input_bf16[i] = bfloat16(input[i]);
        for (int i = 0; i < N; ++i) gamma_bf16[i] = bfloat16(gamma[i]);
        for (int i = 0; i < N; ++i) beta_bf16[i] = bfloat16(beta[i]);
        
        bfloat16* d_input_bf16 = sycl::malloc_device<bfloat16>(M * N, q);
        bfloat16* d_gamma_bf16 = sycl::malloc_device<bfloat16>(N, q);
        bfloat16* d_beta_bf16 = sycl::malloc_device<bfloat16>(N, q);
        bfloat16* d_output_bf16 = sycl::malloc_device<bfloat16>(M * N, q);
        
        q.memcpy(d_input_bf16, input_bf16.data(), M * N * sizeof(bfloat16)).wait();
        q.memcpy(d_gamma_bf16, gamma_bf16.data(), N * sizeof(bfloat16)).wait();
        q.memcpy(d_beta_bf16, beta_bf16.data(), N * sizeof(bfloat16)).wait();
        
        // Warmup
        turbodiffusion::sycl_backend::LayerNormBF16::launch(
            q, d_input_bf16, d_gamma_bf16, d_beta_bf16, d_output_bf16, EPS, M, N
        );
        
        double time_bf16 = benchmark([&]() {
            turbodiffusion::sycl_backend::LayerNormBF16::launch(
                q, d_input_bf16, d_gamma_bf16, d_beta_bf16, d_output_bf16, EPS, M, N
            );
        }, 200);
        
        std::vector<bfloat16> output_bf16(M * N);
        q.memcpy(output_bf16.data(), d_output_bf16, M * N * sizeof(bfloat16)).wait();
        
        std::vector<float> output_bf16_fp32(M * N);
        for (int i = 0; i < M * N; ++i) output_bf16_fp32[i] = float(output_bf16[i]);
        
        std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_bf16 << " ms" << std::endl;
        double err_bf16 = compare_results(output_bf16_fp32, cuda_output);
        verify_stats(input, output_bf16_fp32, M, N, "BF16");
        
        sycl::free(d_input_bf16, q);
        sycl::free(d_gamma_bf16, q);
        sycl::free(d_beta_bf16, q);
        sycl::free(d_output_bf16, q);
        
        // ==================== Test 3: WG Tuning ====================
        std::cout << "\n[Test 3] Work-Group Size Tuning (FP32)" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        float* d_input_wg = sycl::malloc_device<float>(M * N, q);
        float* d_gamma_wg = sycl::malloc_device<float>(N, q);
        float* d_beta_wg = sycl::malloc_device<float>(N, q);
        float* d_output_wg = sycl::malloc_device<float>(M * N, q);
        
        q.memcpy(d_input_wg, input.data(), M * N * sizeof(float)).wait();
        q.memcpy(d_gamma_wg, gamma.data(), N * sizeof(float)).wait();
        q.memcpy(d_beta_wg, beta.data(), N * sizeof(float)).wait();
        
        // Test different WG sizes
        auto test_wg = [&](const char* name, auto&& launch_fn) {
            launch_fn();
            double t = benchmark([&]() { launch_fn(); }, 200);
            
            std::vector<float> out(M * N);
            q.memcpy(out.data(), d_output_wg, M * N * sizeof(float)).wait();
            
            std::cout << "  " << std::left << std::setw(12) << name
                      << std::setw(10) << std::fixed << std::setprecision(4) << t << " ms"
                      << " (speedup: " << std::setprecision(1) << (time_fp32/t) << "x)" << std::endl;
        };
        
        test_wg("WG=64", [&]() {
            turbodiffusion::sycl_backend::LayerNormTuned<64>::launch(
                q, d_input_wg, d_gamma_wg, d_beta_wg, d_output_wg, EPS, M, N);
        });
        
        test_wg("WG=128", [&]() {
            turbodiffusion::sycl_backend::LayerNormTuned<128>::launch(
                q, d_input_wg, d_gamma_wg, d_beta_wg, d_output_wg, EPS, M, N);
        });
        
        test_wg("WG=256", [&]() {
            turbodiffusion::sycl_backend::LayerNormTuned<256>::launch(
                q, d_input_wg, d_gamma_wg, d_beta_wg, d_output_wg, EPS, M, N);
        });
        
        test_wg("WG=512", [&]() {
            turbodiffusion::sycl_backend::LayerNormTuned<512>::launch(
                q, d_input_wg, d_gamma_wg, d_beta_wg, d_output_wg, EPS, M, N);
        });
        
        sycl::free(d_input_wg, q);
        sycl::free(d_gamma_wg, q);
        sycl::free(d_beta_wg, q);
        sycl::free(d_output_wg, q);
        
        // ==================== Summary ====================
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "SUMMARY" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "FP32:  Time=" << time_fp32 << "ms, Error=" << err_fp32 
                  << (err_fp32 < 1e-4 ? " ✓" : " ✗") << std::endl;
        std::cout << "BF16:  Time=" << time_bf16 << "ms, Error=" << err_bf16 
                  << " (speedup: " << (time_fp32/time_bf16) << "x)"
                  << (err_bf16 < 1e-4 ? " ✓" : " ✗") << std::endl;
        
        if (err_fp32 < 1e-4) {
            std::cout << "\n✅ LayerNorm tests PASSED" << std::endl;
            return 0;
        } else {
            std::cout << "\n❌ LayerNorm tests FAILED" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
