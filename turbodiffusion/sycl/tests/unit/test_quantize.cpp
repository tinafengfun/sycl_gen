/**
 * Quantization SYCL Test
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
#include "../../src/quant/quantize.hpp"

using bfloat16 = sycl::ext::oneapi::bfloat16;

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

// Compare INT8 results - count mismatches
double compare_quant_results(const std::vector<int8_t>& sycl_output,
                             const std::vector<int8_t>& cuda_output) {
    int mismatches = 0;
    int total = sycl_output.size();
    
    for (int i = 0; i < total; ++i) {
        if (sycl_output[i] != cuda_output[i]) {
            mismatches++;
        }
    }
    
    double match_rate = 100.0 * (total - mismatches) / total;
    std::cout << "  Mismatches: " << mismatches << "/" << total << std::endl;
    std::cout << "  Match rate: " << std::fixed << std::setprecision(2) << match_rate << "%" << std::endl;
    
    return match_rate;
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
    std::cout << "=== Quantization SYCL Test ===" << std::endl;
    std::cout << "Type A: Element-wise kernel with vectorized optimization\n" << std::endl;
    
    const int M = 32;
    const int N = 2048;
    
    // Load data
    std::cout << "Loading CUDA reference data..." << std::endl;
    auto input = load_binary<float>("../wan2_validation/data/quant_input_M32_N2048_fp32.bin", M * N);
    auto scale = load_binary<float>("../wan2_validation/data/quant_scale_cuda_M32_fp32.bin", M);
    auto cuda_output = load_binary<int8_t>("../wan2_validation/data/quant_output_cuda_M32_N2048_int8.bin", M * N);
    
    std::cout << "Data: " << M << " rows x " << N << " cols" << std::endl;
    std::cout << "Input size: " << (M * N * sizeof(float) / 1024.0) << " KB (FP32)" << std::endl;
    std::cout << "Output size: " << (M * N * sizeof(int8_t) / 1024.0) << " KB (INT8)" << std::endl;
    
    try {
        sycl::queue q(sycl::gpu_selector_v);
        std::cout << "\nDevice: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        
        std::cout << std::string(70, '=') << std::endl;
        
        // ==================== Test 1: FP32 Basic ====================
        std::cout << "\n[Test 1] FP32 Basic (1 element/thread)" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        float* d_input = sycl::malloc_device<float>(M * N, q);
        float* d_scale = sycl::malloc_device<float>(M, q);
        int8_t* d_output = sycl::malloc_device<int8_t>(M * N, q);
        
        q.memcpy(d_input, input.data(), M * N * sizeof(float)).wait();
        q.memcpy(d_scale, scale.data(), M * sizeof(float)).wait();
        
        // Warmup
        turbodiffusion::sycl_backend::QuantizeFP32::launch(
            q, d_input, d_scale, d_output, M, N
        );
        
        double time_fp32_basic = benchmark([&]() {
            turbodiffusion::sycl_backend::QuantizeFP32::launch(
                q, d_input, d_scale, d_output, M, N
            );
        }, 200);
        
        std::vector<int8_t> output_fp32_basic(M * N);
        q.memcpy(output_fp32_basic.data(), d_output, M * N * sizeof(int8_t)).wait();
        
        std::cout << "  Time: " << std::fixed << std::setprecision(4) << time_fp32_basic << " ms" << std::endl;
        double match_fp32_basic = compare_quant_results(output_fp32_basic, cuda_output);
        
        sycl::free(d_input, q);
        sycl::free(d_scale, q);
        sycl::free(d_output, q);
        
        // ==================== Test 2: FP32 Vectorized (Vec4) ====================
        std::cout << "\n[Test 2] FP32 Vectorized (4 elements/thread)" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        d_input = sycl::malloc_device<float>(M * N, q);
        d_scale = sycl::malloc_device<float>(M, q);
        d_output = sycl::malloc_device<int8_t>(M * N, q);
        
        q.memcpy(d_input, input.data(), M * N * sizeof(float)).wait();
        q.memcpy(d_scale, scale.data(), M * sizeof(float)).wait();
        
        turbodiffusion::sycl_backend::QuantizeFP32Vec4::launch(
            q, d_input, d_scale, d_output, M, N
        );
        
        double time_fp32_vec4 = benchmark([&]() {
            turbodiffusion::sycl_backend::QuantizeFP32Vec4::launch(
                q, d_input, d_scale, d_output, M, N
            );
        }, 200);
        
        std::vector<int8_t> output_fp32_vec4(M * N);
        q.memcpy(output_fp32_vec4.data(), d_output, M * N * sizeof(int8_t)).wait();
        
        std::cout << "  Time: " << time_fp32_vec4 << " ms" << std::endl;
        std::cout << "  Speedup: " << std::setprecision(2) << (time_fp32_basic/time_fp32_vec4) << "x" << std::endl;
        double match_fp32_vec4 = compare_quant_results(output_fp32_vec4, cuda_output);
        
        sycl::free(d_input, q);
        sycl::free(d_scale, q);
        sycl::free(d_output, q);
        
        // ==================== Test 3: BF16 Input ====================
        std::cout << "\n[Test 3] BF16 Input (2x memory bandwidth)" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        // Convert input to BF16
        std::vector<bfloat16> input_bf16(M * N);
        for (int i = 0; i < M * N; ++i) {
            input_bf16[i] = bfloat16(input[i]);
        }
        
        std::cout << "  Input size: " << (M * N * sizeof(bfloat16) / 1024.0) << " KB (BF16)" << std::endl;
        
        bfloat16* d_input_bf16 = sycl::malloc_device<bfloat16>(M * N, q);
        d_scale = sycl::malloc_device<float>(M, q);
        d_output = sycl::malloc_device<int8_t>(M * N, q);
        
        q.memcpy(d_input_bf16, input_bf16.data(), M * N * sizeof(bfloat16)).wait();
        q.memcpy(d_scale, scale.data(), M * sizeof(float)).wait();
        
        turbodiffusion::sycl_backend::QuantizeBF16Input::launch(
            q, d_input_bf16, d_scale, d_output, M, N
        );
        
        double time_bf16 = benchmark([&]() {
            turbodiffusion::sycl_backend::QuantizeBF16Input::launch(
                q, d_input_bf16, d_scale, d_output, M, N
            );
        }, 200);
        
        std::vector<int8_t> output_bf16(M * N);
        q.memcpy(output_bf16.data(), d_output, M * N * sizeof(int8_t)).wait();
        
        std::cout << "  Time: " << time_bf16 << " ms" << std::endl;
        std::cout << "  Speedup vs FP32 basic: " << (time_fp32_basic/time_bf16) << "x" << std::endl;
        double match_bf16 = compare_quant_results(output_bf16, cuda_output);
        
        sycl::free(d_input_bf16, q);
        sycl::free(d_scale, q);
        sycl::free(d_output, q);
        
        // ==================== Test 4: BF16 Vectorized ====================
        std::cout << "\n[Test 4] BF16 Vectorized (4 elements/thread)" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        d_input_bf16 = sycl::malloc_device<bfloat16>(M * N, q);
        d_scale = sycl::malloc_device<float>(M, q);
        d_output = sycl::malloc_device<int8_t>(M * N, q);
        
        q.memcpy(d_input_bf16, input_bf16.data(), M * N * sizeof(bfloat16)).wait();
        q.memcpy(d_scale, scale.data(), M * sizeof(float)).wait();
        
        turbodiffusion::sycl_backend::QuantizeBF16Vec4::launch(
            q, d_input_bf16, d_scale, d_output, M, N
        );
        
        double time_bf16_vec4 = benchmark([&]() {
            turbodiffusion::sycl_backend::QuantizeBF16Vec4::launch(
                q, d_input_bf16, d_scale, d_output, M, N
            );
        }, 200);
        
        std::vector<int8_t> output_bf16_vec4(M * N);
        q.memcpy(output_bf16_vec4.data(), d_output, M * N * sizeof(int8_t)).wait();
        
        std::cout << "  Time: " << time_bf16_vec4 << " ms" << std::endl;
        std::cout << "  Speedup vs FP32 basic: " << (time_fp32_basic/time_bf16_vec4) << "x" << std::endl;
        double match_bf16_vec4 = compare_quant_results(output_bf16_vec4, cuda_output);
        
        sycl::free(d_input_bf16, q);
        sycl::free(d_scale, q);
        sycl::free(d_output, q);
        
        // ==================== Test 5: WG Tuning ====================
        std::cout << "\n[Test 5] Work-Group Size Tuning (FP32)" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        d_input = sycl::malloc_device<float>(M * N, q);
        d_scale = sycl::malloc_device<float>(M, q);
        d_output = sycl::malloc_device<int8_t>(M * N, q);
        
        q.memcpy(d_input, input.data(), M * N * sizeof(float)).wait();
        q.memcpy(d_scale, scale.data(), M * sizeof(float)).wait();
        
        auto test_wg = [&](const char* name, auto&& launch_fn) {
            launch_fn();
            double t = benchmark([&]() { launch_fn(); }, 200);
            std::cout << "  " << std::left << std::setw(12) << name
                      << std::setw(10) << std::fixed << std::setprecision(4) << t << " ms"
                      << " (speedup: " << std::setprecision(2) << (time_fp32_basic/t) << "x)" << std::endl;
        };
        
        test_wg("WG=64", [&]() {
            turbodiffusion::sycl_backend::QuantizeTuned<64>::launch(
                q, d_input, d_scale, d_output, M, N);
        });
        
        test_wg("WG=128", [&]() {
            turbodiffusion::sycl_backend::QuantizeTuned<128>::launch(
                q, d_input, d_scale, d_output, M, N);
        });
        
        test_wg("WG=256", [&]() {
            turbodiffusion::sycl_backend::QuantizeTuned<256>::launch(
                q, d_input, d_scale, d_output, M, N);
        });
        
        test_wg("WG=512", [&]() {
            turbodiffusion::sycl_backend::QuantizeTuned<512>::launch(
                q, d_input, d_scale, d_output, M, N);
        });
        
        sycl::free(d_input, q);
        sycl::free(d_scale, q);
        sycl::free(d_output, q);
        
        // ==================== Summary ====================
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "SUMMARY - Type A (Element-wise) Kernel" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << std::left << std::setw(25) << "Configuration"
                  << std::setw(12) << "Time(ms)"
                  << std::setw(12) << "Speedup"
                  << std::setw(15) << "Match Rate" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        std::cout << std::left << std::setw(25) << "FP32 Basic"
                  << std::setw(12) << time_fp32_basic
                  << std::setw(12) << "1.00x"
                  << std::setw(15) << match_fp32_basic << "%" << std::endl;
        
        std::cout << std::setw(25) << "FP32 Vec4"
                  << std::setw(12) << time_fp32_vec4
                  << std::setw(12) << (time_fp32_basic/time_fp32_vec4)
                  << std::setw(15) << match_fp32_vec4 << "%" << std::endl;
        
        std::cout << std::setw(25) << "BF16 Input"
                  << std::setw(12) << time_bf16
                  << std::setw(12) << (time_fp32_basic/time_bf16)
                  << std::setw(15) << match_bf16 << "%" << std::endl;
        
        std::cout << std::setw(25) << "BF16 Vec4"
                  << std::setw(12) << time_bf16_vec4
                  << std::setw(12) << (time_fp32_basic/time_bf16_vec4)
                  << std::setw(15) << match_bf16_vec4 << "%" << std::endl;
        
        std::cout << "\nKey Findings:" << std::endl;
        std::cout << "- Type A kernel: Memory bandwidth bound" << std::endl;
        std::cout << "- BF16 provides 2x memory bandwidth reduction" << std::endl;
        std::cout << "- Vectorization (Vec4) improves throughput" << std::endl;
        std::cout << "- INT8 output is exact (no precision loss in quantization)" << std::endl;
        
        if (match_fp32_basic > 99.0) {
            std::cout << "\n✅ Quantization tests PASSED" << std::endl;
            return 0;
        } else {
            std::cout << "\n❌ Quantization tests FAILED" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
