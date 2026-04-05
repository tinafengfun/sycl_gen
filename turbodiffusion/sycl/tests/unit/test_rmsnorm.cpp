/**
 * RMSNorm SYCL Unit Test
 * 对比CUDA参考输出
 */

#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include "../src/norm/rmsnorm.hpp"

// 加载二进制数据
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

int main() {
    std::cout << "=== RMSNorm SYCL Unit Test ===" << std::endl;
    
    const int M = 32;
    const int N = 2048;
    const float EPS = 1e-6;
    
    std::cout << "Loading CUDA reference data..." << std::endl;
    
    // 加载输入数据
    auto input = load_binary("../wan2_validation/data/rmsnorm_input_M32_N2048_fp32.bin", M * N);
    auto weight = load_binary("../wan2_validation/data/rmsnorm_weight_N2048_fp32.bin", N);
    auto cuda_output = load_binary("../wan2_validation/data/rmsnorm_output_cuda_M32_N2048_fp32.bin", M * N);
    
    std::cout << "Input: " << input.size() << " elements" << std::endl;
    std::cout << "Weight: " << weight.size() << " elements" << std::endl;
    std::cout << "CUDA output: " << cuda_output.size() << " elements" << std::endl;
    
    // 分配SYCL输出缓冲区
    std::vector<float> sycl_output(M * N);
    
    try {
        // 创建SYCL队列
        sycl::queue q(sycl::gpu_selector_v);
        std::cout << "\nDevice: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        
        // 分配设备内存
        float* d_input = sycl::malloc_device<float>(M * N, q);
        float* d_weight = sycl::malloc_device<float>(N, q);
        float* d_output = sycl::malloc_device<float>(M * N, q);
        
        // 拷贝输入数据到设备
        q.memcpy(d_input, input.data(), M * N * sizeof(float)).wait();
        q.memcpy(d_weight, weight.data(), N * sizeof(float)).wait();
        
        std::cout << "\nRunning SYCL RMSNorm..." << std::endl;
        
        // 调用SYCL RMSNorm
        turbodiffusion::sycl_backend::rmsnorm_sycl(
            q, d_input, d_weight, d_output, EPS, M, N
        );
        q.wait();
        
        // 拷贝结果回主机
        q.memcpy(sycl_output.data(), d_output, M * N * sizeof(float)).wait();
        
        // 释放设备内存
        sycl::free(d_input, q);
        sycl::free(d_weight, q);
        sycl::free(d_output, q);
        
    } catch (const std::exception& e) {
        std::cerr << "SYCL Error: " << e.what() << std::endl;
        return 1;
    }
    
    // 对比结果
    std::cout << "\n=== Comparison ===" << std::endl;
    
    double max_diff = 0.0;
    double mean_diff = 0.0;
    int max_diff_idx = 0;
    
    for (int i = 0; i < M * N; ++i) {
        double diff = std::abs(sycl_output[i] - cuda_output[i]);
        mean_diff += diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    mean_diff /= (M * N);
    
    std::cout << "Max absolute error: " << max_diff << std::endl;
    std::cout << "Mean absolute error: " << mean_diff << std::endl;
    std::cout << "Max diff at index: " << max_diff_idx << std::endl;
    std::cout << "  CUDA value: " << cuda_output[max_diff_idx] << std::endl;
    std::cout << "  SYCL value: " << sycl_output[max_diff_idx] << std::endl;
    
    // 验证RMS
    std::cout << "\n=== RMS Verification ===" << std::endl;
    for (int row = 0; row < std::min(3, M); ++row) {
        double sum_sq = 0;
        for (int i = 0; i < N; ++i) {
            sum_sq += sycl_output[row * N + i] * sycl_output[row * N + i];
        }
        float rms = std::sqrt(sum_sq / N);
        std::cout << "Row " << row << " RMS: " << rms;
        if (std::abs(rms - 1.0f) < 0.01f) {
            std::cout << " ✓" << std::endl;
        } else {
            std::cout << " ✗" << std::endl;
        }
    }
    
    // 判断测试是否通过
    std::cout << "\n=== Test Result ===" << std::endl;
    if (max_diff < 1e-4) {
        std::cout << "✅ PASS - Max error " << max_diff << " < 1e-4" << std::endl;
        return 0;
    } else {
        std::cout << "❌ FAIL - Max error " << max_diff << " >= 1e-4" << std::endl;
        return 1;
    }
}