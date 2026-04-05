/**
 * RMSNorm CUDA Dump Program - Fixed Version
 * 修复RMS计算：正确跨所有线程归约
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; \
        exit(1); \
    }

/**
 * RMSNorm CUDA Kernel - Fixed
 * 正确实现：所有线程参与归约
 */
template<typename T>
__global__ void rmsnorm_kernel(
    const T* input,
    const T* weight,
    T* output,
    int m, int n,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Step 1: 每个线程计算局部平方和
    float thread_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = static_cast<float>(input[row * n + i]);
        thread_sum += val * val;
    }
    
    // Step 2: 使用shared memory进行block级归约
    __shared__ float shared_sum[256];  // 假设max 256 threads
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // 树形归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // 计算RMS
    float rms = sqrtf(shared_sum[0] / n + eps);
    
    // Step 3: 所有线程归一化并应用weight
    for (int i = tid; i < n; i += blockDim.x) {
        float val = static_cast<float>(input[row * n + i]);
        float w = static_cast<float>(weight[i]);
        output[row * n + i] = static_cast<T>(w * val / rms);
    }
}

template<typename T>
void save_binary(const std::string& filename, const T* data, size_t count) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    file.write(reinterpret_cast<const char*>(data), count * sizeof(T));
    file.close();
    std::cout << "Saved: " << filename << " (" << count << " elements)" << std::endl;
}

template<typename T>
void generate_test_data(std::vector<T>& input, std::vector<T>& weight, 
                        int m, int n, int seed = 42) {
    srand(seed);
    
    for (int i = 0; i < m * n; ++i) {
        input[i] = static_cast<T>((rand() / float(RAND_MAX)) * 2.0f - 1.0f);
    }
    
    for (int i = 0; i < n; ++i) {
        weight[i] = static_cast<T>(1.0f);
    }
}

int main() {
    std::cout << "=== RMSNorm CUDA Dump (Fixed) ===" << std::endl;
    
    const int M = 32;
    const int N = 2048;
    const float EPS = 1e-6;
    
    std::cout << "Config: M=" << M << ", N=" << N << std::endl;
    
    std::vector<float> h_input(M * N);
    std::vector<float> h_weight(N);
    std::vector<float> h_output(M * N);
    
    generate_test_data(h_input, h_weight, M, N);
    
    std::string out_dir = "/workspace/turbodiffusion/data/";
    save_binary(out_dir + "rmsnorm_input_M32_N2048_fp32.bin", h_input.data(), M * N);
    save_binary(out_dir + "rmsnorm_weight_N2048_fp32.bin", h_weight.data(), N);
    
    float *d_input, *d_weight, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "\nRunning CUDA RMSNorm kernel..." << std::endl;
    dim3 grid(M);
    dim3 block(256);
    
    rmsnorm_kernel<float><<<grid, block>>>(d_input, d_weight, d_output, M, N, EPS);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    save_binary(out_dir + "rmsnorm_output_cuda_M32_N2048_fp32.bin", h_output.data(), M * N);
    
    std::ofstream meta_file(out_dir + "rmsnorm_metadata.txt");
    meta_file << "M: " << M << std::endl;
    meta_file << "N: " << N << std::endl;
    meta_file << "EPS: " << EPS << std::endl;
    meta_file << "DataType: FP32" << std::endl;
    meta_file << "Grid: " << M << std::endl;
    meta_file << "Block: 256" << std::endl;
    meta_file << "GPU: L20" << std::endl;
    meta_file.close();
    std::cout << "Saved: metadata" << std::endl;
    
    // Verification
    std::cout << "\n=== Verification ===" << std::endl;
    
    // 验证RMS = 1.0
    for (int row = 0; row < std::min(5, M); ++row) {
        double sum_sq = 0;
        for (int i = 0; i < N; ++i) {
            sum_sq += h_output[row * N + i] * h_output[row * N + i];
        }
        float rms = std::sqrt(sum_sq / N);
        std::cout << "Row " << row << " RMS: " << rms;
        if (std::abs(rms - 1.0f) < 0.01f) {
            std::cout << " ✓" << std::endl;
        } else {
            std::cout << " ✗ (error: " << std::abs(rms - 1.0f) << ")" << std::endl;
        }
    }
    
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    
    std::cout << "\n=== Dump Complete ===" << std::endl;
    return 0;
}