/**
 * LayerNorm CUDA Dump Program - Fixed
 * 正确实现：使用shared memory进行block级归约
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    }

template<typename T>
__global__ void layernorm_kernel(
    const T* input,
    const T* gamma,
    const T* beta,
    T* output,
    int m, int n,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float shared_data[256];  // 用于归约的shared memory
    
    // Step 1: 计算mean
    float thread_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        thread_sum += static_cast<float>(input[row * n + i]);
    }
    
    shared_data[tid] = thread_sum;
    __syncthreads();
    
    // 树形归约计算总和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_data[0] / n;
    
    // Step 2: 计算variance
    float thread_var = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = static_cast<float>(input[row * n + i]) - mean;
        thread_var += diff * diff;
    }
    
    shared_data[tid] = thread_var;
    __syncthreads();
    
    // 树形归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    float var = shared_data[0] / n;
    float inv_std = 1.0f / sqrtf(var + eps);
    
    // Step 3: 归一化并应用gamma/beta
    for (int i = tid; i < n; i += blockDim.x) {
        float val = (static_cast<float>(input[row * n + i]) - mean) * inv_std;
        float g = static_cast<float>(gamma[i]);
        float b = static_cast<float>(beta[i]);
        output[row * n + i] = static_cast<T>(val * g + b);
    }
}

template<typename T>
void save_binary(const std::string& filename, const T* data, size_t count) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data), count * sizeof(T));
    file.close();
    std::cout << "Saved: " << filename << " (" << count << " elements)" << std::endl;
}

template<typename T>
void generate_test_data(std::vector<T>& input, std::vector<T>& gamma, 
                        std::vector<T>& beta, int m, int n, int seed = 42) {
    srand(seed);
    
    for (int i = 0; i < m * n; ++i) {
        input[i] = static_cast<T>((rand() / float(RAND_MAX)) * 2.0f - 1.0f);
    }
    
    for (int i = 0; i < n; ++i) {
        gamma[i] = static_cast<T>(1.0f);
        beta[i] = static_cast<T>(0.0f);
    }
}

int main() {
    std::cout << "=== LayerNorm CUDA Dump (Fixed) ===" << std::endl;
    
    const int M = 32;
    const int N = 2048;
    const float EPS = 1e-6;
    
    std::cout << "Config: M=" << M << ", N=" << N << std::endl;
    
    std::vector<float> h_input(M * N);
    std::vector<float> h_gamma(N);
    std::vector<float> h_beta(N);
    std::vector<float> h_output(M * N);
    
    generate_test_data(h_input, h_gamma, h_beta, M, N);
    
    std::string out_dir = "/workspace/turbodiffusion/data/";
    save_binary(out_dir + "layernorm_input_M32_N2048_fp32.bin", h_input.data(), M * N);
    save_binary(out_dir + "layernorm_gamma_N2048_fp32.bin", h_gamma.data(), N);
    save_binary(out_dir + "layernorm_beta_N2048_fp32.bin", h_beta.data(), N);
    
    float *d_input, *d_gamma, *d_beta, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "\nRunning CUDA LayerNorm kernel..." << std::endl;
    layernorm_kernel<float><<<M, 256>>>(d_input, d_gamma, d_beta, d_output, M, N, EPS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    save_binary(out_dir + "layernorm_output_cuda_M32_N2048_fp32.bin", h_output.data(), M * N);
    
    std::ofstream meta_file(out_dir + "layernorm_metadata.txt");
    meta_file << "M: " << M << std::endl;
    meta_file << "N: " << N << std::endl;
    meta_file << "EPS: " << EPS << std::endl;
    meta_file << "DataType: FP32" << std::endl;
    meta_file.close();
    std::cout << "Saved: metadata" << std::endl;
    
    // Verification
    std::cout << "\n=== Verification ===" << std::endl;
    for (int row = 0; row < std::min(5, M); ++row) {
        double sum = 0, sum_sq = 0;
        for (int i = 0; i < N; ++i) {
            sum += h_output[row * N + i];
            sum_sq += h_output[row * N + i] * h_output[row * N + i];
        }
        double mean = sum / N;
        double var = sum_sq / N - mean * mean;
        
        std::cout << "Row " << row << ": Mean=" << mean << " Var=" << var;
        if (std::abs(mean) < 0.01 && std::abs(var - 1.0) < 0.01) {
            std::cout << " ✓" << std::endl;
        } else {
            std::cout << " ✗" << std::endl;
        }
    }
    
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output);
    
    std::cout << "\n=== Dump Complete ===" << std::endl;
    return 0;
}