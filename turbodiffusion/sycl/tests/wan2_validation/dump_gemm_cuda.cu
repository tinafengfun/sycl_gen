/**
 * GEMM CUDA Dump Program (Simplified)
 * INT8 input, FP32 output
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    }

// Simplified INT8 GEMM kernel
__global__ void gemm_int8_kernel(
    const int8_t* A,
    const int8_t* B,
    float* C,
    int m, int n, int k
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        int32_t sum = 0;
        for (int i = 0; i < k; ++i) {
            sum += static_cast<int32_t>(A[row * k + i]) * 
                   static_cast<int32_t>(B[i * n + col]);
        }
        C[row * n + col] = static_cast<float>(sum);
    }
}

template<typename T>
void save_binary(const std::string& filename, const T* data, size_t count) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data), count * sizeof(T));
    file.close();
    std::cout << "Saved: " << filename << " (" << count << " elements)" << std::endl;
}

int main() {
    std::cout << "=== GEMM CUDA Dump ===" << std::endl;
    
    const int M = 32;
    const int N = 2048;
    const int K = 2048;
    
    std::cout << "Config: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    
    std::vector<int8_t> h_A(M * K);
    std::vector<int8_t> h_B(K * N);
    std::vector<float> h_C(M * N);
    
    // Generate test data
    srand(42);
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<int8_t>((rand() % 256) - 128);  // [-128, 127]
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<int8_t>((rand() % 256) - 128);
    }
    
    std::string out_dir = "/workspace/turbodiffusion/data/";
    save_binary(out_dir + "gemm_A_M32_K2048_int8.bin", h_A.data(), M * K);
    save_binary(out_dir + "gemm_B_K2048_N2048_int8.bin", h_B.data(), K * N);
    
    int8_t *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(int8_t), cudaMemcpyHostToDevice));
    
    std::cout << "\nRunning CUDA GEMM kernel..." << std::endl;
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_int8_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    save_binary(out_dir + "gemm_output_cuda_M32_N2048_fp32.bin", h_C.data(), M * N);
    
    std::ofstream meta_file(out_dir + "gemm_metadata.txt");
    meta_file << "M: " << M << std::endl;
    meta_file << "N: " << N << std::endl;
    meta_file << "K: " << K << std::endl;
    meta_file << "InputType: INT8" << std::endl;
    meta_file << "OutputType: FP32" << std::endl;
    meta_file.close();
    std::cout << "Saved: metadata" << std::endl;
    
    // Verification
    std::cout << "\n=== Verification ===" << std::endl;
    double sum = 0;
    for (int i = 0; i < M * N; ++i) {
        sum += h_C[i];
    }
    std::cout << "Output sum: " << sum << std::endl;
    std::cout << "Output mean: " << sum / (M * N) << std::endl;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    std::cout << "\n=== Dump Complete ===" << std::endl;
    return 0;
}