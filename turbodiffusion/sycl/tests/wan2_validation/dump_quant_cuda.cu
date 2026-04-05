/**
 * Quantization CUDA Dump Program
 * FP32/BF16 → INT8 with AMMAX
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    }

template<typename T>
__global__ void quantization_kernel(
    const T* input,
    int8_t* output,
    float* scale_out,
    int m, int n
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Step 1: Find AMMAX (absolute max)
    float thread_amax = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = fabsf(static_cast<float>(input[row * n + i]));
        thread_amax = fmaxf(thread_amax, val);
    }
    
    // Warp reduction with XOR shuffle
    __syncwarp();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xffffffff, thread_amax, offset);
        thread_amax = fmaxf(thread_amax, other);
    }
    
    // Store in shared memory
    __shared__ float block_amax;
    if (tid == 0) {
        block_amax = thread_amax;
    }
    __syncthreads();
    
    float scale = block_amax / 127.0f;  // INT8 range
    if (tid == 0) {
        scale_out[row] = scale;
    }
    
    // Step 2: Quantize to INT8
    for (int i = tid; i < n; i += blockDim.x) {
        float val = static_cast<float>(input[row * n + i]);
        int8_t q = static_cast<int8_t>(roundf(val / scale));
        output[row * n + i] = q;
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
    std::cout << "=== Quantization CUDA Dump ===" << std::endl;
    
    const int M = 32;
    const int N = 2048;
    
    std::cout << "Config: M=" << M << ", N=" << N << std::endl;
    
    std::vector<float> h_input(M * N);
    std::vector<int8_t> h_output(M * N);
    std::vector<float> h_scale(M);
    
    // Generate test data
    srand(42);
    for (int i = 0; i < M * N; ++i) {
        h_input[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;  // [-1, 1]
    }
    
    std::string out_dir = "/workspace/turbodiffusion/data/";
    save_binary(out_dir + "quant_input_M32_N2048_fp32.bin", h_input.data(), M * N);
    
    float *d_input;
    int8_t *d_output;
    float *d_scale;
    CUDA_CHECK(cudaMalloc(&d_input, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, M * N * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_scale, M * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "\nRunning CUDA Quantization kernel..." << std::endl;
    quantization_kernel<float><<<M, 256>>>(d_input, d_output, d_scale, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, M * N * sizeof(int8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_scale.data(), d_scale, M * sizeof(float), cudaMemcpyDeviceToHost));
    
    save_binary(out_dir + "quant_output_cuda_M32_N2048_int8.bin", h_output.data(), M * N);
    save_binary(out_dir + "quant_scale_cuda_M32_fp32.bin", h_scale.data(), M);
    
    std::ofstream meta_file(out_dir + "quant_metadata.txt");
    meta_file << "M: " << M << std::endl;
    meta_file << "N: " << N << std::endl;
    meta_file << "DataType: INT8" << std::endl;
    meta_file.close();
    std::cout << "Saved: metadata" << std::endl;
    
    // Verification
    std::cout << "\n=== Verification ===" << std::endl;
    std::cout << "Scale (first 5 rows): ";
    for (int i = 0; i < 5; ++i) {
        std::cout << h_scale[i] << " ";
    }
    std::cout << std::endl;
    
    int max_val = 0;
    for (int i = 0; i < N; ++i) {
        max_val = std::max(max_val, (int)h_output[i]);
    }
    std::cout << "Max INT8 value in first row: " << max_val << " (should be ~127)" << std::endl;
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scale);
    
    std::cout << "\n=== Dump Complete ===" << std::endl;
    return 0;
}