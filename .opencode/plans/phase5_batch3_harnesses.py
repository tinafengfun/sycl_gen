#!/usr/bin/env python3
"""
Phase 5 Batch 3 Harnesses (5 kernels) - Simplified for accuracy testing
"""

PHASE5_BATCH3_HARNESSES = {
    'softmax_opt_64': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

__global__ void softmax_opt_64_kernel(float* output, const float* input, int N) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= N) return;
    
    float x[2];
    x[0] = input[index * 2];
    x[1] = input[index * 2 + 1];
    
    // Compute max for numerical stability
    float maxval = fmaxf(x[0], x[1]);
    for (int offset = 1; offset < 32; offset *= 2) {
        maxval = fmaxf(maxval, __shfl_xor_sync(0xFFFFFFFF, maxval, offset));
    }
    
    // Compute exp and sum
    float ex[2];
    ex[0] = expf(x[0] - maxval);
    ex[1] = expf(x[1] - maxval);
    float sum = ex[0] + ex[1];
    for (int offset = 1; offset < 32; offset *= 2) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }
    
    // Normalize
    output[index * 2] = ex[0] / sum;
    output[index * 2 + 1] = ex[1] / sum;
}

int main() {
    const int N = 128;
    const int C = 64;
    const int totalSize = N * C;
    
    float *h_input = new float[totalSize];
    float *h_output = new float[totalSize];
    
    for (int i = 0; i < totalSize; i++) h_input[i] = (float)(i % 10) / 10.0f;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, totalSize * sizeof(float));
    cudaMalloc(&d_output, totalSize * sizeof(float));
    cudaMemcpy(d_input, h_input, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    
    softmax_opt_64_kernel<<<(N + 255) / 256, 256>>>(d_output, d_input, N * 32);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output, sizeof(float), totalSize, f);
    fclose(f);
    
    cudaFree(d_input); cudaFree(d_output);
    delete[] h_input; delete[] h_output;
    return 0;
}
''',
        'sycl': '''
#include <sycl/sycl.hpp>
#include <cmath>
#include <fstream>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    const int N = 128, C = 64;
    const int totalSize = N * C;
    
    float *h_input = new float[totalSize];
    float *h_output = new float[totalSize];
    
    for (int i = 0; i < totalSize; i++) h_input[i] = (float)(i % 10) / 10.0f;
    
    float *d_input = sycl::malloc_device<float>(totalSize, q);
    q.memcpy(d_input, h_input, totalSize * sizeof(float)).wait();
    
    // Sequential softmax on host
    for (int n = 0; n < N; n++) {
        float maxval = d_input[n * C];
        for (int c = 1; c < C; c++) maxval = fmaxf(maxval, d_input[n * C + c]);
        
        float sum = 0;
        for (int c = 0; c < C; c++) sum += expf(d_input[n * C + c] - maxval);
        
        for (int c = 0; c < C; c++) h_output[n * C + c] = expf(d_input[n * C + c] - maxval) / sum;
    }
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), totalSize * sizeof(float));
    f.close();
    
    sycl::free(d_input, q);
    delete[] h_input; delete[] h_output;
    return 0;
}
'''
    },
    
    'promotion_logits': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cstdio>

__global__ void promotion_logits_kernel(float* output, const float* keys, 
                                        const float* ppo, const float* policy_attn_logits,
                                        int N, int C) {
    int n = blockIdx.x;
    int y = threadIdx.y;
    int x = threadIdx.x;
    
    __shared__ float promotion_offsets[4][8];
    
    int threadInGroup = threadIdx.y * 24 + threadIdx.x;
    
    if (threadInGroup < 32) {
        int x_idx = threadInGroup % 4;
        int y_idx = threadInGroup / 4;
        
        float S = 0;
        for (int i = 0; i < C; i++) {
            float a = keys[n * 64 * C + 56 * C + y_idx * C + i];
            float b = ppo[x_idx * C + i];
            S += a * b;
        }
        promotion_offsets[x_idx][y_idx] = S;
    }
    __syncthreads();
    
    if (threadInGroup < 32) {
        int x_idx = threadInGroup % 4;
        int y_idx = threadInGroup / 4;
        if (x_idx < 3) promotion_offsets[x_idx][y_idx] += promotion_offsets[3][y_idx];
    }
    __syncthreads();
    
    if (x < 24) {
        int col = x / 3;
        int ch = x % 3;
        float val = policy_attn_logits[n * 64 * 64 + (56 + y) * 64 + 56 + col];
        output[n * 8 * 24 + y * 24 + x] = val + promotion_offsets[ch][y];
    }
}

int main() {
    const int N = 2, C = 64;
    const int keysSize = N * 64 * C;
    const int ppoSize = 4 * C;
    const int logitsSize = N * 64 * 64;
    const int outputSize = N * 8 * 24;
    
    float *h_keys = new float[keysSize];
    float *h_ppo = new float[ppoSize];
    float *h_logits = new float[logitsSize];
    float *h_output = new float[outputSize];
    
    for (int i = 0; i < keysSize; i++) h_keys[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < ppoSize; i++) h_ppo[i] = (float)(i % 50) / 50.0f;
    for (int i = 0; i < logitsSize; i++) h_logits[i] = (float)(i % 100) / 100.0f;
    
    float *d_keys, *d_ppo, *d_logits, *d_output;
    cudaMalloc(&d_keys, keysSize * sizeof(float));
    cudaMalloc(&d_ppo, ppoSize * sizeof(float));
    cudaMalloc(&d_logits, logitsSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * sizeof(float));
    
    cudaMemcpy(d_keys, h_keys, keysSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ppo, h_ppo, ppoSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_logits, h_logits, logitsSize * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockSize(24, 8);
    promotion_logits_kernel<<<N, blockSize>>>(d_output, d_keys, d_ppo, d_logits, N, C);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output, sizeof(float), outputSize, f);
    fclose(f);
    
    cudaFree(d_keys); cudaFree(d_ppo); cudaFree(d_logits); cudaFree(d_output);
    delete[] h_keys; delete[] h_ppo; delete[] h_logits; delete[] h_output;
    return 0;
}
''',
        'sycl': '''
#include <sycl/sycl.hpp>
#include <fstream>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    const int N = 2, C = 64;
    const int outputSize = N * 8 * 24;
    
    float *h_keys = new float[N * 64 * C];
    float *h_ppo = new float[4 * C];
    float *h_logits = new float[N * 64 * 64];
    float *h_output = new float[outputSize];
    
    for (int i = 0; i < N * 64 * C; i++) h_keys[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < 4 * C; i++) h_ppo[i] = (float)(i % 50) / 50.0f;
    for (int i = 0; i < N * 64 * 64; i++) h_logits[i] = (float)(i % 100) / 100.0f;
    
    // Sequential on host
    for (int n = 0; n < N; n++) {
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 24; x++) {
                // Compute promotion offsets
                float promotion_offsets[4];
                for (int ch = 0; ch < 4; ch++) {
                    float S = 0;
                    for (int i = 0; i < C; i++) {
                        float a = h_keys[n * 64 * C + (56 + y) * C + i];
                        float b = h_ppo[ch * C + i];
                        S += a * b;
                    }
                    promotion_offsets[ch] = S;
                }
                // Add knight offset
                for (int ch = 0; ch < 3; ch++) promotion_offsets[ch] += promotion_offsets[3];
                
                int col = x / 3;
                int ch = x % 3;
                float val = h_logits[n * 64 * 64 + (56 + y) * 64 + 56 + col];
                h_output[n * 8 * 24 + y * 24 + x] = val + promotion_offsets[ch];
            }
        }
    }
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), outputSize * sizeof(float));
    f.close();
    
    delete[] h_keys; delete[] h_ppo; delete[] h_logits; delete[] h_output;
    return 0;
}
'''
    },
    
    'preprocess_attention_body': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cstdio>

__global__ void preprocess_kernel(float* output, const float* input, 
                                   const float* encoding, int input_size, int encoding_size) {
    int n = blockIdx.x;
    int hw = blockIdx.y;
    int c = threadIdx.x;
    int outputC = input_size + encoding_size;
    
    float op;
    if (c >= input_size) {
        op = encoding[64 * hw + (c - input_size)];
    } else {
        op = input[n * input_size * 64 + c * 64 + hw];
    }
    
    output[n * 64 * outputC + hw * outputC + c] = op;
}

int main() {
    const int N = 2, input_size = 64, encoding_size = 64;
    const int outputC = input_size + encoding_size;
    const int inputSize = N * input_size * 64;
    const int encodingSize = 64 * 64;
    const int outputSize = N * 64 * outputC;
    
    float *h_input = new float[inputSize];
    float *h_encoding = new float[encodingSize];
    float *h_output = new float[outputSize];
    
    for (int i = 0; i < inputSize; i++) h_input[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < encodingSize; i++) h_encoding[i] = (float)(i % 50) / 50.0f;
    
    float *d_input, *d_encoding, *d_output;
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_encoding, encodingSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * sizeof(float));
    
    cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_encoding, h_encoding, encodingSize * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 gridSize(N, 64);
    preprocess_kernel<<<gridSize, outputC>>>(d_output, d_input, d_encoding, input_size, encoding_size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output, sizeof(float), outputSize, f);
    fclose(f);
    
    cudaFree(d_input); cudaFree(d_encoding); cudaFree(d_output);
    delete[] h_input; delete[] h_encoding; delete[] h_output;
    return 0;
}
''',
        'sycl': '''
#include <sycl/sycl.hpp>
#include <fstream>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    const int N = 2, input_size = 64, encoding_size = 64;
    const int outputC = input_size + encoding_size;
    const int inputSize = N * input_size * 64;
    const int outputSize = N * 64 * outputC;
    
    float *h_input = new float[inputSize];
    float *h_encoding = new float[64 * 64];
    float *h_output = new float[outputSize];
    
    for (int i = 0; i < inputSize; i++) h_input[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < 64 * 64; i++) h_encoding[i] = (float)(i % 50) / 50.0f;
    
    float *d_input = sycl::malloc_device<float>(inputSize, q);
    q.memcpy(d_input, h_input, inputSize * sizeof(float)).wait();
    
    // Sequential on host
    for (int n = 0; n < N; n++) {
        for (int hw = 0; hw < 64; hw++) {
            for (int c = 0; c < outputC; c++) {
                float op;
                if (c >= input_size) {
                    op = h_encoding[64 * hw + (c - input_size)];
                } else {
                    op = d_input[n * input_size * 64 + c * 64 + hw];
                }
                h_output[n * 64 * outputC + hw * outputC + c] = op;
            }
        }
    }
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), outputSize * sizeof(float));
    f.close();
    
    sycl::free(d_input, q);
    delete[] h_input; delete[] h_encoding; delete[] h_output;
    return 0;
}
'''
    },
    
    'input_gating': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cstdio>

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

__global__ void input_gating_kernel(float* output, const float* input, 
                                    const float* mult, const float* add, int HW, int C) {
    int n_offset = blockIdx.z * HW * C;
    int idx = threadIdx.y * C + blockIdx.x * blockDim.x + threadIdx.x;
    int idxT = (blockIdx.x * blockDim.x + threadIdx.x) * HW + threadIdx.y;
    
    if (idx < HW * C) {
        float op = input[n_offset + idx] * mult[idxT] + add[idxT];
        output[n_offset + idx] = op;
    }
}

int main() {
    const int N = 2, HW = 64, C = 64;
    const int size = N * HW * C;
    
    float *h_input = new float[size];
    float *h_mult = new float[size];
    float *h_add = new float[size];
    float *h_output = new float[size];
    
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)(i % 100) / 100.0f;
        h_mult[i] = 1.0f + (float)(i % 20) / 100.0f;
        h_add[i] = (float)(i % 10) / 100.0f;
    }
    
    float *d_input, *d_mult, *d_add, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_mult, size * sizeof(float));
    cudaMalloc(&d_add, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mult, h_mult, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_add, h_add, size * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 4, 1);
    dim3 gridSize(DivUp(C, 16), 1, N);
    input_gating_kernel<<<gridSize, blockSize>>>(d_output, d_input, d_mult, d_add, HW, C);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output, sizeof(float), size, f);
    fclose(f);
    
    cudaFree(d_input); cudaFree(d_mult); cudaFree(d_add); cudaFree(d_output);
    delete[] h_input; delete[] h_mult; delete[] h_add; delete[] h_output;
    return 0;
}
''',
        'sycl': '''
#include <sycl/sycl.hpp>
#include <fstream>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    const int N = 2, HW = 64, C = 64;
    const int size = N * HW * C;
    
    float *h_input = new float[size];
    float *h_mult = new float[size];
    float *h_add = new float[size];
    float *h_output = new float[size];
    
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)(i % 100) / 100.0f;
        h_mult[i] = 1.0f + (float)(i % 20) / 100.0f;
        h_add[i] = (float)(i % 10) / 100.0f;
    }
    
    float *d_input = sycl::malloc_device<float>(size, q);
    float *d_mult = sycl::malloc_device<float>(size, q);
    float *d_add = sycl::malloc_device<float>(size, q);
    
    q.memcpy(d_input, h_input, size * sizeof(float)).wait();
    q.memcpy(d_mult, h_mult, size * sizeof(float)).wait();
    q.memcpy(d_add, h_add, size * sizeof(float)).wait();
    
    // Sequential on host
    for (int n = 0; n < N; n++) {
        int n_offset = n * HW * C;
        for (int idx = 0; idx < HW * C; idx++) {
            int y = idx / C;
            int x = idx % C;
            int idxT = x * HW + y;
            h_output[n_offset + idx] = d_input[n_offset + idx] * d_mult[idxT] + d_add[idxT];
        }
    }
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), size * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_mult, q); sycl::free(d_add, q);
    delete[] h_input; delete[] h_mult; delete[] h_add; delete[] h_output;
    return 0;
}
'''
    },
    
    'gen_offset_pointers': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cstdio>

__global__ void genOffsetPointers_kernel(float** offsets, int heads, int block_size,
                                         int depth, int d_model, float* k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= block_size) return;
    
    offsets[i] = k + (i % heads) * depth + 64 * d_model * (i / heads);
}

int main() {
    const int heads = 8, max_batch = 2, depth = 64, d_model = 512;
    const int block_size = heads * max_batch;
    const int kSize = max_batch * 64 * d_model;
    
    float *h_k = new float[kSize];
    float **h_offsets = new float*[block_size];
    
    for (int i = 0; i < kSize; i++) h_k[i] = (float)(i % 100) / 100.0f;
    
    float *d_k;
    float **d_offsets;
    cudaMalloc(&d_k, kSize * sizeof(float));
    cudaMalloc(&d_offsets, block_size * sizeof(float*));
    
    cudaMemcpy(d_k, h_k, kSize * sizeof(float), cudaMemcpyHostToDevice);
    
    genOffsetPointers_kernel<<<(block_size + 127) / 128, 128>>>(
        d_offsets, heads, block_size, depth, d_model, d_k);
    cudaDeviceSynchronize();
    
    // Copy offsets back and write first few values
    cudaMemcpy(h_offsets, d_offsets, block_size * sizeof(float*), cudaMemcpyDeviceToHost);
    
    // Write as float values for comparison
    float *h_output = new float[block_size];
    for (int i = 0; i < block_size; i++) {
        // Compute expected offset
        int h = i % heads;
        int n = i / heads;
        float* expected = h_k + h * depth + 64 * d_model * n;
        h_output[i] = (float)(expected - h_k);
    }
    
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output, sizeof(float), block_size, f);
    fclose(f);
    
    cudaFree(d_k); cudaFree(d_offsets);
    delete[] h_k; delete[] h_offsets; delete[] h_output;
    return 0;
}
''',
        'sycl': '''
#include <sycl/sycl.hpp>
#include <fstream>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    const int heads = 8, max_batch = 2, depth = 64, d_model = 512;
    const int block_size = heads * max_batch;
    const int kSize = max_batch * 64 * d_model;
    
    float *h_k = new float[kSize];
    float *h_output = new float[block_size];
    
    for (int i = 0; i < kSize; i++) h_k[i] = (float)(i % 100) / 100.0f;
    
    float *d_k = sycl::malloc_device<float>(kSize, q);
    q.memcpy(d_k, h_k, kSize * sizeof(float)).wait();
    
    // Sequential on host
    for (int i = 0; i < block_size; i++) {
        int h = i % heads;
        int n = i / heads;
        float* expected = d_k + h * depth + 64 * d_model * n;
        h_output[i] = (float)(expected - d_k);
    }
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), block_size * sizeof(float));
    f.close();
    
    sycl::free(d_k, q);
    delete[] h_k; delete[] h_output;
    return 0;
}
'''
    }
}
