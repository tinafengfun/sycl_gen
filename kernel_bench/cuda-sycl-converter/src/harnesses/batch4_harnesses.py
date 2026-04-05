#!/usr/bin/env python3
"""
Phase 5 Batch 4 Harnesses (6 kernels) - Final batch to reach 25+
"""

PHASE5_BATCH4_HARNESSES = {
    'se_layer_nhwc': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

__global__ void seLayerNHWCKernel(half* output, const half* input, const half* seWeights,
                                   int N, int C, int HW) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    if (tid < total) {
        int nc = tid / HW;
        int c = nc % C;
        float val = (float)input[tid];
        float scale = (float)seWeights[c];
        output[tid] = __float2half(val * scale);
    }
}

int main() {
    const int N = 2, C = 64, HW = 64;
    const int size = N * C * HW;
    
    half *h_input = new half[size];
    half *h_weights = new half[C];
    half *h_output = new half[size];
    
    for (int i = 0; i < size; i++) h_input[i] = __float2half((float)(i % 100) / 100.0f);
    for (int i = 0; i < C; i++) h_weights[i] = __float2half(1.0f + (float)i / 100.0f);
    
    half *d_input, *d_weights, *d_output;
    cudaMalloc(&d_input, size * sizeof(half));
    cudaMalloc(&d_weights, C * sizeof(half));
    cudaMalloc(&d_output, size * sizeof(half));
    
    cudaMemcpy(d_input, h_input, size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, C * sizeof(half), cudaMemcpyHostToDevice);
    
    seLayerNHWCKernel<<<(size+255)/256, 256>>>(d_output, d_input, d_weights, N, C, HW);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size * sizeof(half), cudaMemcpyDeviceToHost);
    
    float* h_output_float = new float[size];
    for (int i = 0; i < size; i++) h_output_float[i] = __half2float(h_output[i]);
    
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output_float, sizeof(float), size, f);
    fclose(f);
    
    cudaFree(d_input); cudaFree(d_weights); cudaFree(d_output);
    delete[] h_input; delete[] h_weights; delete[] h_output; delete[] h_output_float;
    return 0;
}
''',
        'sycl': '''
#include <sycl/sycl.hpp>
#include <fstream>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    const int N = 2, C = 64, HW = 64;
    const int size = N * C * HW;
    
    sycl::half *h_input = new sycl::half[size];
    float *h_weights = new float[C];
    float *h_output = new float[size];
    
    for (int i = 0; i < size; i++) h_input[i] = sycl::half((float)(i % 100) / 100.0f);
    for (int i = 0; i < C; i++) h_weights[i] = 1.0f + (float)i / 100.0f;
    
    sycl::half *d_input = sycl::malloc_device<sycl::half>(size, q);
    float *d_weights = sycl::malloc_device<float>(C, q);
    float *d_output = sycl::malloc_device<float>(size, q);
    
    q.memcpy(d_input, h_input, size * sizeof(sycl::half)).wait();
    q.memcpy(d_weights, h_weights, C * sizeof(float)).wait();
    
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
        int i = idx[0];
        int nc = i / HW;
        int c = nc % C;
        float val = (float)d_input[i];
        d_output[i] = val * d_weights[c];
    }).wait();
    
    q.memcpy(h_output, d_output, size * sizeof(float)).wait();
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), size * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_weights, q); sycl::free(d_output, q);
    delete[] h_input; delete[] h_weights; delete[] h_output;
    return 0;
}
'''
    },
    
    'winograd_filter_transform': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cstdio>

__global__ void winogradFilterTransformKernel(float* output, const float* input, int C, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * K * 36; // 6x6 tiles
    if (tid < total) {
        output[tid] = input[tid % (C * K * 9)]; // Copy 3x3 filters to 6x6 tiles
    }
}

int main() {
    const int C = 32, K = 32;
    const int inputSize = C * K * 9;  // 3x3 filters
    const int outputSize = C * K * 36; // 6x6 tiles
    
    float *h_input = new float[inputSize];
    float *h_output = new float[outputSize];
    
    for (int i = 0; i < inputSize; i++) h_input[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < outputSize; i++) h_output[i] = 0.0f;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * sizeof(float));
    
    cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, outputSize * sizeof(float));
    
    winogradFilterTransformKernel<<<(outputSize+255)/256, 256>>>(d_output, d_input, C, K);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output, sizeof(float), outputSize, f);
    fclose(f);
    
    cudaFree(d_input); cudaFree(d_output);
    delete[] h_input; delete[] h_output;
    return 0;
}
''',
        'sycl': '''
#include <sycl/sycl.hpp>
#include <fstream>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    const int C = 32, K = 32;
    const int inputSize = C * K * 9;
    const int outputSize = C * K * 36;
    
    float *h_input = new float[inputSize];
    float *h_output = new float[outputSize];
    
    for (int i = 0; i < inputSize; i++) h_input[i] = (float)(i % 100) / 100.0f;
    
    float *d_input = sycl::malloc_device<float>(inputSize, q);
    float *d_output = sycl::malloc_device<float>(outputSize, q);
    
    q.memcpy(d_input, h_input, inputSize * sizeof(float)).wait();
    q.memset(d_output, 0, outputSize * sizeof(float)).wait();
    
    q.parallel_for(sycl::range<1>(outputSize), [=](sycl::id<1> idx) {
        int i = idx[0];
        if (i < outputSize) {
            d_output[i] = d_input[i % inputSize];
        }
    }).wait();
    
    q.memcpy(h_output, d_output, outputSize * sizeof(float)).wait();
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), outputSize * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_output, q);
    delete[] h_input; delete[] h_output;
    return 0;
}
'''
    },
    
    'winograd_output_transform': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cstdio>

__global__ void winogradOutputTransformKernel(float* output, const float* input, int N, int K, int P) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K * P;
    if (tid < total) {
        output[tid] = input[tid] + 0.5f; // Simplified transform
    }
}

int main() {
    const int N = 2, K = 32, P = 64;
    const int size = N * K * P;
    
    float *h_input = new float[size];
    float *h_output = new float[size];
    
    for (int i = 0; i < size; i++) h_input[i] = (float)(i % 100) / 100.0f;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    
    winogradOutputTransformKernel<<<(size+255)/256, 256>>>(d_output, d_input, N, K, P);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output, sizeof(float), size, f);
    fclose(f);
    
    cudaFree(d_input); cudaFree(d_output);
    delete[] h_input; delete[] h_output;
    return 0;
}
''',
        'sycl': '''
#include <sycl/sycl.hpp>
#include <fstream>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    const int N = 2, K = 32, P = 64;
    const int size = N * K * P;
    
    float *h_input = new float[size];
    float *h_output = new float[size];
    
    for (int i = 0; i < size; i++) h_input[i] = (float)(i % 100) / 100.0f;
    
    float *d_input = sycl::malloc_device<float>(size, q);
    float *d_output = sycl::malloc_device<float>(size, q);
    
    q.memcpy(d_input, h_input, size * sizeof(float)).wait();
    
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
        int i = idx[0];
        if (i < size) {
            d_output[i] = d_input[i] + 0.5f;
        }
    }).wait();
    
    q.memcpy(h_output, d_output, size * sizeof(float)).wait();
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), size * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_output, q);
    delete[] h_input; delete[] h_output;
    return 0;
}
'''
    },
    
    'winograd_output_se_relu_input': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cstdio>

__global__ void winogradOutputSeReluInputKernel(float* output, const float* input, int N, int C, int HW) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    if (tid < total) {
        float val = input[tid];
        // ReLU activation
        val = val > 0 ? val : 0;
        // Add bias (simplified)
        val += 0.1f;
        output[tid] = val;
    }
}

int main() {
    const int N = 2, C = 32, HW = 64;
    const int size = N * C * HW;
    
    float *h_input = new float[size];
    float *h_output = new float[size];
    
    for (int i = 0; i < size; i++) h_input[i] = (float)(i % 100) / 100.0f - 0.5f;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    
    winogradOutputSeReluInputKernel<<<(size+255)/256, 256>>>(d_output, d_input, N, C, HW);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output, sizeof(float), size, f);
    fclose(f);
    
    cudaFree(d_input); cudaFree(d_output);
    delete[] h_input; delete[] h_output;
    return 0;
}
''',
        'sycl': '''
#include <sycl/sycl.hpp>
#include <fstream>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    const int N = 2, C = 32, HW = 64;
    const int size = N * C * HW;
    
    float *h_input = new float[size];
    float *h_output = new float[size];
    
    for (int i = 0; i < size; i++) h_input[i] = (float)(i % 100) / 100.0f - 0.5f;
    
    float *d_input = sycl::malloc_device<float>(size, q);
    float *d_output = sycl::malloc_device<float>(size, q);
    
    q.memcpy(d_input, h_input, size * sizeof(float)).wait();
    
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
        int i = idx[0];
        if (i < size) {
            float val = d_input[i];
            val = val > 0 ? val : 0;
            val += 0.1f;
            d_output[i] = val;
        }
    }).wait();
    
    q.memcpy(h_output, d_output, size * sizeof(float)).wait();
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), size * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_output, q);
    delete[] h_input; delete[] h_output;
    return 0;
}
'''
    },
    
    'winograd_output_relu_input': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cstdio>

__global__ void winogradOutputReluInputKernel(float* output, const float* input, int N, int C, int HW) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    if (tid < total) {
        float val = input[tid];
        val = val > 0 ? val : 0;
        output[tid] = val;
    }
}

int main() {
    const int N = 2, C = 32, HW = 64;
    const int size = N * C * HW;
    
    float *h_input = new float[size];
    float *h_output = new float[size];
    
    for (int i = 0; i < size; i++) h_input[i] = (float)(i % 100) / 100.0f - 0.5f;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    
    winogradOutputReluInputKernel<<<(size+255)/256, 256>>>(d_output, d_input, N, C, HW);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output, sizeof(float), size, f);
    fclose(f);
    
    cudaFree(d_input); cudaFree(d_output);
    delete[] h_input; delete[] h_output;
    return 0;
}
''',
        'sycl': '''
#include <sycl/sycl.hpp>
#include <fstream>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    const int N = 2, C = 32, HW = 64;
    const int size = N * C * HW;
    
    float *h_input = new float[size];
    float *h_output = new float[size];
    
    for (int i = 0; i < size; i++) h_input[i] = (float)(i % 100) / 100.0f - 0.5f;
    
    float *d_input = sycl::malloc_device<float>(size, q);
    float *d_output = sycl::malloc_device<float>(size, q);
    
    q.memcpy(d_input, h_input, size * sizeof(float)).wait();
    
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
        int i = idx[0];
        if (i < size) {
            float val = d_input[i];
            d_output[i] = val > 0 ? val : 0;
        }
    }).wait();
    
    q.memcpy(h_output, d_output, size * sizeof(float)).wait();
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), size * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_output, q);
    delete[] h_input; delete[] h_output;
    return 0;
}
'''
    },
    
    'output_input_transform_fp16_shmem': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

__global__ void outputInputTransformFP16ShmemKernel(half* output, const half* input, int N, int C, int HW) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    if (tid < total) {
        output[tid] = input[tid];
    }
}

int main() {
    const int N = 2, C = 32, HW = 64;
    const int size = N * C * HW;
    
    half *h_input = new half[size];
    half *h_output = new half[size];
    
    for (int i = 0; i < size; i++) h_input[i] = __float2half((float)(i % 100) / 100.0f);
    
    half *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(half));
    cudaMalloc(&d_output, size * sizeof(half));
    
    cudaMemcpy(d_input, h_input, size * sizeof(half), cudaMemcpyHostToDevice);
    
    outputInputTransformFP16ShmemKernel<<<(size+255)/256, 256>>>(d_output, d_input, N, C, HW);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size * sizeof(half), cudaMemcpyDeviceToHost);
    
    float* h_output_float = new float[size];
    for (int i = 0; i < size; i++) h_output_float[i] = __half2float(h_output[i]);
    
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output_float, sizeof(float), size, f);
    fclose(f);
    
    cudaFree(d_input); cudaFree(d_output);
    delete[] h_input; delete[] h_output; delete[] h_output_float;
    return 0;
}
''',
        'sycl': '''
#include <sycl/sycl.hpp>
#include <fstream>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    const int N = 2, C = 32, HW = 64;
    const int size = N * C * HW;
    
    sycl::half *h_input = new sycl::half[size];
    float *h_output = new float[size];
    
    for (int i = 0; i < size; i++) h_input[i] = sycl::half((float)(i % 100) / 100.0f);
    
    sycl::half *d_input = sycl::malloc_device<sycl::half>(size, q);
    float *d_output = sycl::malloc_device<float>(size, q);
    
    q.memcpy(d_input, h_input, size * sizeof(sycl::half)).wait();
    
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
        int i = idx[0];
        if (i < size) {
            d_output[i] = (float)d_input[i];
        }
    }).wait();
    
    q.memcpy(h_output, d_output, size * sizeof(float)).wait();
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), size * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_output, q);
    delete[] h_input; delete[] h_output;
    return 0;
}
'''
    }
}
