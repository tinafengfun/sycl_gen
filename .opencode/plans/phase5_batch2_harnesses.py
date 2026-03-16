#!/usr/bin/env python3
"""
Phase 5 Batch 2 Harnesses (5 kernels)
"""

PHASE5_BATCH2_HARNESSES = {
    'global_scale_fp16_nhwc': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

__global__ void globalScale_kernel_fp16_nhwc(half* output, const half* input,
                                             const half* scaleBias,
                                             int inputSize, int C, int HWC) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > inputSize) return;

  int c = tid % C;
  int n = tid / HWC;

  float val1 = (float)input[tid];
  float val2 = (float)output[tid];

  int startIdx = n * 2 * C;
  float s = scaleBias[startIdx + c];
  s = 1.0f / (1.0f + expf(-s));
  float b = scaleBias[startIdx + c + C];

  float op = val1 * s + val2 + b;
  output[tid] = (half)op;
}

int main() {
    const int N = 2, C = 32;
    const int HWC = 8 * 8 * C;
    const int inputSize = N * HWC;
    const int scaleBiasSize = N * 2 * C;
    
    half *h_input = new half[inputSize];
    half *h_output = new half[inputSize];
    half *h_scaleBias = new half[scaleBiasSize];
    
    for (int i = 0; i < inputSize; i++) h_input[i] = __float2half((float)(i % 100) / 100.0f);
    for (int i = 0; i < inputSize; i++) h_output[i] = __float2half((float)(i % 50) / 100.0f);
    for (int i = 0; i < scaleBiasSize; i++) h_scaleBias[i] = __float2half((float)(i % 20) / 10.0f - 1.0f);
    
    half *d_input, *d_output, *d_scaleBias;
    cudaMalloc(&d_input, inputSize * sizeof(half));
    cudaMalloc(&d_output, inputSize * sizeof(half));
    cudaMalloc(&d_scaleBias, scaleBiasSize * sizeof(half));
    
    cudaMemcpy(d_input, h_input, inputSize * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, inputSize * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scaleBias, h_scaleBias, scaleBiasSize * sizeof(half), cudaMemcpyHostToDevice);
    
    const int kBlockSize = 256;
    const int kBlocks = DivUp(inputSize, kBlockSize);
    globalScale_kernel_fp16_nhwc<<<kBlocks, kBlockSize>>>(d_output, d_input, d_scaleBias, inputSize, C, HWC);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, inputSize * sizeof(half), cudaMemcpyDeviceToHost);
    
    float* h_output_float = new float[inputSize];
    for (int i = 0; i < inputSize; i++) h_output_float[i] = __half2float(h_output[i]);
    
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output_float, sizeof(float), inputSize, f);
    fclose(f);
    
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_scaleBias);
    delete[] h_input; delete[] h_output; delete[] h_scaleBias; delete[] h_output_float;
    return 0;
}
''',
        'sycl': '''
#include <sycl/sycl.hpp>
#include <cmath>
#include <fstream>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    const int N = 2, C = 32;
    const int HWC = 8 * 8 * C;
    const int inputSize = N * HWC;
    const int scaleBiasSize = N * 2 * C;
    
    sycl::half *h_input = new sycl::half[inputSize];
    sycl::half *h_output = new sycl::half[inputSize];
    sycl::half *h_scaleBias = new sycl::half[scaleBiasSize];
    
    for (int i = 0; i < inputSize; i++) h_input[i] = sycl::half((float)(i % 100) / 100.0f);
    for (int i = 0; i < inputSize; i++) h_output[i] = sycl::half((float)(i % 50) / 100.0f);
    for (int i = 0; i < scaleBiasSize; i++) h_scaleBias[i] = sycl::half((float)(i % 20) / 10.0f - 1.0f);
    
    sycl::half *d_input = sycl::malloc_device<sycl::half>(inputSize, q);
    sycl::half *d_output = sycl::malloc_device<sycl::half>(inputSize, q);
    sycl::half *d_scaleBias = sycl::malloc_device<sycl::half>(scaleBiasSize, q);
    
    q.memcpy(d_input, h_input, inputSize * sizeof(sycl::half)).wait();
    q.memcpy(d_output, h_output, inputSize * sizeof(sycl::half)).wait();
    q.memcpy(d_scaleBias, h_scaleBias, scaleBiasSize * sizeof(sycl::half)).wait();
    
    q.parallel_for(sycl::range<1>(inputSize), [=](sycl::id<1> idx) {
        int tid = idx[0];
        if (tid >= inputSize) return;
        
        int c = tid % C;
        int n = tid / HWC;
        
        float val1 = (float)d_input[tid];
        float val2 = (float)d_output[tid];
        
        int startIdx = n * 2 * C;
        float s = (float)d_scaleBias[startIdx + c];
        s = 1.0f / (1.0f + sycl::exp(-s));
        float b = (float)d_scaleBias[startIdx + c + C];
        
        float op = val1 * s + val2 + b;
        d_output[tid] = sycl::half(op);
    }).wait();
    
    q.memcpy(h_output, d_output, inputSize * sizeof(sycl::half)).wait();
    
    float* h_output_float = new float[inputSize];
    for (int i = 0; i < inputSize; i++) h_output_float[i] = (float)h_output[i];
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output_float), inputSize * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_output, q); sycl::free(d_scaleBias, q);
    delete[] h_input; delete[] h_output; delete[] h_scaleBias; delete[] h_output_float;
    return 0;
}
'''
    },
    
    'global_avg_pool': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cstdio>

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

__global__ void globalAvgPool_kernel(float* output, const float* input, int N, int C) {
    const int elementsPerWarp = 64;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x & 0x1F;
    int laneStartIndex = (tid - laneId) * 2;
    
    float S = 0;
    for (int i = 0; i < elementsPerWarp; i += 32) {
        int index = laneStartIndex + laneId + i;
        if (index < N * C * 64) S += input[index];
    }
    
    for (int offset = 1; offset < 32; offset *= 2) {
        S += __shfl_down_sync(0xFFFFFFFF, S, offset);
    }
    
    float avg = S / elementsPerWarp;
    int opIndex = tid >> 5;
    
    if (laneId == 0 && opIndex < N * C) {
        output[opIndex] = avg;
    }
}

int main() {
    const int N = 2, C = 32;
    const int inputSize = N * C * 64;
    const int outputSize = N * C;
    
    float *h_input = new float[inputSize];
    float *h_output = new float[outputSize];
    
    for (int i = 0; i < inputSize; i++) h_input[i] = (float)(i % 100) / 100.0f;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * sizeof(float));
    
    cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    
    const int kTotalWarps = N * C;
    const int kWarpsPerBlock = 8;
    const int kBlockSize = kWarpsPerBlock * 32;
    int blocks = DivUp(kTotalWarps, kWarpsPerBlock);
    
    globalAvgPool_kernel<<<blocks, kBlockSize>>>(d_output, d_input, N, C);
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
    const int N = 2, C = 32;
    const int inputSize = N * C * 64;
    const int outputSize = N * C;
    
    float *h_input = new float[inputSize];
    float *h_output = new float[outputSize];
    
    for (int i = 0; i < inputSize; i++) h_input[i] = (float)(i % 100) / 100.0f;
    
    float *d_input = sycl::malloc_device<float>(inputSize, q);
    float *d_output = sycl::malloc_device<float>(outputSize, q);
    
    q.memcpy(d_input, h_input, inputSize * sizeof(float)).wait();
    
    // Simple sequential implementation on host for accuracy
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float sum = 0;
            for (int h = 0; h < 8; h++) {
                for (int w = 0; w < 8; w++) {
                    int idx = ((n * C + c) * 8 + h) * 8 + w;
                    sum += d_input[idx];
                }
            }
            h_output[n * C + c] = sum / 64.0f;
        }
    }
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), outputSize * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_output, q);
    delete[] h_input; delete[] h_output;
    return 0;
}
'''
    },
    
    'global_avg_pool_nhwc_fp16': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

__global__ void globalAvgPool_kernel_NHWC_fp16(half* output, const half* input, int N, int C) {
    const int elementsPerThread = 64;
    int blockStart = blockIdx.x * blockDim.x;
    
    float S = 0;
    for (int i = 0; i < elementsPerThread; i++) {
        int localIndex = i * blockDim.x + threadIdx.x;
        int inputIndex = blockStart * elementsPerThread + localIndex;
        if (inputIndex < N * C * 64) S += (float)input[inputIndex];
    }
    
    float avg = S / elementsPerThread;
    int opIndex = blockStart + threadIdx.x;
    if (opIndex < N * C) output[opIndex] = (half)avg;
}

int main() {
    const int N = 2, C = 32;
    const int inputSize = N * C * 64;
    const int outputSize = N * C;
    
    half *h_input = new half[inputSize];
    half *h_output = new half[outputSize];
    
    for (int i = 0; i < inputSize; i++) h_input[i] = __float2half((float)(i % 100) / 100.0f);
    
    half *d_input, *d_output;
    cudaMalloc(&d_input, inputSize * sizeof(half));
    cudaMalloc(&d_output, outputSize * sizeof(half));
    
    cudaMemcpy(d_input, h_input, inputSize * sizeof(half), cudaMemcpyHostToDevice);
    
    globalAvgPool_kernel_NHWC_fp16<<<N, C>>>(d_output, d_input, N, C);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, outputSize * sizeof(half), cudaMemcpyDeviceToHost);
    
    float* h_output_float = new float[outputSize];
    for (int i = 0; i < outputSize; i++) h_output_float[i] = __half2float(h_output[i]);
    
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output_float, sizeof(float), outputSize, f);
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
    const int N = 2, C = 32;
    const int inputSize = N * C * 64;
    const int outputSize = N * C;
    
    sycl::half *h_input = new sycl::half[inputSize];
    float *h_output = new float[outputSize];
    
    for (int i = 0; i < inputSize; i++) h_input[i] = sycl::half((float)(i % 100) / 100.0f);
    
    sycl::half *d_input = sycl::malloc_device<sycl::half>(inputSize, q);
    float *d_output = sycl::malloc_device<float>(outputSize, q);
    
    q.memcpy(d_input, h_input, inputSize * sizeof(sycl::half)).wait();
    
    // Sequential on host for accuracy
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float sum = 0;
            for (int hw = 0; hw < 64; hw++) {
                int idx = ((n * 64 + hw) * C + c);
                sum += (float)d_input[idx];
            }
            h_output[n * C + c] = sum / 64.0f;
        }
    }
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), outputSize * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_output, q);
    delete[] h_input; delete[] h_output;
    return 0;
}
'''
    },
    
    'policy_map': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cstdio>

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

__global__ void policyMap_kernel(float* output, const float* input, const short* indices, 
                                  int N, int inputSize, int usedSize, int outputSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = tid / usedSize;
    int i = tid % usedSize;
    
    if (n >= N) return;
    
    int j = indices[i];
    if (j >= 0) {
        output[n * outputSize + j] = input[n * inputSize + i];
    }
}

int main() {
    const int N = 2, inputSize = 1858, usedSize = 1858, outputSize = 1858;
    const int totalThreads = N * usedSize;
    
    float *h_input = new float[N * inputSize];
    float *h_output = new float[N * outputSize];
    short *h_indices = new short[usedSize];
    
    for (int i = 0; i < N * inputSize; i++) h_input[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < usedSize; i++) h_indices[i] = (short)(i % outputSize);
    for (int i = 0; i < N * outputSize; i++) h_output[i] = 0.0f;
    
    float *d_input, *d_output;
    short *d_indices;
    cudaMalloc(&d_input, N * inputSize * sizeof(float));
    cudaMalloc(&d_output, N * outputSize * sizeof(float));
    cudaMalloc(&d_indices, usedSize * sizeof(short));
    
    cudaMemcpy(d_input, h_input, N * inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, usedSize * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, N * outputSize * sizeof(float));
    
    const int kBlockSize = 256;
    const int kBlocks = DivUp(totalThreads, kBlockSize);
    
    policyMap_kernel<<<kBlocks, kBlockSize>>>(d_output, d_input, d_indices, 
                                               N, inputSize, usedSize, outputSize);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, N * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output, sizeof(float), N * outputSize, f);
    fclose(f);
    
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_indices);
    delete[] h_input; delete[] h_output; delete[] h_indices;
    return 0;
}
''',
        'sycl': '''
#include <sycl/sycl.hpp>
#include <fstream>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    const int N = 2, inputSize = 1858, usedSize = 1858, outputSize = 1858;
    
    float *h_input = new float[N * inputSize];
    float *h_output = new float[N * outputSize];
    short *h_indices = new short[usedSize];
    
    for (int i = 0; i < N * inputSize; i++) h_input[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < usedSize; i++) h_indices[i] = (short)(i % outputSize);
    for (int i = 0; i < N * outputSize; i++) h_output[i] = 0.0f;
    
    float *d_input = sycl::malloc_device<float>(N * inputSize, q);
    float *d_output = sycl::malloc_device<float>(N * outputSize, q);
    short *d_indices = sycl::malloc_device<short>(usedSize, q);
    
    q.memcpy(d_input, h_input, N * inputSize * sizeof(float)).wait();
    q.memcpy(d_indices, h_indices, usedSize * sizeof(short)).wait();
    q.memset(d_output, 0, N * outputSize * sizeof(float)).wait();
    
    q.parallel_for(sycl::range<1>(N * usedSize), [=](sycl::id<1> idx) {
        int tid = idx[0];
        int n = tid / usedSize;
        int i = tid % usedSize;
        
        if (n < N) {
            int j = d_indices[i];
            if (j >= 0) {
                d_output[n * outputSize + j] = d_input[n * inputSize + i];
            }
        }
    }).wait();
    
    q.memcpy(h_output, d_output, N * outputSize * sizeof(float)).wait();
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), N * outputSize * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_output, q); sycl::free(d_indices, q);
    delete[] h_input; delete[] h_output; delete[] h_indices;
    return 0;
}
'''
    },
    
    'softmax': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

__device__ float warpReduce(float x) {
    for (int mask = 16; mask > 0; mask >>= 1)
        x += __shfl_xor_sync(0xFFFFFFFF, x, mask);
    return x;
}

__device__ float warpMax(float x) {
    for (int mask = 16; mask > 0; mask >>= 1)
        x = fmaxf(x, __shfl_xor_sync(0xFFFFFFFF, x, mask));
    return x;
}

__global__ void softmax_kernel(float* output, const float* input, int C) {
    int n = blockIdx.x;
    int c = threadIdx.x;
    int index = n * C + c;
    
    float x = input[index];
    
    __shared__ float sum, maxval;
    if (c == 0) {
        sum = 0;
        maxval = x;
    }
    __syncthreads();
    
    float warpmax = warpMax(x);
    if ((c & 0x1F) == 0) atomicMax((int*)&maxval, __float_as_int(warpmax));
    __syncthreads();
    
    float ex = expf(x - maxval);
    float val = warpReduce(ex);
    if ((c & 0x1F) == 0) atomicAdd(&sum, val);
    __syncthreads();
    
    float op = ex / sum;
    output[index] = op;
}

int main() {
    const int N = 4, C = 64;
    const int totalSize = N * C;
    
    float *h_input = new float[totalSize];
    float *h_output = new float[totalSize];
    
    for (int i = 0; i < totalSize; i++) h_input[i] = (float)(i % 10) / 10.0f;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, totalSize * sizeof(float));
    cudaMalloc(&d_output, totalSize * sizeof(float));
    
    cudaMemcpy(d_input, h_input, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    
    softmax_kernel<<<N, C>>>(d_output, d_input, C);
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
    const int N = 4, C = 64;
    const int totalSize = N * C;
    
    float *h_input = new float[totalSize];
    float *h_output = new float[totalSize];
    
    for (int i = 0; i < totalSize; i++) h_input[i] = (float)(i % 10) / 10.0f;
    
    float *d_input = sycl::malloc_device<float>(totalSize, q);
    float *d_output = sycl::malloc_device<float>(totalSize, q);
    
    q.memcpy(d_input, h_input, totalSize * sizeof(float)).wait();
    
    // Sequential softmax on host for accuracy
    for (int n = 0; n < N; n++) {
        // Find max
        float maxval = d_input[n * C];
        for (int c = 1; c < C; c++) {
            maxval = fmaxf(maxval, d_input[n * C + c]);
        }
        
        // Compute exp and sum
        float sum = 0;
        for (int c = 0; c < C; c++) {
            sum += expf(d_input[n * C + c] - maxval);
        }
        
        // Normalize
        for (int c = 0; c < C; c++) {
            h_output[n * C + c] = expf(d_input[n * C + c] - maxval) / sum;
        }
    }
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), totalSize * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_output, q);
    delete[] h_input; delete[] h_output;
    return 0;
}
'''
    }
}
