#!/usr/bin/env python3
"""
Quick fixes for failed harnesses - Batch 2 and 3 SYCL fixes
"""

FIXES = {
    # Fix 1: global_avg_pool - was accessing device memory directly
    'global_avg_pool': {
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
    
    // Proper SYCL kernel
    q.parallel_for(sycl::range<1>(N * C), [=](sycl::id<1> idx) {
        int nc = idx[0];
        int n = nc / C;
        int c = nc % C;
        
        float sum = 0;
        for (int hw = 0; hw < 64; hw++) {
            sum += d_input[n * C * 64 + c * 64 + hw];
        }
        d_output[nc] = sum / 64.0f;
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
    
    # Fix 2: global_avg_pool_nhwc_fp16
    'global_avg_pool_nhwc_fp16': {
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
    
    q.parallel_for(sycl::range<1>(N * C), [=](sycl::id<1> idx) {
        int nc = idx[0];
        int n = nc / C;
        int c = nc % C;
        
        float sum = 0;
        for (int hw = 0; hw < 64; hw++) {
            int idx_in = n * 64 * C + hw * C + c;
            sum += (float)d_input[idx_in];
        }
        d_output[nc] = sum / 64.0f;
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
    
    # Fix 3: softmax
    'softmax': {
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
    
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        int n = idx[0];
        
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
            d_output[n * C + c] = expf(d_input[n * C + c] - maxval) / sum;
        }
    }).wait();
    
    q.memcpy(h_output, d_output, totalSize * sizeof(float)).wait();
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), totalSize * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_output, q);
    delete[] h_input; delete[] h_output;
    return 0;
}
'''
    },
    
    # Fix 4: preprocess_attention_body
    'preprocess_attention_body': {
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
    float *d_encoding = sycl::malloc_device<float>(64 * 64, q);
    float *d_output = sycl::malloc_device<float>(outputSize, q);
    
    q.memcpy(d_input, h_input, inputSize * sizeof(float)).wait();
    q.memcpy(d_encoding, h_encoding, 64 * 64 * sizeof(float)).wait();
    
    q.parallel_for(sycl::range<2>(N, 64), [=](sycl::id<2> idx) {
        int n = idx[0];
        int hw = idx[1];
        
        for (int c = 0; c < outputC; c++) {
            float op;
            if (c >= input_size) {
                op = d_encoding[64 * hw + (c - input_size)];
            } else {
                op = d_input[n * input_size * 64 + c * 64 + hw];
            }
            d_output[n * 64 * outputC + hw * outputC + c] = op;
        }
    }).wait();
    
    q.memcpy(h_output, d_output, outputSize * sizeof(float)).wait();
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), outputSize * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_encoding, q); sycl::free(d_output, q);
    delete[] h_input; delete[] h_encoding; delete[] h_output;
    return 0;
}
'''
    },
    
    # Fix 5: input_gating
    'input_gating': {
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
    float *d_output = sycl::malloc_device<float>(size, q);
    
    q.memcpy(d_input, h_input, size * sizeof(float)).wait();
    q.memcpy(d_mult, h_mult, size * sizeof(float)).wait();
    q.memcpy(d_add, h_add, size * sizeof(float)).wait();
    
    q.parallel_for(sycl::range<1>(N * HW * C), [=](sycl::id<1> idx) {
        int i = idx[0];
        int n = i / (HW * C);
        int hwc = i % (HW * C);
        int y = hwc / C;
        int x = hwc % C;
        int idxT = x * HW + y;
        
        d_output[i] = d_input[i] * d_mult[idxT] + d_add[idxT];
    }).wait();
    
    q.memcpy(h_output, d_output, size * sizeof(float)).wait();
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), size * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_mult, q); sycl::free(d_add, q); sycl::free(d_output, q);
    delete[] h_input; delete[] h_mult; delete[] h_add; delete[] h_output;
    return 0;
}
'''
    },
    
    # Fix 6: softmax_opt_64
    'softmax_opt_64': {
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
    float *d_output = sycl::malloc_device<float>(totalSize, q);
    
    q.memcpy(d_input, h_input, totalSize * sizeof(float)).wait();
    
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        int n = idx[0];
        
        float maxval = d_input[n * C];
        for (int c = 1; c < C; c++) maxval = fmaxf(maxval, d_input[n * C + c]);
        
        float sum = 0;
        for (int c = 0; c < C; c++) sum += expf(d_input[n * C + c] - maxval);
        
        for (int c = 0; c < C; c++) {
            d_output[n * C + c] = expf(d_input[n * C + c] - maxval) / sum;
        }
    }).wait();
    
    q.memcpy(h_output, d_output, totalSize * sizeof(float)).wait();
    
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

if __name__ == '__main__':
    print("Quick fixes defined for failed harnesses")
    print("\nKernels with fixes:")
    for k in FIXES.keys():
        print(f"  - {k}")
