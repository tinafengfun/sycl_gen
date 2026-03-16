# ============================================================================
# PHASE 2: Improved Placeholder Harnesses
# Fixing add_bias_batched and global_scale
# ============================================================================

PHASE2_IMPROVED_HARNESSES = {
    
    # ============================================================================
    # TASK 2.1: IMPROVED - add_bias_batched (was using full batch_norm)
    # ============================================================================
    'add_bias_batched': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cstdio.h>

// IMPROVED: Pure bias addition (removed scale multiplication from batch_norm)
__global__ void addBiasBatchedKernel(float* output, const float* input, 
                                      const float* bias, int Batch, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Batch * N * C;
    
    if (idx < total) {
        int nc = idx / C;
        int c = nc % C;
        // FIXED: Only add bias, no scale multiplication
        output[idx] = input[idx] + bias[c];
    }
}

int main() {
    const int Batch = 2, N = 4, C = 32;
    const int size = Batch * N * C;
    
    float* h_input = new float[size];
    float* h_bias = new float[C];
    float* h_output = new float[size];
    
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)(i % 100) / 100.0f;
    }
    for (int i = 0; i < C; i++) {
        h_bias[i] = (float)i / 100.0f;
    }
    
    float *d_input, *d_bias, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_bias, C * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, C * sizeof(float), cudaMemcpyHostToDevice);
    
    addBiasBatchedKernel<<<(size + 255) / 256, 256>>>(
        d_output, d_input, d_bias, Batch, N, C);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output, sizeof(float), size, f);
    fclose(f);
    
    cudaFree(d_input); cudaFree(d_bias); cudaFree(d_output);
    delete[] h_input; delete[] h_bias; delete[] h_output;
    return 0;
}
''',
        'sycl': '''
#include <sycl/sycl.hpp>
#include <fstream>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    const int Batch = 2, N = 4, C = 32;
    const int size = Batch * N * C;
    
    float* h_input = new float[size];
    float* h_bias = new float[C];
    float* h_output = new float[size];
    
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)(i % 100) / 100.0f;
    }
    for (int i = 0; i < C; i++) {
        h_bias[i] = (float)i / 100.0f;
    }
    
    float* d_input = sycl::malloc_device<float>(size, q);
    float* d_bias = sycl::malloc_device<float>(C, q);
    float* d_output = sycl::malloc_device<float>(size, q);
    
    q.memcpy(d_input, h_input, size * sizeof(float)).wait();
    q.memcpy(d_bias, h_bias, C * sizeof(float)).wait();
    
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
        int i = idx[0];
        int nc = i / C;
        int c = nc % C;
        // FIXED: Only add bias, no scale
        d_output[i] = d_input[i] + d_bias[c];
    }).wait();
    
    q.memcpy(h_output, d_output, size * sizeof(float)).wait();
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), size * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_bias, q); sycl::free(d_output, q);
    delete[] h_input; delete[] h_bias; delete[] h_output;
    return 0;
}
'''
    },
    
    # ============================================================================
    # TASK 2.2: IMPROVED - global_scale (was using full batch_norm)
    # ============================================================================
    'global_scale': {
        'cuda': '''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio.h>
#include <cmath>

// IMPROVED: Pure global scaling (removed bias addition from batch_norm)
__global__ void globalScaleKernel(float* output, const float* input,
                                   const float* scaleBias, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int planeSize = 64;
    int total = N * C * planeSize;
    
    if (idx < total) {
        int nc = idx / planeSize;
        int n = nc / C;
        int c = nc % C;
        
        float val = input[idx];
        // FIXED: Only scale, no bias addition
        float s = scaleBias[n * 2 * C + c];
        
        // Apply sigmoid to scale (as in original lc0)
        s = 1.0f / (1.0f + expf(-s));
        
        output[idx] = val * s;
    }
}

int main() {
    const int N = 2, C = 32;
    const int planeSize = 64;
    const int inputSize = N * C * planeSize;
    const int scaleBiasSize = N * 2 * C;
    
    float* h_input = new float[inputSize];
    float* h_scaleBias = new float[scaleBiasSize];
    float* h_output = new float[inputSize];
    
    for (int i = 0; i < inputSize; i++) {
        h_input[i] = sinf(i * 0.01f) * 0.5f;
    }
    for (int i = 0; i < scaleBiasSize; i++) {
        h_scaleBias[i] = cosf(i * 0.03f) * 0.2f;
    }
    
    float *d_input, *d_scaleBias, *d_output;
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_scaleBias, scaleBiasSize * sizeof(float));
    cudaMalloc(&d_output, inputSize * sizeof(float));
    
    cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scaleBias, h_scaleBias, scaleBiasSize * sizeof(float), cudaMemcpyHostToDevice);
    
    globalScaleKernel<<<(inputSize + 255) / 256, 256>>>(
        d_output, d_input, d_scaleBias, N, C);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, inputSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    FILE* f = fopen("/workspace/output_cuda.bin", "wb");
    fwrite(h_output, sizeof(float), inputSize, f);
    fclose(f);
    
    cudaFree(d_input); cudaFree(d_scaleBias); cudaFree(d_output);
    delete[] h_input; delete[] h_scaleBias; delete[] h_output;
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
    const int planeSize = 64;
    const int inputSize = N * C * planeSize;
    const int scaleBiasSize = N * 2 * C;
    
    float* h_input = new float[inputSize];
    float* h_scaleBias = new float[scaleBiasSize];
    float* h_output = new float[inputSize];
    
    for (int i = 0; i < inputSize; i++) {
        h_input[i] = sycl::sin(i * 0.01f) * 0.5f;
    }
    for (int i = 0; i < scaleBiasSize; i++) {
        h_scaleBias[i] = sycl::cos(i * 0.03f) * 0.2f;
    }
    
    float* d_input = sycl::malloc_device<float>(inputSize, q);
    float* d_scaleBias = sycl::malloc_device<float>(scaleBiasSize, q);
    float* d_output = sycl::malloc_device<float>(inputSize, q);
    
    q.memcpy(d_input, h_input, inputSize * sizeof(float)).wait();
    q.memcpy(d_scaleBias, h_scaleBias, scaleBiasSize * sizeof(float)).wait();
    
    q.parallel_for(sycl::range<1>(inputSize), [=](sycl::id<1> idx) {
        int i = idx[0];
        int planeSize = 64;
        int nc = i / planeSize;
        int n = nc / C;
        int c = nc % C;
        
        float val = d_input[i];
        float s = d_scaleBias[n * 2 * C + c];
        
        // Apply sigmoid
        s = 1.0f / (1.0f + sycl::exp(-s));
        
        d_output[i] = val * s;
    }).wait();
    
    q.memcpy(h_output, d_output, inputSize * sizeof(float)).wait();
    
    std::ofstream f("/workspace/output_sycl.bin", std::ios::binary);
    f.write(reinterpret_cast<char*>(h_output), inputSize * sizeof(float));
    f.close();
    
    sycl::free(d_input, q); sycl::free(d_scaleBias, q); sycl::free(d_output, q);
    delete[] h_input; delete[] h_scaleBias; delete[] h_output;
    return 0;
}
'''
    }
}

# Phase 2 Summary
PHASE2_SUMMARY = {
    'improved_kernels': ['add_bias_batched', 'global_scale'],
    'improvement_type': 'Removed incorrect operations from batch_norm',
    'total_harnesses': 2,
    'lines_of_code': 180
}

print("=" * 80)
print("✅ Phase 2 Implementation Complete")
print("=" * 80)
print(f"\nImproved Kernels: {PHASE2_SUMMARY['improved_kernels']}")
print(f"Improvement: {PHASE2_SUMMARY['improvement_type']}")
print(f"Total Harnesses: {PHASE2_SUMMARY['total_harnesses']}")
print(f"Lines of Code: ~{PHASE2_SUMMARY['lines_of_code']}")
print("\n🚀 Ready for Phase 3: Create RealAccuracyTester class")
