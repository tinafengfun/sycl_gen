/*
 * Simple vector addition test for CUDA and SYCL
 * This is a simplified example to demonstrate the testing framework
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// Simple kernel for testing
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1024;
    const int bytes = N * sizeof(float);
    
    // Host arrays
    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    std::vector<float> h_c(N, 0.0f);
    
    // Device arrays
    float* d_a;
    float* d_b;
    float* d_c;
    
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy to device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // Copy back
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Verify
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (std::abs(h_c[i] - 3.0f) > 1e-5) {
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "✅ CUDA test passed" << std::endl;
    } else {
        std::cout << "❌ CUDA test failed" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Save output
    FILE* fp = fopen("reference_data/vector_add_cuda_output.bin", "wb");
    fwrite(h_c.data(), sizeof(float), N, fp);
    fclose(fp);
    
    return success ? 0 : 1;
}
