/*
  Test CUDA Kernel - Vector Addition
  Simple test to verify build tools functionality
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace test {

// Simple vector addition kernel for testing
template <typename T>
__global__ void vectorAdd_kernel(T* c, const T* a, const T* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// Wrapper function
template <typename T>
void vectorAdd(T* c, const T* a, const T* b, int n, cudaStream_t stream) {
  const int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  
  vectorAdd_kernel<<<numBlocks, blockSize, 0, stream>>>(c, a, b, n);
}

// Explicit instantiations
template void vectorAdd<float>(float* c, const float* a, const float* b, 
                               int n, cudaStream_t stream);
template void vectorAdd<half>(half* c, const half* a, const half* b, 
                              int n, cudaStream_t stream);

} // namespace test
