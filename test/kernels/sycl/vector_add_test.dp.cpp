/*
  Test SYCL Kernel - Vector Addition
  Simple test to verify build tools functionality
*/

#include <sycl/sycl.hpp>

namespace test {

// Simple vector addition kernel for testing
template <typename T>
void vectorAdd(T* c, const T* a, const T* b, int n, sycl::queue& queue) {
  queue.parallel_for(
    sycl::range<1>(n),
    [=](sycl::id<1> i) {
      c[i] = a[i] + b[i];
    }
  );
}

// Explicit instantiations
template void vectorAdd<float>(float* c, const float* a, const float* b, 
                               int n, sycl::queue& queue);

} // namespace test
