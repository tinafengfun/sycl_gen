/*
 * Simple vector addition test for SYCL
 * SYCL version of the vector add test
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <sycl/sycl.hpp>

int main() {
    const int N = 1024;
    
    // Host arrays
    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    std::vector<float> h_c(N, 0.0f);
    
    // Create SYCL queue
    sycl::queue q;
    
    // Device arrays
    float* d_a = sycl::malloc_device<float>(N, q);
    float* d_b = sycl::malloc_device<float>(N, q);
    float* d_c = sycl::malloc_device<float>(N, q);
    
    // Copy to device
    q.memcpy(d_a, h_a.data(), N * sizeof(float)).wait();
    q.memcpy(d_b, h_b.data(), N * sizeof(float)).wait();
    
    // Launch kernel
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
        d_c[i] = d_a[i] + d_b[i];
    }).wait();
    
    // Copy back
    q.memcpy(h_c.data(), d_c, N * sizeof(float)).wait();
    
    // Verify
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (std::abs(h_c[i] - 3.0f) > 1e-5) {
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "✅ SYCL test passed" << std::endl;
    } else {
        std::cout << "❌ SYCL test failed" << std::endl;
    }
    
    // Cleanup
    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);
    
    // Save output
    FILE* fp = fopen("reference_data/vector_add_sycl_output.bin", "wb");
    fwrite(h_c.data(), sizeof(float), N, fp);
    fclose(fp);
    
    return success ? 0 : 1;
}
