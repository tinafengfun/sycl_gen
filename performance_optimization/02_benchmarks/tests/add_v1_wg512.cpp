#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

enum ActivationFunction { ACTIVATION_NONE };

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void addVectors_kernel(T* c, const T* a, const T* b, int size,
                       ActivationFunction activation,
                       const sycl::nd_item<1> &item) {
    int i = item.get_global_id(0);
    if (i < size) {
        float aVal = a ? (float)a[i] : 0.0f;
        float bVal = b ? (float)b[i] : 0.0f;
        c[i] = (T)(aVal + bVal);
    }
}

template <typename T>
void addVectors(T* c, T* a, T* b, int size,
                ActivationFunction activation, sycl::queue &queue) {
    constexpr int kBlockSize = 512;
    int blocks = DivUp(size, kBlockSize);
    
    queue.parallel_for(
        sycl::nd_range<1>(blocks * kBlockSize, kBlockSize),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            addVectors_kernel(c, a, b, size, activation, item);
        }
    );
}

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    
    std::ofstream csv("results_v1_wg512.csv");
    csv << "Size,Time_ms,GFLOPS,Bandwidth_GB/s\n";
    
    std::cout << "=== Add Vectors V1 (WG=512) ===\n";
    std::cout << "GPU: " << q.get_device().get_info<sycl::info::device::name>() << "\n\n";
    
    std::vector<int> sizes = {256, 512, 1024, 4096, 16384};
    
    for (int n : sizes) {
        float *d_a = sycl::malloc_device<float>(n, q);
        float *d_b = sycl::malloc_device<float>(n, q);
        float *d_c = sycl::malloc_device<float>(n, q);
        
        std::vector<float> h_a(n), h_b(n);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < n; ++i) {
            h_a[i] = dist(gen);
            h_b[i] = dist(gen);
        }
        
        q.memcpy(d_a, h_a.data(), n * sizeof(float)).wait();
        q.memcpy(d_b, h_b.data(), n * sizeof(float)).wait();
        
        // Warmup
        for (int i = 0; i < 10; ++i) {
            addVectors(d_c, d_a, d_b, n, ACTIVATION_NONE, q);
            q.wait();
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            addVectors(d_c, d_a, d_b, n, ACTIVATION_NONE, q);
            q.wait();
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        double avg_time_ms = (elapsed.count() * 1000.0) / 100.0;
        
        double flops = n * 1.0;
        double bytes = n * 3 * sizeof(float);
        double gflops = (flops / (avg_time_ms * 1e-3)) / 1e9;
        double bandwidth = (bytes / (avg_time_ms * 1e-3)) / 1e9;
        
        std::cout << "N=" << n << "\tTime: " << avg_time_ms << " ms\t"
                  << "GFLOPS: " << gflops << "\tBandwidth: " << bandwidth << " GB/s\n";
        
        csv << n << "," << avg_time_ms << "," << gflops << "," << bandwidth << "\n";
        
        sycl::free(d_a, q);
        sycl::free(d_b, q);
        sycl::free(d_c, q);
    }
    
    csv.close();
    return 0;
}