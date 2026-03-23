#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>

enum ActivationFunction { ACTIVATION_NONE };

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// V2: Sub-Group 16 (explicit)
template <typename T>
void addVectors_v2_kernel(T* c, const T* a, const T* b, int size,
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
void addVectors_v2(T* c, T* a, T* b, int size,
                   ActivationFunction activation, sycl::queue &queue) {
    constexpr int kBlockSize = 512;
    int blocks = DivUp(size, kBlockSize);
    
    queue.parallel_for(
        sycl::nd_range<1>(blocks * kBlockSize, kBlockSize),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            addVectors_v2_kernel(c, a, b, size, activation, item);
        }
    );
}

// V3: 4-Wide Vectorization
template <typename T>
void addVectors_v3_kernel(T* c, const T* a, const T* b, int size,
                          ActivationFunction activation,
                          const sycl::nd_item<1> &item) {
    int idx = item.get_global_id(0) * 4;
    
    if (idx + 4 <= size) {
        sycl::vec<float, 4> a_vec, b_vec, c_vec;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            a_vec[i] = a ? (float)a[idx + i] : 0.0f;
            b_vec[i] = b ? (float)b[idx + i] : 0.0f;
        }
        c_vec = a_vec + b_vec;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            c[idx + i] = (T)c_vec[i];
        }
    } else {
        for (int i = idx; i < size; ++i) {
            float aVal = a ? (float)a[i] : 0.0f;
            float bVal = b ? (float)b[i] : 0.0f;
            c[i] = (T)(aVal + bVal);
        }
    }
}

template <typename T>
void addVectors_v3(T* c, T* a, T* b, int size,
                   ActivationFunction activation, sycl::queue &queue) {
    constexpr int kBlockSize = 512;
    int blocks = DivUp(size / 4, kBlockSize);
    
    queue.parallel_for(
        sycl::nd_range<1>(blocks * kBlockSize, kBlockSize),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            addVectors_v3_kernel(c, a, b, size, activation, item);
        }
    );
}

// V4: Large GRF Mode (256KB)
template <typename T>
void addVectors_v4_kernel(T* c, const T* a, const T* b, int size,
                          ActivationFunction activation,
                          const sycl::nd_item<1> &item) {
    int idx = item.get_global_id(0) * 4;
    
    if (idx + 4 <= size) {
        sycl::vec<float, 4> a_vec, b_vec, c_vec;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            a_vec[i] = a ? (float)a[idx + i] : 0.0f;
            b_vec[i] = b ? (float)b[idx + i] : 0.0f;
        }
        c_vec = a_vec + b_vec;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            c[idx + i] = (T)c_vec[i];
        }
    } else {
        for (int i = idx; i < size; ++i) {
            float aVal = a ? (float)a[i] : 0.0f;
            float bVal = b ? (float)b[i] : 0.0f;
            c[i] = (T)(aVal + bVal);
        }
    }
}

template <typename T>
void addVectors_v4(T* c, T* a, T* b, int size,
                   ActivationFunction activation, sycl::queue &queue) {
    constexpr int kBlockSize = 512;
    int blocks = DivUp(size / 4, kBlockSize);
    
    queue.parallel_for(
        sycl::nd_range<1>(blocks * kBlockSize, kBlockSize),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] 
        [[intel::num_simd_work_items(4)]] {
            addVectors_v4_kernel(c, a, b, size, activation, item);
        }
    );
}

// Benchmark function
template<typename Func>
void benchmark_version(sycl::queue& q, const std::string& version_name, 
                       Func kernel_func, std::ofstream& csv) {
    std::vector<int> sizes = {256, 512, 1024, 4096, 16384};
    
    std::cout << "\n=== " << version_name << " ===\n";
    csv << "\n" << version_name << "\n";
    csv << "Size,Time_ms,GFLOPS,Bandwidth_GB/s\n";
    
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
            kernel_func(d_c, d_a, d_b, n);
            q.wait();
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            kernel_func(d_c, d_a, d_b, n);
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
}

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v);
        
        std::cout << "========================================\n";
        std::cout << "Add Vectors Complete Benchmark (V2,V3,V4)\n";
        std::cout << "========================================\n";
        std::cout << "GPU: " << q.get_device().get_info<sycl::info::device::name>() << "\n\n";
        
        std::ofstream csv("add_vectors_v2_v3_v4_results.csv");
        csv << "Add Vectors Benchmark Results\n";
        csv << "GPU: " << q.get_device().get_info<sycl::info::device::name>() << "\n\n";
        
        // Test V2
        benchmark_version(q, "V2: Sub-Group 16", 
            [&](float* c, float* a, float* b, int n) {
                addVectors_v2(c, a, b, n, ACTIVATION_NONE, q);
            }, csv);
        
        // Test V3
        benchmark_version(q, "V3: 4-Wide Vectorization",
            [&](float* c, float* a, float* b, int n) {
                addVectors_v3(c, a, b, n, ACTIVATION_NONE, q);
            }, csv);
        
        // Test V4
        benchmark_version(q, "V4: Large GRF Mode",
            [&](float* c, float* a, float* b, int n) {
                addVectors_v4(c, a, b, n, ACTIVATION_NONE, q);
            }, csv);
        
        csv.close();
        std::cout << "\n✅ All tests completed!\n";
        std::cout << "Results saved to: add_vectors_v2_v3_v4_results.csv\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}