#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <math>
#include <algorithm>
#include <string>

// Include the kernel
#include "../01_kernels/add_vectors/generated/v0_baseline.cpp"

using namespace perf;

struct TestResult {
    int size;
    double time_ms;
    double gflops;
    double bandwidth_gbps;
};

template<typename T>
TestResult benchmark_add_vectors(sycl::queue& q, int n, 
                                  int warmup_iterations = 10,
                                  int test_iterations = 100) {
    // Allocate device memory
    T *d_a = sycl::malloc_device<T>(n, q);
    T *d_b = sycl::malloc_device<T>(n, q);
    T *d_c = sycl::malloc_device<T>(n, q);
    
    // Initialize random data
    std::vector<T> h_a(n), h_b(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<T>(dist(gen));
        h_b[i] = static_cast<T>(dist(gen));
    }
    
    q.memcpy(d_a, h_a.data(), n * sizeof(T)).wait();
    q.memcpy(d_b, h_b.data(), n * sizeof(T)).wait();
    
    // Warmup
    for (int i = 0; i < warmup_iterations; ++i) {
        lczero::sycldnn_backend::addVectors(
            d_c, d_a, d_b, n,
            lczero::sycldnn_backend::ACTIVATION_NONE, q);
        q.wait();
    }
    
    // Benchmark
    std::vector<double> times;
    times.reserve(test_iterations);
    
    for (int i = 0; i < test_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        lczero::sycldnn_backend::addVectors(
            d_c, d_a, d_b, n,
            lczero::sycldnn_backend::ACTIVATION_NONE, q);
        q.wait();
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        times.push_back(elapsed.count() * 1000.0); // Convert to ms
    }
    
    // Calculate statistics
    double avg_time = 0.0;
    for (double t : times) avg_time += t;
    avg_time /= test_iterations;
    
    // Calculate metrics
    int flops_per_element = 1;  // 1 addition
    int bytes_per_element = 3 * sizeof(T);  // read a, b + write c
    
    double gflops = calculate_gflops(n, flops_per_element, avg_time);
    double bandwidth = calculate_bandwidth(n, bytes_per_element, avg_time);
    
    // Cleanup
    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);
    
    TestResult result;
    result.size = n;
    result.time_ms = avg_time;
    result.gflops = gflops;
    result.bandwidth_gbps = bandwidth;
    
    return result;
}

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v);
        
        std::cout << "=== Add Vectors Benchmark (V0 Baseline) ===\n";
        std::cout << "GPU: " << q.get_device().get_info<sycl::info::device::name>() << "\n\n";
        
        std::vector<int> sizes = {256, 512, 1024, 4096, 16384};
        std::ofstream csv("add_vectors_v0_results.csv");
        csv << "Size,Time_ms,GFLOPS,Bandwidth_GB/s\n";
        
        std::cout << "Size\t\tTime(ms)\tGFLOPS\t\tBandwidth(GB/s)\n";
        std::cout << std::string(60, '-') << "\n";
        
        for (int n : sizes) {
            auto result = benchmark_add_vectors<float>(q, n);
            
            std::cout << result.size << "\t\t"
                      << std::fixed << std::setprecision(4) << result.time_ms << "\t\t"
                      << std::setprecision(2) << result.gflops << "\t\t"
                      << result.bandwidth_gbps << "\n";
            
            csv << result.size << ","
                << result.time_ms << ","
                << result.gflops << ","
                << result.bandwidth_gbps << "\n";
        }
        
        csv.close();
        std::cout << "\nResults saved to add_vectors_v0_results.csv\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}