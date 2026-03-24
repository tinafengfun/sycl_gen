#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

// V0: Baseline (WG=256, SG=32 - note: wrong for BMG)
namespace v0 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }
template <typename T>
void globalAvgPool_kernel(T* output, const T* input, int N, int C,
                          const sycl::nd_item<1> &item) {
    int nc = item.get_global_id(0);
    if (nc >= N * C) return;
    float sum = 0.0f;
    for (int i = 0; i < 64; ++i) {
        sum += (float)input[nc * 64 + i];
    }
    output[nc] = (T)(sum / 64.0f);
}
template <typename T>
void globalAvgPool(T* output, const T* input, int N, int C, sycl::queue &queue) {
    int total = N * C;
    int blocks = DivUp(total, 256);
    queue.parallel_for(
        sycl::nd_range<1>(blocks * 256, 256),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            globalAvgPool_kernel(output, input, N, C, item);
        }
    );
}
}

// V1: WG=512 + SG=16
namespace v1 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }
template <typename T>
void globalAvgPool_kernel(T* output, const T* input, int N, int C,
                          const sycl::nd_item<1> &item) {
    int nc = item.get_global_id(0);
    if (nc >= N * C) return;
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        sum += (float)input[nc * 64 + i];
    }
    output[nc] = (T)(sum / 64.0f);
}
template <typename T>
void globalAvgPool(T* output, const T* input, int N, int C, sycl::queue &queue) {
    int total = N * C;
    int blocks = DivUp(total, 512);
    queue.parallel_for(
        sycl::nd_range<1>(blocks * 512, 512),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            globalAvgPool_kernel(output, input, N, C, item);
        }
    );
}
}

// V2: 4-wide vectorized
namespace v2 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }
template <typename T>
void globalAvgPool_kernel(T* output, const T* input, int N, int C,
                          const sycl::nd_item<1> &item) {
    int nc = item.get_global_id(0);
    if (nc >= N * C) return;
    
    sycl::vec<float, 4> sum_vec(0.0f);
    const float* ptr = reinterpret_cast<const float*>(&input[nc * 64]);
    
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        sycl::vec<float, 4> data;
        data.load(0, ptr + i * 4);
        sum_vec += data;
    }
    
    float sum = sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3];
    output[nc] = (T)(sum / 64.0f);
}
template <typename T>
void globalAvgPool(T* output, const T* input, int N, int C, sycl::queue &queue) {
    int total = N * C;
    int blocks = DivUp(total, 512);
    queue.parallel_for(
        sycl::nd_range<1>(blocks * 512, 512),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            globalAvgPool_kernel(output, input, N, C, item);
        }
    );
}
}

struct TestConfig {
    int N, C;
};

template<typename Func>
void test_version(sycl::queue& q, const std::string& name, Func func, 
                  const TestConfig& cfg, std::ofstream& csv, int iterations = 100) {
    int total_out = cfg.N * cfg.C;
    int total_in = total_out * 64;  // 8x8 spatial
    
    float *d_input = sycl::malloc_device<float>(total_in, q);
    float *d_output = sycl::malloc_device<float>(total_out, q);
    
    std::vector<float> h_input(total_in);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_input) v = dist(gen);
    q.memcpy(d_input, h_input.data(), total_in * sizeof(float)).wait();
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        func(d_output, d_input, cfg.N, cfg.C, q);
        q.wait();
    }
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func(d_output, d_input, cfg.N, cfg.C, q);
        q.wait();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    double avg_time_ms = (elapsed.count() * 1000.0) / iterations;
    
    // Calculate metrics
    int flops_per_element = 64;  // 64 additions + 1 division
    int bytes_per_element = sizeof(float) * (64 + 1);  // Read 64, write 1
    
    double flops = total_out * flops_per_element;
    double bytes = total_out * bytes_per_element;
    double gflops = (flops / (avg_time_ms * 1e-3)) / 1e9;
    double bandwidth = (bytes / (avg_time_ms * 1e-3)) / 1e9;
    
    std::cout << name << "\tN=" << cfg.N << " C=" << cfg.C 
              << "\tTime: " << avg_time_ms << " ms\t"
              << "GFLOPS: " << gflops << "\tBW: " << bandwidth << " GB/s" << std::endl;
    
    csv << name << "," << cfg.N << "," << cfg.C << ","
        << avg_time_ms << "," << gflops << "," << bandwidth << "\n";
    
    sycl::free(d_input, q);
    sycl::free(d_output, q);
}

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v);
        
        std::cout << "========================================" << std::endl;
        std::cout << "Global Avg Pool REAL Benchmark" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "GPU: " << q.get_device().get_info<sycl::info::device::name>() << std::endl << std::endl;
        
        std::ofstream csv("global_avg_pool_real_results.csv");
        csv << "Version,N,C,Time_ms,GFLOPS,Bandwidth_GB/s\n";
        
        std::vector<TestConfig> configs = {
            {4, 64}, {8, 64}, {16, 64}, {64, 64}, {256, 64}
        };
        
        std::cout << "=== V0: Baseline (WG=256) ===" << std::endl;
        for (const auto& cfg : configs) {
            test_version(q, "V0", v0::globalAvgPool<float>, cfg, csv);
        }
        
        std::cout << "\n=== V1: WG=512 + SG=16 ===" << std::endl;
        for (const auto& cfg : configs) {
            test_version(q, "V1", v1::globalAvgPool<float>, cfg, csv);
        }
        
        std::cout << "\n=== V2: Vec4 Vectorized ===" << std::endl;
        for (const auto& cfg : configs) {
            test_version(q, "V2", v2::globalAvgPool<float>, cfg, csv);
        }
        
        csv.close();
        std::cout << "\n✅ REAL Global Avg Pool testing completed!" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}