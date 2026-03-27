#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <cmath>

// V0: Baseline
namespace v0 {
enum ActivationFunction { ACTIVATION_NONE };
inline int DivUp(int a, int b) { return (a + b - 1) / b; }
template <typename T>
void softmax_kernel(T* output, const T* input, int N, int C,
                    const sycl::nd_item<1> &item) {
    int n = item.get_group(0);
    int tid = item.get_local_id(0);
    if (n >= N) return;
    float max_val = -1e20f;
    for (int c = tid; c < C; c += item.get_local_range(0)) {
        max_val = sycl::max(max_val, (float)input[n * C + c]);
    }
    auto sg = item.get_sub_group();
    for (int offset = 16 / 2; offset > 0; offset /= 2) {
        max_val = sycl::max(max_val, sycl::permute_group_by_xor(sg, max_val, offset));
    }
    float sum = 0.0f;
    for (int c = tid; c < C; c += item.get_local_range(0)) {
        float val = sycl::exp((float)input[n * C + c] - max_val);
        output[n * C + c] = (T)val;
        sum += val;
    }
    for (int offset = 16 / 2; offset > 0; offset /= 2) {
        sum += sycl::permute_group_by_xor(sg, sum, offset);
    }
    for (int c = tid; c < C; c += item.get_local_range(0)) {
        output[n * C + c] = (T)((float)output[n * C + c] / sum);
    }
}
template <typename T>
void softmax(T* output, const T* input, int N, int C, sycl::queue &queue) {
    queue.parallel_for(
        sycl::nd_range<1>(N * 256, 256),
        [=](sycl::nd_item<1> item) {
            softmax_kernel(output, input, N, C, item);
        }
    );
}
}

// V1: WG=512 + SG=16
namespace v1 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }
template <typename T>
void softmax_kernel(T* output, const T* input, int N, int C,
                    const sycl::nd_item<1> &item) {
    int n = item.get_group(0);
    int tid = item.get_local_id(0);
    if (n >= N) return;
    float max_val = -1e20f;
    for (int c = tid; c < C; c += item.get_local_range(0)) {
        max_val = sycl::max(max_val, (float)input[n * C + c]);
    }
    auto sg = item.get_sub_group();
    for (int offset = 16 / 2; offset > 0; offset /= 2) {
        max_val = sycl::max(max_val, sycl::permute_group_by_xor(sg, max_val, offset));
    }
    float sum = 0.0f;
    for (int c = tid; c < C; c += item.get_local_range(0)) {
        float val = sycl::exp((float)input[n * C + c] - max_val);
        output[n * C + c] = (T)val;
        sum += val;
    }
    for (int offset = 16 / 2; offset > 0; offset /= 2) {
        sum += sycl::permute_group_by_xor(sg, sum, offset);
    }
    for (int c = tid; c < C; c += item.get_local_range(0)) {
        output[n * C + c] = (T)((float)output[n * C + c] / sum);
    }
}
template <typename T>
void softmax(T* output, const T* input, int N, int C, sycl::queue &queue) {
    queue.parallel_for(
        sycl::nd_range<1>(N * 512, 512),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            softmax_kernel(output, input, N, C, item);
        }
    );
}
}

// V2: Optimized reduction
namespace v2 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }
template <typename T>
void softmax_kernel(T* output, const T* input, int N, int C,
                    const sycl::nd_item<1> &item) {
    int n = item.get_group(0);
    int tid = item.get_local_id(0);
    if (n >= N) return;
    
    float max_val = -1e20f;
    #pragma unroll
    for (int c = tid; c < C; c += item.get_local_range(0)) {
        max_val = sycl::max(max_val, (float)input[n * C + c]);
    }
    
    auto sg = item.get_sub_group();
    #pragma unroll
    for (int offset = 16 / 2; offset > 0; offset /= 2) {
        max_val = sycl::max(max_val, sycl::permute_group_by_xor(sg, max_val, offset));
    }
    
    float sum = 0.0f;
    #pragma unroll
    for (int c = tid; c < C; c += item.get_local_range(0)) {
        float val = sycl::exp((float)input[n * C + c] - max_val);
        output[n * C + c] = (T)val;
        sum += val;
    }
    
    #pragma unroll
    for (int offset = 16 / 2; offset > 0; offset /= 2) {
        sum += sycl::permute_group_by_xor(sg, sum, offset);
    }
    
    #pragma unroll
    for (int c = tid; c < C; c += item.get_local_range(0)) {
        output[n * C + c] = (T)((float)output[n * C + c] / sum);
    }
}
template <typename T>
void softmax(T* output, const T* input, int N, int C, sycl::queue &queue) {
    queue.parallel_for(
        sycl::nd_range<1>(N * 512, 512),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            softmax_kernel(output, input, N, C, item);
        }
    );
}
}

// V3: Single-thread-per-output (Type C optimal pattern)
namespace v3 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }
template <typename T>
void softmax_kernel(T* output, const T* input, int N, int C, sycl::item<1> item) {
    int n = item.get_id(0);
    if (n >= N) return;
    
    // Step 1: Find max (for numerical stability)
    float max_val = -1e20f;
    #pragma unroll 8
    for (int c = 0; c < C; c++) {
        max_val = sycl::max(max_val, (float)input[n * C + c]);
    }
    
    // Step 2: Compute exp and sum
    float sum = 0.0f;
    #pragma unroll 8
    for (int c = 0; c < C; c++) {
        float val = sycl::exp((float)input[n * C + c] - max_val);
        output[n * C + c] = (T)val;
        sum += val;
    }
    
    // Step 3: Normalize
    float inv_sum = 1.0f / sum;
    #pragma unroll 8
    for (int c = 0; c < C; c++) {
        output[n * C + c] = (T)((float)output[n * C + c] * inv_sum);
    }
}
template <typename T>
void softmax(T* output, const T* input, int N, int C, sycl::queue &queue) {
    queue.parallel_for(
        sycl::range<1>(N),
        [=](sycl::item<1> item) {
            softmax_kernel(output, input, N, C, item);
        }
    );
}
}

struct TestConfig {
    int N, C;
    int total() const { return N * C; }
};

template<typename Func>
void test_version(sycl::queue& q, const std::string& name, Func func, 
                  const TestConfig& cfg, std::ofstream& csv, int iterations = 100) {
    int total = cfg.total();
    
    float *d_input = sycl::malloc_device<float>(total, q);
    float *d_output = sycl::malloc_device<float>(total, q);
    
    std::vector<float> h_input(total);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_input) v = dist(gen);
    q.memcpy(d_input, h_input.data(), total * sizeof(float)).wait();
    
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
    int flops_per_element = 10;  // Approximate for softmax
    int bytes_per_element = 12;  // Read input + write output
    
    double flops = total * flops_per_element;
    double bytes = total * bytes_per_element;
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
        std::cout << "Softmax REAL Benchmark (V0, V1, V2, V3)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "GPU: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
        std::cout << "Iterations: 100 per test" << std::endl;
        std::cout << "Data sizes: 256, 512, 1024, 4096, 16384" << std::endl << std::endl;
        
        std::ofstream csv("softmax_real_results.csv");
        csv << "Version,N,C,Time_ms,GFLOPS,Bandwidth_GB/s\n";
        
        std::vector<TestConfig> configs = {
            {4, 64},    // 256
            {8, 64},    // 512
            {16, 64},   // 1024
            {64, 64},   // 4096
            {256, 64},  // 16384
        };
        
        std::cout << "=== V0: Baseline (WG=256) ===" << std::endl;
        for (const auto& cfg : configs) {
            test_version(q, "V0", v0::softmax<float>, cfg, csv);
        }
        
        std::cout << "\n=== V1: WG=512 + SG=16 ===" << std::endl;
        for (const auto& cfg : configs) {
            test_version(q, "V1", v1::softmax<float>, cfg, csv);
        }
        
        std::cout << "\n=== V2: Optimized ===" << std::endl;
        for (const auto& cfg : configs) {
            test_version(q, "V2", v2::softmax<float>, cfg, csv);
        }
        
        std::cout << "\n=== V3: Single-thread-per-output ===" << std::endl;
        for (const auto& cfg : configs) {
            test_version(q, "V3", v3::softmax<float>, cfg, csv);
        }
        
        csv.close();
        std::cout << "\n✅ REAL Softmax testing completed!" << std::endl;
        std::cout << "Results saved to: softmax_real_results.csv" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}