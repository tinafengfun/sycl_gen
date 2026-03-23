#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

enum ActivationFunction { ACTIVATION_NONE, ACTIVATION_RELU };

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

inline float activate(float val, ActivationFunction act) {
    if (act == ACTIVATION_RELU) return val > 0 ? val : 0;
    return val;
}

// V0: Baseline
template <typename T>
void batchNorm_v0_kernel(T* output, const T* input, const T* skipInput,
                         int N, int C, int H, int W,
                         const float* means, const float* varMultipliers,
                         ActivationFunction activation,
                         const sycl::nd_item<1> &item) {
    int idx = item.get_global_id(0);
    int total = N * C * H * W;
    if (idx < total) {
        int wIndex = (sizeof(T) == sizeof(float)) ? ((idx / (H * W)) % C) : (idx % C);
        float el = (float)input[idx];
        el -= means[wIndex];
        el *= varMultipliers[wIndex];
        if (skipInput) el += (float)skipInput[idx];
        el = activate(el, activation);
        output[idx] = (T)el;
    }
}

// V1: WG=512
template <typename T>
void batchNorm_v1_kernel(T* output, const T* input, const T* skipInput,
                         int N, int C, int H, int W,
                         const float* means, const float* varMultipliers,
                         ActivationFunction activation,
                         const sycl::nd_item<1> &item) {
    int idx = item.get_global_id(0);
    int total = N * C * H * W;
    if (idx < total) {
        int wIndex = (sizeof(T) == sizeof(float)) ? ((idx / (H * W)) % C) : (idx % C);
        float el = (float)input[idx];
        el -= means[wIndex];
        el *= varMultipliers[wIndex];
        if (skipInput) el += (float)skipInput[idx];
        el = activate(el, activation);
        output[idx] = (T)el;
    }
}

// V4: SLM caching
template <typename T>
void batchNorm_v4_kernel(T* output, const T* input, const T* skipInput,
                         int N, int C, int H, int W,
                         const float* means, const float* varMultipliers,
                         ActivationFunction activation,
                         const sycl::nd_item<1> &item,
                         sycl::local_accessor<float, 1> local_means,
                         sycl::local_accessor<float, 1> local_vars) {
    int tid = item.get_local_id(0);
    int total = N * C * H * W;
    
    // Cache mean/variance in SLM
    for (int i = tid; i < C; i += item.get_local_range(0)) {
        local_means[i] = means[i];
        local_vars[i] = varMultipliers[i];
    }
    item.barrier(sycl::access::fence_space::local_space);
    
    int idx = item.get_global_id(0);
    if (idx < total) {
        int wIndex = (sizeof(T) == sizeof(float)) ? ((idx / (H * W)) % C) : (idx % C);
        float el = (float)input[idx];
        el -= local_means[wIndex];
        el *= local_vars[wIndex];
        if (skipInput) el += (float)skipInput[idx];
        el = activate(el, activation);
        output[idx] = (T)el;
    }
}

// Host functions
template <typename T>
void batchNorm_v0(T* output, const T* input, const T* skipInput,
                  int N, int C, int H, int W,
                  const float* means, const float* varMultipliers,
                  ActivationFunction activation, sycl::queue &q) {
    int total = N * C * H * W;
    int kBlockSize = 256;
    int blocks = DivUp(total, kBlockSize);
    
    q.parallel_for(
        sycl::nd_range<1>(blocks * kBlockSize, kBlockSize),
        [=](sycl::nd_item<1> item) {
            batchNorm_v0_kernel(output, input, skipInput, N, C, H, W,
                              means, varMultipliers, activation, item);
        }
    );
}

template <typename T>
void batchNorm_v1(T* output, const T* input, const T* skipInput,
                  int N, int C, int H, int W,
                  const float* means, const float* varMultipliers,
                  ActivationFunction activation, sycl::queue &q) {
    int total = N * C * H * W;
    int kBlockSize = 512;
    int blocks = DivUp(total, kBlockSize);
    
    q.parallel_for(
        sycl::nd_range<1>(blocks * kBlockSize, kBlockSize),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            batchNorm_v1_kernel(output, input, skipInput, N, C, H, W,
                              means, varMultipliers, activation, item);
        }
    );
}

template <typename T>
void batchNorm_v4(T* output, const T* input, const T* skipInput,
                  int N, int C, int H, int W,
                  const float* means, const float* varMultipliers,
                  ActivationFunction activation, sycl::queue &q) {
    int total = N * C * H * W;
    int kBlockSize = 512;
    int blocks = DivUp(total, kBlockSize);
    
    q.parallel_for(
        sycl::nd_range<1>(blocks * kBlockSize, kBlockSize),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            sycl::local_accessor<float, 1> local_means(C, item);
            sycl::local_accessor<float, 1> local_vars(C, item);
            batchNorm_v4_kernel(output, input, skipInput, N, C, H, W,
                              means, varMultipliers, activation, item,
                              local_means, local_vars);
        }
    );
}

// Benchmark helper
struct TestConfig {
    int N, C, H, W;
    int total() const { return N * C * H * W; }
};

template<typename Func>
void benchmark_batch_norm(sycl::queue& q, const std::string& name, 
                          Func kernel_func, const TestConfig& cfg,
                          std::ofstream& csv) {
    int total = cfg.total();
    int flops_per_element = 5;  // sub, mul, add (skip), activate
    int bytes_per_element = sizeof(float) * (3 + 2);  // input, output, skip + means, vars
    
    // Allocate
    float *d_input = sycl::malloc_device<float>(total, q);
    float *d_output = sycl::malloc_device<float>(total, q);
    float *d_means = sycl::malloc_device<float>(cfg.C, q);
    float *d_vars = sycl::malloc_device<float>(cfg.C, q);
    
    // Init data
    std::vector<float> h_input(total), h_means(cfg.C), h_vars(cfg.C);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_input) v = dist(gen);
    for (auto& v : h_means) v = dist(gen) * 0.1f;
    for (auto& v : h_vars) v = 1.0f + dist(gen);
    
    q.memcpy(d_input, h_input.data(), total * sizeof(float)).wait();
    q.memcpy(d_means, h_means.data(), cfg.C * sizeof(float)).wait();
    q.memcpy(d_vars, h_vars.data(), cfg.C * sizeof(float)).wait();
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        kernel_func(d_output, d_input, nullptr, cfg.N, cfg.C, cfg.H, cfg.W,
                   d_means, d_vars, ACTIVATION_NONE);
        q.wait();
    }
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        kernel_func(d_output, d_input, nullptr, cfg.N, cfg.C, cfg.H, cfg.W,
                   d_means, d_vars, ACTIVATION_NONE);
        q.wait();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    double avg_time_ms = (elapsed.count() * 1000.0) / 100.0;
    
    double flops = total * flops_per_element;
    double bytes = total * bytes_per_element;
    double gflops = (flops / (avg_time_ms * 1e-3)) / 1e9;
    double bandwidth = (bytes / (avg_time_ms * 1e-3)) / 1e9;
    
    std::cout << name << "\tN=" << cfg.N << " C=" << cfg.C 
              << "\tTime: " << avg_time_ms << " ms\t"
              << "GFLOPS: " << gflops << "\tBW: " << bandwidth << " GB/s\n";
    
    csv << name << "," <> cfg.N << "," << avg_time_ms << ","
        << gflops << "," << bandwidth << "\n";
    
    sycl::free(d_input, q);
    sycl::free(d_output, q);
    sycl::free(d_means, q);
    sycl::free(d_vars, q);
}

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v);
        
        std::cout << "========================================\n";
        std::cout << "Batch Norm Benchmark (V0, V1, V4)\n";
        std::cout << "========================================\n";
        std::cout << "GPU: " << q.get_device().get_info<sycl::info::device::name>() << "\n\n";
        
        std::ofstream csv("batch_norm_results.csv");
        csv << "Version,N,Time_ms,GFLOPS,Bandwidth_GB/s\n";
        
        // Test configurations
        std::vector<TestConfig> configs = {
            {1, 64, 8, 8},    // 4096 elements
            {1, 128, 8, 8},   // 8192 elements
            {4, 64, 8, 8},    // 16384 elements
            {4, 128, 8, 8},   // 32768 elements
        };
        
        std::cout << "=== V0: Baseline (WG=256) ===\n";
        for (const auto& cfg : configs) {
            benchmark_batch_norm(q, "V0", 
                [&](auto... args) { batchNorm_v0(args..., q); }, cfg, csv);
        }
        
        std::cout << "\n=== V1: WG=512 + SG=16 ===\n";
        for (const auto& cfg : configs) {
            benchmark_batch_norm(q, "V1",
                [&](auto... args) { batchNorm_v1(args..., q); }, cfg, csv);
        }
        
        std::cout << "\n=== V4: SLM Caching ===\n";
        for (const auto& cfg : configs) {
            benchmark_batch_norm(q, "V4",
                [&](auto... args) { batchNorm_v4(args..., q); }, cfg, csv);
        }
        
        csv.close();
        std::cout << "\n✅ Batch norm tests completed!\n";
        std::cout << "Results saved to: batch_norm_results.csv\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}