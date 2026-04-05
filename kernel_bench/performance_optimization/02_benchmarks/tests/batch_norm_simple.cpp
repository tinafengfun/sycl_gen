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

// Unified kernel with version selector
template <typename T, int VERSION>
void batchNorm_kernel(T* output, const T* input, const T* skipInput,
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

template <typename T, int WG_SIZE, bool USE_SG16>
void batchNorm(T* output, const T* input, const T* skipInput,
               int N, int C, int H, int W,
               const float* means, const float* varMultipliers,
               ActivationFunction activation, sycl::queue &q) {
    int total = N * C * H * W;
    int blocks = DivUp(total, WG_SIZE);
    
    if constexpr (USE_SG16) {
        q.parallel_for(
            sycl::nd_range<1>(blocks * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                batchNorm_kernel<T, 1>(output, input, skipInput, N, C, H, W,
                                        means, varMultipliers, activation, item);
            }
        );
    } else {
        q.parallel_for(
            sycl::nd_range<1>(blocks * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) {
                batchNorm_kernel<T, 0>(output, input, skipInput, N, C, H, W,
                                        means, varMultipliers, activation, item);
            }
        );
    }
}

struct TestConfig {
    int N, C, H, W;
    int total() const { return N * C * H * W; }
};

template<int WG_SIZE, bool USE_SG16>
void benchmark_batch_norm(sycl::queue& q, const std::string& name, 
                          const TestConfig& cfg, std::ofstream& csv) {
    int total = cfg.total();
    int flops_per_element = 5;
    int bytes_per_element = sizeof(float) * 5;
    
    float *d_input = sycl::malloc_device<float>(total, q);
    float *d_output = sycl::malloc_device<float>(total, q);
    float *d_means = sycl::malloc_device<float>(cfg.C, q);
    float *d_vars = sycl::malloc_device<float>(cfg.C, q);
    
    std::vector<float> h_input(total), h_means(cfg.C), h_vars(cfg.C);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_input) v = dist(gen);
    for (auto& v : h_means) v = dist(gen) * 0.1f;
    for (auto& v : h_vars) v = 1.0f + dist(gen);
    
    q.memcpy(d_input, h_input.data(), total * sizeof(float)).wait();
    q.memcpy(d_means, h_means.data(), cfg.C * sizeof(float)).wait();
    q.memcpy(d_vars, h_vars.data(), cfg.C * sizeof(float)).wait();
    
    for (int i = 0; i < 10; ++i) {
        batchNorm<float, WG_SIZE, USE_SG16>(
            d_output, d_input, nullptr, cfg.N, cfg.C, cfg.H, cfg.W,
            d_means, d_vars, ACTIVATION_NONE, q);
        q.wait();
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        batchNorm<float, WG_SIZE, USE_SG16>(
            d_output, d_input, nullptr, cfg.N, cfg.C, cfg.H, cfg.W,
            d_means, d_vars, ACTIVATION_NONE, q);
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
    
    csv << name << "," << cfg.N << "," << cfg.C << ","
        << avg_time_ms << "," << gflops << "," << bandwidth << "\n";
    
    sycl::free(d_input, q);
    sycl::free(d_output, q);
    sycl::free(d_means, q);
    sycl::free(d_vars, q);
}

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v);
        
        std::cout << "========================================\n";
        std::cout << "Batch Norm Benchmark (V0, V1, V2)\n";
        std::cout << "========================================\n";
        std::cout << "GPU: " << q.get_device().get_info<sycl::info::device::name>() << "\n\n";
        
        std::ofstream csv("batch_norm_results.csv");
        csv << "Version,N,C,Time_ms,GFLOPS,Bandwidth_GB/s\n";
        
        std::vector<TestConfig> configs = {
            {1, 64, 8, 8},
            {1, 128, 8, 8},
            {4, 64, 8, 8},
            {4, 128, 8, 8},
        };
        
        std::cout << "=== V0: Baseline (WG=256) ===\n";
        for (const auto& cfg : configs) {
            benchmark_batch_norm<256, false>(q, "V0", cfg, csv);
        }
        
        std::cout << "\n=== V1: WG=512 ===\n";
        for (const auto& cfg : configs) {
            benchmark_batch_norm<512, false>(q, "V1", cfg, csv);
        }
        
        std::cout << "\n=== V2: WG=512 + SG=16 ===\n";
        for (const auto& cfg : configs) {
            benchmark_batch_norm<512, true>(q, "V2", cfg, csv);
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