#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <cmath>

// Include all softmax versions
namespace v0 {
    #include "../01_kernels/softmax/generated/v0_baseline.cpp"
}
namespace v1 {
    #include "../01_kernels/softmax/generated/v1_wg512.cpp"
}
namespace v2 {
    #include "../01_kernels/softmax/generated/v2_sg16.cpp"
}
namespace v3 {
    #include "../01_kernels/softmax/generated/v3_shuffle.cpp"
}
namespace v4 {
    #include "../01_kernels/softmax/generated/v4_vec4.cpp"
}
namespace v5 {
    #include "../01_kernels/softmax/generated/v5_optimized.cpp"
}

using namespace lczero::sycldnn_backend;

struct TestConfig {
    int N, C;
};

bool validate_softmax(float* output, int N, int C) {
    for (int n = 0; n < N; ++n) {
        float sum = 0;
        for (int c = 0; c < C; ++c) {
            sum += output[n * C + c];
        }
        if (std::abs(sum - 1.0f) > 0.01f) return false;
    }
    return true;
}

template<typename Func>
void test_version(sycl::queue& q, const std::string& name, Func func, 
                const TestConfig& cfg, std::ofstream& csv) {
    int total = cfg.N * cfg.C;
    int iterations = 100;
    
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
    int flops_per_element = 10;  // Approximate
    int bytes_per_element = 12;  // Approximate
    
    double flops = total * flops_per_element;
    double bytes = total * bytes_per_element;
    double gflops = (flops / (avg_time_ms * 1e-3)) / 1e9;
    double bandwidth = (bytes / (avg_time_ms * 1e-3)) / 1e9;
    
    // Validation
    std::vector<float> h_output(total);
    q.memcpy(h_output.data(), d_output, total * sizeof(float)).wait();
    bool passed = validate_softmax(h_output.data(), cfg.N, cfg.C);
    
    std::cout << name << "\tN=" << cfg.N << " C=" << cfg.C 
              << "\tTime: " << avg_time_ms << " ms\t"
              << "GFLOPS: " << gflops << "\tBW: " << bandwidth << " GB/s"
              << (passed ? " ✅" : " ❌") << std::endl;
    
    csv << name << "," << cfg.N << "," << cfg.C << ","
        << avg_time_ms << "," << gflops << "," << bandwidth << ","
        << (passed ? "PASS" : "FAIL") << "\n";
    
    sycl::free(d_input, q);
    sycl::free(d_output, q);
}

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v);
        
        std::cout << "========================================\n";
        std::cout << "Softmax Complete Benchmark (All 6 Versions)\n";
        std::cout << "========================================\n";
        std::cout << "GPU: " << q.get_device().get_info<sycl::info::device::name>() << "\n\n";
        
        std::ofstream csv("softmax_all_versions_results.csv");
        csv << "Version,N,C,Time_ms,GFLOPS,Bandwidth_GB/s,Status\n";
        
        // Test configurations: vary N, keep C=64
        std::vector<TestConfig> configs = {
            {4, 64},   // 256 elements
            {8, 64},   // 512 elements
            {16, 64},  // 1024 elements
            {64, 64},  // 4096 elements
            {256, 64}, // 16384 elements
        };
        
        std::cout << "=== V0: Baseline ===\n";
        for (const auto& cfg : configs) {
            test_version(q, "V0", v0::softmax<float>, cfg, csv);
        }
        
        std::cout << "\n=== V1: WG=512 ===\n";
        for (const auto& cfg : configs) {
            test_version(q, "V1", v1::softmax<float>, cfg, csv);
        }
        
        std::cout << "\n=== V2: SG=16 ===\n";
        for (const auto& cfg : configs) {
            test_version(q, "V2", v2::softmax<float>, cfg, csv);
        }
        
        std::cout << "\n=== V3: Shuffle ===\n";
        for (const auto& cfg : configs) {
            test_version(q, "V3", v3::softmax<float>, cfg, csv);
        }
        
        std::cout << "\n=== V4: Vec4 ===\n";
        for (const auto& cfg : configs) {
            test_version(q, "V4", v4::softmax<float>, cfg, csv);
        }
        
        std::cout << "\n=== V5: Optimized ===\n";
        for (const auto& cfg : configs) {
            test_version(q, "V5", v5::softmax<float>, cfg, csv);
        }
        
        csv.close();
        std::cout << "\n✅ Softmax all versions completed!\n";
        std::cout << "Results saved to: softmax_all_versions_results.csv\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}