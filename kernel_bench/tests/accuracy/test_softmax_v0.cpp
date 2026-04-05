#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <cmath>

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

struct TestConfig { int N, C; };

bool validate_softmax(float* output, int N, int C) {
    for (int n = 0; n < N; ++n) {
        float sum = 0;
        for (int c = 0; c < C; ++c) sum += output[n * C + c];
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
    for (int i = 0; i < 10; ++i) {
        func(d_output, d_input, cfg.N, cfg.C, q);
        q.wait();
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func(d_output, d_input, cfg.N, cfg.C, q);
        q.wait();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double avg_time_ms = (elapsed.count() * 1000.0) / iterations;
    int flops_per_element = 10;
    int bytes_per_element = 12;
    double flops = total * flops_per_element;
    double bytes = total * bytes_per_element;
    double gflops = (flops / (avg_time_ms * 1e-3)) / 1e9;
    double bandwidth = (bytes / (avg_time_ms * 1e-3)) / 1e9;
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
        std::cout << "Softmax Complete Benchmark\n";
        std::cout << "========================================\n";
        std::cout << "GPU: " << q.get_device().get_info<sycl::info::device::name>() << "\n\n";
        std::ofstream csv("softmax_v0_test.csv");
        csv << "Version,N,C,Time_ms,GFLOPS,Bandwidth_GB/s,Status\n";
        std::vector<TestConfig> configs = {
            {4, 64}, {8, 64}, {16, 64}, {64, 64}, {256, 64}
        };
        std::cout << "=== V0: Baseline ===\n";
        for (const auto& cfg : configs) {
            test_version(q, "V0", v0::softmax<float>, cfg, csv);
        }
        csv.close();
        std::cout << "\n✅ Softmax V0 completed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}