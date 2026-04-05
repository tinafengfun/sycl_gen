#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>

// Kernel using explicit sub-group size via attributes
template <int SG_SIZE>
void sg_test_kernel(float* output, const float* input, int n, 
                    sycl::nd_item<1> item) {
    auto sg = item.get_sub_group();
    int i = item.get_global_id(0);
    
    if (i < n) {
        float val = input[i];
        // Use sub-group reduction via permute
        for (int offset = SG_SIZE / 2; offset > 0; offset /= 2) {
            float tmp = sycl::permute_group_by_xor(sg, val, offset);
            val += tmp;
        }
        output[i] = val;
    }
}

template <int SG_SIZE>
float benchmark_sg_size(sycl::queue& q, int n, int iterations = 100) {
    float *d_input = sycl::malloc_device<float>(n, q);
    float *d_output = sycl::malloc_device<float>(n, q);
    
    // Initialize
    std::vector<float> h_input(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n; ++i) h_input[i] = dist(gen);
    q.memcpy(d_input, h_input.data(), n * sizeof(float)).wait();
    
    constexpr int WG_SIZE = 256;
    int num_wg = (n + WG_SIZE - 1) / WG_SIZE;
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        q.parallel_for(
            sycl::nd_range<1>(num_wg * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                sg_test_kernel<SG_SIZE>(d_output, d_input, n, item);
            }
        ).wait();
    }
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        q.parallel_for(
            sycl::nd_range<1>(num_wg * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                sg_test_kernel<SG_SIZE>(d_output, d_input, n, item);
            }
        ).wait();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    float avg_time_ms = (elapsed.count() * 1000.0f) / iterations;
    
    sycl::free(d_input, q);
    sycl::free(d_output, q);
    
    return avg_time_ms;
}

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    
    std::ofstream out("sg_size_test_results.txt");
    out << "=== Sub-Group Size Test Results ===\n\n";
    out << "GPU: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    out << "Testing sub-group reduction performance\n\n";
    
    std::vector<int> sg_sizes = {16, 32};
    std::vector<int> data_sizes = {256, 512, 1024, 4096, 16384, 65536};
    const int iterations = 100;
    
    out << "Data Size -> |";
    for (auto n : data_sizes) {
        out << " " << n << " |";
    }
    out << "\n";
    out << "-------------|";
    for (size_t i = 0; i < data_sizes.size(); ++i) {
        out << "---------|";
    }
    out << "\n";
    
    std::vector<std::vector<double>> results(sg_sizes.size(), 
                                      std::vector<double>(data_sizes.size()));
    
    for (size_t sg_idx = 0; sg_idx < sg_sizes.size(); ++sg_idx) {
        int sg = sg_sizes[sg_idx];
        out << "SG=" << sg << "      |";
        
        for (size_t ds_idx = 0; ds_idx < data_sizes.size(); ++ds_idx) {
            int n = data_sizes[ds_idx];
            double avg_time = 0.0;
            
            if (sg == 16) {
                avg_time = benchmark_sg_size<16>(q, n, iterations);
            } else {
                avg_time = benchmark_sg_size<32>(q, n, iterations);
            }
            
            results[sg_idx][ds_idx] = avg_time;
            out << " " << std::fixed << std::setprecision(3) << avg_time << " |";
        }
        out << "\n";
    }
    
    // Analysis
    out << "\n=== Analysis ===\n\n";
    
    out << "Best sub-group size for each data size:\n";
    for (size_t ds_idx = 0; ds_idx < data_sizes.size(); ++ds_idx) {
        double best_time = results[0][ds_idx];
        int best_sg = sg_sizes[0];
        
        for (size_t sg_idx = 1; sg_idx < sg_sizes.size(); ++sg_idx) {
            if (results[sg_idx][ds_idx] < best_time) {
                best_time = results[sg_idx][ds_idx];
                best_sg = sg_sizes[sg_idx];
            }
        }
        
        double baseline = results[0][ds_idx]; // SG=16 as baseline
        double speedup = baseline / best_time;
        
        out << "  N=" << data_sizes[ds_idx] << ": SG=" << best_sg 
            << " (" << std::fixed << std::setprecision(3) << best_time 
            << " ms, speedup vs SG=16: " << std::setprecision(2) << speedup << "x)\n";
    }
    
    out << "\nRecommendation: For BMG B60 (Xe2), use sub-group size 16\n";
    
    out.close();
    std::cout << "Sub-group size test completed. Results saved to sg_size_test_results.txt\n";
    
    return 0;
}