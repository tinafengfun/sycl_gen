#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>

// Simple vector addition kernel with configurable work-group size
template <int WG_SIZE>
void vector_add_kernel(float* c, const float* a, const float* b, int n, 
                       sycl::nd_item<1> item) {
    int i = item.get_global_id(0);
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

template <int WG_SIZE>
float benchmark_wg_size(sycl::queue& q, int n, int iterations = 100) {
    // Allocate device memory
    float *d_a = sycl::malloc_device<float>(n, q);
    float *d_b = sycl::malloc_device<float>(n, q);
    float *d_c = sycl::malloc_device<float>(n, q);
    
    // Initialize data
    std::vector<float> h_a(n), h_b(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        h_a[i] = dist(gen);
        h_b[i] = dist(gen);
    }
    
    q.memcpy(d_a, h_a.data(), n * sizeof(float)).wait();
    q.memcpy(d_b, h_b.data(), n * sizeof(float)).wait();
    
    int num_wg = (n + WG_SIZE - 1) / WG_SIZE;
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        q.parallel_for(
            sycl::nd_range<1>(num_wg * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) {
                vector_add_kernel<WG_SIZE>(d_c, d_a, d_b, n, item);
            }
        ).wait();
    }
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        q.parallel_for(
            sycl::nd_range<1>(num_wg * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) {
                vector_add_kernel<WG_SIZE>(d_c, d_a, d_b, n, item);
            }
        ).wait();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    float avg_time_ms = (elapsed.count() * 1000.0f) / iterations;
    
    // Cleanup
    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);
    
    return avg_time_ms;
}

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    
    std::ofstream out("wg_size_sweep_results.txt");
    out << "=== Work-Group Size Sweep Results ===\n\n";
    out << "GPU: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    out << "Testing vector add kernel with different work-group sizes\n\n";
    
    // Test configurations
    std::vector<int> wg_sizes = {64, 128, 256, 512, 1024};
    std::vector<int> data_sizes = {256, 512, 1024, 4096, 16384, 65536, 262144};
    
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
    
    // Store results for analysis
    std::vector<std::vector<double>> results(wg_sizes.size(), 
                                      std::vector<double>(data_sizes.size()));
    
    for (size_t wg_idx = 0; wg_idx < wg_sizes.size(); ++wg_idx) {
        int wg = wg_sizes[wg_idx];
        out << "WG=" << wg << "     |";
        
        for (size_t ds_idx = 0; ds_idx < data_sizes.size(); ++ds_idx) {
            int n = data_sizes[ds_idx];
            double avg_time = 0.0;
            
            switch (wg) {
                case 64:   avg_time = benchmark_wg_size<64>(q, n, iterations); break;
                case 128:  avg_time = benchmark_wg_size<128>(q, n, iterations); break;
                case 256:  avg_time = benchmark_wg_size<256>(q, n, iterations); break;
                case 512:  avg_time = benchmark_wg_size<512>(q, n, iterations); break;
                case 1024: avg_time = benchmark_wg_size<1024>(q, n, iterations); break;
            }
            
            results[wg_idx][ds_idx] = avg_time;
            out << " " << std::fixed << std::setprecision(3) << avg_time << " |";
        }
        out << "\n";
    }
    
    // Analysis
    out << "\n=== Analysis ===\n\n";
    
    out << "Best work-group size for each data size:\n";
    for (size_t ds_idx = 0; ds_idx < data_sizes.size(); ++ds_idx) {
        double best_time = results[0][ds_idx];
        int best_wg = wg_sizes[0];
        
        for (size_t wg_idx = 1; wg_idx < wg_sizes.size(); ++wg_idx) {
            if (results[wg_idx][ds_idx] < best_time) {
                best_time = results[wg_idx][ds_idx];
                best_wg = wg_sizes[wg_idx];
            }
        }
        
        double baseline = results[2][ds_idx]; // WG=256 as baseline
        double speedup = baseline / best_time;
        
        out << "  N=" << data_sizes[ds_idx] << ": WG=" << best_wg 
            << " (" << std::fixed << std::setprecision(3) << best_time 
            << " ms, speedup vs WG=256: " << std::setprecision(2) << speedup << "x)\n";
    }
    
    out << "\nOverall best work-group size: ";
    double total_best = 1e9;
    int overall_best_wg = 256;
    for (size_t wg_idx = 0; wg_idx < wg_sizes.size(); ++wg_idx) {
        double total_time = 0;
        for (size_t ds_idx = 0; ds_idx < data_sizes.size(); ++ds_idx) {
            total_time += results[wg_idx][ds_idx];
        }
        if (total_time < total_best) {
            total_best = total_time;
            overall_best_wg = wg_sizes[wg_idx];
        }
    }
    out << overall_best_wg << "\n";
    
    out.close();
    std::cout << "Work-group size sweep completed. Results saved to wg_size_sweep_results.txt\n";
    
    return 0;
}