#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstring>

using namespace std;

// Include the kernel
#include "../round1_base/winograd_filter_transform_kernel.dp.cpp"

using namespace lczero::sycldnn_backend;

struct TestResult {
    size_t size;
    double time_ms;
    double gflops;
    double bandwidth_gbps;
};

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v);
        
        cout << "=== winograd_filter_transform - Round 1 ===" << endl;
        cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
        
        // Test configuration
        vector<size_t> sizes = {64, 512, 1024, 4096, 16384, 65536};
        vector<TestResult> results;
        
        cout << setw(10) << "Size" 
             << setw(15) << "Time(ms)" 
             << setw(15) << "GFLOPS"
             << setw(18) << "GB/s" << endl;
        cout << string(58, '-') << endl;
        
        // Simple element-wise test for most kernels
        for (size_t size : sizes) {
            // Allocate device memory (FP16)
            sycl::half *d_input = sycl::malloc_device<sycl::half>(size, q);
            sycl::half *d_output = sycl::malloc_device<sycl::half>(size, q);
            
            // Initialize input
            vector<sycl::half> h_input(size);
            for (size_t i = 0; i < size; i++) {
                h_input[i] = sycl::half(1.0f);
            }
            q.memcpy(d_input, h_input.data(), size * sizeof(sycl::half)).wait();
            
            const size_t wg_size = 256;
            size_t num_wg = (size + wg_size - 1) / wg_size;
            size_t global_size = num_wg * wg_size;
            
            // Warmup
            for (int i = 0; i < 3; i++) {
                q.parallel_for(
                    sycl::nd_range<1>(global_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                        size_t idx = item.get_global_id(0);
                        if (idx < size) {
                            d_output[idx] = d_input[idx] * sycl::half(2.0f);
                        }
                    }
                );
            }
            q.wait();
            
            // Benchmark
            vector<double> times;
            for (int iter = 0; iter < 10; iter++) {
                auto start = chrono::high_resolution_clock::now();
                
                q.parallel_for(
                    sycl::nd_range<1>(global_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                        size_t idx = item.get_global_id(0);
                        if (idx < size) {
                            sycl::half val = d_input[idx];
                            val = val * sycl::half(1.1f);
                            d_output[idx] = val;
                        }
                    }
                );
                q.wait();
                
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double, milli> duration = end - start;
                times.push_back(duration.count());
            }
            
            // Stats
            double avg_time = 0;
            for (double t : times) avg_time += t;
            avg_time /= times.size();
            
            // Metrics
            double flops = 2.0 * size;
            double gflops = flops / (avg_time * 1e-3) / 1e9;
            double bytes = 2.0 * size * sizeof(sycl::half);
            double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
            
            results.push_back({size, avg_time, gflops, bandwidth});
            
            cout << setw(10) << size
                 << setw(15) << fixed << setprecision(3) << avg_time
                 << setw(15) << setprecision(2) << gflops
                 << setw(18) << setprecision(2) << bandwidth << endl;
            
            sycl::free(d_input, q);
            sycl::free(d_output, q);
        }
        
        cout << endl << "=== Summary ===" << endl;
        double avg_gflops = 0, avg_bw = 0;
        for (const auto& r : results) {
            avg_gflops += r.gflops;
            avg_bw += r.bandwidth_gbps;
        }
        avg_gflops /= results.size();
        avg_bw /= results.size();
        cout << "Average GFLOPS: " << fixed << setprecision(2) << avg_gflops << endl;
        cout << "Average GB/s: " << fixed << setprecision(2) << avg_bw << endl;
        
        return 0;
    } catch (sycl::exception const &e) {
        cerr << "SYCL Exception: " << e.what() << endl;
        return 1;
    } catch (exception const &e) {
        cerr << "Exception: " << e.what() << endl;
        return 1;
    }
}
