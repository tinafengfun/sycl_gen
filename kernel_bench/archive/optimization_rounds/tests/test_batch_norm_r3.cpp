#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

using namespace std;

// Include the kernel
#include "../round3_slm_optimized/batch_norm_kernel.dp.cpp"

using namespace lczero::sycldnn_backend;

// Helper to allocate and initialize device memory
template<typename T>
T* alloc_and_init(sycl::queue& q, size_t size, T init_val) {
    T* d_ptr = sycl::malloc_device<T>(size, q);
    vector<T> h_data(size, init_val);
    q.memcpy(d_ptr, h_data.data(), size * sizeof(T)).wait();
    return d_ptr;
}

template<typename T>
void verify_and_cleanup(sycl::queue& q, T* d_ptr, size_t size, const char* name) {
    vector<T> h_data(size);
    q.memcpy(h_data.data(), d_ptr, size * sizeof(T)).wait();
    
    // Simple verification - check if values are reasonable
    bool has_nan = false;
    for (size_t i = 0; i < min(size_t(100), size); i++) {
        if (std::isnan((float)h_data[i])) {
            has_nan = true;
            break;
        }
    }
    
    if (has_nan) {
        cerr << "WARNING: " << name << " contains NaN values!" << endl;
    }
    
    sycl::free(d_ptr, q);
}

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v);
        
        cout << "=== batch_norm - Round 3 ===" << endl;
        cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
        
        // Test configurations
        struct TestConfig {
            int N, C, H, W;
            size_t total_size;
        };
        
        vector<TestConfig> configs = {
            {1, 64, 8, 8},      // 4096 elements
            {1, 128, 16, 16},  // 32768 elements  
            {1, 256, 32, 32},  // 262144 elements
            {4, 256, 32, 32},  // 1048576 elements (~1M)
        };
        
        // Calculate sizes
        for (auto& cfg : configs) {
            cfg.total_size = (size_t)cfg.N * cfg.C * cfg.H * cfg.W;
        }
        
        cout << setw(15) << "Config (NCHW)" 
             << setw(15) << "Size"
             << setw(15) << "Time(ms)" 
             << setw(15) << "GFLOPS"
             << setw(18) << "GB/s" << endl;
        cout << string(78, '-') << endl;
        
        for (const auto& cfg : configs) {
            const size_t size = cfg.total_size;
            
            // Allocate device memory
            sycl::half* d_input = alloc_and_init<sycl::half>(q, size, sycl::half(0.5f));
            sycl::half* d_output = alloc_and_init<sycl::half>(q, size, sycl::half(0.0f));
            
            // Allocate parameter arrays (for batch_norm, layer_norm, etc.)
            float* d_means = sycl::malloc_device<float>(cfg.C, q);
            float* d_var_multipliers = sycl::malloc_device<float>(cfg.C, q);
            vector<float> h_params(cfg.C, 1.0f);
            q.memcpy(d_means, h_params.data(), cfg.C * sizeof(float)).wait();
            q.memcpy(d_var_multipliers, h_params.data(), cfg.C * sizeof(float)).wait();
            
            // Warmup
            for (int i = 0; i < 3; i++) {
                batchNorm(d_output, d_input, nullptr, cfg.N, cfg.C, cfg.H, cfg.W,
                         d_means, d_var_multipliers, ACTIVATION_RELU, q);
            }
            q.wait();
            
            // Benchmark
            vector<double> times;
            for (int iter = 0; iter < 10; iter++) {
                auto start = chrono::high_resolution_clock::now();
                
                batchNorm(d_output, d_input, nullptr, cfg.N, cfg.C, cfg.H, cfg.W,
                         d_means, d_var_multipliers, ACTIVATION_RELU, q);
                q.wait();
                
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double, milli> duration = end - start;
                times.push_back(duration.count());
            }
            
            // Stats
            double avg_time = 0;
            for (double t : times) avg_time += t;
            avg_time /= times.size();
            
            // Calculate metrics
            double flops = 4 * size;
            double gflops = flops / (avg_time * 1e-3) / 1e9;
            
            double bytes = 2.0 * size * sizeof(sycl::half);  // read + write
            double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
            
            cout << setw(15) << (to_string(cfg.N) + "x" + to_string(cfg.C) + "x" + to_string(cfg.H) + "x" + to_string(cfg.W))
                 << setw(15) << size
                 << setw(15) << fixed << setprecision(3) << avg_time
                 << setw(15) << setprecision(2) << gflops
                 << setw(18) << setprecision(2) << bandwidth << endl;
            
            // Cleanup
            verify_and_cleanup(q, d_input, size, "input");
            verify_and_cleanup(q, d_output, size, "output");
            sycl::free(d_means, q);
            sycl::free(d_var_multipliers, q);
        }
        
        cout << endl << "Test completed successfully!" << endl;
        return 0;
    } catch (sycl::exception const &e) {
        cerr << "SYCL Exception: " << e.what() << endl;
        return 1;
    } catch (exception const &e) {
        cerr << "Exception: " << e.what() << endl;
        return 1;
    }
}
