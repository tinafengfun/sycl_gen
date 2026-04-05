#!/usr/bin/env python3
"""
Generate real test code for each kernel based on its specific interface
"""

import os
from pathlib import Path

BASE_DIR = Path("optimization_rounds")

# Kernel interface definitions
KERNEL_INTERFACES = {
    "add_vectors": {
        "template": "T",
        "params": ["c", "a", "b", "size", "asize", "bsize", "activation", "queue"],
        "types": ["T*", "T*", "T*", "int", "int", "int", "ActivationFunction", "sycl::queue&"],
        "input_sizes": ["size"],
        "dimensions": 1,
        "flops_per_element": 2,  # add + activation
    },
    "batch_norm": {
        "template": "T",
        "params": ["output", "input", "skipInput", "N", "C", "H", "W", "means", "var_multipliers", "activation", "queue"],
        "types": ["T*", "const T*", "const T*", "int", "int", "int", "int", "const float*", "const float*", "ActivationFunction", "sycl::queue&"],
        "input_sizes": ["N*C*H*W"],
        "dimensions": 4,
        "flops_per_element": 4,  # sub + mul + add + activation
    },
    "layer_norm": {
        "template": "T",
        "params": ["output", "input", "N", "C", "H", "W", "means", "var_multipliers", "activation", "queue"],
        "types": ["T*", "const T*", "int", "int", "int", "int", "const float*", "const float*", "ActivationFunction", "sycl::queue&"],
        "input_sizes": ["N*C*H*W"],
        "dimensions": 4,
        "flops_per_element": 4,
    },
    "global_scale": {
        "template": "T",
        "params": ["output", "input", "N", "C", "H", "W", "scale", "activation", "queue"],
        "types": ["T*", "const T*", "int", "int", "int", "int", "float", "ActivationFunction", "sycl::queue&"],
        "input_sizes": ["N*C*H*W"],
        "dimensions": 4,
        "flops_per_element": 2,  # mul + activation
    },
    "global_avg_pool": {
        "template": "T",
        "params": ["output", "input", "N", "C", "H", "W", "queue"],
        "types": ["T*", "const T*", "int", "int", "int", "int", "sycl::queue&"],
        "input_sizes": ["N*C*H*W"],
        "output_sizes": ["N*C"],
        "dimensions": 4,
        "flops_per_element": 3,  # add + div
    },
}

def generate_test_for_kernel(kernel_name, round_num, round_dir):
    """Generate specialized test code for a kernel"""
    
    if kernel_name not in KERNEL_INTERFACES:
        # Generic test for unknown kernels
        return generate_generic_test(kernel_name, round_num, round_dir)
    
    interface = KERNEL_INTERFACES[kernel_name]
    
    # Generate test code
    test_code = f'''#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

using namespace std;

// Include the kernel
#include "../{round_dir}/{kernel_name}_kernel.dp.cpp"

using namespace lczero::sycldnn_backend;

// Helper to allocate and initialize device memory
template<typename T>
T* alloc_and_init(sycl::queue& q, size_t size, T init_val) {{
    T* d_ptr = sycl::malloc_device<T>(size, q);
    vector<T> h_data(size, init_val);
    q.memcpy(d_ptr, h_data.data(), size * sizeof(T)).wait();
    return d_ptr;
}}

template<typename T>
void verify_and_cleanup(sycl::queue& q, T* d_ptr, size_t size, const char* name) {{
    vector<T> h_data(size);
    q.memcpy(h_data.data(), d_ptr, size * sizeof(T)).wait();
    
    // Simple verification - check if values are reasonable
    bool has_nan = false;
    for (size_t i = 0; i < min(size_t(100), size); i++) {{
        if (std::isnan((float)h_data[i])) {{
            has_nan = true;
            break;
        }}
    }}
    
    if (has_nan) {{
        cerr << "WARNING: " << name << " contains NaN values!" << endl;
    }}
    
    sycl::free(d_ptr, q);
}}

int main() {{
    try {{
        sycl::queue q(sycl::gpu_selector_v);
        
        cout << "=== {kernel_name} - Round {round_num} ===" << endl;
        cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
        
        // Test configurations
        struct TestConfig {{
            int N, C, H, W;
            size_t total_size;
        }};
        
        vector<TestConfig> configs = {{
            {{1, 64, 8, 8}},      // 4096 elements
            {{1, 128, 16, 16}},  // 32768 elements  
            {{1, 256, 32, 32}},  // 262144 elements
            {{4, 256, 32, 32}},  // 1048576 elements (~1M)
        }};
        
        // Calculate sizes
        for (auto& cfg : configs) {{
            cfg.total_size = (size_t)cfg.N * cfg.C * cfg.H * cfg.W;
        }}
        
        cout << setw(15) << "Config (NCHW)" 
             << setw(15) << "Size"
             << setw(15) << "Time(ms)" 
             << setw(15) << "GFLOPS"
             << setw(18) << "GB/s" << endl;
        cout << string(78, '-') << endl;
        
        for (const auto& cfg : configs) {{
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
            for (int i = 0; i < 3; i++) {{
'''
    
    # Add kernel-specific call
    if kernel_name == "add_vectors":
        test_code += f'''                addVectors(d_output, d_input, d_input, size, size, size, 
                          ACTIVATION_RELU, q);
'''
    elif kernel_name == "batch_norm":
        test_code += f'''                batchNorm(d_output, d_input, nullptr, cfg.N, cfg.C, cfg.H, cfg.W,
                         d_means, d_var_multipliers, ACTIVATION_RELU, q);
'''
    elif kernel_name == "layer_norm":
        test_code += f'''                layerNorm(d_output, d_input, cfg.N, cfg.C, cfg.H, cfg.W,
                         d_means, d_var_multipliers, ACTIVATION_RELU, q);
'''
    elif kernel_name == "global_scale":
        test_code += f'''                globalScale(d_output, d_input, cfg.N, cfg.C, cfg.H, cfg.W,
                           1.5f, ACTIVATION_RELU, q);
'''
    elif kernel_name == "global_avg_pool":
        test_code += f'''                // Global avg pool outputs N*C elements
                sycl::half* d_pool_output = sycl::malloc_device<sycl::half>(cfg.N * cfg.C, q);
                globalAvgPool(d_pool_output, d_input, cfg.N, cfg.C, cfg.H, cfg.W, q);
                sycl::free(d_pool_output, q);
'''
    
    test_code += f'''            }}
            q.wait();
            
            // Benchmark
            vector<double> times;
            for (int iter = 0; iter < 10; iter++) {{
                auto start = chrono::high_resolution_clock::now();
                
'''
    
    # Repeat kernel call
    if kernel_name == "add_vectors":
        test_code += f'''                addVectors(d_output, d_input, d_input, size, size, size, 
                          ACTIVATION_RELU, q);
'''
    elif kernel_name == "batch_norm":
        test_code += f'''                batchNorm(d_output, d_input, nullptr, cfg.N, cfg.C, cfg.H, cfg.W,
                         d_means, d_var_multipliers, ACTIVATION_RELU, q);
'''
    elif kernel_name == "layer_norm":
        test_code += f'''                layerNorm(d_output, d_input, cfg.N, cfg.C, cfg.H, cfg.W,
                         d_means, d_var_multipliers, ACTIVATION_RELU, q);
'''
    elif kernel_name == "global_scale":
        test_code += f'''                globalScale(d_output, d_input, cfg.N, cfg.C, cfg.H, cfg.W,
                           1.5f, ACTIVATION_RELU, q);
'''
    elif kernel_name == "global_avg_pool":
        test_code += f'''                sycl::half* d_pool_output = sycl::malloc_device<sycl::half>(cfg.N * cfg.C, q);
                globalAvgPool(d_pool_output, d_input, cfg.N, cfg.C, cfg.H, cfg.W, q);
                sycl::free(d_pool_output, q);
'''
    
    flops_per_element = interface.get("flops_per_element", 2)
    
    test_code += f'''                q.wait();
                
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double, milli> duration = end - start;
                times.push_back(duration.count());
            }}
            
            // Stats
            double avg_time = 0;
            for (double t : times) avg_time += t;
            avg_time /= times.size();
            
            // Calculate metrics
            double flops = {flops_per_element} * size;
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
        }}
        
        cout << endl << "Test completed successfully!" << endl;
        return 0;
    }} catch (sycl::exception const &e) {{
        cerr << "SYCL Exception: " << e.what() << endl;
        return 1;
    }} catch (exception const &e) {{
        cerr << "Exception: " << e.what() << endl;
        return 1;
    }}
}}
'''
    
    return test_code

def generate_generic_test(kernel_name, round_num, round_dir):
    """Generate a generic test for kernels without specific interface definition"""
    
    return f'''#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace std;

// Include the kernel (may need adjustment based on actual interface)
// #include "../{round_dir}/{kernel_name}_kernel.dp.cpp"

int main() {{
    cout << "=== {kernel_name} - Round {round_num} ===" << endl;
    cout << "Note: Generic test - kernel interface not yet defined" << endl;
    cout << "Kernel file: ../{round_dir}/{kernel_name}_kernel.dp.cpp" << endl;
    
    // Placeholder - this kernel needs custom test based on its specific interface
    cout << "Please check the kernel file and implement appropriate test." << endl;
    
    return 0;
}}
'''

def main():
    # Generate tests for known kernels first
    kernels_to_test = ["add_vectors", "batch_norm", "layer_norm", "global_scale", "global_avg_pool"]
    
    for kernel_name in kernels_to_test:
        print(f"Generating test for {kernel_name}...")
        
        # Round 1
        test_code = generate_test_for_kernel(kernel_name, 1, "round1_base")
        test_file = BASE_DIR / "tests" / f"test_{kernel_name}_r1.cpp"
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        # Round 2-5 will use same test but link to different kernel versions
        for round_num in range(2, 6):
            round_dir = f"round{round_num}_" + ["memory_optimized", "slm_optimized", "xmx_optimized", "final"][round_num-2]
            test_code = generate_test_for_kernel(kernel_name, round_num, round_dir)
            test_file = BASE_DIR / "tests" / f"test_{kernel_name}_r{round_num}.cpp"
            with open(test_file, 'w') as f:
                f.write(test_code)
    
    print(f"\nGenerated tests for {len(kernels_to_test)} kernels × 5 rounds = {len(kernels_to_test) * 5} test files")
    print(f"Test files saved to: {BASE_DIR}/tests/")

if __name__ == "__main__":
    main()
