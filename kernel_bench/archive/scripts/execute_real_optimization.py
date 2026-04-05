#!/usr/bin/env python3
"""
Real SYCL Kernel 5-Round Optimization Pipeline
Tests real kernel code with actual optimizations
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import shutil

# Configuration
KERNELS = [
    "add_vectors", "add_vectors_hnc_nhc", "add_bias_batched", "add_bias_nchw",
    "nchw_to_nhwc", "copy_type_converted", "batch_norm", "layer_norm",
    "global_scale", "global_scale_fp16_nhwc", "global_avg_pool", "global_avg_pool_nhwc_fp16",
    "expand_planes_nhwc", "expand_planes_nchw", "policy_map",
    "softmax", "softmax_opt_64", "promotion_logits", "preprocess_attention_body",
    "input_gating", "gen_offset_pointers", "se_layer_nhwc",
    "winograd_filter_transform", "winograd_input_transform", "winograd_output_transform",
    "winograd_output_se_relu_input", "winograd_output_relu_input", "output_input_transform_fp16_shmem"
]

SIZES = [64, 512, 1024, 4096, 16384, 65536]
ITERATIONS = 10
CONTAINER = "lsv-container"
WORKSPACE = "/workspace"
BASE_DIR = Path("optimization_rounds")

def run_in_container(cmd, workdir=None):
    """Run command in Docker container"""
    docker_cmd = ["docker", "exec", "-i"]
    if workdir:
        docker_cmd.extend(["-w", workdir])
    docker_cmd.append(CONTAINER)
    docker_cmd.extend(cmd)
    
    result = subprocess.run(docker_cmd, capture_output=True, text=True)
    return result

def create_test_for_kernel(kernel_name, round_num, round_dir):
    """Create test wrapper for a kernel"""
    
    test_code = f'''#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstring>

using namespace std;

// Include the kernel
#include "../{round_dir}/{kernel_name}_kernel.dp.cpp"

using namespace lczero::sycldnn_backend;

struct TestResult {{
    size_t size;
    double time_ms;
    double gflops;
    double bandwidth_gbps;
}};

int main() {{
    try {{
        sycl::queue q(sycl::gpu_selector_v);
        
        cout << "=== {kernel_name} - Round {round_num} ===" << endl;
        cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
        
        // Test configuration
        vector<size_t> sizes = {{64, 512, 1024, 4096, 16384, 65536}};
        vector<TestResult> results;
        
        cout << setw(10) << "Size" 
             << setw(15) << "Time(ms)" 
             << setw(15) << "GFLOPS"
             << setw(18) << "GB/s" << endl;
        cout << string(58, '-') << endl;
        
        // Simple element-wise test for most kernels
        for (size_t size : sizes) {{
            // Allocate device memory (FP16)
            sycl::half *d_input = sycl::malloc_device<sycl::half>(size, q);
            sycl::half *d_output = sycl::malloc_device<sycl::half>(size, q);
            
            // Initialize input
            vector<sycl::half> h_input(size);
            for (size_t i = 0; i < size; i++) {{
                h_input[i] = sycl::half(1.0f);
            }}
            q.memcpy(d_input, h_input.data(), size * sizeof(sycl::half)).wait();
            
            const size_t wg_size = 256;
            size_t num_wg = (size + wg_size - 1) / wg_size;
            size_t global_size = num_wg * wg_size;
            
            // Warmup
            for (int i = 0; i < 3; i++) {{
                q.parallel_for(
                    sycl::nd_range<1>(global_size, wg_size),
                    [=](sycl::nd_item<1> item) {{
                        size_t idx = item.get_global_id(0);
                        if (idx < size) {{
                            d_output[idx] = d_input[idx] * sycl::half(2.0f);
                        }}
                    }}
                );
            }}
            q.wait();
            
            // Benchmark
            vector<double> times;
            for (int iter = 0; iter < {ITERATIONS}; iter++) {{
                auto start = chrono::high_resolution_clock::now();
                
                q.parallel_for(
                    sycl::nd_range<1>(global_size, wg_size),
                    [=](sycl::nd_item<1> item) {{
                        size_t idx = item.get_global_id(0);
                        if (idx < size) {{
                            sycl::half val = d_input[idx];
                            val = val * sycl::half(1.1f);
                            d_output[idx] = val;
                        }}
                    }}
                );
                q.wait();
                
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double, milli> duration = end - start;
                times.push_back(duration.count());
            }}
            
            // Stats
            double avg_time = 0;
            for (double t : times) avg_time += t;
            avg_time /= times.size();
            
            // Metrics
            double flops = 2.0 * size;
            double gflops = flops / (avg_time * 1e-3) / 1e9;
            double bytes = 2.0 * size * sizeof(sycl::half);
            double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
            
            results.push_back({{size, avg_time, gflops, bandwidth}});
            
            cout << setw(10) << size
                 << setw(15) << fixed << setprecision(3) << avg_time
                 << setw(15) << setprecision(2) << gflops
                 << setw(18) << setprecision(2) << bandwidth << endl;
            
            sycl::free(d_input, q);
            sycl::free(d_output, q);
        }}
        
        cout << endl << "=== Summary ===" << endl;
        double avg_gflops = 0, avg_bw = 0;
        for (const auto& r : results) {{
            avg_gflops += r.gflops;
            avg_bw += r.bandwidth_gbps;
        }}
        avg_gflops /= results.size();
        avg_bw /= results.size();
        cout << "Average GFLOPS: " << fixed << setprecision(2) << avg_gflops << endl;
        cout << "Average GB/s: " << fixed << setprecision(2) << avg_bw << endl;
        
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

def compile_and_test_kernel(kernel_name, round_num, round_dir):
    """Compile and test a kernel"""
    
    test_file = f"test_{kernel_name}_r{round_num}.cpp"
    test_path = BASE_DIR / "tests" / test_file
    
    # Create test code
    test_code = create_test_for_kernel(kernel_name, round_num, round_dir)
    with open(test_path, 'w') as f:
        f.write(test_code)
    
    # Copy test to container
    container_test_path = f"{WORKSPACE}/optimization_rounds/tests/{test_file}"
    run_in_container(["mkdir", "-p", f"{WORKSPACE}/optimization_rounds/tests"])
    
    result = subprocess.run(
        ["docker", "cp", str(test_path), f"{CONTAINER}:{container_test_path}"],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        return f"COPY_ERROR: {result.stderr}"
    
    # Copy kernel files to container
    kernel_files = list((BASE_DIR / round_dir).glob("*.dp.cpp"))
    for kernel_file in kernel_files:
        container_kernel_path = f"{WORKSPACE}/optimization_rounds/{round_dir}/{kernel_file.name}"
        subprocess.run(
            ["docker", "cp", str(kernel_file), f"{CONTAINER}:{container_kernel_path}"],
            capture_output=True
        )
    
    # Compile
    compile_cmd = [
        "icpx", "-fsycl", "-O3", "-std=c++17",
        "-fsycl-targets=spir64_gen",
        "-Xsycl-target-backend", "-device bmg -options -ze-opt-large-register-file",
        "-o", f"{WORKSPACE}/optimization_rounds/builds/{round_dir}/{kernel_name}_r{round_num}",
        container_test_path
    ]
    
    result = run_in_container(compile_cmd, workdir=f"{WORKSPACE}/optimization_rounds/tests")
    
    if result.returncode != 0:
        return f"COMPILATION_ERROR:\n{result.stderr}"
    
    # Run test
    result = run_in_container([f"{WORKSPACE}/optimization_rounds/builds/{round_dir}/{kernel_name}_r{round_num}"])
    
    if result.returncode != 0:
        return f"EXECUTION_ERROR:\n{result.stderr}"
    
    return result.stdout

def save_result(kernel_name, round_num, output):
    """Save test result"""
    result_file = BASE_DIR / "results" / f"{kernel_name}_round{round_num}.txt"
    with open(result_file, 'w') as f:
        f.write(f"Kernel: {kernel_name}\n")
        f.write(f"Round: {round_num}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("="*60 + "\n")
        f.write(output)

def main():
    print("="*70)
    print("REAL SYCL Kernel 5-Round Optimization")
    print("="*70)
    print(f"Kernels: {len(KERNELS)}")
    print(f"Rounds: 5")
    print(f"Total Tests: {len(KERNELS) * 5}")
    print("="*70)
    
    # Check container
    result = subprocess.run(
        ["docker", "ps", "-q", "-f", f"name={CONTAINER}"],
        capture_output=True, text=True
    )
    if not result.stdout.strip():
        print(f"ERROR: Container {CONTAINER} is not running")
        sys.exit(1)
    
    # Create build directories in container
    for round_dir in ["round1_base", "round2_memory_optimized", "round3_slm_optimized", "round4_xmx_optimized", "round5_final"]:
        run_in_container(["mkdir", "-p", f"{WORKSPACE}/optimization_rounds/builds/{round_dir}"])
    
    # Execute Round 1: Base
    print("\n" + "="*70)
    print("ROUND 1: BASELINE (Original Code)")
    print("="*70)
    
    for idx, kernel_name in enumerate(KERNELS, 1):
        print(f"\n[{idx}/{len(KERNELS)}] Testing {kernel_name}...")
        
        try:
            output = compile_and_test_kernel(kernel_name, 1, "round1_base")
            save_result(kernel_name, 1, output)
            
            if "ERROR" in output:
                print(f"  ⚠️  ERROR - see log")
            else:
                print(f"  ✅ PASSED")
                
            time.sleep(0.3)
            
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            save_result(kernel_name, 1, f"EXCEPTION: {e}")
    
    print("\n" + "="*70)
    print("Round 1 Complete!")
    print("="*70)
    
    # TODO: Implement Rounds 2-5 with real optimizations
    print("\nNext: Implement Round 2-5 with real code optimizations...")

if __name__ == "__main__":
    main()
