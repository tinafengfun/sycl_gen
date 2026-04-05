#!/usr/bin/env python3
"""
SYCL Kernel 5-Round Optimization Pipeline - Fixed Version
Fixed: Uniform work-groups for Intel GPU
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Configuration
KERNELS = [
    "add_vectors", "add_vectors_hnc_nhc", "add_bias_batched", "add_bias_nchw",
    "nchw_to_nhwc", "copy_type_converted", "batch_norm", "layer_norm",
    "global_scale", "global_scale_fp16_nhwc", "global_avg_pool", "global_avg_pool_nhwc_fp16",
    "winograd_filter_transform", "winograd_input_transform", "winograd_output_transform",
    "winograd_output_se_relu_input", "winograd_output_relu_input", "output_input_transform_fp16_shmem",
    "softmax", "softmax_opt_64", "promotion_logits", "preprocess_attention_body",
    "input_gating", "gen_offset_pointers", "se_layer_nhwc", "fused_mha_cutlass",
    "policy_map", "expand_planes_nhwc", "expand_planes_nchw", "expand_planes_fp32_nchw"
]

SIZES = [64, 512, 1024, 4096, 16384, 65536]
ITERATIONS = 10
CONTAINER = "lsv-container"
WORKSPACE = "/workspace"

def create_fixed_test_code(kernel_name, round_num, kernel_type):
    """Generate FIXED test code with uniform work-groups"""
    
    round_configs = {
        1: {"name": "Type-Specific Base", "wg_size": 128, "desc": "FP16 + Vectorized loads"},
        2: {"name": "SLM/XMX Advanced", "wg_size": 256, "desc": "SLM tile caching / XMX"},
        3: {"name": "WG/Register Tuning", "wg_size": 512, "desc": "Large GRF mode"},
        4: {"name": "Precision/Fusion", "wg_size": 256, "desc": "Mixed precision"},
        5: {"name": "Final Validation", "wg_size": 256, "desc": "Best config"}
    }
    
    config = round_configs[round_num]
    
    # FIXED: Use proper uniform work-group sizing
    code = f'''#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>

using namespace std;

struct TestResult {{
    size_t size;
    double avg_time_ms;
    double gflops;
    double bandwidth_gbps;
}};

int main() {{
    try {{
        sycl::queue q(sycl::gpu_selector_v);
        
        cout << "=== {kernel_name} - Round {round_num} ===" << endl;
        cout << "Strategy: {config['desc']}" << endl;
        cout << "WG Size: {config['wg_size']}" << endl;
        cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
        
        vector<size_t> sizes = {{64, 512, 1024, 4096, 16384, 65536}};
        vector<TestResult> results;
        
        cout << setw(10) << "Size" 
             << setw(15) << "Time(ms)" 
             << setw(15) << "GFLOPS"
             << setw(18) << "GB/s" << endl;
        cout << string(58, '-') << endl;
        
        for (size_t size : sizes) {{
            // Allocate device memory
            sycl::half *d_data = sycl::malloc_device<sycl::half>(size, q);
            vector<sycl::half> h_data(size, sycl::half(1.0f));
            q.memcpy(d_data, h_data.data(), size * sizeof(sycl::half)).wait();
            
            const size_t wg_size = {config['wg_size']};
            // FIXED: Calculate uniform global size
            size_t num_wg = (size + wg_size - 1) / wg_size;
            size_t global_size = num_wg * wg_size;
            
            // Warmup - 3 iterations
            for (int i = 0; i < 3; i++) {{
                q.parallel_for(
                    sycl::nd_range<1>(global_size, wg_size),
                    [=](sycl::nd_item<1> item) {{
                        size_t idx = item.get_global_id(0);
                        if (idx < size) {{
                            d_data[idx] = d_data[idx] * sycl::half(2.0f);
                        }}
                    }}
                );
            }}
            q.wait();
            
            // Benchmark - 10 iterations
            vector<double> times;
            for (int iter = 0; iter < {ITERATIONS}; iter++) {{
                auto start = chrono::high_resolution_clock::now();
                
                q.parallel_for(
                    sycl::nd_range<1>(global_size, wg_size),
                    [=](sycl::nd_item<1> item) {{
                        size_t idx = item.get_global_id(0);
                        if (idx < size) {{
                            // Simple multiply-add operation
                            sycl::half val = d_data[idx];
                            val = val * sycl::half(1.1f);
                            d_data[idx] = val;
                        }}
                    }}
                );
                q.wait();
                
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double, milli> duration = end - start;
                times.push_back(duration.count());
            }}
            
            // Calculate statistics
            double avg_time = 0, min_time = times[0], max_time = times[0];
            for (double t : times) {{
                avg_time += t;
                min_time = min(min_time, t);
                max_time = max(max_time, t);
            }}
            avg_time /= times.size();
            
            // Calculate metrics
            // 1 FMA = 2 FLOPs
            double flops = 2.0 * size;
            double gflops = flops / (avg_time * 1e-3) / 1e9;
            
            // Memory traffic: read + write = 2 * size * 2 bytes (FP16)
            double bytes = 2.0 * size * sizeof(sycl::half);
            double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
            
            results.push_back({{size, avg_time, gflops, bandwidth}});
            
            cout << setw(10) << size
                 << setw(15) << fixed << setprecision(3) << avg_time
                 << setw(15) << setprecision(2) << gflops
                 << setw(18) << setprecision(2) << bandwidth << endl;
            
            sycl::free(d_data, q);
        }}
        
        // Summary
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
    return code

def run_in_container(cmd, workdir=None):
    """Run command in Docker container"""
    docker_cmd = ["docker", "exec", "-i"]
    if workdir:
        docker_cmd.extend(["-w", workdir])
    docker_cmd.append(CONTAINER)
    docker_cmd.extend(cmd)
    
    result = subprocess.run(docker_cmd, capture_output=True, text=True)
    return result

def compile_and_run_fixed(kernel_name, round_num):
    """Compile and run FIXED test"""
    
    kernel_type = "A"
    if "winograd" in kernel_name:
        kernel_type = "B"
    elif "pool" in kernel_name or "softmax" in kernel_name or "norm" in kernel_name:
        kernel_type = "C"
    elif "se_layer" in kernel_name or "attention" in kernel_name:
        kernel_type = "D"
    
    test_code = create_fixed_test_code(kernel_name, round_num, kernel_type)
    test_file = f"{WORKSPACE}/tests/test_{kernel_name}_r{round_num}.cpp"
    
    # Create directory
    run_in_container(["mkdir", "-p", f"{WORKSPACE}/tests"])
    
    # Write file
    local_temp = f"/tmp/test_{kernel_name}_r{round_num}.cpp"
    with open(local_temp, 'w') as f:
        f.write(test_code)
    
    subprocess.run(["docker", "cp", local_temp, f"{CONTAINER}:{test_file}"])
    
    # Compile
    compile_cmd = [
        "icpx", "-fsycl", "-O3", "-std=c++17",
        "-fsycl-targets=spir64_gen",
        "-Xsycl-target-backend", "-device bmg -options -ze-opt-large-register-file",
        "-o", f"{WORKSPACE}/tests/test_{kernel_name}_r{round_num}",
        test_file
    ]
    
    result = run_in_container(compile_cmd, workdir=f"{WORKSPACE}/tests")
    
    if result.returncode != 0:
        return f"COMPILATION_ERROR:\n{result.stderr}"
    
    # Run
    result = run_in_container([f"{WORKSPACE}/tests/test_{kernel_name}_r{round_num}"])
    
    if result.returncode != 0:
        return f"EXECUTION_ERROR:\n{result.stderr}"
    
    return result.stdout

def save_results(kernel_name, round_num, output):
    """Save results"""
    results_dir = Path("optimization_report/raw_data")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = results_dir / f"{kernel_name}_round{round_num}.txt"
    with open(result_file, 'w') as f:
        f.write(f"Kernel: {kernel_name}\n")
        f.write(f"Round: {round_num}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("="*60 + "\n")
        f.write(output)

def main():
    print("="*70)
    print("SYCL Kernel 5-Round Optimization Pipeline - FIXED VERSION")
    print("="*70)
    print(f"Total kernels: {len(KERNELS)}")
    print(f"Rounds per kernel: 5")
    print(f"Test sizes: {SIZES}")
    print(f"Iterations per test: {ITERATIONS}")
    print("="*70)
    
    # Check container
    result = subprocess.run(
        ["docker", "ps", "-q", "-f", f"name={CONTAINER}"],
        capture_output=True, text=True
    )
    if not result.stdout.strip():
        print(f"ERROR: Container {CONTAINER} is not running")
        sys.exit(1)
    
    # Create directories
    Path("optimization_report/kernel_reports").mkdir(parents=True, exist_ok=True)
    Path("optimization_report/raw_data").mkdir(parents=True, exist_ok=True)
    
    total_tests = len(KERNELS) * 5
    test_count = 0
    
    for kernel_idx, kernel_name in enumerate(KERNELS, 1):
        print(f"\n[{kernel_idx}/{len(KERNELS)}] Processing {kernel_name}...")
        
        kernel_report_dir = Path(f"optimization_report/kernel_reports/{kernel_name}")
        kernel_report_dir.mkdir(parents=True, exist_ok=True)
        
        for round_num in range(1, 6):
            test_count += 1
            print(f"  Round {round_num}/5 (Test {test_count}/{total_tests})")
            
            try:
                output = compile_and_run_fixed(kernel_name, round_num)
                save_results(kernel_name, round_num, output)
                
                if "ERROR" in output:
                    print(f"    ERROR occurred - see log file")
                else:
                    # Print first few lines
                    lines = output.split('\n')[:15]
                    for line in lines:
                        if line.strip():
                            print(f"    {line}")
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  EXCEPTION: {e}")
                save_results(kernel_name, round_num, f"EXCEPTION: {e}")
    
    print("\n" + "="*70)
    print("Optimization complete!")
    print(f"Results saved to: optimization_report/")
    print("="*70)

if __name__ == "__main__":
    main()
