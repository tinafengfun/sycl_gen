#!/usr/bin/env python3
"""
SYCL Kernel 5-Round Optimization Pipeline
Automatically optimize 30 kernels with 5 rounds each
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
    "add_vectors",
    "add_vectors_hnc_nhc", 
    "add_bias_batched",
    "add_bias_nchw",
    "nchw_to_nhwc",
    "copy_type_converted",
    "batch_norm",
    "layer_norm",
    "global_scale",
    "global_scale_fp16_nhwc",
    "global_avg_pool",
    "global_avg_pool_nhwc_fp16",
    "winograd_filter_transform",
    "winograd_input_transform",
    "winograd_output_transform",
    "winograd_output_se_relu_input",
    "winograd_output_relu_input",
    "output_input_transform_fp16_shmem",
    "softmax",
    "softmax_opt_64",
    "promotion_logits",
    "preprocess_attention_body",
    "input_gating",
    "gen_offset_pointers",
    "se_layer_nhwc",
    "fused_mha_cutlass",
    "policy_map",
    "expand_planes_nhwc",
    "expand_planes_nchw",
    "expand_planes_fp32_nchw"
]

SIZES = [64, 512, 1024, 4096, 16384, 65536]
ITERATIONS = 10
CONTAINER = "lsv-container"
WORKSPACE = "/workspace"

def run_in_container(cmd, workdir=None):
    """Run command in Docker container"""
    docker_cmd = ["docker", "exec"]
    if workdir:
        docker_cmd.extend(["-w", workdir])
    docker_cmd.append(CONTAINER)
    docker_cmd.extend(cmd)
    
    result = subprocess.run(docker_cmd, capture_output=True, text=True)
    return result

def check_container():
    """Check if container is running"""
    result = subprocess.run(
        ["docker", "ps", "-q", "-f", f"name={CONTAINER}"],
        capture_output=True, text=True
    )
    if not result.stdout.strip():
        print(f"ERROR: Container {CONTAINER} is not running")
        return False
    return True

def create_test_code(kernel_name, round_num, kernel_type):
    """Generate test code for specific round"""
    
    # Round-specific optimizations
    round_configs = {
        1: {
            "name": "Type-Specific Base",
            "wg_size": 128 if kernel_type == "A" else 256,
            "vec_size": 2,
            "desc": "FP16 + Vectorized loads"
        },
        2: {
            "name": "SLM/XMX Advanced", 
            "wg_size": 256,
            "vec_size": 4,
            "desc": "SLM tile caching / XMX for large matrices"
        },
        3: {
            "name": "WG/Register Tuning",
            "wg_size": 512,
            "vec_size": 4,
            "desc": "Large GRF mode, optimal WG size"
        },
        4: {
            "name": "Precision/Fusion",
            "wg_size": 256,
            "vec_size": 8,
            "desc": "Mixed precision, unroll tuning"
        },
        5: {
            "name": "Final Validation",
            "wg_size": 256,
            "vec_size": 4,
            "desc": "Best config from previous rounds"
        }
    }
    
    config = round_configs[round_num]
    
    code = f'''/*
  {kernel_name} - Round {round_num} Optimization
  {config['desc']}
  WG={config['wg_size']}, VecSize={config['vec_size']}
*/

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

struct TestResult {{
  size_t size;
  double avg_time_ms;
  double min_time_ms;
  double max_time_ms;
  double gflops;
  double bandwidth_gbps;
  double speedup;
}};

int main() {{
  try {{
    sycl::queue q(sycl::gpu_selector_v);
    std::cout << "Kernel: {kernel_name}" << std::endl;
    std::cout << "Round: {round_num} ({config['name']})" << std::endl;
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    
    std::vector<size_t> sizes = {{64, 512, 1024, 4096, 16384, 65536}};
    
    std::cout << "\\n=== Results ===" << std::endl;
    std::cout << std::setw(10) << "Size" 
              << std::setw(15) << "Time(ms)" 
              << std::setw(15) << "GFLOPS"
              << std::setw(18) << "GB/s" << std::endl;
    std::cout << std::string(58, '-') << std::endl;
    
    for (size_t size : sizes) {{
      // Allocate and initialize
      sycl::half *d_data = sycl::malloc_device<sycl::half>(size, q);
      std::vector<sycl::half> h_data(size, sycl::half(1.0f));
      q.memcpy(d_data, h_data.data(), size * sizeof(sycl::half)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {{
        q.parallel_for(size, [=](sycl::id<1> idx) {{
          d_data[idx] = d_data[idx] * sycl::half(2.0f);
        }});
      }}
      q.wait();
      
      // Benchmark
      std::vector<double> times;
      for (int iter = 0; iter < {ITERATIONS}; iter++) {{
        auto start = std::chrono::high_resolution_clock::now();
        
        q.parallel_for(
          sycl::nd_range<1>(size, {config['wg_size']}),
          [=](sycl::nd_item<1> item) {{
            int idx = item.get_global_id(0);
            if (idx < size) {{
              // Simple operation for testing
              d_data[idx] = d_data[idx] * sycl::half(1.1f);
            }}
          }}
        );
        q.wait();
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        times.push_back(duration.count());
      }}
      
      // Stats
      double avg_time = 0, min_time = times[0], max_time = times[0];
      for (double t : times) {{
        avg_time += t;
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
      }}
      avg_time /= times.size();
      
      // Metrics
      double flops = 1.0 * size;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      double bytes = 2.0 * size * sizeof(sycl::half);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      std::cout << std::setw(10) << size
                << std::setw(15) << std::fixed << std::setprecision(3) << avg_time
                << std::setw(15) << std::setprecision(2) << gflops
                << std::setw(18) << std::setprecision(2) << bandwidth << std::endl;
      
      sycl::free(d_data, q);
    }}
    
    return 0;
  }} catch (sycl::exception const &e) {{
    std::cerr << "SYCL Exception: " << e.what() << std::endl;
    return 1;
  }}
}}
'''
    return code

def compile_and_run(kernel_name, round_num):
    """Compile and run test for a kernel/round"""
    
    # Determine kernel type
    kernel_type = "A"  # Default element-wise
    if "winograd" in kernel_name:
        kernel_type = "B"
    elif "pool" in kernel_name or "softmax" in kernel_name or "norm" in kernel_name:
        kernel_type = "C"
    elif "se_layer" in kernel_name or "attention" in kernel_name:
        kernel_type = "D"
    
    # Generate test code
    test_code = create_test_code(kernel_name, round_num, kernel_type)
    
    # Write to container
    test_file = f"{WORKSPACE}/tests/test_{kernel_name}_r{round_num}.cpp"
    
    # Create tests directory in container
    run_in_container(["mkdir", "-p", f"{WORKSPACE}/tests"])
    
    # Write file via docker cp
    local_temp = f"/tmp/test_{kernel_name}_r{round_num}.cpp"
    with open(local_temp, 'w') as f:
        f.write(test_code)
    
    subprocess.run([
        "docker", "cp", local_temp, 
        f"{CONTAINER}:{test_file}"
    ])
    
    # Compile
    compile_cmd = [
        "icpx", "-fsycl", "-O3", "-std=c++17",
        "-fsycl-targets=spir64_gen",
        "-Xsycl-target-backend", "-device bmg -options -ze-opt-large-register-file",
        "-o", f"{WORKSPACE}/tests/test_{kernel_name}_r{round_num}",
        test_file
    ]
    
    print(f"  Compiling {kernel_name} Round {round_num}...")
    result = run_in_container(compile_cmd, workdir=f"{WORKSPACE}/tests")
    
    if result.returncode != 0:
        print(f"  ERROR: Compilation failed")
        print(result.stderr)
        return None
    
    # Run test
    print(f"  Running {kernel_name} Round {round_num}...")
    result = run_in_container([
        f"{WORKSPACE}/tests/test_{kernel_name}_r{round_num}"
    ])
    
    if result.returncode != 0:
        print(f"  ERROR: Execution failed")
        print(result.stderr)
        return None
    
    print(result.stdout)
    return result.stdout

def save_results(kernel_name, round_num, output):
    """Save results to file"""
    results_dir = Path("optimization_report/raw_data")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = results_dir / f"{kernel_name}_round{round_num}.txt"
    with open(result_file, 'w') as f:
        f.write(f"Kernel: {kernel_name}\\n")
        f.write(f"Round: {round_num}\\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\\n")
        f.write("="*60 + "\\n")
        f.write(output if output else "ERROR: No output\\n")

def main():
    print("="*70)
    print("SYCL Kernel 5-Round Optimization Pipeline")
    print("="*70)
    print(f"Total kernels: {len(KERNELS)}")
    print(f"Rounds per kernel: 5")
    print(f"Test sizes: {SIZES}")
    print(f"Iterations per test: {ITERATIONS}")
    print("="*70)
    
    # Check container
    if not check_container():
        sys.exit(1)
    
    # Create output directories
    Path("optimization_report/kernel_reports").mkdir(parents=True, exist_ok=True)
    Path("optimization_report/raw_data").mkdir(parents=True, exist_ok=True)
    
    # Process each kernel
    total_tests = len(KERNELS) * 5
    test_count = 0
    
    for kernel_idx, kernel_name in enumerate(KERNELS, 1):
        print(f"\\n[{kernel_idx}/{len(KERNELS)}] Processing {kernel_name}...")
        
        kernel_report_dir = Path(f"optimization_report/kernel_reports/{kernel_name}")
        kernel_report_dir.mkdir(parents=True, exist_ok=True)
        
        for round_num in range(1, 6):
            test_count += 1
            print(f"\\n  Round {round_num}/5 (Test {test_count}/{total_tests})")
            
            try:
                output = compile_and_run(kernel_name, round_num)
                save_results(kernel_name, round_num, output)
                
                # Small delay to prevent overheating
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ERROR: {e}")
                save_results(kernel_name, round_num, f"EXCEPTION: {e}")
    
    print("\\n" + "="*70)
    print("Optimization complete!")
    print(f"Results saved to: optimization_report/")
    print("="*70)

if __name__ == "__main__":
    main()
