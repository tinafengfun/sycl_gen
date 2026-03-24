#!/usr/bin/env python3
"""
Comprehensive Kernel Benchmark Executor
Tests all 6 versions of a kernel across 5 data sizes
"""

import subprocess
import json
import time
import sys
from pathlib import Path

class KernelTester:
    def __init__(self, kernel_name, state_file="test_state.json"):
        self.kernel_name = kernel_name
        self.state_file = Path(state_file)
        self.versions = ["V0", "V1", "V2", "V3", "V4", "V5"]
        self.sizes = [256, 512, 1024, 4096, 16384]
        self.results = []
        
    def generate_test_cpp(self, version):
        """Generate test C++ file for a specific version"""
        version_file = Path(f"01_kernels/{self.kernel_name}/generated/{version.lower()}.cpp")
        if not version_file.exists():
            # Handle special naming
            version_map = {
                "V0": "v0_baseline",
                "V1": "v1_wg512", 
                "V2": "v2_sg16",
                "V3": "v3_shuffle" if "softmax" in self.kernel_name else "v3_vec4" if "global_avg_pool" in self.kernel_name else "v3_slm_tiling",
                "V4": "v4_vec4" if "softmax" in self.kernel_name else "v4_slm" if "global_avg_pool" in self.kernel_name else "v4_large_grf",
                "V5": "v5_optimized" if "softmax" in self.kernel_name or "global_avg_pool" in self.kernel_name else "v5_xmx_dpas"
            }
            version_file = Path(f"01_kernels/{self.kernel_name}/generated/{version_map.get(version, version.lower())}.cpp")
        
        test_cpp = f'''
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <cmath>

// Include kernel
#include "{version_file}"

using namespace lczero::sycldnn_backend;

struct TestResult {{
    int size;
    double time_ms;
    double gflops;
    double bandwidth_gbps;
    bool passed;
    std::string error;
}};

// Validation function (kernel-specific)
bool validate_softmax(float* output, int N, int C) {{
    for (int n = 0; n < N; ++n) {{
        float sum = 0;
        for (int c = 0; c < C; ++c) {{
            sum += output[n * C + c];
        }}
        if (std::abs(sum - 1.0f) > 0.01f) return false;
    }}
    return true;
}}

bool validate_global_avg_pool(float* output, float* input, int N, int C) {{
    // Check output is in reasonable range
    for (int i = 0; i < N * C; ++i) {{
        if (output[i] < 0 || output[i] > 1) return false;
    }}
    return true;
}}

bool validate_winograd(float* output, float* input, int N, int C) {{
    // Basic sanity check
    for (int i = 0; i < N * C * 64; ++i) {{
        if (std::isnan(output[i]) || std::isinf(output[i])) return false;
    }}
    return true;
}}

TestResult test_kernel(int N, int C, int iterations = 100) {{
    TestResult result;
    result.size = N * C;
    result.passed = true;
    
    try {{
        sycl::queue q(sycl::gpu_selector_v);
        
        int total = N * C;
        float *d_input = sycl::malloc_device<float>(total, q);
        float *d_output = sycl::malloc_device<float>(total, q);
        
        // Init data
        std::vector<float> h_input(total);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : h_input) v = dist(gen);
        q.memcpy(d_input, h_input.data(), total * sizeof(float)).wait();
        
        // Kernel-specific setup
        #ifdef TEST_SOFTMAX
        // Softmax: input is (N, C), output is (N, C)
        #elif defined(TEST_GLOBAL_AVG_POOL)
        // Global avg pool: input is (N, C, 8, 8), output is (N, C)
        int spatial = 64; // 8x8
        sycl::free(d_input, q);
        sycl::free(d_output, q);
        d_input = sycl::malloc_device<float>(total * spatial, q);
        d_output = sycl::malloc_device<float>(total, q);
        
        std::vector<float> h_input_spatial(total * spatial);
        for (auto& v : h_input_spatial) v = dist(gen);
        q.memcpy(d_input, h_input_spatial.data(), total * spatial * sizeof(float)).wait();
        #elif defined(TEST_WINOGRAD)
        // Winograd: input is (N, C, 8, 8), output is (N, C, 6, 6, 4)
        int output_size = total * 6 * 6 * 4;
        sycl::free(d_input, q);
        sycl::free(d_output, q);
        d_input = sycl::malloc_device<float>(total * 64, q);
        d_output = sycl::malloc_device<float>(output_size, q);
        
        std::vector<float> h_input_win(total * 64);
        for (auto& v : h_input_win) v = dist(gen);
        q.memcpy(d_input, h_input_win.data(), total * 64 * sizeof(float)).wait();
        #endif
        
        // Warmup
        for (int i = 0; i < 10; ++i) {{
            #ifdef TEST_SOFTMAX
            softmax(d_output, d_input, N, C, q);
            #elif defined(TEST_GLOBAL_AVG_POOL)
            globalAvgPool(d_output, d_input, N, C, q);
            #elif defined(TEST_WINOGRAD)
            winogradInputTransform(d_output, d_input, N, C, 8, 8, q);
            #endif
            q.wait();
        }}
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {{
            #ifdef TEST_SOFTMAX
            softmax(d_output, d_input, N, C, q);
            #elif defined(TEST_GLOBAL_AVG_POOL)
            globalAvgPool(d_output, d_input, N, C, q);
            #elif defined(TEST_WINOGRAD)
            winogradInputTransform(d_output, d_input, N, C, 8, 8, q);
            #endif
            q.wait();
        }}
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        result.time_ms = (elapsed.count() * 1000.0) / iterations;
        
        // Calculate metrics
        int flops_per_element = 10;  // Approximate
        int bytes_per_element = 12;  // Approximate
        
        double flops = total * flops_per_element;
        double bytes = total * bytes_per_element;
        result.gflops = (flops / (result.time_ms * 1e-3)) / 1e9;
        result.bandwidth_gbps = (bytes / (result.time_ms * 1e-3)) / 1e9;
        
        // Validation
        #ifdef TEST_SOFTMAX
        std::vector<float> h_output(total);
        q.memcpy(h_output.data(), d_output, total * sizeof(float)).wait();
        result.passed = validate_softmax(h_output.data(), N, C);
        #elif defined(TEST_GLOBAL_AVG_POOL)
        std::vector<float> h_output(total);
        q.memcpy(h_output.data(), d_output, total * sizeof(float)).wait();
        result.passed = validate_global_avg_pool(h_output.data(), nullptr, N, C);
        #elif defined(TEST_WINOGRAD)
        std::vector<float> h_output(total * 6 * 6 * 4);
        q.memcpy(h_output.data(), d_output, total * 6 * 6 * 4 * sizeof(float)).wait();
        result.passed = validate_winograd(h_output.data(), nullptr, N, C);
        #endif
        
        sycl::free(d_input, q);
        sycl::free(d_output, q);
        
    }} catch (const std::exception& e) {{
        result.passed = false;
        result.error = e.what();
    }}
    
    return result;
}}

int main(int argc, char** argv) {{
    if (argc < 2) {{
        std::cerr << "Usage: " << argv[0] << " <size_multiplier>" << std::endl;
        return 1;
    }}
    
    int multiplier = std::atoi(argv[1]);
    int N = multiplier;
    int C = 64;  // Fixed channels
    
    auto result = test_kernel(N, C);
    
    if (result.passed) {{
        std::cout << "PASSED," << result.size << "," << result.time_ms << "," 
                  << result.gflops << "," << result.bandwidth_gbps << std::endl;
        return 0;
    }} else {{
        std::cout << "FAILED," << result.size << ",0,0,0," << result.error << std::endl;
        return 1;
    }}
}}
'''
        return test_cpp
    
    def run_test(self, version, size_multiplier):
        """Compile and run test for a specific version and size"""
        test_file = f"/tmp/test_{self.kernel_name}_{version}.cpp"
        binary_file = f"/tmp/test_{self.kernel_name}_{version}"
        
        # Generate test file
        cpp_code = self.generate_test_cpp(version)
        with open(test_file, 'w') as f:
            f.write(cpp_code)
        
        # Copy kernel file to container
        version_file = Path(f"01_kernels/{self.kernel_name}/generated/{version.lower()}.cpp")
        if not version_file.exists():
            version_map = {
                "V0": "v0_baseline",
                "V1": "v1_wg512",
                "V2": "v2_sg16",
                "V3": "v3_shuffle" if "softmax" in self.kernel_name else "v3_vec4" if "global_avg_pool" in self.kernel_name else "v3_slm_tiling",
                "V4": "v4_vec4" if "softmax" in self.kernel_name else "v4_slm" if "global_avg_pool" in self.kernel_name else "v4_large_grf",
                "V5": "v5_optimized" if "softmax" in self.kernel_name or "global_avg_pool" in self.kernel_name else "v5_xmx_dpas"
            }
            version_file = Path(f"01_kernels/{self.kernel_name}/generated/{version_map.get(version, version.lower())}.cpp")
        
        # Copy to container
        subprocess.run(["docker", "cp", str(version_file), f"lsv-container:/workspace/kernel.cpp"], check=True)
        subprocess.run(["docker", "cp", test_file, f"lsv-container:/workspace/test.cpp"], check=True)
        
        # Compile
        compile_cmd = [
            "docker", "exec", "lsv-container",
            "bash", "-c",
            f"cd /workspace && icpx -fsycl -O2 -std=c++17 -DTEST_{self.kernel_name.upper().replace('_', '_')} test.cpp -o test_binary 2>&1"
        ]
        
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {"status": "COMPILE_ERROR", "error": result.stderr}
        
        # Run test
        run_cmd = [
            "docker", "exec", "lsv-container",
            "/workspace/test_binary", str(size_multiplier)
        ]
        
        result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return {"status": "RUNTIME_ERROR", "error": result.stderr}
        
        # Parse output
        parts = result.stdout.strip().split(',')
        if parts[0] == "PASSED":
            return {
                "status": "PASSED",
                "size": int(parts[1]),
                "time_ms": float(parts[2]),
                "gflops": float(parts[3]),
                "bandwidth_gbps": float(parts[4])
            }
        else:
            return {"status": "FAILED", "error": parts[5] if len(parts) > 5 else "Unknown"}
    
    def run_all_tests(self):
        """Run all tests for this kernel"""
        print(f"\n{'='*60}")
        print(f"Testing Kernel: {self.kernel_name}")
        print(f"{'='*60}\n")
        
        total_tests = len(self.versions) * len(self.sizes)
        test_num = 0
        
        for version in self.versions:
            print(f"\n--- Version {version} ---")
            for size in self.sizes:
                test_num += 1
                size_multiplier = size // 64  # Convert to N (assuming C=64)
                
                print(f"[{test_num}/{total_tests}] Testing size={size}... ", end='', flush=True)
                
                # Update state
                self.update_state(version, size, "RUNNING")
                
                try:
                    result = self.run_test(version, size_multiplier)
                    
                    if result["status"] == "PASSED":
                        print(f"✅ PASSED - {result['time_ms']:.3f}ms, {result['gflops']:.2f} GFLOPS")
                        self.results.append({
                            "kernel": self.kernel_name,
                            "version": version,
                            "size": size,
                            **result
                        })
                    else:
                        print(f"❌ FAILED - {result.get('error', 'Unknown')}")
                        self.results.append({
                            "kernel": self.kernel_name,
                            "version": version,
                            "size": size,
                            "status": result["status"],
                            "error": result.get("error", "")
                        })
                        
                        # Pause on error
                        if result["status"] in ["COMPILE_ERROR", "RUNTIME_ERROR"]:
                            print(f"\n⚠️  ERROR DETECTED! Pausing for inspection...")
                            print(f"Kernel: {self.kernel_name}, Version: {version}, Size: {size}")
                            print(f"Error: {result.get('error', 'Unknown')}")
                            input("Press Enter to continue or Ctrl+C to abort...")
                    
                except Exception as e:
                    print(f"❌ EXCEPTION - {str(e)}")
                    self.results.append({
                        "kernel": self.kernel_name,
                        "version": version,
                        "size": size,
                        "status": "EXCEPTION",
                        "error": str(e)
                    })
                    input("Exception occurred. Press Enter to continue or Ctrl+C to abort...")
                
                self.update_state(version, size, "COMPLETED")
        
        # Save results
        self.save_results()
        print(f"\n✅ All tests completed for {self.kernel_name}")
        print(f"Results saved to: {self.kernel_name}_results.json\n")
    
    def update_state(self, version, size, test_status):
        """Update global test state"""
        state_data = {
            "current_kernel": self.kernel_name,
            "current_version": version,
            "current_size": size,
            "status": test_status,
            "timestamp": time.time()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state_data, f)
    
    def save_results(self):
        """Save test results"""
        output_file = f"{self.kernel_name}_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_runner.py <kernel_name>")
        print("Example: python3 test_runner.py softmax")
        sys.exit(1)
    
    kernel_name = sys.argv[1]
    tester = KernelTester(kernel_name)
    tester.run_all_tests()
