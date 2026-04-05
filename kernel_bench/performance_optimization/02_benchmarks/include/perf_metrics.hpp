#pragma once
#ifndef PERF_METRICS_HPP
#define PERF_METRICS_HPP

#include <string>
#include <vector>

namespace perf {

struct TestConfig {
    std::string kernel_name;
    std::string version;
    int data_size;
    int warmup_iterations;
    int test_iterations;
    bool use_fp16;
};

struct PerfResult {
    double time_ms;           // Average execution time
    double std_dev_ms;        // Standard deviation
    double min_ms;            // Minimum time
    double max_ms;            // Maximum time
    double gflops;            // GFLOPS achieved
    double bandwidth_gbps;    // Memory bandwidth GB/s
    double theoretical_gflops;
    double theoretical_bw_gbps;
    double efficiency_percent;
    bool passed;              // Test passed?
    std::string error_msg;    // Error message if failed
};

struct KernelMetrics {
    int flops_per_element;
    int bytes_per_element;
    std::vector<int> test_sizes;
};

// Utility functions
inline double calculate_gflops(int elements, int flops_per_element, double time_ms) {
    double flops = static_cast<double>(elements) * flops_per_element;
    return (flops / (time_ms * 1e-3)) / 1e9;
}

inline double calculate_bandwidth(int elements, int bytes_per_element, double time_ms) {
    double bytes = static_cast<double>(elements) * bytes_per_element;
    return (bytes / (time_ms * 1e-3)) / 1e9;
}

} // namespace perf

#endif // PERF_METRICS_HPP