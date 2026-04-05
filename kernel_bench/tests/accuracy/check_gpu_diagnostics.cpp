#include <sycl/sycl.hpp>
#include <iostream>
#include <cmath>

// ============================================
// GPU Diagnostics Tool
// Check actual GPU specs and verify calculations
// ============================================

int main() {
    sycl::queue queue(sycl::gpu_selector_v);
    auto device = queue.get_device();
    
    std::cout << "========================================" << std::endl;
    std::cout << "GPU Diagnostics Report" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Device info
    std::cout << "\n1. DEVICE INFO:" << std::endl;
    std::cout << "   Name: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "   Vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
    std::cout << "   Driver Version: " << device.get_info<sycl::info::device::driver_version>() << std::endl;
    
    // Compute units
    int compute_units = device.get_info<sycl::info::device::max_compute_units>();
    std::cout << "   Compute Units: " << compute_units << std::endl;
    
    // Clock frequency
    uint32_t freq_khz = device.get_info<sycl::info::device::max_clock_frequency>();
    float freq_ghz = freq_khz / 1000.0f;
    std::cout << "   Clock Frequency: " << freq_ghz << " GHz" << std::endl;
    
    // Memory info
    std::cout << "\n2. MEMORY INFO:" << std::endl;
    size_t global_mem = device.get_info<sycl::info::device::global_mem_size>();
    std::cout << "   Global Memory: " << (global_mem / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    
    size_t local_mem = device.get_info<sycl::info::device::local_mem_size>();
    std::cout << "   Local Memory (SLM): " << (local_mem / 1024.0) << " KB" << std::endl;
    
    // Subgroup info
    std::cout << "\n3. SUBGROUP INFO:" << std::endl;
    auto sg_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
    std::cout << "   Supported subgroup sizes: ";
    for (auto sz : sg_sizes) std::cout << sz << " ";
    std::cout << std::endl;
    
    // Work group size
    size_t wg_size = device.get_info<sycl::info::device::max_work_group_size>();
    std::cout << "   Max Work-Group Size: " << wg_size << std::endl;
    
    // FP16 support
    std::cout << "\n4. FP16 SUPPORT:" << std::endl;
    bool fp16_support = device.has(sycl::aspect::fp16);
    std::cout << "   FP16: " << (fp16_support ? "YES" : "NO") << std::endl;
    
    // Theoretical performance calculation
    std::cout << "\n5. THEORETICAL PERFORMANCE:" << std::endl;
    
    // Intel Xe architecture: 8 EUs per subslice, each EU has 8 FP16 ops/cycle
    // For 160 CUs (EUs), at 2.4 GHz
    // Peak = CUs * ops_per_cycle * clock_frequency * 2 (for FMA)
    
    int ops_per_cycle_per_eu = 16;  // DPAS instruction can do 16 FP16 ops per cycle
    double theoretical_tflops = compute_units * ops_per_cycle_per_eu * freq_ghz / 1000.0;
    
    std::cout << "   Compute Units: " << compute_units << std::endl;
    std::cout << "   Ops/Cycle/EU: " << ops_per_cycle_per_eu << std::endl;
    std::cout << "   Clock: " << freq_ghz << " GHz" << std::endl;
    std::cout << "   Theoretical Peak: " << theoretical_tflops << " TFLOPS" << std::endl;
    std::cout << "   (Note: Actual may be lower due to SKU binning)" << std::endl;
    
    // Verify GFLOPS calculation formula
    std::cout << "\n6. GFLOPS CALCULATION VERIFICATION:" << std::endl;
    int M = 8192, N = 8192, K = 8192;
    double ops = 2.0 * M * N * K;  // 2 FLOPs per multiply-add
    double time_ms = 73.84;  // Actual measured time
    double gflops = ops / (time_ms * 1e-3) / 1e9;
    
    std::cout << "   Matrix: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "   Total Ops: 2*" << M << "*" << N << "*" << K << " = " << (ops/1e12) << " Tera-ops" << std::endl;
    std::cout << "   Time: " << time_ms << " ms" << std::endl;
    std::cout << "   GFLOPS = ops / (time * 1e-3) / 1e9" << std::endl;
    std::cout << "          = " << ops << " / (" << time_ms << " * 0.001) / 1e9" << std::endl;
    std::cout << "          = " << gflops << " GFLOPS" << std::endl;
    std::cout << "          = " << (gflops/1000.0) << " TFLOPS" << std::endl;
    
    // Memory bandwidth check
    std::cout << "\n7. MEMORY BANDWIDTH ANALYSIS:" << std::endl;
    double bytes_accessed = (M * K + K * N + M * N) * sizeof(sycl::half);
    double bandwidth_gbps = bytes_accessed / (time_ms * 1e-3) / 1e9;
    std::cout << "   Data moved: " << (bytes_accessed/1e9) << " GB" << std::endl;
    std::cout << "   Bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "   (B60 typically has ~500-800 GB/s memory bandwidth)" << std::endl;
    
    // Arithmetic intensity
    double arithmetic_intensity = ops / bytes_accessed;
    std::cout << "   Arithmetic Intensity: " << arithmetic_intensity << " FLOPs/byte" << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "CONCLUSION:" << std::endl;
    std::cout << "- Measured: " << (gflops/1000.0) << " TFLOPS" << std::endl;
    std::cout << "- Theoretical: " << theoretical_tflops << " TFLOPS" << std::endl;
    std::cout << "- Efficiency: " << ((gflops/1000.0)/theoretical_tflops*100.0) << "%" << std::endl;
    std::cout << "\nNote: GEMM performance is often limited by memory bandwidth," << std::endl;
    std::cout << "      not compute. Peak compute is only achievable with" << std::endl;
    std::cout << "      high arithmetic intensity operations." << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
