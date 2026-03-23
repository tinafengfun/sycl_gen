#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main() {
    std::ofstream out("device_query_results.txt");
    
    auto platforms = sycl::platform::get_platforms();
    out << "=== SYCL Platform and Device Query ===\n\n";
    out << "Number of platforms: " << platforms.size() << "\n\n";
    
    for (size_t i = 0; i < platforms.size(); ++i) {
        auto& platform = platforms[i];
        out << "Platform " << i << ": " << platform.get_info<sycl::info::platform::name>() << "\n";
        out << "  Vendor: " << platform.get_info<sycl::info::platform::vendor>() << "\n";
        out << "  Version: " << platform.get_info<sycl::info::platform::version>() << "\n\n";
        
        auto devices = platform.get_devices();
        out << "  Devices: " << devices.size() << "\n";
        
        for (size_t j = 0; j < devices.size(); ++j) {
            auto& device = devices[j];
            out << "\n  Device " << j << ": " << device.get_info<sycl::info::device::name>() << "\n";
            out << "    Type: " << (device.is_gpu() ? "GPU" : (device.is_cpu() ? "CPU" : "Other")) << "\n";
            out << "    Vendor: " << device.get_info<sycl::info::device::vendor>() << "\n";
            out << "    Driver Version: " << device.get_info<sycl::info::device::driver_version>() << "\n";
            
            // Memory info
            out << "    Global Memory: " << device.get_info<sycl::info::device::global_mem_size>() / (1024*1024) << " MB\n";
            out << "    Local Memory: " << device.get_info<sycl::info::device::local_mem_size>() / 1024 << " KB\n";
            out << "    Max Work Group Size: " << device.get_info<sycl::info::device::max_work_group_size>() << "\n";
            
            // Sub-group info
            auto sg_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
            out << "    Sub-group Sizes: ";
            for (size_t k = 0; k < sg_sizes.size(); ++k) {
                out << sg_sizes[k];
                if (k < sg_sizes.size() - 1) out << ", ";
            }
            out << "\n";
            
            // Vector widths
            out << "    Preferred Vector Width (float): " << device.get_info<sycl::info::device::preferred_vector_width_float>() << "\n";
            out << "    Native Vector Width (float): " << device.get_info<sycl::info::device::native_vector_width_float>() << "\n";
            
            // Compute units
            out << "    Max Compute Units: " << device.get_info<sycl::info::device::max_compute_units>() << "\n";
            
            // Half precision support
            out << "    Half Precision Support: " << (device.has(sycl::aspect::fp16) ? "Yes" : "No") << "\n";
            
            // Double precision support
            out << "    Double Precision Support: " << (device.has(sycl::aspect::fp64) ? "Yes" : "No") << "\n";
            
            // Atomic memory order capabilities
            out << "    Atomic Memory Order: ";
            if (device.has(sycl::aspect::atomic64)) {
                out << "64-bit";
            } else {
                out << "32-bit";
            }
            out << "\n";
        }
        out << "\n";
    }
    
    out << "\n=== BMG B60 Optimization Recommendations ===\n\n";
    out << "Target Hardware: Intel BMG B60 (Xe2 Architecture)\n";
    out << "Sub-group Size: 16 (optimal for BMG)\n";
    out << "SLM Size: 256 KB per XeCore\n";
    out << "L2 Cache: 18 MB\n";
    out << "Memory Bandwidth: ~500 GB/s\n";
    out << "Max Work Group Size: 1024\n\n";
    
    out << "Optimization Checklist:\n";
    out << "[ ] Work-group size: 256-512 threads\n";
    out << "[ ] Sub-group size: 16 lanes\n";
    out << "[ ] Memory access: 64-byte aligned, coalesced\n";
    out << "[ ] Vector width: 16 for compute-heavy kernels\n";
    out << "[ ] SLM usage: Stay within 256 KB per work-group\n";
    out << "[ ] L2 cache: Tile data to fit in 18 MB\n\n";
    
    out.close();
    std::cout << "Device query completed. Results saved to 00_validation/device_query_results.txt\n";
    
    return 0;
}