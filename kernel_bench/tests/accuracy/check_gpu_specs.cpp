#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    auto dev = q.get_device();
    
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Max compute units: " << dev.get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "Max clock frequency: " << dev.get_info<sycl::info::device::max_clock_frequency>() << " MHz" << std::endl;
    std::cout << "Max work-group size: " << dev.get_info<sycl::info::device::max_work_group_size>() << std::endl;
    
    auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    std::cout << "Sub-group sizes: ";
    for (auto s : sg_sizes) std::cout << s << " ";
    std::cout << std::endl;
    
    // Check XMX support
    bool has_matrix = dev.has(sycl::aspect::ext_intel_matrix);
    std::cout << "XMX/DPAS support: " << (has_matrix ? "YES" : "NO") << std::endl;
    
    return 0;
}
