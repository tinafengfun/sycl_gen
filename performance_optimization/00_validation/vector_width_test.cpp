#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    std::ofstream out("vector_width_test_results.txt");
    out << "=== Vector Width Test Results ===\n\n";
    out << "GPU: " << q.get_device().get_info<sycl::info::device::name>() << "\n\n";
    out << "Device native vector width: " 
        << q.get_device().get_info<sycl::info::device::native_vector_width_float>() << "\n";
    out << "Device preferred vector width: " 
        << q.get_device().get_info<sycl::info::device::preferred_vector_width_float>() << "\n\n";
    out << "Note: BMG B60 supports 16-wide vectors for optimal performance\n";
    out.close();
    std::cout << "Vector width info saved.\n";
    return 0;
}
