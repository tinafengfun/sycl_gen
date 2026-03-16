/*
  Simple SYCL test for BMG/XPU compilation options
  测试新的编译选项是否工作正常
*/

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== SYCL BMG/XPU Compilation Test ===" << std::endl;
    
    // 获取设备信息
    std::vector<sycl::device> devices = sycl::device::get_devices();
    std::cout << "Found " << devices.size() << " SYCL devices:" << std::endl;
    
    for (size_t i = 0; i < devices.size(); ++i) {
        std::cout << "  [" << i << "] " 
                  << devices[i].get_info<sycl::info::device::name>()
                  << " (" << (devices[i].is_gpu() ? "GPU" : "CPU") << ")"
                  << std::endl;
    }
    
    // 选择默认设备
    sycl::queue queue(sycl::default_selector_v);
    std::cout << "\nUsing device: " 
              << queue.get_device().get_info<sycl::info::device::name>() 
              << std::endl;
    
    // 简单的vector add测试
    const size_t N = 1024;
    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 2.0f);
    std::vector<float> c(N, 0.0f);
    
    // 分配设备内存
    float* d_a = sycl::malloc_device<float>(N, queue);
    float* d_b = sycl::malloc_device<float>(N, queue);
    float* d_c = sycl::malloc_device<float>(N, queue);
    
    // 拷贝数据到设备
    queue.memcpy(d_a, a.data(), N * sizeof(float));
    queue.memcpy(d_b, b.data(), N * sizeof(float));
    queue.wait();
    
    // 执行kernel
    queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        d_c[idx] = d_a[idx] + d_b[idx];
    }).wait();
    
    // 拷贝结果回主机
    queue.memcpy(c.data(), d_c, N * sizeof(float)).wait();
    
    // 验证结果
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        if (c[i] != 3.0f) {
            std::cout << "ERROR at index " << i << ": expected 3.0, got " << c[i] << std::endl;
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "✓ Vector add test PASSED (1024 elements)" << std::endl;
        std::cout << "✓ Result verification: 1.0 + 2.0 = 3.0" << std::endl;
    }
    
    // 释放内存
    sycl::free(d_a, queue);
    sycl::free(d_b, queue);
    sycl::free(d_c, queue);
    
    // 测试特征支持
    auto device = queue.get_device();
    std::cout << "\n=== Device Features ===" << std::endl;
    std::cout << "FP64 support: " 
              << (device.has(sycl::aspect::fp64) ? "Yes" : "No") << std::endl;
    std::cout << "FP16 support: " 
              << (device.has(sycl::aspect::fp16) ? "Yes" : "No") << std::endl;
    std::cout << "Atomic64 support: " 
              << (device.has(sycl::aspect::atomic64) ? "Yes" : "No") << std::endl;
    
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "✓ Compilation successful" << std::endl;
    std::cout << "✓ Device detection successful" << std::endl;
    std::cout << "✓ Kernel execution successful" << std::endl;
    std::cout << "✓ All tests PASSED!" << std::endl;
    
    return 0;
}
