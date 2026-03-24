#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

// Add Vectors Kernel - Element-wise addition
// V0: Baseline (WG=256)
namespace v0 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void addVectors_kernel(T* output, const T* a, const T* b, int N, const sycl::nd_item<1> &item) {
    int idx = item.get_global_id(0);
    if (idx >= N) return;
    output[idx] = a[idx] + b[idx];
}

template <typename T>
void addVectors(T* output, const T* a, const T* b, int N, sycl::queue &queue) {
    int wg_size = 256;
    int blocks = DivUp(N, wg_size);
    queue.parallel_for(
        sycl::nd_range<1>(blocks * wg_size, wg_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            addVectors_kernel(output, a, b, N, item);
        }
    );
}
}

// V1: WG=128 (previously optimal for element-wise)
namespace v1 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void addVectors_kernel(T* output, const T* a, const T* b, int N, const sycl::nd_item<1> &item) {
    int idx = item.get_global_id(0);
    if (idx >= N) return;
    output[idx] = a[idx] + b[idx];
}

template <typename T>
void addVectors(T* output, const T* a, const T* b, int N, sycl::queue &queue) {
    int wg_size = 128;
    int blocks = DivUp(N, wg_size);
    queue.parallel_for(
        sycl::nd_range<1>(blocks * wg_size, wg_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            addVectors_kernel(output, a, b, N, item);
        }
    );
}
}

// V2: Grid-stride with unrolling
namespace v2 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void addVectors_kernel(T* output, const T* a, const T* b, int N, const sycl::nd_item<1> &item) {
    int tid = item.get_global_id(0);
    int grid_size = item.get_global_range(0);
    
    #pragma unroll 4
    for (int idx = tid; idx < N; idx += grid_size) {
        output[idx] = a[idx] + b[idx];
    }
}

template <typename T>
void addVectors(T* output, const T* a, const T* b, int N, sycl::queue &queue) {
    int wg_size = 256;
    int blocks = DivUp(N, wg_size);
    queue.parallel_for(
        sycl::nd_range<1>(blocks * wg_size, wg_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            addVectors_kernel(output, a, b, N, item);
        }
    );
}
}

int main() {
    sycl::queue queue(sycl::gpu_selector_v);
    auto device = queue.get_device();
    std::cout << "========================================" << std::endl;
    std::cout << "Add Vectors Kernel Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "GPU: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << std::endl;

    std::vector<int> sizes = {256, 512, 1024, 4096, 16384};
    int iterations = 100;
    std::ofstream csv("add_vectors_results.csv");
    csv << "Version,N,Time_ms,GFLOPS,Bandwidth_GB/s" << std::endl;

    for (int N : sizes) {
        std::vector<float> h_a(N);
        std::vector<float> h_b(N);
        std::vector<float> h_output(N);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : h_a) v = dist(gen);
        for (auto &v : h_b) v = dist(gen);

        float *d_a = sycl::malloc_device<float>(N, queue);
        float *d_b = sycl::malloc_device<float>(N, queue);
        float *d_output = sycl::malloc_device<float>(N, queue);
        
        queue.memcpy(d_a, h_a.data(), N * sizeof(float));
        queue.memcpy(d_b, h_b.data(), N * sizeof(float));
        queue.wait();

        double total_ops = (double)N * 1;  // 1 add per element
        double total_bytes = (N * 3) * sizeof(float);

        auto run_test = [&](const char* name, auto &&kernel_func) {
            for (int i = 0; i < 10; ++i) kernel_func();
            queue.wait();
            
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) kernel_func();
            queue.wait();
            auto end = std::chrono::high_resolution_clock::now();
            
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
            double gflops = (total_ops / (time_ms * 1e-3)) / 1e9;
            double bw = (total_bytes / (time_ms * 1e-3)) / 1e9;
            
            std::cout << name << "\tN=" << N << "\tTime: " << time_ms << " ms\t"
                      << "GFLOPS: " << gflops << "\tBW: " << bw << " GB/s" << std::endl;
            csv << name << "," << N << "," << time_ms << "," << gflops << "," << bw << std::endl;
        };

        std::cout << "=== Testing N=" << N << " ===" << std::endl;
        run_test("V0", [&]() { v0::addVectors(d_output, d_a, d_b, N, queue); });
        run_test("V1", [&]() { v1::addVectors(d_output, d_a, d_b, N, queue); });
        run_test("V2", [&]() { v2::addVectors(d_output, d_a, d_b, N, queue); });

        sycl::free(d_a, queue);
        sycl::free(d_b, queue);
        sycl::free(d_output, queue);
    }

    csv.close();
    std::cout << std::endl << "✅ Add Vectors testing completed!" << std::endl;
    return 0;
}
