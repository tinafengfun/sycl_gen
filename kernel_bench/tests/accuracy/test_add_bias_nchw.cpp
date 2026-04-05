#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

// Add Bias NCHW Kernel - Element-wise bias addition for NCHW layout
// V0: Baseline (simple 1D)
namespace v0 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void addBiasNCHW_kernel(T* output, const T* input, const T* bias,
                        int N, int C, int H, int W, const sycl::nd_item<1> &item) {
    int idx = item.get_global_id(0);
    int total = N * C * H * W;
    if (idx >= total) return;
    
    int c = (idx / (H * W)) % C;
    output[idx] = input[idx] + bias[c];
}

template <typename T>
void addBiasNCHW(T* output, const T* input, const T* bias,
                 int N, int C, int H, int W, sycl::queue &queue) {
    int total = N * C * H * W;
    int wg_size = 256;
    int blocks = DivUp(total, wg_size);
    queue.parallel_for(
        sycl::nd_range<1>(blocks * wg_size, wg_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            addBiasNCHW_kernel(output, input, bias, N, C, H, W, item);
        }
    );
}
}

// V1: WG=128 optimized for element-wise
namespace v1 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void addBiasNCHW_kernel(T* output, const T* input, const T* bias,
                        int N, int C, int H, int W, const sycl::nd_item<1> &item) {
    int idx = item.get_global_id(0);
    int total = N * C * H * W;
    if (idx >= total) return;
    
    int c = (idx / (H * W)) % C;
    output[idx] = input[idx] + bias[c];
}

template <typename T>
void addBiasNCHW(T* output, const T* input, const T* bias,
                 int N, int C, int H, int W, sycl::queue &queue) {
    int total = N * C * H * W;
    int wg_size = 128;
    int blocks = DivUp(total, wg_size);
    queue.parallel_for(
        sycl::nd_range<1>(blocks * wg_size, wg_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            addBiasNCHW_kernel(output, input, bias, N, C, H, W, item);
        }
    );
}
}

// V2: Grid-stride with unrolling
namespace v2 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void addBiasNCHW_kernel(T* output, const T* input, const T* bias,
                        int N, int C, int H, int W, int total, const sycl::nd_item<1> &item) {
    int tid = item.get_global_id(0);
    int grid_size = item.get_global_range(0);
    int spatial = H * W;
    
    #pragma unroll 4
    for (int idx = tid; idx < total; idx += grid_size) {
        int c = (idx / spatial) % C;
        output[idx] = input[idx] + bias[c];
    }
}

template <typename T>
void addBiasNCHW(T* output, const T* input, const T* bias,
                 int N, int C, int H, int W, sycl::queue &queue) {
    int total = N * C * H * W;
    int wg_size = 256;
    int blocks = DivUp(total, wg_size);
    queue.parallel_for(
        sycl::nd_range<1>(blocks * wg_size, wg_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            addBiasNCHW_kernel(output, input, bias, N, C, H, W, total, item);
        }
    );
}
}

int main() {
    sycl::queue queue(sycl::gpu_selector_v);
    auto device = queue.get_device();
    std::cout << "========================================" << std::endl;
    std::cout << "Add Bias NCHW Kernel Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "GPU: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << std::endl;

    std::vector<int> sizes = {64, 128, 256, 512, 1024};
    int iterations = 100;
    std::ofstream csv("add_bias_nchw_results.csv");
    csv << "Version,N,C,H,W,Time_ms,GFLOPS,Bandwidth_GB/s" << std::endl;

    int C = 64, H = 8, W = 8;

    for (int N : sizes) {
        int total_size = N * C * H * W;
        int bias_size = C;

        std::vector<float> h_input(total_size);
        std::vector<float> h_bias(bias_size);
        std::vector<float> h_output(total_size);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : h_input) v = dist(gen);
        for (auto &v : h_bias) v = dist(gen) * 0.1f;

        float *d_input = sycl::malloc_device<float>(total_size, queue);
        float *d_output = sycl::malloc_device<float>(total_size, queue);
        float *d_bias = sycl::malloc_device<float>(bias_size, queue);
        
        queue.memcpy(d_input, h_input.data(), total_size * sizeof(float));
        queue.memcpy(d_bias, h_bias.data(), bias_size * sizeof(float));
        queue.wait();

        double total_ops = (double)total_size * 1;
        double total_bytes = (total_size * 2 + bias_size) * sizeof(float);

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
            csv << name << "," << N << "," << C << "," << H << "," << W << ","
                << time_ms << "," << gflops << "," << bw << std::endl;
        };

        std::cout << "=== Testing N=" << N << " ===" << std::endl;
        run_test("V0", [&]() { v0::addBiasNCHW(d_output, d_input, d_bias, N, C, H, W, queue); });
        run_test("V1", [&]() { v1::addBiasNCHW(d_output, d_input, d_bias, N, C, H, W, queue); });
        run_test("V2", [&]() { v2::addBiasNCHW(d_output, d_input, d_bias, N, C, H, W, queue); });

        sycl::free(d_input, queue);
        sycl::free(d_output, queue);
        sycl::free(d_bias, queue);
    }

    csv.close();
    std::cout << std::endl << "✅ Add Bias NCHW testing completed!" << std::endl;
    return 0;
}
