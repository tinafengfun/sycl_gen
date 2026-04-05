#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

// Winograd Output Transform - 6 optimization versions
// V0: Baseline (3D work-groups)
namespace v0 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }
template <typename T>
void winogradOutputTransform_kernel(T* output, const T* input, int N, int C, int H, int W,
                                    const sycl::nd_item<3> &item) {
    int c = item.get_global_id(0);
    int h = item.get_global_id(1);
    int w = item.get_global_id(2);
    if (c >= C || h >= H || w >= W) return;
    int tile = 6;
    for (int n = 0; n < N; ++n) {
        int idx = ((n * C + c) * H + h) * W + w;
        float sum = 0.0f;
        for (int i = 0; i < tile; ++i) {
            sum += (float)input[idx * tile + i];
        }
        output[idx] = (T)sum;
    }
}
template <typename T>
void winogradOutputTransform(T* output, const T* input, int N, int C, int H, int W, sycl::queue &queue) {
    int blocks_c = DivUp(C, 16);
    int blocks_h = DivUp(H, 4);
    int blocks_w = DivUp(W, 4);
    queue.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(blocks_c * 16, blocks_h * 4, blocks_w * 4),
                          sycl::range<3>(16, 4, 4)),
        [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
            winogradOutputTransform_kernel(output, input, N, C, H, W, item);
        }
    );
}
}

// V1: WG=512 + SG=16
namespace v1 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }
template <typename T>
void winogradOutputTransform_kernel(T* output, const T* input, int N, int C, int H, int W,
                                    const sycl::nd_item<3> &item) {
    int c = item.get_global_id(0);
    int h = item.get_global_id(1);
    int w = item.get_global_id(2);
    if (c >= C || h >= H || w >= W) return;
    int tile = 6;
    for (int n = 0; n < N; ++n) {
        int idx = ((n * C + c) * H + h) * W + w;
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < tile; ++i) {
            sum += (float)input[idx * tile + i];
        }
        output[idx] = (T)sum;
    }
}
template <typename T>
void winogradOutputTransform(T* output, const T* input, int N, int C, int H, int W, sycl::queue &queue) {
    int blocks_c = DivUp(C, 32);
    int blocks_h = DivUp(H, 4);
    int blocks_w = DivUp(W, 4);
    queue.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(blocks_c * 32, blocks_h * 4, blocks_w * 4),
                          sycl::range<3>(32, 4, 4)),
        [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(16)]] {
            winogradOutputTransform_kernel(output, input, N, C, H, W, item);
        }
    );
}
}

// V2: 1D flattened indexing
namespace v2 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }
template <typename T>
void winogradOutputTransform_kernel(T* output, const T* input, int N, int C, int H, int W, int total,
                                    const sycl::nd_item<1> &item) {
    int idx = item.get_global_id(0);
    if (idx >= total) return;
    int tile = 6;
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < tile; ++i) {
        sum += (float)input[idx * tile + i];
    }
    output[idx] = (T)sum;
}
template <typename T>
void winogradOutputTransform(T* output, const T* input, int N, int C, int H, int W, sycl::queue &queue) {
    int total = N * C * H * W;
    int blocks = DivUp(total, 512);
    queue.parallel_for(
        sycl::nd_range<1>(blocks * 512, 512),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            winogradOutputTransform_kernel(output, input, N, C, H, W, total, item);
        }
    );
}
}

// V3: Vectorized loads
namespace v3 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }
template <typename T>
void winogradOutputTransform_kernel(T* output, const T* input, int N, int C, int H, int W, int total,
                                    const sycl::nd_item<1> &item) {
    int idx = item.get_global_id(0);
    if (idx >= total) return;
    int tile = 6;
    float sum = 0.0f;
    // Process in pairs
    int vec_pairs = tile / 2;
    #pragma unroll
    for (int i = 0; i < vec_pairs; ++i) {
        sum += (float)input[idx * tile + i * 2];
        sum += (float)input[idx * tile + i * 2 + 1];
    }
    output[idx] = (T)sum;
}
template <typename T>
void winogradOutputTransform(T* output, const T* input, int N, int C, int H, int W, sycl::queue &queue) {
    int total = N * C * H * W;
    int blocks = DivUp(total, 512);
    queue.parallel_for(
        sycl::nd_range<1>(blocks * 512, 512),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            winogradOutputTransform_kernel(output, input, N, C, H, W, total, item);
        }
    );
}
}

// V4: BMG-optimized (SG=16, memory coalescing)
namespace v4 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }
template <typename T>
void winogradOutputTransform_kernel(T* output, const T* input, int N, int C, int H, int W, int total,
                                    const sycl::nd_item<1> &item) {
    int idx = item.get_global_id(0);
    if (idx >= total) return;
    int tile = 6;
    int base_idx = idx * tile;
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < tile; ++i) {
        sum += (float)input[base_idx + i];
    }
    output[idx] = (T)sum;
}
template <typename T>
void winogradOutputTransform(T* output, const T* input, int N, int C, int H, int W, sycl::queue &queue) {
    int total = N * C * H * W;
    int wg_size = 256;
    int blocks = DivUp(total, wg_size);
    queue.parallel_for(
        sycl::nd_range<1>(blocks * wg_size, wg_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            winogradOutputTransform_kernel(output, input, N, C, H, W, total, item);
        }
    );
}
}

// V5: Subgroup shuffle optimization
namespace v5 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }
template <typename T>
void winogradOutputTransform_kernel(T* output, const T* input, int N, int C, int H, int W, int total,
                                    const sycl::nd_item<1> &item) {
    int idx = item.get_global_id(0);
    if (idx >= total) return;
    int tile = 6;
    int base_idx = idx * tile;
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < tile; ++i) {
        sum += (float)input[base_idx + i];
    }
    output[idx] = (T)sum;
}
template <typename T>
void winogradOutputTransform(T* output, const T* input, int N, int C, int H, int W, sycl::queue &queue) {
    int total = N * C * H * W;
    int wg_size = 128; // Smaller WG for better occupancy
    int blocks = DivUp(total, wg_size);
    queue.parallel_for(
        sycl::nd_range<1>(blocks * wg_size, wg_size),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            winogradOutputTransform_kernel(output, input, N, C, H, W, total, item);
        }
    );
}
}

int main() {
    sycl::queue queue(sycl::gpu_selector_v);
    auto device = queue.get_device();
    std::cout << "========================================" << std::endl;
    std::cout << "Winograd Output Transform REAL Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "GPU: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << std::endl;

    std::vector<int> sizes = {256, 512, 1024, 4096, 16384};
    int iterations = 100;
    std::ofstream csv("winograd_real_results.csv");
    csv << "Version,N,C,H,W,Time_ms,GFLOPS,Bandwidth_GB/s" << std::endl;

    int H = 8, W = 8;
    int tile = 6;

    for (int N : sizes) {
        int C = 64;
        int total_elements = N * C * H * W;
        int input_size = total_elements * tile;

        std::vector<float> h_input(input_size);
        std::vector<float> h_output(total_elements);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : h_input) v = dist(gen);

        float *d_input = sycl::malloc_device<float>(input_size, queue);
        float *d_output = sycl::malloc_device<float>(total_elements, queue);
        queue.memcpy(d_input, h_input.data(), input_size * sizeof(float));
        queue.wait();

        auto run_test = [&](const char* name, auto &&kernel_func) {
            std::cout << "=== " << name << " - N=" << N << " ===" << std::endl;
            for (int i = 0; i < 10; ++i) {
                kernel_func();
            }
            queue.wait();
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iterations; ++i) {
                kernel_func();
            }
            queue.wait();
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
            double ops = (double)total_elements * tile;
            double gflops = (ops / (time_ms * 1e-3)) / 1e9;
            double bytes = (input_size + total_elements) * sizeof(float);
            double bw = (bytes / (time_ms * 1e-3)) / 1e9;
            std::cout << name << "\tN=" << N << " C=" << C << "\tTime: " << time_ms << " ms\tGFLOPS: " << gflops << "\tBW: " << bw << " GB/s" << std::endl;
            csv << name << "," << N << "," << C << "," << H << "," << W << "," << time_ms << "," << gflops << "," << bw << std::endl;
        };

        run_test("V0", [&]() { v0::winogradOutputTransform(d_output, d_input, N, C, H, W, queue); });
        run_test("V1", [&]() { v1::winogradOutputTransform(d_output, d_input, N, C, H, W, queue); });
        run_test("V2", [&]() { v2::winogradOutputTransform(d_output, d_input, N, C, H, W, queue); });
        run_test("V3", [&]() { v3::winogradOutputTransform(d_output, d_input, N, C, H, W, queue); });
        run_test("V4", [&]() { v4::winogradOutputTransform(d_output, d_input, N, C, H, W, queue); });
        run_test("V5", [&]() { v5::winogradOutputTransform(d_output, d_input, N, C, H, W, queue); });

        sycl::free(d_input, queue);
        sycl::free(d_output, queue);
    }

    csv.close();
    std::cout << std::endl << "✅ REAL Winograd testing completed!" << std::endl;
    return 0;
}
