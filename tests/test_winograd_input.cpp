#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <cmath>

// Winograd Input Transform
// V0: Baseline
namespace v0 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void inputTransform_kernel(T* output, const T* input, int N, int C, int H, int W,
                           const sycl::nd_item<3> &item) {
    int c = item.get_global_id(0);
    int h = item.get_global_id(1);
    int w = item.get_global_id(2);
    
    if (c >= C || h >= H/4 || w >= W/4) return;
    
    for (int n = 0; n < N; ++n) {
        // Read 4x4 input tile
        T inTile[4][4];
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                int in_h = h * 4 + y;
                int in_w = w * 4 + x;
                if (in_h < H && in_w < W) {
                    inTile[y][x] = input[((n * C + c) * H + in_h) * W + in_w];
                } else {
                    inTile[y][x] = 0;
                }
            }
        }
        
        // Transform to 6x6
        T outTile[6][6];
        for (int y = 0; y < 6; ++y) {
            for (int x = 0; x < 6; ++x) {
                outTile[y][x] = 0;
                // Simplified transform
                int sy = (y < 4) ? y : 3;
                int sx = (x < 4) ? x : 3;
                outTile[y][x] = inTile[sy][sx];
            }
        }
        
        // Write output
        for (int y = 0; y < 6; ++y) {
            for (int x = 0; x < 6; ++x) {
                output[((n * C + c) * 36 + y * 6 + x) * (H/4) * (W/4) + h * (W/4) + w] = outTile[y][x];
            }
        }
    }
}

template <typename T>
void inputTransform(T* output, const T* input, int N, int C, int H, int W, sycl::queue &queue) {
    int blocks_c = DivUp(C, 16);
    int blocks_h = DivUp(H/4, 4);
    int blocks_w = DivUp(W/4, 4);
    queue.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(blocks_c * 16, blocks_h * 4, blocks_w * 4),
                          sycl::range<3>(16, 4, 4)),
        [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(16)]] {
            inputTransform_kernel(output, input, N, C, H, W, item);
        }
    );
}
}

// V1: Loop unrolled
namespace v1 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void inputTransform_kernel(T* output, const T* input, int N, int C, int H, int W,
                           const sycl::nd_item<3> &item) {
    int c = item.get_global_id(0);
    int h = item.get_global_id(1);
    int w = item.get_global_id(2);
    
    if (c >= C || h >= H/4 || w >= W/4) return;
    
    #pragma unroll
    for (int n = 0; n < N; ++n) {
        T inTile[4][4];
        #pragma unroll
        for (int y = 0; y < 4; ++y) {
            #pragma unroll
            for (int x = 0; x < 4; ++x) {
                int in_h = h * 4 + y;
                int in_w = w * 4 + x;
                inTile[y][x] = (in_h < H && in_w < W) ? 
                    input[((n * C + c) * H + in_h) * W + in_w] : 0;
            }
        }
        
        T outTile[6][6];
        #pragma unroll
        for (int y = 0; y < 6; ++y) {
            #pragma unroll
            for (int x = 0; x < 6; ++x) {
                int sy = (y < 4) ? y : 3;
                int sx = (x < 4) ? x : 3;
                outTile[y][x] = inTile[sy][sx];
            }
        }
        
        #pragma unroll
        for (int y = 0; y < 6; ++y) {
            #pragma unroll
            for (int x = 0; x < 6; ++x) {
                output[((n * C + c) * 36 + y * 6 + x) * (H/4) * (W/4) + h * (W/4) + w] = outTile[y][x];
            }
        }
    }
}

template <typename T>
void inputTransform(T* output, const T* input, int N, int C, int H, int W, sycl::queue &queue) {
    int blocks_c = DivUp(C, 16);
    int blocks_h = DivUp(H/4, 4);
    int blocks_w = DivUp(W/4, 4);
    queue.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(blocks_c * 16, blocks_h * 4, blocks_w * 4),
                          sycl::range<3>(16, 4, 4)),
        [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(16)]] {
            inputTransform_kernel(output, input, N, C, H, W, item);
        }
    );
}
}

// V2: 1D flattened
namespace v2 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void inputTransform_kernel(T* output, const T* input, int N, int C, int H, int W, int total_tiles,
                           const sycl::nd_item<1> &item) {
    int idx = item.get_global_id(0);
    if (idx >= total_tiles) return;
    
    int tiles_per_channel = (H/4) * (W/4);
    int c = idx / tiles_per_channel;
    int tile_idx = idx % tiles_per_channel;
    int h = tile_idx / (W/4);
    int w = tile_idx % (W/4);
    
    if (c >= C) return;
    
    #pragma unroll
    for (int n = 0; n < N; ++n) {
        T inTile[4][4];
        #pragma unroll
        for (int y = 0; y < 4; ++y) {
            #pragma unroll
            for (int x = 0; x < 4; ++x) {
                int in_h = h * 4 + y;
                int in_w = w * 4 + x;
                inTile[y][x] = (in_h < H && in_w < W) ? 
                    input[((n * C + c) * H + in_h) * W + in_w] : 0;
            }
        }
        
        T outTile[6][6];
        #pragma unroll
        for (int y = 0; y < 6; ++y) {
            #pragma unroll
            for (int x = 0; x < 6; ++x) {
                int sy = (y < 4) ? y : 3;
                int sx = (x < 4) ? x : 3;
                outTile[y][x] = inTile[sy][sx];
            }
        }
        
        #pragma unroll
        for (int y = 0; y < 6; ++y) {
            #pragma unroll
            for (int x = 0; x < 6; ++x) {
                output[((n * C + c) * 36 + y * 6 + x) * (H/4) * (W/4) + tile_idx] = outTile[y][x];
            }
        }
    }
}

template <typename T>
void inputTransform(T* output, const T* input, int N, int C, int H, int W, sycl::queue &queue) {
    int total_tiles = C * (H/4) * (W/4);
    int blocks = DivUp(total_tiles, 256);
    queue.parallel_for(
        sycl::nd_range<1>(blocks * 256, 256),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            inputTransform_kernel(output, input, N, C, H, W, total_tiles, item);
        }
    );
}
}

int main() {
    sycl::queue queue(sycl::gpu_selector_v);
    auto device = queue.get_device();
    std::cout << "========================================" << std::endl;
    std::cout << "Winograd Input Transform Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "GPU: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << std::endl;

    std::vector<int> sizes = {64, 128, 256};
    int iterations = 50;
    std::ofstream csv("winograd_input_transform_results.csv");
    csv << "Version,N,C,H,W,Time_ms,GFLOPS,Bandwidth_GB/s" << std::endl;

    int C = 64, H = 8, W = 8;

    for (int N : sizes) {
        int input_size = N * C * H * W;
        int output_size = N * C * 36 * (H/4) * (W/4);
        
        std::vector<float> h_input(input_size);
        std::vector<float> h_output(output_size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : h_input) v = dist(gen);

        float *d_input = sycl::malloc_device<float>(input_size, queue);
        float *d_output = sycl::malloc_device<float>(output_size, queue);
        queue.memcpy(d_input, h_input.data(), input_size * sizeof(float));
        queue.wait();

        double total_ops = (double)N * C * (H/4) * (W/4) * 36 * 2;  // Approximate
        double total_bytes = (input_size + output_size) * sizeof(float);

        auto run_test = [&](const char* name, auto &&kernel_func) {
            for (int i = 0; i < 5; ++i) kernel_func();
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

        run_test("V0", [&]() { v0::inputTransform(d_output, d_input, N, C, H, W, queue); });
        run_test("V1", [&]() { v1::inputTransform(d_output, d_input, N, C, H, W, queue); });
        run_test("V2", [&]() { v2::inputTransform(d_output, d_input, N, C, H, W, queue); });

        sycl::free(d_input, queue);
        sycl::free(d_output, queue);
    }

    csv.close();
    std::cout << std::endl << "Testing completed!" << std::endl;
    return 0;
}
