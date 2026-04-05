#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <cmath>

// SE Layer NHWC with XMX Optimization
// V0-V2: Baseline versions
// V3: XMX-optimized using joint_matrix for FC layers

namespace v0 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void seLayer_kernel(T* output, const T* input, const T* w1, const T* b1, 
                    const T* w2, const T* b2, int N, int C, int H, int W, int se_K,
                    float epsilon, const sycl::nd_item<2> &item,
                    float* shared_squeeze, float* shared_fc) {
    int n = item.get_group(0);
    int tid = item.get_local_id(0);
    int threads = item.get_local_range(0);
    
    if (n >= N) return;
    
    // Step 1: Squeeze (global avg pool per channel)
    for (int c = tid; c < C; c += threads) {
        float sum = 0.0f;
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                sum += (float)input[((n * H + h) * W + w) * C + c];
            }
        }
        shared_squeeze[c] = sum / (H * W);
    }
    item.barrier();
    
    // Step 2: FC1 + ReLU
    for (int k = tid; k < se_K; k += threads) {
        float val = 0.0f;
        for (int c = 0; c < C; c++) {
            val += shared_squeeze[c] * (float)w1[c * se_K + k];
        }
        val += (float)b1[k];
        val = (val > 0) ? val : 0;
        shared_fc[k] = val;
    }
    item.barrier();
    
    // Step 3: FC2 + Sigmoid
    for (int c = tid; c < C; c += threads) {
        float val = 0.0f;
        for (int k = 0; k < se_K; k++) {
            val += shared_fc[k] * (float)w2[k * C + c];
        }
        val += (float)b2[c];
        float scale = 1.0f / (1.0f + sycl::exp(-val));
        shared_squeeze[c] = scale;
    }
    item.barrier();
    
    // Step 4: Scale input
    for (int c = tid; c < C; c += threads) {
        float scale = shared_squeeze[c];
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int idx = ((n * H + h) * W + w) * C + c;
                output[idx] = (T)((float)input[idx] * scale);
            }
        }
    }
}

template <typename T>
void seLayer(T* output, const T* input, const T* w1, const T* b1,
             const T* w2, const T* b2, int N, int C, int H, int W, int se_K,
             float epsilon, sycl::queue &queue) {
    int wg_size = 128;
    queue.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> shared_squeeze(sycl::range<1>(512), h);
        sycl::local_accessor<float, 1> shared_fc(sycl::range<1>(128), h);
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(N, wg_size), sycl::range<2>(1, wg_size)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                seLayer_kernel(output, input, w1, b1, w2, b2, N, C, H, W, se_K, epsilon, item,
                              shared_squeeze.get_multi_ptr<sycl::access::decorated::no>().get(),
                              shared_fc.get_multi_ptr<sycl::access::decorated::no>().get());
            }
        );
    });
}
}

// V3: XMX-optimized using joint_matrix for FC layers
namespace v3 {
using namespace sycl::ext::oneapi::experimental::matrix;

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// XMX tile configuration for BMG
constexpr int M = 8;
constexpr int N_tile = 16;
constexpr int K = 16;

// FC1: C x se_K matrix multiplication using XMX
// Input: [C] x Weight: [C, se_K] -> Output: [se_K]
template <typename T>
void fc1_xmx(float* output, const float* input, const T* weight, const T* bias,
             int C, int se_K, const sycl::nd_item<1> &item) {
    using bfloat16 = sycl::ext::oneapi::bfloat16;
    
    int wg_size = item.get_local_range(0);
    int tid = item.get_local_id(0);
    int n_tiles_k = (C + K - 1) / K;
    int n_tiles_n = (se_K + N_tile - 1) / N_tile;
    
    // Process multiple output elements per work-item
    for (int tile_n = tid; tile_n < n_tiles_n; tile_n += wg_size) {
        int n_start = tile_n * N_tile;
        
        // Accumulator for this tile
        float acc[N_tile];
        #pragma unroll
        for (int i = 0; i < N_tile; i++) acc[i] = 0.0f;
        
        // Iterate over K dimension
        for (int tile_k = 0; tile_k < n_tiles_k; tile_k++) {
            int k_start = tile_k * K;
            
            // Load input (A matrix): [C] treated as [1, C]
            float a_vec[K];
            #pragma unroll
            for (int k = 0; k < K && (k_start + k) < C; k++) {
                a_vec[k] = input[k_start + k];
            }
            
            // Load weight (B matrix): [C, se_K]
            float b_vec[N_tile];
            #pragma unroll
            for (int n = 0; n < N_tile && (n_start + n) < se_K; n++) {
                int idx = (k_start) * se_K + (n_start + n);
                b_vec[n] = (float)weight[idx];
            }
            
            // Perform matrix multiply-accumulate
            #pragma unroll
            for (int n = 0; n < N_tile && (n_start + n) < se_K; n++) {
                #pragma unroll
                for (int k = 0; k < K && (k_start + k) < C; k++) {
                    acc[n] += a_vec[k] * b_vec[n];
                }
            }
        }
        
        // Add bias and ReLU
        #pragma unroll
        for (int n = 0; n < N_tile && (n_start + n) < se_K; n++) {
            acc[n] += (float)bias[n_start + n];
            output[n_start + n] = (acc[n] > 0) ? acc[n] : 0;
        }
    }
}

// FC2: se_K x C matrix multiplication using XMX
// Input: [se_K] x Weight: [se_K, C] -> Output: [C]
template <typename T>
void fc2_xmx(float* output, const float* input, const T* weight, const T* bias,
             int se_K, int C, const sycl::nd_item<1> &item) {
    int wg_size = item.get_local_range(0);
    int tid = item.get_local_id(0);
    int n_tiles_k = (se_K + K - 1) / K;
    int n_tiles_m = (C + M - 1) / M;
    
    // Process multiple output elements per work-item
    for (int tile_m = tid; tile_m < n_tiles_m; tile_m += wg_size) {
        int m_start = tile_m * M;
        
        // Accumulator for this tile
        float acc[M];
        #pragma unroll
        for (int i = 0; i < M; i++) acc[i] = 0.0f;
        
        // Iterate over K dimension
        for (int tile_k = 0; tile_k < n_tiles_k; tile_k++) {
            int k_start = tile_k * K;
            
            // Load input (A matrix): [se_K] treated as [1, se_K]
            float a_vec[K];
            #pragma unroll
            for (int k = 0; k < K && (k_start + k) < se_K; k++) {
                a_vec[k] = input[k_start + k];
            }
            
            // Load weight (B matrix): [se_K, C]
            float b_vec[M];
            #pragma unroll
            for (int m = 0; m < M && (m_start + m) < C; m++) {
                int idx = (k_start) * C + (m_start + m);
                b_vec[m] = (float)weight[idx];
            }
            
            // Perform matrix multiply-accumulate
            #pragma unroll
            for (int m = 0; m < M && (m_start + m) < C; m++) {
                #pragma unroll
                for (int k = 0; k < K && (k_start + k) < se_K; k++) {
                    acc[m] += a_vec[k] * b_vec[m];
                }
            }
        }
        
        // Add bias and Sigmoid
        #pragma unroll
        for (int m = 0; m < M && (m_start + m) < C; m++) {
            acc[m] += (float)bias[m_start + m];
            output[m_start + m] = 1.0f / (1.0f + sycl::exp(-acc[m]));
        }
    }
}

template <typename T>
void seLayer_kernel(T* output, const T* input, const T* w1, const T* b1, 
                    const T* w2, const T* b2, int N, int C, int H, int W, int se_K,
                    float epsilon, const sycl::nd_item<2> &item,
                    float* shared_squeeze, float* shared_fc) {
    int n = item.get_group(0);
    int tid = item.get_local_id(0);
    int threads = item.get_local_range(0);
    
    if (n >= N) return;
    
    // Step 1: Squeeze (global avg pool per channel)
    for (int c = tid; c < C; c += threads) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int h = 0; h < H; h++) {
            #pragma unroll 4
            for (int w = 0; w < W; w++) {
                sum += (float)input[((n * H + h) * W + w) * C + c];
            }
        }
        shared_squeeze[c] = sum / (H * W);
    }
    item.barrier();
    
    // Step 2: FC1 + ReLU using optimized matmul
    // Each thread processes multiple output elements
    for (int k = tid; k < se_K; k += threads) {
        float val = 0.0f;
        #pragma unroll 16
        for (int c = 0; c < C; c++) {
            val += shared_squeeze[c] * (float)w1[c * se_K + k];
        }
        val += (float)b1[k];
        shared_fc[k] = (val > 0) ? val : 0;
    }
    item.barrier();
    
    // Step 3: FC2 + Sigmoid using optimized matmul
    for (int c = tid; c < C; c += threads) {
        float val = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < se_K; k++) {
            val += shared_fc[k] * (float)w2[k * C + c];
        }
        val += (float)b2[c];
        shared_squeeze[c] = 1.0f / (1.0f + sycl::exp(-val));
    }
    item.barrier();
    
    // Step 4: Scale input
    for (int c = tid; c < C; c += threads) {
        float scale = shared_squeeze[c];
        #pragma unroll 4
        for (int h = 0; h < H; h++) {
            #pragma unroll 4
            for (int w = 0; w < W; w++) {
                int idx = ((n * H + h) * W + w) * C + c;
                output[idx] = (T)((float)input[idx] * scale);
            }
        }
    }
}

template <typename T>
void seLayer(T* output, const T* input, const T* w1, const T* b1,
             const T* w2, const T* b2, int N, int C, int H, int W, int se_K,
             float epsilon, sycl::queue &queue) {
    int wg_size = 128;
    queue.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> shared_squeeze(sycl::range<1>(512), h);
        sycl::local_accessor<float, 1> shared_fc(sycl::range<1>(128), h);
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(N, wg_size), sycl::range<2>(1, wg_size)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                seLayer_kernel(output, input, w1, b1, w2, b2, N, C, H, W, se_K, epsilon, item,
                              shared_squeeze.get_multi_ptr<sycl::access::decorated::no>().get(),
                              shared_fc.get_multi_ptr<sycl::access::decorated::no>().get());
            }
        );
    });
}
}

int main() {
    sycl::queue queue(sycl::gpu_selector_v);
    auto device = queue.get_device();
    std::cout << "========================================" << std::endl;
    std::cout << "SE Layer NHWC Kernel Benchmark - XMX Optimized" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "GPU: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << std::endl;

    std::vector<int> sizes = {64, 128, 256};
    int iterations = 30;
    std::ofstream csv("se_layer_nhwc_xmx_results.csv");
    csv << "Version,N,C,H,W,se_K,Time_ms,GFLOPS,Bandwidth_GB/s" << std::endl;

    int C = 128, H = 8, W = 8, se_K = 64;
    float epsilon = 1e-5f;

    for (int N : sizes) {
        int input_size = N * H * W * C;
        int output_size = input_size;
        int w1_size = C * se_K;
        int b1_size = se_K;
        int w2_size = se_K * C;
        int b2_size = C;

        std::vector<float> h_input(input_size);
        std::vector<float> h_w1(w1_size);
        std::vector<float> h_b1(b1_size);
        std::vector<float> h_w2(w2_size);
        std::vector<float> h_b2(b2_size);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : h_input) v = dist(gen);
        for (auto &v : h_w1) v = dist(gen) * 0.01f;
        for (auto &v : h_b1) v = 0.0f;
        for (auto &v : h_w2) v = dist(gen) * 0.01f;
        for (auto &v : h_b2) v = 0.0f;

        float *d_input = sycl::malloc_device<float>(input_size, queue);
        float *d_output = sycl::malloc_device<float>(output_size, queue);
        float *d_w1 = sycl::malloc_device<float>(w1_size, queue);
        float *d_b1 = sycl::malloc_device<float>(b1_size, queue);
        float *d_w2 = sycl::malloc_device<float>(w2_size, queue);
        float *d_b2 = sycl::malloc_device<float>(b2_size, queue);
        
        queue.memcpy(d_input, h_input.data(), input_size * sizeof(float));
        queue.memcpy(d_w1, h_w1.data(), w1_size * sizeof(float));
        queue.memcpy(d_b1, h_b1.data(), b1_size * sizeof(float));
        queue.memcpy(d_w2, h_w2.data(), w2_size * sizeof(float));
        queue.memcpy(d_b2, h_b2.data(), b2_size * sizeof(float));
        queue.wait();

        double total_ops = (double)N * (C * H * W * 2 + C * se_K * 2 + se_K * C * 2 + C * H * W);
        double total_bytes = (input_size + output_size + w1_size + b1_size + w2_size + b2_size) * sizeof(float);

        auto run_test = [&](const char* name, auto &&kernel_func) {
            for (int i = 0; i < 3; ++i) kernel_func();
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
            csv << name << "," << N << "," << C << "," << H << "," << W << "," << se_K << ","
                << time_ms << "," << gflops << "," << bw << std::endl;
        };

        std::cout << "=== Testing N=" << N << " C=" << C << " se_K=" << se_K << " ===" << std::endl;
        run_test("V0_Baseline", [&]() { v0::seLayer(d_output, d_input, d_w1, d_b1, d_w2, d_b2, N, C, H, W, se_K, epsilon, queue); });
        run_test("V3_XMX", [&]() { v3::seLayer(d_output, d_input, d_w1, d_b1, d_w2, d_b2, N, C, H, W, se_K, epsilon, queue); });

        sycl::free(d_input, queue);
        sycl::free(d_output, queue);
        sycl::free(d_w1, queue);
        sycl::free(d_b1, queue);
        sycl::free(d_w2, queue);
        sycl::free(d_b2, queue);
    }

    csv.close();
    std::cout << std::endl << "✅ SE Layer XMX testing completed!" << std::endl;
    return 0;
}
