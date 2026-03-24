#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <cmath>

// Layer Normalization - More complex than batch norm
// V0: Baseline (two-pass algorithm)
namespace v0 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void layerNorm_kernel(T* output, const T* input, const T* gamma, const T* beta,
                      int N, int C, float epsilon, const sycl::nd_item<1> &item,
                      float* shared_mem) {
    int n = item.get_group(0);
    int tid = item.get_local_id(0);
    int threads = item.get_local_range(0);
    
    if (n >= N) return;
    
    // Pass 1: Compute mean
    float sum = 0.0f;
    for (int c = tid; c < C; c += threads) {
        sum += (float)input[n * C + c];
    }
    shared_mem[tid] = sum;
    item.barrier();
    
    // Reduce to get mean
    for (int offset = threads / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared_mem[tid] += shared_mem[tid + offset];
        }
        item.barrier();
    }
    float mean = shared_mem[0] / C;
    
    // Pass 2: Compute variance
    float var_sum = 0.0f;
    for (int c = tid; c < C; c += threads) {
        float diff = (float)input[n * C + c] - mean;
        var_sum += diff * diff;
    }
    shared_mem[tid] = var_sum;
    item.barrier();
    
    for (int offset = threads / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared_mem[tid] += shared_mem[tid + offset];
        }
        item.barrier();
    }
    float variance = shared_mem[0] / C;
    float inv_std = 1.0f / std::sqrt(variance + epsilon);
    
    // Pass 3: Normalize and scale
    for (int c = tid; c < C; c += threads) {
        float normalized = ((float)input[n * C + c] - mean) * inv_std;
        output[n * C + c] = (T)(normalized * (float)gamma[c] + (float)beta[c]);
    }
}

template <typename T>
void layerNorm(T* output, const T* input, const T* gamma, const T* beta,
               int N, int C, float epsilon, sycl::queue &queue) {
    int wg_size = 256;
    int blocks = N;
    queue.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(wg_size), h);
        h.parallel_for(
            sycl::nd_range<1>(blocks * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                layerNorm_kernel(output, input, gamma, beta, N, C, epsilon, item,
                                shared_mem.template get_multi_ptr<sycl::access::decorated::no>().get());
            }
        );
    });
}
}

// V1: Smaller WG=128 with unrolled reduction
namespace v1 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void layerNorm_kernel(T* output, const T* input, const T* gamma, const T* beta,
                      int N, int C, float epsilon, const sycl::nd_item<1> &item,
                      float* shared_mem) {
    int n = item.get_group(0);
    int tid = item.get_local_id(0);
    int threads = item.get_local_range(0);
    
    if (n >= N) return;
    
    // Pass 1: Compute mean (unrolled)
    float sum = 0.0f;
    #pragma unroll 4
    for (int c = tid; c < C; c += threads) {
        sum += (float)input[n * C + c];
    }
    shared_mem[tid] = sum;
    item.barrier();
    
    // Tree reduction
    #pragma unroll
    for (int offset = threads / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared_mem[tid] += shared_mem[tid + offset];
        }
        item.barrier();
    }
    float mean = shared_mem[0] / C;
    
    // Pass 2: Compute variance
    float var_sum = 0.0f;
    #pragma unroll 4
    for (int c = tid; c < C; c += threads) {
        float diff = (float)input[n * C + c] - mean;
        var_sum += diff * diff;
    }
    shared_mem[tid] = var_sum;
    item.barrier();
    
    #pragma unroll
    for (int offset = threads / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared_mem[tid] += shared_mem[tid + offset];
        }
        item.barrier();
    }
    float variance = shared_mem[0] / C;
    float inv_std = 1.0f / std::sqrt(variance + epsilon);
    
    // Pass 3: Normalize and scale
    #pragma unroll 4
    for (int c = tid; c < C; c += threads) {
        float normalized = ((float)input[n * C + c] - mean) * inv_std;
        output[n * C + c] = (T)(normalized * (float)gamma[c] + (float)beta[c]);
    }
}

template <typename T>
void layerNorm(T* output, const T* input, const T* gamma, const T* beta,
               int N, int C, float epsilon, sycl::queue &queue) {
    int wg_size = 128;
    int blocks = N;
    queue.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(wg_size), h);
        h.parallel_for(
            sycl::nd_range<1>(blocks * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                layerNorm_kernel(output, input, gamma, beta, N, C, epsilon, item,
                                shared_mem.template get_multi_ptr<sycl::access::decorated::no>().get());
            }
        );
    });
}
}

// V2: Single-pass Welford algorithm
namespace v2 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void layerNorm_kernel(T* output, const T* input, const T* gamma, const T* beta,
                      int N, int C, float epsilon, const sycl::nd_item<1> &item,
                      float* shared_mean, float* shared_var) {
    int n = item.get_group(0);
    int tid = item.get_local_id(0);
    int threads = item.get_local_range(0);
    
    if (n >= N) return;
    
    // Single-pass Welford
    float mean = 0.0f;
    float M2 = 0.0f;
    int count = 0;
    
    #pragma unroll 4
    for (int c = tid; c < C; c += threads) {
        float x = (float)input[n * C + c];
        count++;
        float delta = x - mean;
        mean += delta / count;
        float delta2 = x - mean;
        M2 += delta * delta2;
    }
    
    shared_mean[tid] = mean;
    shared_var[tid] = M2;
    item.barrier();
    
    // Reduce mean and M2
    #pragma unroll
    for (int offset = threads / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            // Combine two Welford estimates
            float n1 = C / threads;
            float n2 = C / threads;
            float mean1 = shared_mean[tid];
            float mean2 = shared_mean[tid + offset];
            float m2_1 = shared_var[tid];
            float m2_2 = shared_var[tid + offset];
            
            float delta = mean2 - mean1;
            shared_mean[tid] = mean1 + delta * n2 / (n1 + n2);
            shared_var[tid] = m2_1 + m2_2 + delta * delta * n1 * n2 / (n1 + n2);
        }
        item.barrier();
    }
    
    mean = shared_mean[0];
    float variance = shared_var[0] / C;
    float inv_std = 1.0f / std::sqrt(variance + epsilon);
    
    // Normalize and scale
    #pragma unroll 4
    for (int c = tid; c < C; c += threads) {
        float normalized = ((float)input[n * C + c] - mean) * inv_std;
        output[n * C + c] = (T)(normalized * (float)gamma[c] + (float)beta[c]);
    }
}

template <typename T>
void layerNorm(T* output, const T* input, const T* gamma, const T* beta,
               int N, int C, float epsilon, sycl::queue &queue) {
    int wg_size = 256;
    int blocks = N;
    queue.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> shared_mean(sycl::range<1>(wg_size), h);
        sycl::local_accessor<float, 1> shared_var(sycl::range<1>(wg_size), h);
        h.parallel_for(
            sycl::nd_range<1>(blocks * wg_size, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                layerNorm_kernel(output, input, gamma, beta, N, C, epsilon, item,
                                shared_mean.template get_multi_ptr<sycl::access::decorated::no>().get(),
                                shared_var.template get_multi_ptr<sycl::access::decorated::no>().get());
            }
        );
    });
}
}

int main() {
    sycl::queue queue(sycl::gpu_selector_v);
    auto device = queue.get_device();
    std::cout << "========================================" << std::endl;
    std::cout << "Layer Normalization Kernel Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "GPU: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << std::endl;

    std::vector<int> sizes = {64, 128, 256};
    int iterations = 50;
    std::ofstream csv("layer_norm_results.csv");
    csv << "Version,N,C,Time_ms,GFLOPS,Bandwidth_GB/s" << std::endl;

    int C = 128;
    float epsilon = 1e-5f;

    for (int N : sizes) {
        int data_size = N * C;

        std::vector<float> h_input(data_size);
        std::vector<float> h_gamma(C);
        std::vector<float> h_beta(C);
        std::vector<float> h_output(data_size);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : h_input) v = dist(gen);
        for (auto &v : h_gamma) v = 1.0f + dist(gen) * 0.1f;
        for (auto &v : h_beta) v = dist(gen) * 0.1f;

        float *d_input = sycl::malloc_device<float>(data_size, queue);
        float *d_output = sycl::malloc_device<float>(data_size, queue);
        float *d_gamma = sycl::malloc_device<float>(C, queue);
        float *d_beta = sycl::malloc_device<float>(C, queue);
        
        queue.memcpy(d_input, h_input.data(), data_size * sizeof(float));
        queue.memcpy(d_gamma, h_gamma.data(), C * sizeof(float));
        queue.memcpy(d_beta, h_beta.data(), C * sizeof(float));
        queue.wait();

        // Layer norm: ~5 ops per element (mean, var, sqrt, normalize, scale)
        double total_ops = (double)N * C * 5;
        double total_bytes = (data_size * 2 + C * 2) * sizeof(float);

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
            
            std::cout << name << "\tN=" << N << " C=" << C << "\tTime: " << time_ms << " ms\t"
                      << "GFLOPS: " << gflops << "\tBW: " << bw << " GB/s" << std::endl;
            csv << name << "," << N << "," << C << ","
                << time_ms << "," << gflops << "," << bw << std::endl;
        };

        std::cout << "=== Testing N=" << N << " C=" << C << " ===" << std::endl;
        run_test("V0", [&]() { v0::layerNorm(d_output, d_input, d_gamma, d_beta, N, C, epsilon, queue); });
        run_test("V1", [&]() { v1::layerNorm(d_output, d_input, d_gamma, d_beta, N, C, epsilon, queue); });
        run_test("V2", [&]() { v2::layerNorm(d_output, d_input, d_gamma, d_beta, N, C, epsilon, queue); });

        sycl::free(d_input, queue);
        sycl::free(d_output, queue);
        sycl::free(d_gamma, queue);
        sycl::free(d_beta, queue);
    }

    csv.close();
    std::cout << std::endl << "✅ Layer Norm testing completed!" << std::endl;
    return 0;
}
