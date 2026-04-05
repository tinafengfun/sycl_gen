#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <cmath>

// Batch Normalization - 另一个困难kernel
// 涉及：均值、方差计算，归一化，缩放/偏移

// V0: Baseline - 直接实现
namespace v0 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void batch_norm_kernel(T* output, const T* input, const T* gamma, const T* beta,
                       const T* mean, const T* variance, int N, int C, int H, int W,
                       float epsilon, const sycl::nd_item<2> &item,
                       float* shared_mean, float* shared_var) {
    int c = item.get_global_id(0);
    int n = item.get_group(1);
    
    if (c >= C || n >= N) return;
    
    // 计算当前channel在当前sample的均值和方差
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            int idx = ((n * C + c) * H + h) * W + w;
            float val = (float)input[idx];
            sum += val;
            sq_sum += val * val;
        }
    }
    
    float local_mean = sum / (H * W);
    float local_var = sq_sum / (H * W) - local_mean * local_mean;
    
    // 使用全局统计量（推理模式）或局部统计量（训练模式）
    // 这里使用混合：全局mean/var + 局部归一化
    float final_mean = (float)mean[c];
    float final_var = (float)variance[c];
    float std_dev = sycl::sqrt(final_var + epsilon);
    
    // 归一化并应用gamma/beta
    float g = (float)gamma[c];
    float b = (float)beta[c];
    
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            int idx = ((n * C + c) * H + h) * W + w;
            float val = (float)input[idx];
            float normalized = (val - final_mean) / std_dev;
            float out = normalized * g + b;
            output[idx] = (T)out;
        }
    }
}

template <typename T>
void batchNorm(T* output, const T* input, const T* gamma, const T* beta,
               const T* mean, const T* variance, int N, int C, int H, int W,
               float epsilon, sycl::queue &queue) {
    queue.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> shared_mean(sycl::range<1>(128), h);
        sycl::local_accessor<float, 1> shared_var(sycl::range<1>(128), h);
        
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(C, N), sycl::range<2>(1, 1)),
            [=](sycl::nd_item<2> item) {
                batch_norm_kernel(output, input, gamma, beta, mean, variance,
                                 N, C, H, W, epsilon, item,
                                 shared_mean.get_multi_ptr<sycl::access::decorated::no>().get(),
                                 shared_var.get_multi_ptr<sycl::access::decorated::no>().get());
            }
        );
    });
}
}

// V1: Loop unrolling
namespace v1 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void batch_norm_kernel(T* output, const T* input, const T* gamma, const T* beta,
                       const T* mean, const T* variance, int N, int C, int H, int W,
                       float epsilon, const sycl::nd_item<2> &item) {
    int c = item.get_global_id(0);
    int n = item.get_group(1);
    
    if (c >= C || n >= N) return;
    
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    #pragma unroll 4
    for (int h = 0; h < H; h++) {
        #pragma unroll 4
        for (int w = 0; w < W; w++) {
            int idx = ((n * C + c) * H + h) * W + w;
            float val = (float)input[idx];
            sum += val;
            sq_sum += val * val;
        }
    }
    
    float final_mean = (float)mean[c];
    float final_var = (float)variance[c];
    float std_dev = sycl::sqrt(final_var + epsilon);
    
    float g = (float)gamma[c];
    float b = (float)beta[c];
    
    #pragma unroll 4
    for (int h = 0; h < H; h++) {
        #pragma unroll 4
        for (int w = 0; w < W; w++) {
            int idx = ((n * C + c) * H + h) * W + w;
            float val = (float)input[idx];
            float normalized = (val - final_mean) / std_dev;
            float out = normalized * g + b;
            output[idx] = (T)out;
        }
    }
}

template <typename T>
void batchNorm(T* output, const T* input, const T* gamma, const T* beta,
               const T* mean, const T* variance, int N, int C, int H, int W,
               float epsilon, sycl::queue &queue) {
    queue.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(C, N), sycl::range<2>(1, 1)),
        [=](sycl::nd_item<2> item) {
            batch_norm_kernel(output, input, gamma, beta, mean, variance,
                             N, C, H, W, epsilon, item);
        }
    );
}
}

// V2: Vectorized memory access
namespace v2 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void batch_norm_kernel(T* output, const T* input, const T* gamma, const T* beta,
                       const T* mean, const T* variance, int N, int C, int H, int W,
                       float epsilon, const sycl::nd_item<2> &item) {
    int c = item.get_global_id(0);
    int n = item.get_group(1);
    
    if (c >= C || n >= N) return;
    
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    // Process 4 elements at a time
    int vec_size = 4;
    int vec_limit = (H * W) / vec_size * vec_size;
    
    for (int i = 0; i < vec_limit; i += vec_size) {
        int h = i / W;
        int w = i % W;
        #pragma unroll
        for (int j = 0; j < vec_size; j++) {
            int idx = ((n * C + c) * H + h) * W + w + j;
            float val = (float)input[idx];
            sum += val;
            sq_sum += val * val;
        }
    }
    
    // Remainder
    for (int i = vec_limit; i < H * W; i++) {
        int h = i / W;
        int w = i % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = (float)input[idx];
        sum += val;
        sq_sum += val * val;
    }
    
    float final_mean = (float)mean[c];
    float final_var = (float)variance[c];
    float std_dev = sycl::sqrt(final_var + epsilon);
    
    float g = (float)gamma[c];
    float b = (float)beta[c];
    
    // Vectorized write back
    for (int i = 0; i < vec_limit; i += vec_size) {
        int h = i / W;
        int w = i % W;
        #pragma unroll
        for (int j = 0; j < vec_size; j++) {
            int idx = ((n * C + c) * H + h) * W + w + j;
            float val = (float)input[idx];
            float normalized = (val - final_mean) / std_dev;
            float out = normalized * g + b;
            output[idx] = (T)out;
        }
    }
    
    for (int i = vec_limit; i < H * W; i++) {
        int h = i / W;
        int w = i % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = (float)input[idx];
        float normalized = (val - final_mean) / std_dev;
        float out = normalized * g + b;
        output[idx] = (T)out;
    }
}

template <typename T>
void batchNorm(T* output, const T* input, const T* gamma, const T* beta,
               const T* mean, const T* variance, int N, int C, int H, int W,
               float epsilon, sycl::queue &queue) {
    queue.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(C, N), sycl::range<2>(1, 1)),
        [=](sycl::nd_item<2> item) {
            batch_norm_kernel(output, input, gamma, beta, mean, variance,
                             N, C, H, W, epsilon, item);
        }
    );
}
}

int main() {
    sycl::queue queue(sycl::gpu_selector_v);
    auto device = queue.get_device();
    std::cout << "========================================" << std::endl;
    std::cout << "Batch Normalization - HARD Kernel Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "GPU: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << std::endl;

    std::vector<int> sizes = {64, 128, 256};
    int iterations = 50;
    std::ofstream csv("hard_batch_norm_results.csv");
    csv << "Version,N,C,H,W,Time_ms,GFLOPS,Bandwidth_GB/s" << std::endl;

    int C = 128;
    int H = 8;
    int W = 8;
    float epsilon = 1e-5f;

    for (int N : sizes) {
        std::cout << "=== Testing N=" << N << ", C=" << C << ", H=W=" << H << " ===" << std::endl;
        
        int data_size = N * C * H * W;
        int param_size = C;
        
        std::vector<float> h_input(data_size);
        std::vector<float> h_output(data_size);
        std::vector<float> h_gamma(param_size);
        std::vector<float> h_beta(param_size);
        std::vector<float> h_mean(param_size);
        std::vector<float> h_var(param_size);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for (auto &v : h_input) v = dist(gen);
        for (auto &v : h_gamma) v = 1.0f + dist(gen) * 0.1f;
        for (auto &v : h_beta) v = dist(gen) * 0.1f;
        for (auto &v : h_mean) v = dist(gen) * 0.1f;
        for (auto &v : h_var) v = 0.5f + dist(gen) * 0.1f;
        
        float *d_input = sycl::malloc_device<float>(data_size, queue);
        float *d_output = sycl::malloc_device<float>(data_size, queue);
        float *d_gamma = sycl::malloc_device<float>(param_size, queue);
        float *d_beta = sycl::malloc_device<float>(param_size, queue);
        float *d_mean = sycl::malloc_device<float>(param_size, queue);
        float *d_var = sycl::malloc_device<float>(param_size, queue);
        
        queue.memcpy(d_input, h_input.data(), data_size * sizeof(float));
        queue.memcpy(d_gamma, h_gamma.data(), param_size * sizeof(float));
        queue.memcpy(d_beta, h_beta.data(), param_size * sizeof(float));
        queue.memcpy(d_mean, h_mean.data(), param_size * sizeof(float));
        queue.memcpy(d_var, h_var.data(), param_size * sizeof(float));
        queue.wait();
        
        // Each element: 2 reads + 1 write, plus some compute
        double total_ops = (double)N * C * H * W * 8;  // 8 FLOPs per element
        double total_bytes = (data_size * 2 + param_size * 4) * sizeof(float);
        
        auto run_test = [&](const char* name, auto &&kernel_func) {
            for (int i = 0; i < 5; ++i) {
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
            double gflops = (total_ops / (time_ms * 1e-3)) / 1e9;
            double bw = (total_bytes / (time_ms * 1e-3)) / 1e9;
            
            std::cout << name << "\tTime: " << time_ms << " ms\t"
                      << "GFLOPS: " << gflops << "\t"
                      << "BW: " << bw << " GB/s" << std::endl;
            
            csv << name << "," << N << "," << C << "," << H << "," << W << ","
                << time_ms << "," << gflops << "," << bw << std::endl;
        };
        
        run_test("V0", [&]() { v0::batchNorm(d_output, d_input, d_gamma, d_beta,
                                                  d_mean, d_var, N, C, H, W, epsilon, queue); });
        run_test("V1", [&]() { v1::batchNorm(d_output, d_input, d_gamma, d_beta,
                                                  d_mean, d_var, N, C, H, W, epsilon, queue); });
        run_test("V2", [&]() { v2::batchNorm(d_output, d_input, d_gamma, d_beta,
                                                  d_mean, d_var, N, C, H, W, epsilon, queue); });
        
        sycl::free(d_input, queue);
        sycl::free(d_output, queue);
        sycl::free(d_gamma, queue);
        sycl::free(d_beta, queue);
        sycl::free(d_mean, queue);
        sycl::free(d_var, queue);
    }

    csv.close();
    std::cout << std::endl << "Batch Norm testing completed!" << std::endl;
    return 0;
}
