#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <cmath>

// 困难Kernel测试：Winograd Output + SE + ReLU + Input Transform (融合4个操作)
// 这是最难优化的kernel之一

// V0: Baseline - 直接翻译，一个线程处理一个channel
namespace v0 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void fused_kernel(T* output, const T* input, const T* skip, const T* bias,
                  const T* w1, const T* b1, const T* w2, const T* b2,
                  int N, int C, int se_K, const sycl::nd_item<2> &item,
                  float* shared_data) {
    int k = item.get_local_id(0);
    int n = item.get_group(0);
    
    if (k >= C || n >= N) return;
    
    T board[8][8];
    T b = bias[k];
    
    for (int h = 0; h < 8; h += 4) {
        for (int w = 0; w < 8; w += 4) {
            T tile[6][6];
            for (int y = 0; y < 6; y++) {
                for (int x = 0; x < 6; x++) {
                    tile[y][x] = input[((n * 36 + y * 6 + x) * C + k)];
                }
            }
            
            T outEl[4][4];
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    outEl[y][x] = tile[y][x] + tile[y+1][x] + tile[y+2][x];
                    outEl[y][x] += tile[y][x+1] + tile[y+1][x+1] + tile[y+2][x+1];
                    outEl[y][x] += tile[y][x+2] + tile[y+1][x+2] + tile[y+2][x+2];
                }
            }
            
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    board[h + y][w + x] = outEl[y][x] + b;
                }
            }
        }
    }
    
    float S = 0;
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            S += (float)board[y][x];
        }
    }
    
    shared_data[k] = S / 64.0f;
    item.barrier();
    
    float se_fc1[64];
    for (int i = 0; i < se_K; i++) {
        float val = 0;
        for (int c = 0; c < C; c++) {
            val += shared_data[c] * (float)w1[c * se_K + i];
        }
        val += (float)b1[i];
        if (val < 0) val = 0;
        se_fc1[i] = val;
    }
    
    float se_weight = 0;
    for (int i = 0; i < se_K; i++) {
        se_weight += se_fc1[i] * (float)w2[i * C + k];
    }
    se_weight += (float)b2[k];
    se_weight = 1.0f / (1.0f + sycl::exp(-se_weight));
    
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            float val = (float)board[y][x] * se_weight;
            val += (float)skip[((n * 8 + y) * 8 + x) * C + k];
            if (val < 0) val = 0;
            board[y][x] = (T)val;
        }
    }
    
    for (int h = 0; h < 8; h += 4) {
        for (int w = 0; w < 8; w += 4) {
            T inTile[4][4];
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    inTile[y][x] = board[h + y][w + x];
                }
            }
            
            T outTile[6][6] = {};
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    outTile[y][x] = inTile[y][x];
                }
            }
            
            for (int y = 0; y < 6; y++) {
                for (int x = 0; x < 6; x++) {
                    output[((n * 36 + y * 6 + x) * C + k)] = outTile[y][x];
                }
            }
        }
    }
}

template <typename T>
void fusedTransform(T* output, const T* input, const T* skip, const T* bias,
                    const T* w1, const T* b1, const T* w2, const T* b2,
                    int N, int C, int se_K, sycl::queue &queue) {
    int threads = C;
    queue.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> shared_data(sycl::range<1>(256), h);
        
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(N, threads), sycl::range<2>(1, threads)),
            [=](sycl::nd_item<2> item) {
                fused_kernel(output, input, skip, bias, w1, b1, w2, b2,
                            N, C, se_K, item,
                            shared_data.template get_multi_ptr<sycl::access::decorated::no>().get());
            }
        );
    });
}
}

// V1: Loop unrolling
namespace v1 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void fused_kernel(T* output, const T* input, const T* skip, const T* bias,
                  const T* w1, const T* b1, const T* w2, const T* b2,
                  int N, int C, int se_K, const sycl::nd_item<2> &item,
                  float* shared_data) {
    int k = item.get_local_id(0);
    int n = item.get_group(0);
    
    if (k >= C || n >= N) return;
    
    T board[8][8];
    T b = bias[k];
    
    #pragma unroll
    for (int h = 0; h < 8; h += 4) {
        #pragma unroll
        for (int w = 0; w < 8; w += 4) {
            T tile[6][6];
            #pragma unroll
            for (int y = 0; y < 6; y++) {
                #pragma unroll
                for (int x = 0; x < 6; x++) {
                    tile[y][x] = input[((n * 36 + y * 6 + x) * C + k)];
                }
            }
            
            T outEl[4][4];
            #pragma unroll
            for (int y = 0; y < 4; y++) {
                #pragma unroll
                for (int x = 0; x < 4; x++) {
                    outEl[y][x] = tile[y][x] + tile[y+1][x] + tile[y+2][x];
                    outEl[y][x] += tile[y][x+1] + tile[y+1][x+1] + tile[y+2][x+1];
                    outEl[y][x] += tile[y][x+2] + tile[y+1][x+2] + tile[y+2][x+2];
                }
            }
            
            #pragma unroll
            for (int y = 0; y < 4; y++) {
                #pragma unroll
                for (int x = 0; x < 4; x++) {
                    board[h + y][w + x] = outEl[y][x] + b;
                }
            }
        }
    }
    
    float S = 0;
    #pragma unroll
    for (int y = 0; y < 8; y++) {
        #pragma unroll
        for (int x = 0; x < 8; x++) {
            S += (float)board[y][x];
        }
    }
    
    shared_data[k] = S / 64.0f;
    item.barrier();
    
    float se_fc1[64];
    for (int i = 0; i < se_K; i++) {
        float val = 0;
        #pragma unroll 8
        for (int c = 0; c < C; c++) {
            val += shared_data[c] * (float)w1[c * se_K + i];
        }
        val += (float)b1[i];
        val = (val > 0) ? val : 0;
        se_fc1[i] = val;
    }
    
    float se_weight = 0;
    #pragma unroll 8
    for (int i = 0; i < se_K; i++) {
        se_weight += se_fc1[i] * (float)w2[i * C + k];
    }
    se_weight += (float)b2[k];
    se_weight = 1.0f / (1.0f + sycl::exp(-se_weight));
    
    #pragma unroll
    for (int y = 0; y < 8; y++) {
        #pragma unroll
        for (int x = 0; x < 8; x++) {
            float val = (float)board[y][x] * se_weight;
            val += (float)skip[((n * 8 + y) * 8 + x) * C + k];
            val = (val > 0) ? val : 0;
            board[y][x] = (T)val;
        }
    }
    
    #pragma unroll
    for (int h = 0; h < 8; h += 4) {
        #pragma unroll
        for (int w = 0; w < 8; w += 4) {
            T inTile[4][4];
            #pragma unroll
            for (int y = 0; y < 4; y++) {
                #pragma unroll
                for (int x = 0; x < 4; x++) {
                    inTile[y][x] = board[h + y][w + x];
                }
            }
            
            T outTile[6][6] = {};
            #pragma unroll
            for (int y = 0; y < 4; y++) {
                #pragma unroll
                for (int x = 0; x < 4; x++) {
                    outTile[y][x] = inTile[y][x];
                }
            }
            
            #pragma unroll
            for (int y = 0; y < 6; y++) {
                #pragma unroll
                for (int x = 0; x < 6; x++) {
                    output[((n * 36 + y * 6 + x) * C + k)] = outTile[y][x];
                }
            }
        }
    }
}

template <typename T>
void fusedTransform(T* output, const T* input, const T* skip, const T* bias,
                    const T* w1, const T* b1, const T* w2, const T* b2,
                    int N, int C, int se_K, sycl::queue &queue) {
    int threads = C;
    queue.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> shared_data(sycl::range<1>(256), h);
        
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(N, threads), sycl::range<2>(1, threads)),
            [=](sycl::nd_item<2> item) {
                fused_kernel(output, input, skip, bias, w1, b1, w2, b2,
                            N, C, se_K, item,
                            shared_data.template get_multi_ptr<sycl::access::decorated::no>().get());
            }
        );
    });
}
}

// V2: Multi-thread per channel
namespace v2 {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
void fused_kernel(T* output, const T* input, const T* skip, const T* bias,
                  const T* w1, const T* b1, const T* w2, const T* b2,
                  int N, int C, int se_K, const sycl::nd_item<2> &item,
                  float* shared_data) {
    int tid = item.get_local_id(0);
    int threads = item.get_local_range(0);
    int n = item.get_group(0);
    
    if (n >= N) return;
    
    for (int k = tid; k < C; k += threads) {
        T board[8][8];
        T b = bias[k];
        
        for (int h = 0; h < 8; h += 4) {
            for (int w = 0; w < 8; w += 4) {
                T tile[6][6];
                for (int y = 0; y < 6; y++) {
                    for (int x = 0; x < 6; x++) {
                        tile[y][x] = input[((n * 36 + y * 6 + x) * C + k)];
                    }
                }
                
                T outEl[4][4];
                for (int y = 0; y < 4; y++) {
                    for (int x = 0; x < 4; x++) {
                        outEl[y][x] = tile[y][x] + tile[y+1][x] + tile[y+2][x];
                        outEl[y][x] += tile[y][x+1] + tile[y+1][x+1] + tile[y+2][x+1];
                        outEl[y][x] += tile[y][x+2] + tile[y+1][x+2] + tile[y+2][x+2];
                        board[h + y][w + x] = outEl[y][x] + b;
                    }
                }
            }
        }
        
        float S = 0;
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                S += (float)board[y][x];
            }
        }
        shared_data[k] = S / 64.0f;
    }
    
    item.barrier();
    
    for (int k = tid; k < C; k += threads) {
        float se_weight = 0;
        for (int i = 0; i < se_K; i++) {
            float fc1 = 0;
            for (int c = 0; c < C; c++) {
                fc1 += shared_data[c] * (float)w1[c * se_K + i];
            }
            fc1 += (float)b1[i];
            if (fc1 < 0) fc1 = 0;
            se_weight += fc1 * (float)w2[i * C + k];
        }
        se_weight += (float)b2[k];
        se_weight = 1.0f / (1.0f + sycl::exp(-se_weight));
        shared_data[C + k] = se_weight;
    }
    
    item.barrier();
    
    for (int k = tid; k < C; k += threads) {
        T board[8][8];
        T b = bias[k];
        
        for (int h = 0; h < 8; h += 4) {
            for (int w = 0; w < 8; w += 4) {
                T tile[6][6];
                for (int y = 0; y < 6; y++) {
                    for (int x = 0; x < 6; x++) {
                        tile[y][x] = input[((n * 36 + y * 6 + x) * C + k)];
                    }
                }
                
                for (int h2 = 0; h2 < 4; h2++) {
                    for (int w2 = 0; w2 < 4; w2++) {
                        T outEl = tile[h2][w2] + tile[h2+1][w2] + tile[h2+2][w2];
                        outEl += tile[h2][w2+1] + tile[h2+1][w2+1] + tile[h2+2][w2+1];
                        outEl += tile[h2][w2+2] + tile[h2+1][w2+2] + tile[h2+2][w2+2];
                        board[h + h2][w + w2] = outEl + b;
                    }
                }
            }
        }
        
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                float val = (float)board[y][x] * shared_data[C + k];
                val += (float)skip[((n * 8 + y) * 8 + x) * C + k];
                if (val < 0) val = 0;
                board[y][x] = (T)val;
            }
        }
        
        for (int h = 0; h < 8; h += 4) {
            for (int w = 0; w < 8; w += 4) {
                T outTile[6][6] = {};
                for (int y = 0; y < 4; y++) {
                    for (int x = 0; x < 4; x++) {
                        outTile[y][x] = board[h + y][w + x];
                    }
                }
                
                for (int y = 0; y < 6; y++) {
                    for (int x = 0; x < 6; x++) {
                        output[((n * 36 + y * 6 + x) * C + k)] = outTile[y][x];
                    }
                }
            }
        }
    }
}

template <typename T>
void fusedTransform(T* output, const T* input, const T* skip, const T* bias,
                    const T* w1, const T* b1, const T* w2, const T* b2,
                    int N, int C, int se_K, sycl::queue &queue) {
    int wg_size = 128;
    queue.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> shared_data(sycl::range<1>(512), h);
        
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(N, wg_size), sycl::range<2>(1, wg_size)),
            [=](sycl::nd_item<2> item) {
                fused_kernel(output, input, skip, bias, w1, b1, w2, b2,
                            N, C, se_K, item,
                            shared_data.template get_multi_ptr<sycl::access::decorated::no>().get());
            }
        );
    });
}
}

int main() {
    sycl::queue queue(sycl::gpu_selector_v);
    auto device = queue.get_device();
    std::cout << "================================================" << std::endl;
    std::cout << "FUSED Winograd+SE+ReLU+Input - HARD Kernel Test" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "GPU: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Difficulty: 4 fused operations with complex control flow" << std::endl;
    std::cout << std::endl;

    std::vector<int> sizes = {64, 128, 256};
    int iterations = 20;
    std::ofstream csv("hard_fused_kernel_results.csv");
    csv << "Version,N,C,se_K,Time_ms,GFLOPS,Bandwidth_GB/s" << std::endl;

    int C = 128;
    int se_K = 64;

    for (int N : sizes) {
        std::cout << "=== N=" << N << ", C=" << C << ", se_K=" << se_K << " ===" << std::endl;
        
        int input_size = N * 36 * C;
        int output_size = input_size;
        int skip_size = N * 8 * 8 * C;
        int bias_size = C;
        int w1_size = C * se_K;
        int b1_size = se_K;
        int w2_size = se_K * C;
        int b2_size = C;
        
        std::vector<float> h_input(input_size);
        std::vector<float> h_skip(skip_size);
        std::vector<float> h_bias(bias_size);
        std::vector<float> h_w1(w1_size);
        std::vector<float> h_b1(b1_size);
        std::vector<float> h_w2(w2_size);
        std::vector<float> h_b2(b2_size);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for (auto &v : h_input) v = dist(gen);
        for (auto &v : h_skip) v = dist(gen) * 0.1f;
        for (auto &v : h_bias) v = dist(gen) * 0.1f;
        for (auto &v : h_w1) v = dist(gen) * 0.01f;
        for (auto &v : h_b1) v = 0.0f;
        for (auto &v : h_w2) v = dist(gen) * 0.01f;
        for (auto &v : h_b2) v = 0.0f;
        
        float *d_input = sycl::malloc_device<float>(input_size, queue);
        float *d_output = sycl::malloc_device<float>(output_size, queue);
        float *d_skip = sycl::malloc_device<float>(skip_size, queue);
        float *d_bias = sycl::malloc_device<float>(bias_size, queue);
        float *d_w1 = sycl::malloc_device<float>(w1_size, queue);
        float *d_b1 = sycl::malloc_device<float>(b1_size, queue);
        float *d_w2 = sycl::malloc_device<float>(w2_size, queue);
        float *d_b2 = sycl::malloc_device<float>(b2_size, queue);
        
        queue.memcpy(d_input, h_input.data(), input_size * sizeof(float));
        queue.memcpy(d_skip, h_skip.data(), skip_size * sizeof(float));
        queue.memcpy(d_bias, h_bias.data(), bias_size * sizeof(float));
        queue.memcpy(d_w1, h_w1.data(), w1_size * sizeof(float));
        queue.memcpy(d_b1, h_b1.data(), b1_size * sizeof(float));
        queue.memcpy(d_w2, h_w2.data(), w2_size * sizeof(float));
        queue.memcpy(d_b2, h_b2.data(), b2_size * sizeof(float));
        queue.wait();
        
        double total_ops = (double)N * C * (400 + 2 * se_K);
        double total_bytes = (input_size + output_size + skip_size + bias_size + 
                             w1_size + b1_size + w2_size + b2_size) * sizeof(float);
        
        auto run_test = [&](const char* name, auto &&kernel_func) {
            for (int i = 0; i < 3; ++i) {
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
            
            csv << name << "," << N << "," << C << "," << se_K << ","
                << time_ms << "," << gflops << "," << bw << std::endl;
        };
        
        run_test("V0", [&]() { v0::fusedTransform(d_output, d_input, d_skip, d_bias,
                                                      d_w1, d_b1, d_w2, d_b2, N, C, se_K, queue); });
        run_test("V1", [&]() { v1::fusedTransform(d_output, d_input, d_skip, d_bias,
                                                      d_w1, d_b1, d_w2, d_b2, N, C, se_K, queue); });
        run_test("V2", [&]() { v2::fusedTransform(d_output, d_input, d_skip, d_bias,
                                                      d_w1, d_b1, d_w2, d_b2, N, C, se_K, queue); });
        
        sycl::free(d_input, queue);
        sycl::free(d_output, queue);
        sycl::free(d_skip, queue);
        sycl::free(d_bias, queue);
        sycl::free(d_w1, queue);
        sycl::free(d_b1, queue);
        sycl::free(d_w2, queue);
        sycl::free(d_b2, queue);
    }

    csv.close();
    std::cout << std::endl << "Hard kernel testing completed!" << std::endl;
    return 0;
}
