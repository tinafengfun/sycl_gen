#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace std;

enum ActivationFunction {
  ACTIVATION_NONE,
  ACTIVATION_RELU,
  ACTIVATION_RELU_2,
  ACTIVATION_TANH,
  ACTIVATION_SIGMOID,
  ACTIVATION_SELU,
  ACTIVATION_SWISH,
  ACTIVATION_MISH
};

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
inline void copyAs(void* dst, const void* src) {
  *((T*)(dst)) = *((const T*)(src));
}

inline float warpReduce(float x, sycl::sub_group sg) {
  for (int mask = 16; mask > 0; mask >>= 1)
    x += sycl::permute_group_by_xor(sg, x, mask);
  return x;
}

inline float mishActivate(float el) {
  auto e = sycl::exp(el);
  auto n = e * e + 2.0f * e;
  auto d = el / (n + 2.0f);
  return (el <= -0.6f) ? n * d : el - 2.0f * d;
}

inline float activate(float cVal, ActivationFunction activation) {
  switch (activation) {
    case ACTIVATION_RELU:
      cVal = cVal < 0 ? 0 : cVal;
      break;
    case ACTIVATION_RELU_2:
      cVal = cVal < 0 ? 0 : cVal;
      cVal *= cVal;
      break;
    case ACTIVATION_TANH:
      cVal = sycl::tanh(cVal);
      break;
    case ACTIVATION_SIGMOID:
      cVal = 1.0f / (1.0f + sycl::exp(-cVal));
      break;
    case ACTIVATION_SELU: {
      float alpha = 1.67326324f, scale = 1.05070098f;
      cVal = cVal > 0 ? scale * cVal : scale * alpha * (sycl::exp(cVal) - 1.0f);
      break;
    }
    case ACTIVATION_MISH:
      cVal = mishActivate(cVal);
      break;
    case ACTIVATION_SWISH:
      cVal /= (1.0f + sycl::exp(-cVal));
      break;
    case ACTIVATION_NONE:
      break;
    default:
      cVal = 0;
      break;
  }
  return cVal;
}

template <typename T>
class layer_norm_kernel;

template <typename T>
void LayerNorm(int N, int C, T* output, const T* input, const T* bias,
               const T* skip, const T* gammas, const T* betas, float ep,
               float alpha, ActivationFunction act, sycl::queue& stream) {
  if (C % 16 != 0) throw std::runtime_error("unsupported filter size");
  if (C > 16384) throw std::runtime_error("unsupported filter size");

  constexpr int block_x = 32;
  int block_y = DivUp(C / 16, block_x);
  int block_z = std::min(std::max(512 / (block_x * block_y), 1), N);
  int grid_x = DivUp(N, block_z);

  sycl::range<3> grid(grid_x, 1, 1);
  sycl::range<3> block(block_z, block_y, block_x);

  auto event = stream.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 2> sum_acc({16, 16}, cgh);

    cgh.parallel_for<layer_norm_kernel<T>>(
        sycl::nd_range<3>(grid * block, block),
        [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
          int n = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
          if (n >= N) return;
          int c = (item.get_local_id(1) * 32 + item.get_local_id(2)) * 16;
          bool oobThread = c >= C;

          int biasIndex = c;
          int tensorIndex = n * C + c;

          float val[16] = {0};
          float oth[16] = {0};

          const bool fp16 = std::is_same<sycl::half, T>::value;
          if (!oobThread) {
            if (fp16) {
              sycl::half inp[8];
              copyAs<sycl::uint4>(&inp[0], &input[tensorIndex]);
              for (int i = 0; i < 8; i++) val[i] = (float)inp[i];
              copyAs<sycl::uint4>(&inp[0], &input[tensorIndex + 8]);
              for (int i = 0; i < 8; i++) val[i + 8] = (float)inp[i];
              copyAs<sycl::uint4>(&inp[0], &bias[biasIndex]);
              for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
              copyAs<sycl::uint4>(&inp[0], &bias[biasIndex + 8]);
              for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
              for (int i = 0; i < 16; i++) val[i] += oth[i];
            } else {
              copyAs<sycl::uint4>(&val[0], &input[tensorIndex]);
              copyAs<sycl::uint4>(&val[4], &input[tensorIndex + 4]);
              copyAs<sycl::uint4>(&val[8], &input[tensorIndex + 8]);
              copyAs<sycl::uint4>(&val[12], &input[tensorIndex + 12]);
              copyAs<sycl::uint4>(&oth[0], &bias[biasIndex]);
              copyAs<sycl::uint4>(&oth[4], &bias[biasIndex + 4]);
              copyAs<sycl::uint4>(&oth[8], &bias[biasIndex + 8]);
              copyAs<sycl::uint4>(&oth[12], &bias[biasIndex + 12]);
              for (int i = 0; i < 16; i++) val[i] += oth[i];
            }
          }

          float s = 0;
          if (!oobThread) {
            for (int i = 0; i < 16; i++) {
              val[i] = activate(val[i], act) * alpha;
              s += val[i];
            }
          }

          auto sg = item.get_sub_group();
          s = warpReduce(s, sg);
          if (item.get_local_id(2) == 0) {
            sum_acc[item.get_local_id(0)][item.get_local_id(1)] = s;
          }
          item.barrier(sycl::access::fence_space::local_space);

          if (item.get_local_id(2) == 0 && item.get_local_id(1) == 0) {
            float cSum = 0;
            for (int j = 0; j < item.get_local_range(1); j++)
              cSum += sum_acc[item.get_local_id(0)][j];
            sum_acc[item.get_local_id(0)][0] = cSum;
          }
          item.barrier(sycl::access::fence_space::local_space);

          float mean = sum_acc[item.get_local_id(0)][0] / C;

          s = 0;
          if (!oobThread) {
            for (int i = 0; i < 16; i++) {
              float d = val[i] - mean;
              s += d * d;
            }
          }

          s = warpReduce(s, sg);
          if (item.get_local_id(2) == 0) {
            sum_acc[item.get_local_id(0)][item.get_local_id(1)] = s;
          }
          item.barrier(sycl::access::fence_space::local_space);

          if (item.get_local_id(2) == 0 && item.get_local_id(1) == 0) {
            float cSum = 0;
            for (int j = 0; j < item.get_local_range(1); j++)
              cSum += sum_acc[item.get_local_id(0)][j];
            sum_acc[item.get_local_id(0)][0] = cSum;
          }
          item.barrier(sycl::access::fence_space::local_space);

          float var = sum_acc[item.get_local_id(0)][0] / C;

          if (!oobThread) {
            if (fp16) {
              sycl::half inp[8];
              copyAs<sycl::uint4>(&inp[0], &gammas[biasIndex]);
              for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
              copyAs<sycl::uint4>(&inp[0], &gammas[biasIndex + 8]);
              for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
            } else {
              copyAs<sycl::uint4>(&oth[0], &gammas[biasIndex]);
              copyAs<sycl::uint4>(&oth[4], &gammas[biasIndex + 4]);
              copyAs<sycl::uint4>(&oth[8], &gammas[biasIndex + 8]);
              copyAs<sycl::uint4>(&oth[12], &gammas[biasIndex + 12]);
            }
          }

          for (int i = 0; i < 16; i++) {
            float d = val[i] - mean;
            float norm = d / sycl::sqrt(var + ep);
            val[i] = norm * oth[i];
          }

          if (!oobThread) {
            if (fp16) {
              sycl::half inp[8];
              copyAs<sycl::uint4>(&inp[0], &betas[biasIndex]);
              for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
              copyAs<sycl::uint4>(&inp[0], &betas[biasIndex + 8]);
              for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
            } else {
              copyAs<sycl::uint4>(&oth[0], &betas[biasIndex]);
              copyAs<sycl::uint4>(&oth[4], &betas[biasIndex + 4]);
              copyAs<sycl::uint4>(&oth[8], &betas[biasIndex + 8]);
              copyAs<sycl::uint4>(&oth[12], &betas[biasIndex + 12]);
            }
          }

          for (int i = 0; i < 16; i++) {
            val[i] += oth[i];
          }

          if (!oobThread) {
            if (fp16) {
              sycl::half op[8];
              for (int i = 0; i < 8; i++) op[i] = (sycl::half)val[i];
              copyAs<sycl::uint4>(&output[tensorIndex], &op[0]);
              for (int i = 0; i < 8; i++) op[i] = (sycl::half)val[i + 8];
              copyAs<sycl::uint4>(&output[tensorIndex + 8], &op[0]);
            } else {
              copyAs<sycl::uint4>(&output[tensorIndex], &val[0]);
              copyAs<sycl::uint4>(&output[tensorIndex + 4], &val[4]);
              copyAs<sycl::uint4>(&output[tensorIndex + 8], &val[8]);
              copyAs<sycl::uint4>(&output[tensorIndex + 12], &val[12]);
            }
          }
        });
  });
  event.wait_and_throw();
}

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== layer_norm - Round 1 (Real Kernel) ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    
    // Test configurations: (N, C) - C must be multiple of 16
    struct TestConfig {
      int N, C;
      int totalElements;
    };
    
    vector<TestConfig> configs = {
      {64, 64},      // 4096 elements
      {256, 256},    // 65536 elements
      {1024, 1024},  // 1048576 elements (1M)
      {4096, 1024},  // 4194304 elements (4M)
    };
    
    for (auto& cfg : configs) {
      cfg.totalElements = cfg.N * cfg.C;
    }
    
    cout << setw(15) << "Config (NxC)" 
         << setw(15) << "Total"
         << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS"
         << setw(18) << "GB/s" << endl;
    cout << string(78, '-') << endl;
    
    for (const auto& cfg : configs) {
      const int N = cfg.N;
      const int C = cfg.C;
      const int total = cfg.totalElements;
      
      // Allocate device memory
      sycl::half* d_input = sycl::malloc_device<sycl::half>(total, q);
      sycl::half* d_output = sycl::malloc_device<sycl::half>(total, q);
      sycl::half* d_bias = sycl::malloc_device<sycl::half>(C, q);
      sycl::half* d_gammas = sycl::malloc_device<sycl::half>(C, q);
      sycl::half* d_betas = sycl::malloc_device<sycl::half>(C, q);
      
      // Initialize
      vector<sycl::half> h_input(total, sycl::half(1.0f));
      vector<sycl::half> h_bias(C, sycl::half(0.5f));
      vector<sycl::half> h_gammas(C, sycl::half(1.0f));
      vector<sycl::half> h_betas(C, sycl::half(0.0f));
      
      q.memcpy(d_input, h_input.data(), total * sizeof(sycl::half)).wait();
      q.memcpy(d_bias, h_bias.data(), C * sizeof(sycl::half)).wait();
      q.memcpy(d_gammas, h_gammas.data(), C * sizeof(sycl::half)).wait();
      q.memcpy(d_betas, h_betas.data(), C * sizeof(sycl::half)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        LayerNorm(N, C, d_output, d_input, d_bias, (const sycl::half*)nullptr, d_gammas, d_betas, 1e-5f, 1.0f, ACTIVATION_RELU, q);
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        
        LayerNorm(N, C, d_output, d_input, d_bias, (const sycl::half*)nullptr, d_gammas, d_betas, 1e-5f, 1.0f, ACTIVATION_RELU, q);
        
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        times.push_back(duration.count());
      }
      
      // Stats
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      // Metrics: Complex layer norm with multiple operations
      // Approx: activation + sum + mean + var + norm + scale + shift = ~20 FLOPs per element
      double flops = 20.0 * total;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      
      // Memory: read input + read bias + read gammas + read betas + write output
      double bytes = (total * 3 + C * 3) * sizeof(sycl::half);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(15) << (to_string(N) + "x" + to_string(C))
           << setw(15) << total
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_input, q);
      sycl::free(d_output, q);
      sycl::free(d_bias, q);
      sycl::free(d_gammas, q);
      sycl::free(d_betas, q);
    }
    
    cout << endl << "Test completed!" << endl;
    return 0;
  } catch (sycl::exception const &e) {
    cerr << "SYCL Exception: " << e.what() << endl;
    return 1;
  } catch (exception const &e) {
    cerr << "Exception: " << e.what() << endl;
    return 1;
  }
}
