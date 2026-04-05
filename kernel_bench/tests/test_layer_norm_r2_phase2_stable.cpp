// Phase 2: layer_norm - Stable Vectorized Version
// Conservative optimization with proper bounds checking

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

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

inline float mishActivate(float val) {
  auto e = sycl::exp(val);
  auto n = e * e + 2.0f * e;
  auto d = val / (n + 2.0f);
  return (val <= -0.6f) ? n * d : val - 2.0f * d;
}

inline float applyActivation(float val, ActivationFunction act) {
  switch (act) {
    case ACTIVATION_RELU:
      return val < 0 ? 0 : val;
    case ACTIVATION_RELU_2:
      val = val < 0 ? 0 : val;
      return val * val;
    case ACTIVATION_TANH:
      return sycl::tanh(val);
    case ACTIVATION_SIGMOID:
      return 1.0f / (1.0f + sycl::exp(-val));
    case ACTIVATION_SELU: {
      constexpr float alpha = 1.67326324f, scale = 1.05070098f;
      return val > 0 ? scale * val : scale * alpha * (sycl::exp(val) - 1.0f);
    }
    case ACTIVATION_MISH:
      return mishActivate(val);
    case ACTIVATION_SWISH:
      return val / (1.0f + sycl::exp(-val));
    default:
      return val;
  }
}

// Stable warp reduction
inline float warpReduce(float x, sycl::sub_group sg) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += sycl::permute_group_by_xor(sg, x, offset);
  }
  return x;
}

template <typename T>
class layer_norm_phase2_kernel;

template <typename T>
void LayerNormPhase2(int N, int C, T* output, const T* input, const T* bias,
                     const T* skip, const T* gammas, const T* betas,
                     float epsilon, float alpha, ActivationFunction act,
                     sycl::queue& stream) {
  // Require C to be multiple of 32 for vectorization
  if (C % 32 != 0) {
    throw runtime_error("Channel count must be multiple of 32");
  }
  
  // Grid configuration: each block handles one batch
  const int subgroup_size = 32;
  const int threads_per_block = 256;  // 8 warps
  const int warps_per_block = threads_per_block / subgroup_size;
  const int channels_per_warp = 32;   // Each warp processes 32 channels
  
  int total_warps = DivUp(C, channels_per_warp);
  int grid_x = N * total_warps;  // Each block handles one warp's work
  
  // Use 1D grid for simplicity
  sycl::range<1> global_size(grid_x * threads_per_block);
  sycl::range<1> local_size(threads_per_block);
  
  auto event = stream.submit([&](sycl::handler& cgh) {
    // SLM for reduction
    sycl::local_accessor<float, 1> slm_sum(warps_per_block, cgh);
    sycl::local_accessor<float, 1> slm_sq_sum(warps_per_block, cgh);
    
    cgh.parallel_for<layer_norm_phase2_kernel<T>>(
      sycl::nd_range<1>(global_size, local_size),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const int tid = item.get_local_id(0);
        const int wid = tid / subgroup_size;
        const int lid = tid % subgroup_size;
        const int bid = item.get_group(0);
        
        // Decode batch and warp
        const int n = bid / total_warps;
        const int warp_idx = bid % total_warps;
        const int c_start = warp_idx * channels_per_warp + lid * 2;
        
        if (n >= N || c_start >= C) return;
        
        // Each thread loads 2 half elements (32 bits total)
        const int tensor_idx = n * C + c_start;
        const bool valid = c_start < C;
        
        float vals[2] = {0.0f, 0.0f};
        float thread_sum = 0.0f;
        float thread_sq = 0.0f;
        
        if (valid && c_start + 1 < C) {
          // Load 2 half values as uint
          uint16_t input_pair = *reinterpret_cast<const uint16_t*>(&input[tensor_idx]);
          uint16_t bias_pair = *reinterpret_cast<const uint16_t*>(&bias[c_start]);
          
          sycl::half* input_half = reinterpret_cast<sycl::half*>(&input_pair);
          sycl::half* bias_half = reinterpret_cast<sycl::half*>(&bias_pair);
          
          #pragma unroll
          for (int i = 0; i < 2; i++) {
            float val = (float)input_half[i] + (float)bias_half[i];
            val = applyActivation(val, act) * alpha;
            vals[i] = val;
            thread_sum += val;
            thread_sq += val * val;
          }
        } else if (valid) {
          // Single element (last one)
          float val = (float)input[tensor_idx] + (float)bias[c_start];
          val = applyActivation(val, act) * alpha;
          vals[0] = val;
          thread_sum += val;
          thread_sq += val * val;
        }
        
        // Intra-subgroup reduction
        auto sg = item.get_sub_group();
        float warp_sum = warpReduce(thread_sum, sg);
        float warp_sq = warpReduce(thread_sq, sg);
        
        // Store to SLM
        if (lid == 0) {
          slm_sum[wid] = warp_sum;
          slm_sq_sum[wid] = warp_sq;
        }
        item.barrier(sycl::access::fence_space::local_space);
        
        // Reduce across warps (lane 0 only)
        if (tid == 0) {
          float total = 0.0f;
          float total_sq = 0.0f;
          for (int i = 0; i < warps_per_block; i++) {
            total += slm_sum[i];
            total_sq += slm_sq_sum[i];
          }
          slm_sum[0] = total;
          slm_sq_sum[0] = total_sq;
        }
        item.barrier(sycl::access::fence_space::local_space);
        
        float total_sum = slm_sum[0];
        float total_sq_sum = slm_sq_sum[0];
        float mean = total_sum / C;
        float variance = (total_sq_sum / C) - (mean * mean);
        float inv_std = sycl::rsqrt(variance + epsilon);
        
        // Normalize and write back
        if (valid && c_start + 1 < C) {
          uint16_t gamma_pair = *reinterpret_cast<const uint16_t*>(&gammas[c_start]);
          uint16_t beta_pair = *reinterpret_cast<const uint16_t*>(&betas[c_start]);
          sycl::half* gamma_half = reinterpret_cast<sycl::half*>(&gamma_pair);
          sycl::half* beta_half = reinterpret_cast<sycl::half*>(&beta_pair);
          
          uint16_t out_pair;
          sycl::half* out_half = reinterpret_cast<sycl::half*>(&out_pair);
          
          #pragma unroll
          for (int i = 0; i < 2; i++) {
            float normalized = (vals[i] - mean) * inv_std;
            float scaled = normalized * (float)gamma_half[i] + (float)beta_half[i];
            out_half[i] = (sycl::half)scaled;
          }
          
          *reinterpret_cast<uint16_t*>(&output[tensor_idx]) = out_pair;
        } else if (valid) {
          float normalized = (vals[0] - mean) * inv_std;
          float scaled = normalized * (float)gammas[c_start] + (float)betas[c_start];
          output[tensor_idx] = (sycl::half)scaled;
        }
      }
    );
  });
  
  event.wait_and_throw();
}

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== layer_norm - Phase 2 Optimization (Stable) ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl;
    cout << "Optimization: Vectorized loads (2 elems/thread)" << endl << endl;
    
    struct TestConfig {
      int N, C;
    };
    
    vector<TestConfig> configs = {
      {64, 64},
      {256, 256},
      {1024, 1024},
      {4096, 1024},
    };
    
    cout << setw(15) << "Config (NxC)" 
         << setw(15) << "Total"
         << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS"
         << setw(18) << "GB/s" << endl;
    cout << string(78, '-') << endl;
    
    for (const auto& cfg : configs) {
      const int N = cfg.N;
      const int C = cfg.C;
      const int total = N * C;
      
      // Allocate
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
        LayerNormPhase2(N, C, d_output, d_input, d_bias,
                       (const sycl::half*)nullptr, d_gammas, d_betas,
                       1e-5f, 1.0f, ACTIVATION_RELU, q);
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        
        LayerNormPhase2(N, C, d_output, d_input, d_bias,
                       (const sycl::half*)nullptr, d_gammas, d_betas,
                       1e-5f, 1.0f, ACTIVATION_RELU, q);
        
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(end - start).count());
      }
      
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      double flops = 22.0 * total;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      
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
    
    cout << endl << "Phase 2 layer_norm optimization complete!" << endl;
    return 0;
    
  } catch (sycl::exception const &e) {
    cerr << "SYCL Exception: " << e.what() << endl;
    return 1;
  } catch (exception const &e) {
    cerr << "Exception: " << e.what() << endl;
    return 1;
  }
}
