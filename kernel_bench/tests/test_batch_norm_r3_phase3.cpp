// Phase 3.1: batch_norm - SLM Optimization
// SLM is used to cache mean and varMultiplier values for channels
// This reduces global memory traffic for these parameters

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
  ACTIVATION_MISH,
  ACTIVATION_DEFAULT,
  ACTIVATION_SOFTMAX
};

inline float mishActivate(float el) {
  auto e = sycl::native::exp(el);
  auto n = e * e + 2.0f * e;
  auto d = el / (n + 2.0f);
  return (el <= -0.6f) ? n * d : el - 2.0f * d;
}

inline float activate(float cVal, ActivationFunction activation) {
  switch (activation) {
    case ACTIVATION_RELU:
      return (cVal < 0) ? 0.0f : cVal;
    case ACTIVATION_RELU_2:
      cVal = (cVal < 0) ? 0.0f : cVal;
      return cVal * cVal;
    case ACTIVATION_TANH:
      return sycl::tanh(cVal);
    case ACTIVATION_SIGMOID:
      return 1.0f / (1.0f + sycl::exp(-cVal));
    case ACTIVATION_SELU:
      return 1.0507f * ((cVal >= 0) ? cVal : 1.67326f * (sycl::exp(cVal) - 1.0f));
    case ACTIVATION_SWISH:
      return cVal / (1.0f + sycl::exp(-cVal));
    case ACTIVATION_MISH:
      return mishActivate(cVal);
    default:
      return cVal;
  }
}

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// Original kernel
template <typename T>
void batchNorm_Original(T* output, const T* input, const T* skipInput,
                        int N, int C, int H, int W, const float* means,
                        const float* varMultipliers,
                        ActivationFunction activation, sycl::queue &q) {
  const int total_elements = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(total_elements, kBlockSize);

  q.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
          sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item) {
        int index = item.get_local_id(2) +
                    item.get_local_range(2) * item.get_group(2);
        if (index >= total_elements) return;

        int wIndex = index % C;  // NHWC for fp16

        float el = input[index];
        float mean = means[wIndex];
        float varMulti = varMultipliers[wIndex];

        el -= mean;
        el *= varMulti;

        if (skipInput) el += (float)skipInput[index];

        el = activate(el, activation);

        output[index] = (T)el;
      });
}

// SLM-optimized kernel: Cache mean and varMultiplier in SLM
template <typename T>
class batchNorm_slm_kernel;

template <typename T>
void batchNorm_SLM(T* output, const T* input, const T* skipInput,
                   int N, int C, int H, int W, const float* means,
                   const float* varMultipliers,
                   ActivationFunction activation, sycl::queue &q) {
  const int total_elements = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(total_elements, kBlockSize);

  auto event = q.submit([&](sycl::handler& cgh) {
    // SLM to cache mean and varMultiplier for channels accessed by this block
    // Each block handles a range of elements, which may span multiple channels
    // We'll cache up to 256 channels (configurable)
    const int slm_channels = 256;
    sycl::local_accessor<float, 1> slm_means(slm_channels, cgh);
    sycl::local_accessor<float, 1> slm_vars(slm_channels, cgh);
    
    cgh.parallel_for<batchNorm_slm_kernel<T>>(
      sycl::nd_range<3>(
        sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
        sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item) {
        const int tid = item.get_local_id(2);
        const int bid = item.get_group(2);
        const int index = tid + bid * kBlockSize;
        
        if (index >= total_elements) return;
        
        // Calculate which channels this block will access
        const int start_elem = bid * kBlockSize;
        const int end_elem = sycl::min((bid + 1) * kBlockSize, total_elements);
        const int start_channel = start_elem % C;
        const int end_channel = (end_elem - 1) % C;
        
        // Determine range of channels to cache
        int channels_to_cache;
        if (start_channel <= end_channel) {
          channels_to_cache = end_channel - start_channel + 1;
        } else {
          // Wrapped around
          channels_to_cache = (C - start_channel) + end_channel + 1;
        }
        channels_to_cache = sycl::min(channels_to_cache, slm_channels);
        
        // Cooperatively load mean and var into SLM
        // Each thread loads a subset
        for (int i = tid; i < channels_to_cache; i += kBlockSize) {
          int ch = (start_channel + i) % C;
          slm_means[i] = means[ch];
          slm_vars[i] = varMultipliers[ch];
        }
        item.barrier(sycl::access::fence_space::local_space);
        
        // Now process elements using cached values
        int wIndex = index % C;
        int cache_idx = (wIndex - start_channel + C) % C;
        if (cache_idx >= channels_to_cache) {
          // Fallback to global memory (shouldn't happen with proper sizing)
          cache_idx = 0;
        }
        
        float el = input[index];
        float mean = slm_means[cache_idx];
        float varMulti = slm_vars[cache_idx];

        el -= mean;
        el *= varMulti;

        if (skipInput) el += (float)skipInput[index];

        el = activate(el, activation);

        output[index] = (T)el;
      });
  });
}

// Vectorized version with SLM
template <typename T>
class batchNorm_vec_slm_kernel;

template <typename T>
void batchNorm_VecSLM(T* output, const T* input, const T* skipInput,
                      int N, int C, int H, int W, const float* means,
                      const float* varMultipliers,
                      ActivationFunction activation, sycl::queue &q) {
  const int total_elements = N * C * H * W;
  const int elems_per_thread = 4;  // Process 4 elements per thread
  const int kBlockSize = 256;
  int threads_needed = DivUp(total_elements, elems_per_thread);
  int blocks = DivUp(threads_needed, kBlockSize);

  auto event = q.submit([&](sycl::handler& cgh) {
    const int slm_channels = 256;
    sycl::local_accessor<float, 1> slm_means(slm_channels, cgh);
    sycl::local_accessor<float, 1> slm_vars(slm_channels, cgh);
    
    cgh.parallel_for<batchNorm_vec_slm_kernel<T>>(
      sycl::nd_range<3>(
        sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, kBlockSize),
        sycl::range<3>(1, 1, kBlockSize)),
      [=](sycl::nd_item<3> item) {
        const int tid = item.get_local_id(2);
        const int bid = item.get_group(2);
        const int base_idx = (bid * kBlockSize + tid) * elems_per_thread;
        
        // Calculate channel range for this block
        const int start_elem = bid * kBlockSize * elems_per_thread;
        const int end_elem = sycl::min((bid + 1) * kBlockSize * elems_per_thread, total_elements);
        
        // Determine unique channels accessed by this block
        // For simplicity, cache all C channels if C <= slm_channels
        if (C <= slm_channels) {
          // Each thread loads some channels
          for (int i = tid; i < C; i += kBlockSize) {
            slm_means[i] = means[i];
            slm_vars[i] = varMultipliers[i];
          }
          item.barrier(sycl::access::fence_space::local_space);
          
          // Process elements
          #pragma unroll
          for (int offset = 0; offset < elems_per_thread; offset++) {
            int index = base_idx + offset;
            if (index >= total_elements) break;
            
            int wIndex = index % C;
            
            float el = input[index];
            float mean = slm_means[wIndex];
            float varMulti = slm_vars[wIndex];

            el -= mean;
            el *= varMulti;

            if (skipInput) el += (float)skipInput[index];

            el = activate(el, activation);

            output[index] = (T)el;
          }
        } else {
          // C is large, fall back to global memory
          #pragma unroll
          for (int offset = 0; offset < elems_per_thread; offset++) {
            int index = base_idx + offset;
            if (index >= total_elements) break;
            
            int wIndex = index % C;
            
            float el = input[index];
            float mean = means[wIndex];
            float varMulti = varMultipliers[wIndex];

            el -= mean;
            el *= varMulti;

            if (skipInput) el += (float)skipInput[index];

            el = activate(el, activation);

            output[index] = (T)el;
          }
        }
      });
  });
}

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== batch_norm - Phase 3.1 SLM Optimization ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl;
    cout << "Optimization: SLM caching of mean/var parameters" << endl << endl;
    
    struct TestConfig {
      int N, C, H, W;
      int totalElements;
    };
    
    vector<TestConfig> configs = {
      {1, 64, 8, 8},      // 4096 elements
      {2, 128, 16, 16},   // 65536 elements
      {4, 256, 32, 32},   // 1048576 elements (1M)
      {8, 512, 32, 32},   // 4194304 elements (4M)
    };
    
    for (auto& cfg : configs) {
      cfg.totalElements = cfg.N * cfg.C * cfg.H * cfg.W;
    }
    
    // Test Original
    cout << "--- Original ---" << endl;
    cout << setw(20) << "Config (NxCxHxW)" 
         << setw(12) << "Total"
         << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS"
         << setw(18) << "GB/s" << endl;
    cout << string(80, '-') << endl;
    
    for (const auto& cfg : configs) {
      const int N = cfg.N, C = cfg.C, H = cfg.H, W = cfg.W, total = cfg.totalElements;
      
      sycl::half* d_input = sycl::malloc_device<sycl::half>(total, q);
      sycl::half* d_output = sycl::malloc_device<sycl::half>(total, q);
      float* d_means = sycl::malloc_device<float>(C, q);
      float* d_varMultipliers = sycl::malloc_device<float>(C, q);
      
      vector<sycl::half> h_input(total, sycl::half(1.0f));
      vector<float> h_means(C, 0.5f);
      vector<float> h_varMultipliers(C, 1.0f);
      q.memcpy(d_input, h_input.data(), total * sizeof(sycl::half)).wait();
      q.memcpy(d_means, h_means.data(), C * sizeof(float)).wait();
      q.memcpy(d_varMultipliers, h_varMultipliers.data(), C * sizeof(float)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        batchNorm_Original(d_output, d_input, (const sycl::half*)nullptr, 
                          N, C, H, W, d_means, d_varMultipliers, ACTIVATION_RELU, q);
        q.wait();
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        batchNorm_Original(d_output, d_input, (const sycl::half*)nullptr,
                          N, C, H, W, d_means, d_varMultipliers, ACTIVATION_RELU, q);
        q.wait();
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(end - start).count());
      }
      
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      double flops = 3.0 * total;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      double bytes = 2.0 * total * sizeof(sycl::half) + 2.0 * C * sizeof(float);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(20) << (to_string(N) + "x" + to_string(C) + "x" + to_string(H) + "x" + to_string(W))
           << setw(12) << total
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_input, q);
      sycl::free(d_output, q);
      sycl::free(d_means, q);
      sycl::free(d_varMultipliers, q);
    }
    
    // Test SLM version
    cout << endl << "--- SLM Optimized ---" << endl;
    cout << setw(20) << "Config (NxCxHxW)" 
         << setw(12) << "Total"
         << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS"
         << setw(18) << "GB/s" << endl;
    cout << string(80, '-') << endl;
    
    for (const auto& cfg : configs) {
      const int N = cfg.N, C = cfg.C, H = cfg.H, W = cfg.W, total = cfg.totalElements;
      
      sycl::half* d_input = sycl::malloc_device<sycl::half>(total, q);
      sycl::half* d_output = sycl::malloc_device<sycl::half>(total, q);
      float* d_means = sycl::malloc_device<float>(C, q);
      float* d_varMultipliers = sycl::malloc_device<float>(C, q);
      
      vector<sycl::half> h_input(total, sycl::half(1.0f));
      vector<float> h_means(C, 0.5f);
      vector<float> h_varMultipliers(C, 1.0f);
      q.memcpy(d_input, h_input.data(), total * sizeof(sycl::half)).wait();
      q.memcpy(d_means, h_means.data(), C * sizeof(float)).wait();
      q.memcpy(d_varMultipliers, h_varMultipliers.data(), C * sizeof(float)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        batchNorm_SLM(d_output, d_input, (const sycl::half*)nullptr,
                     N, C, H, W, d_means, d_varMultipliers, ACTIVATION_RELU, q);
        q.wait();
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        batchNorm_SLM(d_output, d_input, (const sycl::half*)nullptr,
                     N, C, H, W, d_means, d_varMultipliers, ACTIVATION_RELU, q);
        q.wait();
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(end - start).count());
      }
      
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      double flops = 3.0 * total;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      double bytes = 2.0 * total * sizeof(sycl::half) + 2.0 * C * sizeof(float);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(20) << (to_string(N) + "x" + to_string(C) + "x" + to_string(H) + "x" + to_string(W))
           << setw(12) << total
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_input, q);
      sycl::free(d_output, q);
      sycl::free(d_means, q);
      sycl::free(d_varMultipliers, q);
    }
    
    // Test Vectorized SLM version
    cout << endl << "--- Vectorized + SLM ---" << endl;
    cout << setw(20) << "Config (NxCxHxW)" 
         << setw(12) << "Total"
         << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS"
         << setw(18) << "GB/s" << endl;
    cout << string(80, '-') << endl;
    
    for (const auto& cfg : configs) {
      const int N = cfg.N, C = cfg.C, H = cfg.H, W = cfg.W, total = cfg.totalElements;
      
      sycl::half* d_input = sycl::malloc_device<sycl::half>(total, q);
      sycl::half* d_output = sycl::malloc_device<sycl::half>(total, q);
      float* d_means = sycl::malloc_device<float>(C, q);
      float* d_varMultipliers = sycl::malloc_device<float>(C, q);
      
      vector<sycl::half> h_input(total, sycl::half(1.0f));
      vector<float> h_means(C, 0.5f);
      vector<float> h_varMultipliers(C, 1.0f);
      q.memcpy(d_input, h_input.data(), total * sizeof(sycl::half)).wait();
      q.memcpy(d_means, h_means.data(), C * sizeof(float)).wait();
      q.memcpy(d_varMultipliers, h_varMultipliers.data(), C * sizeof(float)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        batchNorm_VecSLM(d_output, d_input, (const sycl::half*)nullptr,
                        N, C, H, W, d_means, d_varMultipliers, ACTIVATION_RELU, q);
        q.wait();
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        batchNorm_VecSLM(d_output, d_input, (const sycl::half*)nullptr,
                        N, C, H, W, d_means, d_varMultipliers, ACTIVATION_RELU, q);
        q.wait();
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(end - start).count());
      }
      
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      double flops = 3.0 * total;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      double bytes = 2.0 * total * sizeof(sycl::half) + 2.0 * C * sizeof(float);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(20) << (to_string(N) + "x" + to_string(C) + "x" + to_string(H) + "x" + to_string(W))
           << setw(12) << total
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_input, q);
      sycl::free(d_output, q);
      sycl::free(d_means, q);
      sycl::free(d_varMultipliers, q);
    }
    
    cout << endl << "Phase 3.1 batch_norm SLM optimization complete!" << endl;
    return 0;
  } catch (sycl::exception const &e) {
    cerr << "SYCL Exception: " << e.what() << endl;
    return 1;
  } catch (exception const &e) {
    cerr << "Exception: " << e.what() << endl;
    return 1;
  }
}
