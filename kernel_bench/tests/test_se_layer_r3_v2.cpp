// se_layer_nhwc - Optimized with Cooperative Pooling
// Strategy: Use SLM for cooperative global average pooling
// Each block processes one (n, h, w) location collaboratively

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
using namespace std;

namespace lczero {
namespace sycldnn_backend {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// Original version
template <typename T>
void seLayerNHWC_Original(T* output, const T* input, const T* skip,
    const T* w1, const T* b1, const T* w2, const T* b2, int N, int C, int se_K,
    sycl::queue& q) {
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(N, 8, 8) * sycl::range<3>(1, 4, 64), sycl::range<3>(1, 4, 64)),
      [=](sycl::nd_item<3> item) {
        int n = item.get_group(0), h = item.get_group(1) * 4 + item.get_local_id(1), 
            w = item.get_group(2) * 64 + item.get_local_id(2);
        if (h >= 8 || w >= 8) return;
        int idx = ((n * 8 + h) * 8 + w) * C;
        float S = 0;
        for (int c = 0; c < C; c++) S += (float)input[idx + c];
        float avg = S / C;
        float gamma = 0, beta = 0;
        for (int k = 0; k < se_K; k++) {
          float val = avg * (float)w1[k] + (float)b1[k];
          val = (val > 0) ? val : 0;
          gamma += val * (float)w2[k];
          beta += val * (float)w2[k + se_K];
        }
        gamma = 1.0f / (1.0f + sycl::exp(-gamma));
        for (int c = 0; c < C; c++) {
          float val = (float)input[idx + c] * gamma + beta + (skip ? (float)skip[idx + c] : 0.0f);
          output[idx + c] = (T)((val > 0) ? val : 0);
        }
      });
  });
  q.wait();
}

// Optimized: Cooperative pooling with SLM
template <typename T>
class se_layer_optimized_kernel;

template <typename T>
void seLayerNHWC_Optimized(T* output, const T* input, const T* skip,
    const T* w1, const T* b1, const T* w2, const T* b2, int N, int C, int se_K,
    sycl::queue& q) {
  const int block_size = 256;
  const int tiles_per_block = 4; // Each block processes 4 spatial locations
  
  q.submit([&](sycl::handler& cgh) {
    // SLM for partial sums during pooling
    sycl::local_accessor<float, 1> slm_sum(block_size, cgh);
    
    cgh.parallel_for<se_layer_optimized_kernel<T>>(
      sycl::nd_range<3>(
        sycl::range<3>(N * 8 * 8 / tiles_per_block, 1, 1) * sycl::range<3>(1, 1, block_size),
        sycl::range<3>(1, 1, block_size)
      ),
      [=](sycl::nd_item<3> item) {
        const int tid = item.get_local_id(2);
        const int bid = item.get_group(0);
        
        // Decode spatial position
        const int spatial_idx = bid * tiles_per_block;
        const int n = spatial_idx / 64;
        const int base_hw = spatial_idx % 64;
        
        // Process multiple spatial locations per block
        for (int t = 0; t < tiles_per_block; t++) {
          int hw = base_hw + t;
          if (hw >= 64 || n >= N) continue;
          
          int h = hw / 8;
          int w = hw % 8;
          int idx = ((n * 8 + h) * 8 + w) * C;
          
          // Step 1: Cooperative global average pooling
          float local_sum = 0;
          
          // Each thread sums a subset of channels
          for (int c = tid; c < C; c += block_size) {
            local_sum += (float)input[idx + c];
          }
          slm_sum[tid] = local_sum;
          item.barrier(sycl::access::fence_space::local_space);
          
          // Parallel reduction in SLM
          for (int stride = block_size / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
              slm_sum[tid] += slm_sum[tid + stride];
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          
          float avg = slm_sum[0] / C;
          
          // Step 2: FC layers (computed by all threads in parallel for different k)
          // Each thread computes a subset of the FC outputs
          float gamma_partial = 0, beta_partial = 0;
          
          for (int k = tid; k < se_K; k += block_size) {
            float val = avg * (float)w1[k] + (float)b1[k];
            val = (val > 0) ? val : 0;
            gamma_partial += val * (float)w2[k];
            beta_partial += val * (float)w2[k + se_K];
          }
          
          // Reduce partial results
          // Use warp shuffle for reduction
          auto sg = item.get_sub_group();
          for (int offset = 16; offset > 0; offset >>= 1) {
            gamma_partial += sycl::permute_group_by_xor(sg, gamma_partial, offset);
            beta_partial += sycl::permute_group_by_xor(sg, beta_partial, offset);
          }
          
          // Broadcast results
          float gamma = 0, beta = 0;
          if (tid == 0) {
            gamma = 1.0f / (1.0f + sycl::exp(-gamma_partial));
            beta = beta_partial;
            slm_sum[0] = gamma;
            slm_sum[1] = beta;
          }
          item.barrier(sycl::access::fence_space::local_space);
          
          gamma = slm_sum[0];
          beta = slm_sum[1];
          
          // Step 3: Apply scaling and write output
          for (int c = tid; c < C; c += block_size) {
            float val = (float)input[idx + c] * gamma + beta;
            if (skip) val += (float)skip[idx + c];
            output[idx + c] = (T)((val > 0) ? val : 0);
          }
          
          item.barrier(sycl::access::fence_space::local_space);
        }
      }
    );
  });
  q.wait();
}

} } // namespaces

using namespace lczero::sycldnn_backend;

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    cout << "=== se_layer_nhwc - Optimized with Cooperative Pooling ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    
    struct Config { int N, C, se_K, elements; };
    vector<Config> cfgs = {
      {16, 64, 16, 16*64*64}, 
      {64, 128, 32, 64*128*64}, 
      {256, 256, 64, 256*256*64}
    };
    
    cout << "--- Original ---" << endl;
    cout << setw(8) << "N" << setw(8) << "C" << setw(10) << "se_K" 
         << setw(15) << "Elements" << setw(15) << "Time(ms)" << setw(15) << "GFLOPS" << endl;
    cout << string(76, '-') << endl;
    
    for (auto& c : cfgs) {
      sycl::half *out=sycl::malloc_device<sycl::half>(c.elements,q), 
                 *in=sycl::malloc_device<sycl::half>(c.elements,q),
                 *w1=sycl::malloc_device<sycl::half>(c.se_K,q), 
                 *b1=sycl::malloc_device<sycl::half>(c.se_K,q),
                 *w2=sycl::malloc_device<sycl::half>(c.se_K*c.C*2,q), 
                 *b2=sycl::malloc_device<sycl::half>(c.C*2,q);
      
      for (int i=0; i<5; i++) 
        seLayerNHWC_Original(out, in, (const sycl::half*)nullptr, w1, b1, w2, b2, c.N, c.C, c.se_K, q);
      
      vector<double> times;
      for (int i=0; i<10; i++) { 
        auto s=chrono::high_resolution_clock::now(); 
        seLayerNHWC_Original(out, in, (const sycl::half*)nullptr, w1, b1, w2, b2, c.N, c.C, c.se_K, q); 
        auto e=chrono::high_resolution_clock::now(); 
        times.push_back(chrono::duration<double,milli>(e-s).count()); 
      }
      
      double avg=0; 
      for (double t:times) avg+=t; 
      avg/=times.size();
      double flops=(c.C*2+c.se_K*2)*c.elements/c.C; 
      double gflops=flops/(avg*1e-3)/1e9;
      
      cout << setw(8) << c.N << setw(8) << c.C << setw(10) << c.se_K 
           << setw(15) << c.elements << setw(15) << fixed << setprecision(3) << avg 
           << setw(15) << setprecision(2) << gflops << endl;
      
      sycl::free(out,q); sycl::free(in,q); sycl::free(w1,q); 
      sycl::free(b1,q); sycl::free(w2,q); sycl::free(b2,q);
    }
    
    cout << endl << "--- Cooperative SLM Optimized ---" << endl;
    cout << setw(8) << "N" << setw(8) << "C" << setw(10) << "se_K" 
         << setw(15) << "Elements" << setw(15) << "Time(ms)" << setw(15) << "GFLOPS" << endl;
    cout << string(76, '-') << endl;
    
    for (auto& c : cfgs) {
      sycl::half *out=sycl::malloc_device<sycl::half>(c.elements,q), 
                 *in=sycl::malloc_device<sycl::half>(c.elements,q),
                 *w1=sycl::malloc_device<sycl::half>(c.se_K,q), 
                 *b1=sycl::malloc_device<sycl::half>(c.se_K,q),
                 *w2=sycl::malloc_device<sycl::half>(c.se_K*c.C*2,q), 
                 *b2=sycl::malloc_device<sycl::half>(c.C*2,q);
      
      for (int i=0; i<5; i++) 
        seLayerNHWC_Optimized(out, in, (const sycl::half*)nullptr, w1, b1, w2, b2, c.N, c.C, c.se_K, q);
      
      vector<double> times;
      for (int i=0; i<10; i++) { 
        auto s=chrono::high_resolution_clock::now(); 
        seLayerNHWC_Optimized(out, in, (const sycl::half*)nullptr, w1, b1, w2, b2, c.N, c.C, c.se_K, q); 
        auto e=chrono::high_resolution_clock::now(); 
        times.push_back(chrono::duration<double,milli>(e-s).count()); 
      }
      
      double avg=0; 
      for (double t:times) avg+=t; 
      avg/=times.size();
      double flops=(c.C*2+c.se_K*2)*c.elements/c.C; 
      double gflops=flops/(avg*1e-3)/1e9;
      
      cout << setw(8) << c.N << setw(8) << c.C << setw(10) << c.se_K 
           << setw(15) << c.elements << setw(15) << fixed << setprecision(3) << avg 
           << setw(15) << setprecision(2) << gflops << endl;
      
      sycl::free(out,q); sycl::free(in,q); sycl::free(w1,q); 
      sycl::free(b1,q); sycl::free(w2,q); sycl::free(b2,q);
    }
    
    cout << endl << "Optimization complete!" << endl;
  } catch (exception const &e) { 
    cerr << "Exception: " << e.what() << endl; 
    return 1; 
  }
  return 0;
}
