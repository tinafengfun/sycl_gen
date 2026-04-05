// Phase 3.3: se_layer_nhwc - SLM Optimized
// Cache FC weights (w1, b1, w2, b2) in SLM
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
template <typename T> void seLayerNHWC_Original(T* output, const T* input, const T* skip,
    const T* w1, const T* b1, const T* w2, const T* b2, int N, int C, int se_K, sycl::queue& q) {
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(N, 8, 8) * sycl::range<3>(1, 4, 64), sycl::range<3>(1, 4, 64)),
      [=](sycl::nd_item<3> item) {
        int n = item.get_group(0), h = item.get_group(1) * 4 + item.get_local_id(1), w = item.get_group(2) * 64 + item.get_local_id(2);
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

// SLM optimized - cache FC weights
template <typename T> class se_layer_slm_kernel;
template <typename T> void seLayerNHWC_SLM(T* output, const T* input, const T* skip,
    const T* w1, const T* b1, const T* w2, const T* b2, int N, int C, int se_K, sycl::queue& q) {
  q.submit([&](sycl::handler& cgh) {
    // SLM for FC weights
    sycl::local_accessor<float, 1> slm_w1(se_K, cgh);
    sycl::local_accessor<float, 1> slm_b1(se_K, cgh);
    sycl::local_accessor<float, 1> slm_w2(se_K * 2, cgh);
    sycl::local_accessor<float, 1> slm_b2(C * 2, cgh);
    
    cgh.parallel_for<se_layer_slm_kernel<T>>(
      sycl::nd_range<3>(sycl::range<3>(N, 8, 8) * sycl::range<3>(1, 4, 64), sycl::range<3>(1, 4, 64)),
      [=](sycl::nd_item<3> item) {
        int tid = item.get_local_id(2);
        int block_size = item.get_local_range(2);
        
        // Cooperative loading of weights into SLM
        for (int i = tid; i < se_K; i += block_size) {
          slm_w1[i] = (float)w1[i];
          slm_b1[i] = (float)b1[i];
        }
        for (int i = tid; i < se_K * 2; i += block_size) {
          slm_w2[i] = (float)w2[i];
        }
        for (int i = tid; i < C * 2; i += block_size) {
          slm_b2[i] = (float)b2[i];
        }
        item.barrier(sycl::access::fence_space::local_space);
        
        int n = item.get_group(0), h = item.get_group(1) * 4 + item.get_local_id(1), 
            w = item.get_group(2) * 64 + item.get_local_id(2);
        if (h >= 8 || w >= 8) return;
        int idx = ((n * 8 + h) * 8 + w) * C;
        float S = 0;
        for (int c = 0; c < C; c++) S += (float)input[idx + c];
        float avg = S / C;
        float gamma = 0, beta = 0;
        for (int k = 0; k < se_K; k++) {
          float val = avg * slm_w1[k] + slm_b1[k];
          val = (val > 0) ? val : 0;
          gamma += val * slm_w2[k];
          beta += val * slm_w2[k + se_K];
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
}} // namespaces
using namespace lczero::sycldnn_backend;
int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    cout << "=== se_layer_nhwc - Phase 3.3 SLM Optimization ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    struct Config { int N, C, se_K, elements; };
    vector<Config> cfgs = {{16, 64, 16, 16*64*64}, {64, 128, 32, 64*128*64}, {256, 256, 64, 256*256*64}};
    
    cout << "--- Original ---" << endl;
    cout << setw(8) << "N" << setw(8) << "C" << setw(10) << "se_K" << setw(15) << "Elements" << setw(15) << "Time(ms)" << setw(15) << "GFLOPS" << endl;
    cout << string(71, '-') << endl;
    for (auto& c : cfgs) {
      sycl::half *out=sycl::malloc_device<sycl::half>(c.elements,q), *in=sycl::malloc_device<sycl::half>(c.elements,q),
        *w1=sycl::malloc_device<sycl::half>(c.se_K,q), *b1=sycl::malloc_device<sycl::half>(c.se_K,q),
        *w2=sycl::malloc_device<sycl::half>(c.se_K*c.C*2,q), *b2=sycl::malloc_device<sycl::half>(c.C*2,q);
      for (int i=0;i<5;i++) seLayerNHWC_Original(out, in, (const sycl::half*)nullptr, w1, b1, w2, b2, c.N, c.C, c.se_K, q);
      vector<double> times;
      for (int i=0;i<10;i++) { auto s=chrono::high_resolution_clock::now(); seLayerNHWC_Original(out, in, (const sycl::half*)nullptr, w1, b1, w2, b2, c.N, c.C, c.se_K, q); auto e=chrono::high_resolution_clock::now(); times.push_back(chrono::duration<double,milli>(e-s).count()); }
      double avg=0; for (double t:times) avg+=t; avg/=times.size();
      double flops=(c.C*2+c.se_K*2)*c.elements/c.C; double gflops=flops/(avg*1e-3)/1e9;
      cout << setw(8) << c.N << setw(8) << c.C << setw(10) << c.se_K << setw(15) << c.elements << setw(15) << fixed << setprecision(3) << avg << setw(15) << setprecision(2) << gflops << endl;
      sycl::free(out,q); sycl::free(in,q); sycl::free(w1,q); sycl::free(b1,q); sycl::free(w2,q); sycl::free(b2,q);
    }
    
    cout << endl << "--- SLM Optimized ---" << endl;
    cout << setw(8) << "N" << setw(8) << "C" << setw(10) << "se_K" << setw(15) << "Elements" << setw(15) << "Time(ms)" << setw(15) << "GFLOPS" << endl;
    cout << string(71, '-') << endl;
    for (auto& c : cfgs) {
      sycl::half *out=sycl::malloc_device<sycl::half>(c.elements,q), *in=sycl::malloc_device<sycl::half>(c.elements,q),
        *w1=sycl::malloc_device<sycl::half>(c.se_K,q), *b1=sycl::malloc_device<sycl::half>(c.se_K,q),
        *w2=sycl::malloc_device<sycl::half>(c.se_K*c.C*2,q), *b2=sycl::malloc_device<sycl::half>(c.C*2,q);
      for (int i=0;i<5;i++) seLayerNHWC_SLM(out, in, (const sycl::half*)nullptr, w1, b1, w2, b2, c.N, c.C, c.se_K, q);
      vector<double> times;
      for (int i=0;i<10;i++) { auto s=chrono::high_resolution_clock::now(); seLayerNHWC_SLM(out, in, (const sycl::half*)nullptr, w1, b1, w2, b2, c.N, c.C, c.se_K, q); auto e=chrono::high_resolution_clock::now(); times.push_back(chrono::duration<double,milli>(e-s).count()); }
      double avg=0; for (double t:times) avg+=t; avg/=times.size();
      double flops=(c.C*2+c.se_K*2)*c.elements/c.C; double gflops=flops/(avg*1e-3)/1e9;
      cout << setw(8) << c.N << setw(8) << c.C << setw(10) << c.se_K << setw(15) << c.elements << setw(15) << fixed << setprecision(3) << avg << setw(15) << setprecision(2) << gflops << endl;
      sycl::free(out,q); sycl::free(in,q); sycl::free(w1,q); sycl::free(b1,q); sycl::free(w2,q); sycl::free(b2,q);
    }
    cout << endl << "Phase 3.3 se_layer_nhwc optimization complete!" << endl;
  } catch (exception const &e) { cerr << "Exception: " << e.what() << endl; return 1; }
  return 0;
}
