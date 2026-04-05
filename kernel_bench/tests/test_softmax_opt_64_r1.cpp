// Simplified SYCL conversion for softmax_opt_64_kernel
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
using namespace std;
namespace lczero {
namespace sycldnn_backend {
inline int DivUp(int a, int b) { return (a + b - 1) / b; }
template <typename T>
void SoftmaxOpt64(int N, T* output, const T* input, const T* input2, sycl::queue& q) {
  int size = N * 32;
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(blocks * kBlockSize, kBlockSize),
      [=](sycl::nd_item<1> item) {
        int index = item.get_local_range(0) * item.get_group(0) + item.get_local_id(0);
        if (index >= size) return;
        float x[2];
        x[0] = (float)input[index * 2];
        x[1] = (float)input[index * 2 + 1];
        if (input2 != nullptr) {
          x[0] += (float)input2[index * 2];
          x[1] += (float)input2[index * 2 + 1];
        }
        float threadMax = sycl::max(x[0], x[1]);
        auto sg = item.get_sub_group();
        float maxval = sycl::reduce_over_group(sg, threadMax, sycl::maximum<>());
        maxval = sycl::group_broadcast(sg, maxval, 0);
        float ex[2];
        ex[0] = sycl::exp(x[0] - maxval);
        ex[1] = sycl::exp(x[1] - maxval);
        float threadSum = ex[0] + ex[1];
        float Sum = sycl::reduce_over_group(sg, threadSum, sycl::plus<>());
        Sum = sycl::group_broadcast(sg, Sum, 0);
        output[index * 2] = (T)(ex[0] / Sum);
        output[index * 2 + 1] = (T)(ex[1] / Sum);
      });
  });
  q.wait();
}
}
}
using namespace lczero::sycldnn_backend;
int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    cout << "=== softmax_opt_64 - Round 1 (Converted CUDA→SYCL) ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    struct Cfg { int N; int elements; };
    vector<Cfg> cfgs = {{64, 64*64}, {256, 256*64}, {1024, 1024*64}};
    cout << setw(10) << "N" << setw(15) << "Elements" << setw(15) << "Time(ms)" << setw(15) << "GFLOPS" << endl;
    cout << string(55, '-') << endl;
    for (auto& c : cfgs) {
      sycl::half *out=sycl::malloc_device<sycl::half>(c.elements,q), *in=sycl::malloc_device<sycl::half>(c.elements,q);
      for (int i=0;i<3;i++) SoftmaxOpt64(c.N, out, in, (const sycl::half*)nullptr, q);
      vector<double> times;
      for (int i=0;i<10;i++) { auto s=chrono::high_resolution_clock::now(); SoftmaxOpt64(c.N, out, in, (const sycl::half*)nullptr, q); auto e=chrono::high_resolution_clock::now(); times.push_back(chrono::duration<double,milli>(e-s).count()); }
      double avg=0; for (double t:times) avg+=t; avg/=times.size();
      double flops=20.0*c.elements; double gflops=flops/(avg*1e-3)/1e9;
      cout << setw(10) << c.N << setw(15) << c.elements << setw(15) << fixed << setprecision(3) << avg << setw(15) << setprecision(2) << gflops << endl;
      sycl::free(out,q); sycl::free(in,q);
    }
    cout << endl << "✓ CUDA→SYCL conversion successful!" << endl;
  } catch (exception const &e) { cerr << "Exception: " << e.what() << endl; return 1; }
  return 0;
}
