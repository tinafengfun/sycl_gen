#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
using namespace std;
namespace lczero {
namespace sycldnn_backend {
template <typename T> void winogradFilterTransform(T* transformed_filter, const T* filter, int C, int K, sycl::queue& q) {
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(K, C) * sycl::range<2>(1, 1), sycl::range<2>(1, 1)),
      [=](sycl::nd_item<2> item) {
        int k = item.get_global_id(0), c = item.get_global_id(1);
        if (k >= K || c >= C) return;
        // Simplified 3x3 filter transform
        T src[3][3];
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
            src[i][j] = filter[((k * C + c) * 3 + i) * 3 + j];
        // Transform and write to output (simplified)
        for (int h = 0; h < 6; h++)
          for (int w = 0; w < 6; w++)
            transformed_filter[((k * C + c) * 6 + h) * 6 + w] = src[h/2][w/2] * ((h%2==0 && w%2==0) ? 1.0f : 0.5f);
      });
  });
  q.wait();
}
} }
using namespace lczero::sycldnn_backend;
int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    cout << "=== winograd_filter_transform - Round 1 ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    struct Cfg { int C, K; int filterSize, outSize; };
    vector<Cfg> cfgs = {{64, 64, 64*64*3*3, 64*64*6*6}, {128, 128, 128*128*3*3, 128*128*6*6}, {256, 256, 256*256*3*3, 256*256*6*6}};
    cout << setw(8) << "C" << setw(8) << "K" << setw(15) << "Time(ms)" << setw(15) << "GFLOPS" << endl;
    cout << string(46, '-') << endl;
    for (auto& c : cfgs) {
      sycl::half *out=sycl::malloc_device<sycl::half>(c.outSize,q), *filter=sycl::malloc_device<sycl::half>(c.filterSize,q);
      for (int i=0;i<3;i++) winogradFilterTransform(out, filter, c.C, c.K, q);
      vector<double> times;
      for (int i=0;i<10;i++) { auto s=chrono::high_resolution_clock::now(); winogradFilterTransform(out, filter, c.C, c.K, q); auto e=chrono::high_resolution_clock::now(); times.push_back(chrono::duration<double,milli>(e-s).count()); }
      double avg=0; for (double t:times) avg+=t; avg/=times.size();
      double flops=36.0*c.C*c.K; double gflops=flops/(avg*1e-3)/1e9;
      cout << setw(8) << c.C << setw(8) << c.K << setw(15) << fixed << setprecision(3) << avg << setw(15) << setprecision(2) << gflops << endl;
      sycl::free(out,q); sycl::free(filter,q);
    }
    cout << endl << "Test completed!" << endl;
  } catch (exception const &e) { cerr << "Exception: " << e.what() << endl; return 1; }
  return 0;
}
