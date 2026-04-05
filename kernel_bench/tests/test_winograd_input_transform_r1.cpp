#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
using namespace std;
namespace lczero {
namespace sycldnn_backend {
template <typename T> void winogradInputTransform(T* transformed_input, const T* input, int N, int C, int H, int W, sycl::queue& q) {
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(N, H/4, W/4) * sycl::range<3>(C, 1, 1), sycl::range<3>(C, 1, 1)),
      [=](sycl::nd_item<3> item) {
        int n = item.get_group(0), h = item.get_group(1) * 4, w = item.get_group(2) * 4, c = item.get_local_id(0);
        if (n >= N || h >= H || w >= W || c >= C) return;
        // Simplified input transform (4x4 tiles)
        T tile[4][4];
        for (int i = 0; i < 4; i++)
          for (int j = 0; j < 4; j++)
            tile[i][j] = input[((n * C + c) * H + h + i) * W + w + j];
        // Write transformed output (simplified)
        for (int i = 0; i < 6; i++)
          for (int j = 0; j < 6; j++)
            transformed_input[((((n * C + c) * (H/4) + item.get_group(1)) * (W/4) + item.get_group(2)) * 6 + i) * 6 + j] = tile[i/2][j/2] * 0.5f;
      });
  });
  q.wait();
}
} }
using namespace lczero::sycldnn_backend;
int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    cout << "=== winograd_input_transform - Round 1 ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    struct Cfg { int N, C, H, W; int inSize, outSize; };
    vector<Cfg> cfgs = {{16, 64, 64, 64, 16*64*64*64, 16*64*16*16*36}, {64, 128, 32, 32, 64*128*32*32, 64*128*8*8*36}};
    cout << setw(8) << "N" << setw(8) << "C" << setw(8) << "H" << setw(8) << "W" << setw(15) << "Time(ms)" << setw(15) << "GFLOPS" << endl;
    cout << string(62, '-') << endl;
    for (auto& c : cfgs) {
      sycl::half *out=sycl::malloc_device<sycl::half>(c.outSize,q), *in=sycl::malloc_device<sycl::half>(c.inSize,q);
      for (int i=0;i<3;i++) winogradInputTransform(out, in, c.N, c.C, c.H, c.W, q);
      vector<double> times;
      for (int i=0;i<10;i++) { auto s=chrono::high_resolution_clock::now(); winogradInputTransform(out, in, c.N, c.C, c.H, c.W, q); auto e=chrono::high_resolution_clock::now(); times.push_back(chrono::duration<double,milli>(e-s).count()); }
      double avg=0; for (double t:times) avg+=t; avg/=times.size();
      double flops=144.0*c.N*c.C*(c.H/4)*(c.W/4); double gflops=flops/(avg*1e-3)/1e9;
      cout << setw(8) << c.N << setw(8) << c.C << setw(8) << c.H << setw(8) << c.W << setw(15) << fixed << setprecision(3) << avg << setw(15) << setprecision(2) << gflops << endl;
      sycl::free(out,q); sycl::free(in,q);
    }
    cout << endl << "Test completed!" << endl;
  } catch (exception const &e) { cerr << "Exception: " << e.what() << endl; return 1; }
  return 0;
}
