#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
using namespace std;
namespace lczero {
namespace sycldnn_backend {
template <typename T> void ComputePromotionLogits(int N, int C, T* output, const T* keys,
    const T* ppo, const T* policy_attn_logits, sycl::queue& stream) {
  stream.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 2> promotion_offsets(sycl::range<2>(4, 8), cgh);
    cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(N * 24, 8), sycl::range<2>(24, 8)),
      [=](sycl::nd_item<2> item) {
        int n = item.get_group(0);
        int y = item.get_local_id(1);
        int x = item.get_local_id(0);
        int threadInGroup = y * 24 + x;
        const T* keys_start = keys + n * 64 * C + C * 56;
        if (threadInGroup < 32) {
          int xi = threadInGroup % 4;
          int yi = threadInGroup / 4;
          float S = 0;
          for (int i = 0; i < C; i++) {
            float a = (float)keys_start[yi * C + i];
            float b = (float)ppo[xi * C + i];
            S += a * b;
          }
          promotion_offsets[xi][yi] = S;
        }
        item.barrier(sycl::access::fence_space::local_space);
        if (threadInGroup < 32) {
          int xi = threadInGroup % 4;
          int yi = threadInGroup / 4;
          if (xi < 3) promotion_offsets[xi][yi] += promotion_offsets[3][yi];
        }
        item.barrier(sycl::access::fence_space::local_space);
        int output_stride = 64 * 64 + 8 * 24;
        int w = x / 3;
        int c = x % 3;
        float n_promo_logit = (float)policy_attn_logits[n * output_stride + (48 + y) * 64 + (56 + w)];
        float promo_offset = promotion_offsets[c][w];
        output[n * output_stride + threadInGroup] = (T)(n_promo_logit + promo_offset);
      });
  });
  stream.wait();
}
} }
using namespace lczero::sycldnn_backend;
int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    cout << "=== promotion_logits - Round 1 ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    struct Cfg { int N, C, output_stride; };
    vector<Cfg> cfgs = {{64, 64, 64*64+8*24}, {256, 128, 256*64+8*24}, {1024, 256, 1024*64+8*24}};
    cout << setw(8) << "N" << setw(8) << "C" << setw(15) << "Time(ms)" << setw(15) << "GFLOPS" << endl;
    cout << string(46, '-') << endl;
    for (auto& c : cfgs) {
      int keysSize = c.N * 64 * c.C, ppoSize = 4 * c.C, logitsSize = c.N * (64*64+8*24);
      sycl::half *out=sycl::malloc_device<sycl::half>(logitsSize,q), *keys=sycl::malloc_device<sycl::half>(keysSize,q),
        *ppo=sycl::malloc_device<sycl::half>(ppoSize,q), *logits=sycl::malloc_device<sycl::half>(logitsSize,q);
      for (int i=0;i<3;i++) ComputePromotionLogits(c.N, c.C, out, keys, ppo, logits, q);
      vector<double> times;
      for (int i=0;i<10;i++) { auto s=chrono::high_resolution_clock::now(); ComputePromotionLogits(c.N, c.C, out, keys, ppo, logits, q); auto e=chrono::high_resolution_clock::now(); times.push_back(chrono::duration<double,milli>(e-s).count()); }
      double avg=0; for (double t:times) avg+=t; avg/=times.size();
      double flops=2.0*c.N*32*4*c.C; double gflops=flops/(avg*1e-3)/1e9;
      cout << setw(8) << c.N << setw(8) << c.C << setw(15) << fixed << setprecision(3) << avg << setw(15) << setprecision(2) << gflops << endl;
      sycl::free(out,q); sycl::free(keys,q); sycl::free(ppo,q); sycl::free(logits,q);
    }
    cout << endl << "Test completed!" << endl;
  } catch (exception const &e) { cerr << "Exception: " << e.what() << endl; return 1; }
  return 0;
}
