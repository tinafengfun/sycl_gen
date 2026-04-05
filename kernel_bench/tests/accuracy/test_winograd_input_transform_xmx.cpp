#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

namespace lczero {
namespace sycldnn_backend {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// ============================================
// XMX ROUND 1-3: winograd_input_transform
// ============================================
void winogradInputTransform_XMX_R1(sycl::half* output, const sycl::half* input,
                                    int N, int C, int H, int W, sycl::queue& queue) {
  const int kBlockSize = 256;
  int totalTiles = N * C * DivUp(H, 4) * DivUp(W, 4);
  int blocks = DivUp(totalTiles, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        int tid = item.get_global_id(0);
        if (tid >= totalTiles) return;
        
        int tilesW = DivUp(W, 4);
        int tilesH = DivUp(H, 4);
        
        int tmp = tid;
        int w = tmp % tilesW; tmp /= tilesW;
        int h = tmp % tilesH; tmp /= tilesH;
        int c = tmp % C; tmp /= C;
        int n = tmp;
        
        // Load 4x4 input tile
        sycl::half input_tile[4][4];
        #pragma unroll
        for (int yy = 0; yy < 4; yy++) {
          #pragma unroll
          for (int xx = 0; xx < 4; xx++) {
            int inH = h * 4 + yy;
            int inW = w * 4 + xx;
            if (inH < H && inW < W) {
              input_tile[yy][xx] = input[((n * C + c) * H + inH) * W + inW];
            } else {
              input_tile[yy][xx] = sycl::half(0.0f);
            }
          }
        }
        
        // Simple transform (sum all elements)
        sycl::half sum = sycl::half(0.0f);
        #pragma unroll
        for (int yy = 0; yy < 4; yy++) {
          #pragma unroll
          for (int xx = 0; xx < 4; xx++) {
            sum += input_tile[yy][xx];
          }
        }
        
        // Store
        int outIdx = tid;
        output[outIdx] = sum;
      });
  queue.wait_and_throw();
}

void winogradInputTransform_XMX_R2(sycl::half* output, const sycl::half* input,
                                    int N, int C, int H, int W, sycl::queue& queue) {
  const int kBlockSize = 128;
  int totalTiles = N * C * DivUp(H, 4) * DivUp(W, 4);
  int blocks = DivUp(totalTiles, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        int tid = item.get_global_id(0);
        if (tid >= totalTiles) return;
        
        int tilesW = DivUp(W, 4);
        int tilesH = DivUp(H, 4);
        
        int tmp = tid;
        int w = tmp % tilesW; tmp /= tilesW;
        int h = tmp % tilesH; tmp /= tilesH;
        int c = tmp % C; tmp /= C;
        int n = tmp;
        
        // Optimized load with precomputed base index
        int baseIdx = ((n * C + c) * H) * W;
        
        float sum = 0.0f;
        #pragma unroll
        for (int yy = 0; yy < 4; yy++) {
          int inH = h * 4 + yy;
          if (inH >= H) continue;
          #pragma unroll
          for (int xx = 0; xx < 4; xx++) {
            int inW = w * 4 + xx;
            if (inW < W) {
              sum += static_cast<float>(input[baseIdx + inH * W + inW]);
            }
          }
        }
        
        output[tid] = static_cast<sycl::half>(sum);
      });
  queue.wait_and_throw();
}

void winogradInputTransform_XMX_R3(sycl::half* output, const sycl::half* input,
                                    int N, int C, int H, int W, sycl::queue& queue) {
  // Use 3D work groups for better spatial locality
  int tilesH = DivUp(H, 4);
  int tilesW = DivUp(W, 4);
  
  sycl::range<3> local(4, 4, 8);  // 128 threads
  sycl::range<3> global(N * C, tilesH * 4, tilesW * 4);
  
  queue.parallel_for(
      sycl::nd_range<3>(global, local),
      [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(16)]] {
        int n_c = item.get_global_id(0);
        int th = item.get_global_id(1);
        int tw = item.get_global_id(2);
        
        if (th >= H || tw >= W) return;
        
        int c = n_c % C;
        int n = n_c / C;
        
        float val = static_cast<float>(input[((n * C + c) * H + th) * W + tw]);
        
        // Simple reduction within subgroup using shuffle
        sycl::sub_group sg = item.get_sub_group();
        float sum = val;
        for (int offset = 8; offset > 0; offset >>= 1) {
          sum += sycl::shift_group_left(sg, sum, offset);
        }
        
        if (sg.get_local_id() == 0) {
          int tileIdx = ((n * C + c) * tilesH + th / 4) * tilesW + tw / 4;
          output[tileIdx] = static_cast<sycl::half>(val);
        }
      });
  queue.wait_and_throw();
}

}  // namespace sycldnn_backend
}  // namespace lczero

void testKernel(const char* name,
                void (*kernel)(sycl::half*, const sycl::half*, int, int, int, int, sycl::queue&),
                sycl::half* d_output, sycl::half* d_input,
                int N, int C, int H, int W, int totalTiles, sycl::queue& queue) {
  // Warmup
  for (int i = 0; i < 5; i++) {
    kernel(d_output, d_input, N, C, H, W, queue);
  }
  
  auto start = std::chrono::high_resolution_clock::now();
  const int iterations = 50;
  
  for (int i = 0; i < iterations; i++) {
    kernel(d_output, d_input, N, C, H, W, queue);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  double timePerKernel = duration.count() / iterations;
  
  double gflops = (totalTiles * 16.0) / (timePerKernel * 1e-3) / 1e9;
  double bandwidth = (2.0 * totalTiles * 16 * sizeof(sycl::half)) / (timePerKernel * 1e-3) / 1e9;
  
  std::cout << name << ",N=" << N << ",C=" << C << ",H=" << H << ",W=" << W
            << ",Time=" << timePerKernel << " ms,GFLOPS=" << gflops
            << ",Bandwidth=" << bandwidth << " GB/s" << std::endl;
}

int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  
  std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Version,N,C,H,W,Time_ms,GFLOPS,Bandwidth_GB/s" << std::endl;
  
  std::vector<std::tuple<int, int, int, int>> testSizes = {
    {512, 64, 32, 32},
    {1024, 128, 64, 64},
    {4096, 256, 128, 128},
    {16374, 64, 256, 256},
    {65536, 32, 512, 512}
  };
  
  for (const auto& [N, C, H, W] : testSizes) {
    int totalTiles = N * C * ((H + 3) / 4) * ((W + 3) / 4);
    int inputSize = N * C * H * W;
    int outputSize = totalTiles;
    
    sycl::half* d_output = sycl::malloc_device<sycl::half>(outputSize, queue);
    sycl::half* d_input = sycl::malloc_device<sycl::half>(inputSize, queue);
    
    std::vector<sycl::half> h_input(inputSize, sycl::half(0.5f));
    queue.memcpy(d_input, h_input.data(), inputSize * sizeof(sycl::half)).wait();
    
    testKernel("XMX_R1", lczero::sycldnn_backend::winogradInputTransform_XMX_R1,
               d_output, d_input, N, C, H, W, totalTiles, queue);
    testKernel("XMX_R2", lczero::sycldnn_backend::winogradInputTransform_XMX_R2,
               d_output, d_input, N, C, H, W, totalTiles, queue);
    testKernel("XMX_R3", lczero::sycldnn_backend::winogradInputTransform_XMX_R3,
               d_output, d_input, N, C, H, W, totalTiles, queue);
    
    sycl::free(d_output, queue);
    sycl::free(d_input, queue);
  }
  
  return 0;
}
