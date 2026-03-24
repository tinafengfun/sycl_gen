#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

namespace lczero {
namespace sycldnn_backend {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// Simplified winograd output transform with relu input
// V0: Baseline
void winogradOutputReluInput_V0(float* output, const float* input, 
                                int N, int C, int H, int W,
                                sycl::queue& queue) {
  const int kBlockSize = 256;
  int totalTiles = N * C * DivUp(H, 4) * DivUp(W, 4);
  int blocks = DivUp(totalTiles, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid >= totalTiles) return;
        
        // Simplified: just apply relu to input and copy
        float val = input[tid];
        if (val < 0) val = 0;
        output[tid] = val;
      });
  queue.wait_and_throw();
}

// V1: Single-thread per tile with unroll
void winogradOutputReluInput_V1(float* output, const float* input,
                                int N, int C, int H, int W,
                                sycl::queue& queue) {
  // Use single work-item per tile - process full 4x4 output
  int tilesH = DivUp(H, 4);
  int tilesW = DivUp(W, 4);
  int totalTiles = N * C * tilesH * tilesW;
  
  queue.parallel_for(
      sycl::range<1>(totalTiles),
      [=](sycl::item<1> item) {
        int tid = item.get_id(0);
        
        // Decode tile index
        int tmp = tid;
        int w = tmp % tilesW; tmp /= tilesW;
        int h = tmp % tilesH; tmp /= tilesH;
        int c = tmp % C; tmp /= C;
        int n = tmp;
        
        // Process 4x4 output tile with unroll
        #pragma unroll
        for (int yy = 0; yy < 4; yy++) {
          #pragma unroll
          for (int xx = 0; xx < 4; xx++) {
            int outH = h * 4 + yy;
            int outW = w * 4 + xx;
            if (outH < H && outW < W) {
              int inpIdx = ((n * C + c) * H + outH) * W + outW;
              float val = input[inpIdx];
              if (val < 0) val = 0;  // ReLU
              output[inpIdx] = val;
            }
          }
        }
      });
  queue.wait_and_throw();
}

// V2: 3D work-group
void winogradOutputReluInput_V2(float* output, const float* input,
                                int N, int C, int H, int W,
                                sycl::queue& queue) {
  int tilesH = DivUp(H, 4);
  int tilesW = DivUp(W, 4);
  
  sycl::range<3> local(4, 4, 4);
  sycl::range<3> global(N * C, tilesH * 4, tilesW * 4);
  
  queue.parallel_for(
      sycl::nd_range<3>(global, local),
      [=](sycl::nd_item<3> item) {
        int n_c = item.get_global_id(0);
        int th = item.get_global_id(1);
        int tw = item.get_global_id(2);
        
        if (th >= H || tw >= W) return;
        
        int c = n_c % C;
        int n = n_c / C;
        
        int inpIdx = ((n * C + c) * H + th) * W + tw;
        float val = input[inpIdx];
        if (val < 0) val = 0;
        output[inpIdx] = val;
      });
  queue.wait_and_throw();
}

}  // namespace sycldnn_backend
}  // namespace lczero

void testKernel(const char* name,
                void (*kernel)(float*, const float*, int, int, int, int, sycl::queue&),
                float* d_output, float* d_input,
                int N, int C, int H, int W, int totalElements,
                sycl::queue& queue) {
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
  
  double gflops = (totalElements * 1.0) / (timePerKernel * 1e-3) / 1e9;
  double bandwidth = (2.0 * totalElements * sizeof(float)) / (timePerKernel * 1e-3) / 1e9;
  
  std::cout << name << ",N=" << N << ",C=" << C << ",H=" << H << ",W=" << W
            << ",Time=" << timePerKernel << " ms,GFLOPS=" << gflops
            << ",Bandwidth=" << bandwidth << " GB/s" << std::endl;
}

int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  
  std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "Version,N,C,H,W,Time_ms,GFLOPS,Bandwidth_GB/s" << std::endl;
  
  std::vector<std::tuple<int, int, int, int>> testSizes = {
    {4, 64, 8, 8},
    {8, 128, 16, 16},
    {16, 256, 32, 32},
    {32, 512, 64, 64}
  };
  
  for (const auto& [N, C, H, W] : testSizes) {
    int totalElements = N * C * H * W;
    
    float* d_output = sycl::malloc_device<float>(totalElements, queue);
    float* d_input = sycl::malloc_device<float>(totalElements, queue);
    
    // Initialize
    std::vector<float> h_input(totalElements, 0.5f);
    queue.memcpy(d_input, h_input.data(), totalElements * sizeof(float)).wait();
    
    testKernel("V0", lczero::sycldnn_backend::winogradOutputReluInput_V0,
               d_output, d_input, N, C, H, W, totalElements, queue);
    testKernel("V1", lczero::sycldnn_backend::winogradOutputReluInput_V1,
               d_output, d_input, N, C, H, W, totalElements, queue);
    testKernel("V2", lczero::sycldnn_backend::winogradOutputReluInput_V2,
               d_output, d_input, N, C, H, W, totalElements, queue);
    
    sycl::free(d_output, queue);
    sycl::free(d_input, queue);
  }
  
  return 0;
}
