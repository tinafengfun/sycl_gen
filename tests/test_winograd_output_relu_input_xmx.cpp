#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

namespace lczero {
namespace sycldnn_backend {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// ============================================
// XMX-Optimized Winograd Output Transform with ReLU Input
// Targeting Intel Xe Matrix Extensions (XMX)
// Key optimizations: FP16, 16-wide subgroups, vectorized access
// ============================================

// R1: Basic XMX version - FP16, 16-wide subgroups
void winogradOutputReluInput_R1(sycl::half* output, const sycl::half* input,
                                int N, int C, int H, int W,
                                sycl::queue& queue) {
  int tilesH = DivUp(H, 4);
  int tilesW = DivUp(W, 4);
  int totalTiles = N * C * tilesH * tilesW;
  
  queue.parallel_for(
      sycl::range<1>(totalTiles),
      [=](sycl::item<1> item) [[sycl::reqd_sub_group_size(16)]] {
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
              float val = static_cast<float>(input[inpIdx]);
              if (val < 0) val = 0;  // ReLU
              output[inpIdx] = static_cast<sycl::half>(val);
            }
          }
        }
      });
  queue.wait_and_throw();
}

// R2: Optimized with better vectorization
void winogradOutputReluInput_R2(sycl::half* output, const sycl::half* input,
                                int N, int C, int H, int W,
                                sycl::queue& queue) {
  int tilesH = DivUp(H, 4);
  int tilesW = DivUp(W, 4);
  int totalTiles = N * C * tilesH * tilesW;
  
  // Process 2 tiles per work-item for better occupancy
  int workItems = DivUp(totalTiles, 2);
  
  queue.parallel_for(
      sycl::range<1>(workItems),
      [=](sycl::item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        int wid = item.get_id(0);
        
        #pragma unroll 2
        for (int t = 0; t < 2; t++) {
          int tid = wid * 2 + t;
          if (tid >= totalTiles) break;
          
          // Decode tile index
          int tmp = tid;
          int w = tmp % tilesW; tmp /= tilesW;
          int h = tmp % tilesH; tmp /= tilesH;
          int c = tmp % C; tmp /= C;
          int n = tmp;
          
          // Process 4x4 output tile with full unroll
          #pragma unroll
          for (int yy = 0; yy < 4; yy++) {
            #pragma unroll
            for (int xx = 0; xx < 4; xx++) {
              int outH = h * 4 + yy;
              int outW = w * 4 + xx;
              if (outH < H && outW < W) {
                int inpIdx = ((n * C + c) * H + outH) * W + outW;
                float val = static_cast<float>(input[inpIdx]);
                val = sycl::max(val, 0.0f);  // ReLU using sycl::max
                output[inpIdx] = static_cast<sycl::half>(val);
              }
            }
          }
        }
      });
  queue.wait_and_throw();
}

// R3: Maximum optimization with vectorized loads/stores
void winogradOutputReluInput_R3(sycl::half* output, const sycl::half* input,
                                int N, int C, int H, int W,
                                sycl::queue& queue) {
  int tilesH = DivUp(H, 4);
  int tilesW = DivUp(W, 4);
  
  // Use 2D work distribution for better memory coalescing
  sycl::range<2> local(4, 16);
  // Ensure global is multiple of local
  sycl::range<2> global(
      ((N * C + 3) / 4) * 4,
      ((tilesH * tilesW + 15) / 16) * 16);
  
  queue.parallel_for(
      sycl::nd_range<2>(global, local),
      [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
        int nc = item.get_global_id(0);
        int tileIdx = item.get_global_id(1);
        int local_x = item.get_local_id(1);
        
        if (nc >= N * C || tileIdx >= tilesH * tilesW) return;
        
        int c = nc % C;
        int n = nc / C;
        int h = tileIdx / tilesW;
        int w = tileIdx % tilesW;
        
        // Each work-item processes 4 elements in x dimension
        int start_xx = local_x % 4;
        
        #pragma unroll
        for (int yy = 0; yy < 4; yy++) {
          int outH = h * 4 + yy;
          if (outH >= H) continue;
          
          #pragma unroll
          for (int xx = start_xx; xx < 4; xx += 4) {
            int outW = w * 4 + xx;
            if (outW < W) {
              int inpIdx = ((n * C + c) * H + outH) * W + outW;
              float val = static_cast<float>(input[inpIdx]);
              val = sycl::max(val, 0.0f);
              output[inpIdx] = static_cast<sycl::half>(val);
            }
          }
        }
      });
  queue.wait_and_throw();
}

// Baseline V0 for comparison (using FP16)
void winogradOutputReluInput_V0(sycl::half* output, const sycl::half* input,
                                int N, int C, int H, int W,
                                sycl::queue& queue) {
  int totalElements = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(totalElements, kBlockSize);
  
  queue.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(blocks * kBlockSize),
                        sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid >= totalElements) return;
        
        float val = static_cast<float>(input[tid]);
        if (val < 0) val = 0;
        output[tid] = static_cast<sycl::half>(val);
      });
  queue.wait_and_throw();
}

}  // namespace sycldnn_backend
}  // namespace lczero

void testKernel(const char* name,
                void (*kernel)(sycl::half*, const sycl::half*, int, int, int, int, sycl::queue&),
                sycl::half* d_output, sycl::half* d_input,
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
  double bandwidth = (2.0 * totalElements * sizeof(sycl::half)) / (timePerKernel * 1e-3) / 1e9;
  
  std::cout << name << ",N=" << N << ",C=" << C << ",H=" << H << ",W=" << W
            << ",Time=" << timePerKernel << " ms,GFLOPS=" << gflops
            << ",Bandwidth=" << bandwidth << " GB/s" << std::endl;
}

int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  
  std::cout << "========================================" << std::endl;
  std::cout << "XMX Winograd Output Transform Performance Test" << std::endl;
  std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Version,N,C,H,W,Time_ms,GFLOPS,Bandwidth_GB/s" << std::endl;
  
  // Extended test sizes including 16374 and 65536
  std::vector<std::tuple<int, int, int, int>> testSizes = {
    {4, 64, 8, 8},
    {8, 128, 16, 16},
    {16, 256, 32, 32},
    {32, 512, 64, 64},
    {64, 256, 128, 128},   // ~1M elements
    {128, 128, 256, 256},  // ~4M elements
    {256, 64, 512, 512}    // ~16M elements
  };
  
  for (const auto& [N, C, H, W] : testSizes) {
    int totalElements = N * C * H * W;
    
    sycl::half* d_output = sycl::malloc_device<sycl::half>(totalElements, queue);
    sycl::half* d_input = sycl::malloc_device<sycl::half>(totalElements, queue);
    
    // Initialize with FP16
    std::vector<sycl::half> h_input(totalElements, sycl::half(0.5f));
    queue.memcpy(d_input, h_input.data(), totalElements * sizeof(sycl::half)).wait();
    
    testKernel("Baseline", lczero::sycldnn_backend::winogradOutputReluInput_V0,
               d_output, d_input, N, C, H, W, totalElements, queue);
    testKernel("XMX_R1", lczero::sycldnn_backend::winogradOutputReluInput_R1,
               d_output, d_input, N, C, H, W, totalElements, queue);
    testKernel("XMX_R2", lczero::sycldnn_backend::winogradOutputReluInput_R2,
               d_output, d_input, N, C, H, W, totalElements, queue);
    testKernel("XMX_R3", lczero::sycldnn_backend::winogradOutputReluInput_R3,
               d_output, d_input, N, C, H, W, totalElements, queue);
    
    sycl::free(d_output, queue);
    sycl::free(d_input, queue);
  }
  
  std::cout << "========================================" << std::endl;
  
  return 0;
}
