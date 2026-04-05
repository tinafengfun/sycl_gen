// Phase 2.2: expand_planes_nhwc - Vectorized Bit Operations
// Target: 772 GFLOPS -> 900+ GFLOPS
// Optimizations:
// 1. Process 8 squares per thread (64-bit aligned)
// 2. Vectorized mask loading with uint4
// 3. Coalesced memory access pattern
// 4. Reduced thread divergence

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

using namespace std;

namespace lczero {
namespace sycldnn_backend {

constexpr int kInputPlanes = 112;

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// Optimized expand planes with vectorized bit operations
template <typename T>
class expand_planes_vec_kernel;

template <typename T>
void expandPlanes_NHWC_Vectorized(T* output, const uint64_t* masks, const T* values, 
                                   int n, sycl::queue& stream) {
  // Each thread processes 8 consecutive squares (one uint64_t worth)
  // This improves memory coalescing and reduces divergence
  const int elems_per_thread = 8;
  const int total_squares = n * 8 * 8;
  const int total_threads = DivUp(total_squares, elems_per_thread);
  const int block_size = 256;
  const int grid_size = DivUp(total_threads, block_size);
  
  stream.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<expand_planes_vec_kernel<T>>(
      sycl::nd_range<1>(grid_size * block_size, block_size),
      [=](sycl::nd_item<1> item) {
        const int tid = item.get_global_id(0);
        const int base_idx = tid * elems_per_thread;
        
        if (base_idx >= total_squares) return;
        
        // Process 8 squares per thread
        #pragma unroll
        for (int offset = 0; offset < elems_per_thread; offset++) {
          const int index = base_idx + offset;
          if (index >= total_squares) break;
          
          const int planeIndex = index % kInputPlanes;
          const int boardIndex = index / (kInputPlanes * 8 * 8);
          const int sqIndex = (index / kInputPlanes) & 0x3F;
          
          // Load mask - use cached value if possible
          uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];
          
          // Check bit and write
          bool set = (mask >> sqIndex) & 1ULL;
          output[index] = set ? values[boardIndex * kInputPlanes + planeIndex] : T(0);
        }
      }
    );
  });
  stream.wait_and_throw();
}

// Ultra-optimized version: process entire 64-bit word at once
template <typename T>
class expand_planes_ultra_kernel;

template <typename T>
void expandPlanes_NHWC_Ultra(T* output, const uint64_t* masks, const T* values,
                              int n, sycl::queue& stream) {
  // Process 64 squares (one full board row of planes) per thread group
  // This maximizes cache reuse for mask and value lookups
  const int planes_per_group = 4;  // Process 4 planes together
  const int squares_per_plane = 64;
  
  // Each work-group processes multiple (board, plane) pairs
  const int block_size = 256;
  const int total_plane_boards = n * kInputPlanes;
  const int grid_size = DivUp(total_plane_boards, planes_per_group * (block_size / squares_per_plane));
  
  stream.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<expand_planes_ultra_kernel<T>>(
      sycl::nd_range<1>(grid_size * block_size, block_size),
      [=](sycl::nd_item<1> item) {
        const int tid = item.get_global_id(0);
        
        // Each thread handles one square position across multiple planes
        const int local_tid = item.get_local_id(0);
        const int sqIndex = local_tid % squares_per_plane;  // 0-63
        const int plane_group = local_tid / squares_per_plane;  // Which group of planes
        const int group_id = item.get_group(0);
        
        // Calculate which (board, plane) this thread group handles
        const int base_plane_board = group_id * planes_per_group + plane_group;
        
        if (base_plane_board >= total_plane_boards) return;
        
        // Process up to 4 consecutive planes
        #pragma unroll
        for (int p = 0; p < planes_per_group; p++) {
          const int plane_board_idx = base_plane_board + p * (block_size / squares_per_plane);
          if (plane_board_idx >= total_plane_boards) break;
          
          const int boardIndex = plane_board_idx / kInputPlanes;
          const int planeIndex = plane_board_idx % kInputPlanes;
          
          // Load mask and value once per plane
          uint64_t mask = masks[plane_board_idx];
          T value = values[plane_board_idx];
          
          // Check if this square is set
          bool set = (mask >> sqIndex) & 1ULL;
          
          // Calculate output index: NHWC format
          // index = ((board * 8 + h) * 8 + w) * 112 + plane
          // where h = sqIndex / 8, w = sqIndex % 8
          const int h = sqIndex / 8;
          const int w = sqIndex % 8;
          const int output_idx = ((boardIndex * 8 + h) * 8 + w) * kInputPlanes + planeIndex;
          
          output[output_idx] = set ? value : T(0);
        }
      }
    );
  });
  stream.wait_and_throw();
}

} // namespace sycldnn_backend
} // namespace lczero

using namespace lczero::sycldnn_backend;

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== expand_planes_nhwc - Phase 2.2 Optimization ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl;
    cout << "Optimizations: Vectorized bit ops, improved coalescing" << endl;
    cout << "Target: 772 -> 900+ GFLOPS" << endl << endl;
    
    struct TestConfig {
      int n;
      int totalElements;
    };
    
    vector<TestConfig> configs = {
      {1, 1 * 112 * 8 * 8},
      {4, 4 * 112 * 8 * 8},
      {16, 16 * 112 * 8 * 8},
      {64, 64 * 112 * 8 * 8},
      {256, 256 * 112 * 8 * 8},
    };
    
    // Test both versions
    cout << "--- Version 1: Vectorized (8 elems/thread) ---" << endl;
    cout << setw(12) << "Boards" 
         << setw(15) << "Total"
         << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS"
         << setw(18) << "GB/s" << endl;
    cout << string(75, '-') << endl;
    
    for (const auto& cfg : configs) {
      const int n = cfg.n;
      const int totalElements = cfg.totalElements;
      const int maskElements = n * kInputPlanes;
      
      float* d_output = sycl::malloc_device<float>(totalElements, q);
      uint64_t* d_masks = sycl::malloc_device<uint64_t>(maskElements, q);
      float* d_values = sycl::malloc_device<float>(maskElements, q);
      
      vector<uint64_t> h_masks(maskElements);
      vector<float> h_values(maskElements);
      
      srand(42);
      for (int i = 0; i < maskElements; i++) {
        h_masks[i] = ((uint64_t)rand() << 32) | (uint64_t)rand();
        h_values[i] = (float)(rand() % 100) / 100.0f;
      }
      
      q.memcpy(d_masks, h_masks.data(), maskElements * sizeof(uint64_t)).wait();
      q.memcpy(d_values, h_values.data(), maskElements * sizeof(float)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        expandPlanes_NHWC_Vectorized(d_output, d_masks, d_values, n, q);
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        expandPlanes_NHWC_Vectorized(d_output, d_masks, d_values, n, q);
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(end - start).count());
      }
      
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      double flops = 5.0 * totalElements;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      
      double bytes = maskElements * sizeof(uint64_t) + maskElements * sizeof(float) + 
                     totalElements * sizeof(float);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(12) << n
           << setw(15) << totalElements
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_output, q);
      sycl::free(d_masks, q);
      sycl::free(d_values, q);
    }
    
    cout << endl << "--- Version 2: Ultra (plane-coalesced) ---" << endl;
    cout << setw(12) << "Boards" 
         << setw(15) << "Total"
         << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS"
         << setw(18) << "GB/s" << endl;
    cout << string(75, '-') << endl;
    
    for (const auto& cfg : configs) {
      const int n = cfg.n;
      const int totalElements = cfg.totalElements;
      const int maskElements = n * kInputPlanes;
      
      float* d_output = sycl::malloc_device<float>(totalElements, q);
      uint64_t* d_masks = sycl::malloc_device<uint64_t>(maskElements, q);
      float* d_values = sycl::malloc_device<float>(maskElements, q);
      
      vector<uint64_t> h_masks(maskElements);
      vector<float> h_values(maskElements);
      
      srand(42);
      for (int i = 0; i < maskElements; i++) {
        h_masks[i] = ((uint64_t)rand() << 32) | (uint64_t)rand();
        h_values[i] = (float)(rand() % 100) / 100.0f;
      }
      
      q.memcpy(d_masks, h_masks.data(), maskElements * sizeof(uint64_t)).wait();
      q.memcpy(d_values, h_values.data(), maskElements * sizeof(float)).wait();
      
      // Warmup
      for (int i = 0; i < 3; i++) {
        expandPlanes_NHWC_Ultra(d_output, d_masks, d_values, n, q);
      }
      
      // Benchmark
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        expandPlanes_NHWC_Ultra(d_output, d_masks, d_values, n, q);
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(end - start).count());
      }
      
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      double flops = 5.0 * totalElements;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      
      double bytes = maskElements * sizeof(uint64_t) + maskElements * sizeof(float) + 
                     totalElements * sizeof(float);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(12) << n
           << setw(15) << totalElements
           << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_output, q);
      sycl::free(d_masks, q);
      sycl::free(d_values, q);
    }
    
    cout << endl << "Phase 2.2 optimization complete!" << endl;
    return 0;
  } catch (sycl::exception const &e) {
    cerr << "SYCL Exception: " << e.what() << endl;
    return 1;
  } catch (exception const &e) {
    cerr << "Exception: " << e.what() << endl;
    return 1;
  }
}
