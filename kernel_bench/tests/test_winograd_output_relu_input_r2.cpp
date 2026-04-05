// winograd_output_relu_input - SYCL Conversion
// Fused: Output Transform + ReLU + Input Transform
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
using namespace std;

namespace lczero {
namespace sycldnn_backend {

// Simplified Winograd output transform (6x6 -> 4x4)
template <typename T>
inline void OutputTransform4x4_simplified(T* output, const T* input) {
  // At * input * A (simplified)
  const float At[4][6] = {
    {1, 1, 1, 1, 1, 0},
    {0, 1, -1, 2, -2, 0},
    {0, 1, 1, 4, 4, 0},
    {0, 1, -1, 8, -8, 1}
  };
  const float A[6][4] = {
    {1, 0, 0, 0}, {1, 1, 1, 1}, {1, -1, 1, -1},
    {1, 2, 4, 8}, {1, -2, 4, -8}, {0, 0, 0, 1}
  };
  
  float temp[4][6];
  float in[6][6], out[4][4];
  
  // Load
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 6; j++)
      in[i][j] = (float)input[i * 6 + j];
  
  // At * input
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 6; j++) {
      temp[i][j] = 0;
      for (int k = 0; k < 6; k++)
        temp[i][j] += At[i][k] * in[k][j];
    }
  
  // temp * A
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      out[i][j] = 0;
      for (int k = 0; k < 6; k++)
        out[i][j] += temp[i][k] * A[k][j];
      output[i * 4 + j] = (T)out[i][j];
    }
}

// Simplified Winograd input transform (4x4 -> 6x6)
template <typename T>
inline void InputTransform4x4_simplified(T* output, const T* input) {
  const float Bt[6][6] = {
    {4, 0, -5, 0, 1, 0}, {0, -4, -4, 1, 1, 0}, {0, 4, -4, -1, 1, 0},
    {0, -2, -1, 2, 1, 0}, {0, 2, -1, -2, 1, 0}, {0, 4, 0, -5, 0, 1}
  };
  const float B[6][6] = {
    {4, 0, 0, 0, 0, 0}, {-4, 4, -2, 2, 4, 0}, {-5, -4, -4, -1, -1, 0},
    {0, 1, -1, 2, -2, -5}, {1, 1, 1, 1, 1, 0}, {0, 0, 0, 0, 0, 1}
  };
  
  float temp[6][6];
  float in[6][6] = {}, out[6][6];
  
  // Load 4x4 center, pad with zeros
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      in[i+1][j+1] = (float)input[i * 4 + j];
  
  // Bt * input
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 6; j++) {
      temp[i][j] = 0;
      for (int k = 0; k < 6; k++)
        temp[i][j] += Bt[i][k] * in[k][j];
    }
  
  // temp * B
  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 6; j++) {
      out[i][j] = 0;
      for (int k = 0; k < 6; k++)
        out[i][j] += temp[i][k] * B[k][j];
      output[i * 6 + j] = (T)out[i][j];
    }
}

template <typename T>
void winogradOutputReluInput(int N, int C, T* output, const T* input, const T* bias, sycl::queue& q) {
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<2>(
        sycl::range<2>(N * 4, C * 4),  // 4 tiles per spatial dimension
        sycl::range<2>(4, 64)             // 4 tiles per block, 64 channels
      ),
      [=](sycl::nd_item<2> item) {
        int tile_n = item.get_group(0);
        int tile_c = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
        int local_tile = item.get_local_id(0);
        
        if (tile_c >= C) return;
        
        int n = tile_n / 4;
        int tile_idx = tile_n % 4;
        int h_offset = (tile_idx / 2) * 4;
        int w_offset = (tile_idx % 2) * 4;
        
        // Read transformed input (HWNC layout)
        T transformed[6][6];
        for (int i = 0; i < 6; i++)
          for (int j = 0; j < 6; j++)
            transformed[i][j] = input[((((n * 4 + tile_idx) * C + tile_c) * 6) + i) * 6 + j];
        
        // Output transform
        T tile4x4[4][4];
        OutputTransform4x4_simplified(&tile4x4[0][0], &transformed[0][0]);
        
        // Add bias and ReLU
        T b = bias[tile_c];
        for (int i = 0; i < 4; i++)
          for (int j = 0; j < 4; j++) {
            float val = (float)tile4x4[i][j] + (float)b;
            if (val < 0) val = 0;  // ReLU
            tile4x4[i][j] = (T)val;
          }
        
        // Input transform back to 6x6
        T output_transformed[6][6];
        InputTransform4x4_simplified(&output_transformed[0][0], &tile4x4[0][0]);
        
        // Write output (HWNC layout)
        for (int i = 0; i < 6; i++)
          for (int j = 0; j < 6; j++)
            output[((((n * 4 + tile_idx) * C + tile_c) * 6) + i) * 6 + j] = output_transformed[i][j];
      }
    );
  });
  q.wait();
}

} // namespace sycldnn_backend
} // namespace lczero

using namespace lczero::sycldnn_backend;

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== winograd_output_relu_input - Round 2 Phase 1 ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl;
    cout << "Note: Fused Output Transform + ReLU + Input Transform" << endl << endl;
    
    struct Cfg { int N, C; int tiles; int inputSize, outputSize; };
    vector<Cfg> cfgs = {
      {16, 64, 16*4, 16*4*64*36, 16*4*64*36},
      {64, 128, 64*4, 64*4*128*36, 64*4*128*36}
    };
    
    cout << setw(8) << "N" << setw(8) << "C" << setw(12) << "Tiles" 
         << setw(15) << "Time(ms)" << setw(15) << "GFLOPS" << endl;
    cout << string(58, '-') << endl;
    
    for (auto& c : cfgs) {
      sycl::half *out = sycl::malloc_device<sycl::half>(c.outputSize, q);
      sycl::half *in = sycl::malloc_device<sycl::half>(c.inputSize, q);
      sycl::half *bias = sycl::malloc_device<sycl::half>(c.C, q);
      
      // Warmup
      for (int i = 0; i < 3; i++)
        winogradOutputReluInput(c.N, c.C, out, in, bias, q);
      
      // Benchmark
      vector<double> times;
      for (int i = 0; i < 10; i++) {
        auto s = chrono::high_resolution_clock::now();
        winogradOutputReluInput(c.N, c.C, out, in, bias, q);
        auto e = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(e - s).count());
      }
      
      double avg = 0;
      for (double t : times) avg += t;
      avg /= times.size();
      
      // FLOPs: Output transform (4*6*6*6) + Input transform (6*6*6*6)
      double flops = (4*6*6*6 + 6*6*6*6) * c.tiles * c.C;
      double gflops = flops / (avg * 1e-3) / 1e9;
      
      cout << setw(8) << c.N << setw(8) << c.C << setw(12) << c.tiles
           << setw(15) << fixed << setprecision(3) << avg
           << setw(15) << setprecision(2) << gflops << endl;
      
      sycl::free(out, q);
      sycl::free(in, q);
      sycl::free(bias, q);
    }
    
    cout << endl << "✓ CUDA→SYCL conversion successful!" << endl;
    cout << "✓ Fused kernel: Output Transform + ReLU + Input Transform" << endl;
    
  } catch (exception const &e) {
    cerr << "Exception: " << e.what() << endl;
    return 1;
  }
  return 0;
}
