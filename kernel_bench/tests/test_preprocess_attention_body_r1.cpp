#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>

using namespace std;

namespace lczero {
namespace sycldnn_backend {

template <typename T>
void inputPreprocessForAttentionBody(T* output, const T* input,
                                     const T* encoding, int N, int input_size,
                                     int encoding_size,
                                     bool is_pe_dense_embedding,
                                     sycl::queue& stream) {
  int outputC = input_size + encoding_size;
  sycl::range<2> gridSize(N, 64);
  int blockSize = input_size + encoding_size;
  
  stream.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<2>(gridSize * sycl::range<2>(1, blockSize), sycl::range<2>(1, blockSize)),
      [=](sycl::nd_item<2> item) {
        int n = item.get_group(0);
        int hw = item.get_group(1);
        int c = item.get_local_id(1);

        T op;
        if (c >= input_size) {
          if (is_pe_dense_embedding) {
            op = (T)(encoding[n * 64 * encoding_size + hw * encoding_size +
                              (c - input_size)]);
          } else {
            op = (T)(encoding[64 * hw + (c - input_size)]);
          }
        } else {
          op = input[n * input_size * 64 + c * 64 + hw];
        }

        output[n * 64 * outputC + hw * outputC + c] = op;
      });
  });
  stream.wait();
}

} // namespace sycldnn_backend
} // namespace lczero

using namespace lczero::sycldnn_backend;

int main() {
  try {
    sycl::queue q(sycl::gpu_selector_v);
    
    cout << "=== preprocess_attention_body - Round 1 ===" << endl;
    cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
    
    struct TestConfig { 
      int N, input_size, encoding_size, outputC; 
      int inputElements, encodingElements, outputElements;
    };
    
    vector<TestConfig> configs = {
      {64, 64, 64, 128, 64 * 64 * 64, 64 * 64, 64 * 64 * 128},
      {256, 128, 64, 192, 256 * 128 * 64, 64 * 64, 256 * 64 * 192},
    };
    
    cout << setw(10) << "N" << setw(15) << "InputSize" << setw(15) << "EncodingSize"
         << setw(15) << "OutputC" << setw(15) << "Time(ms)" 
         << setw(15) << "GFLOPS" << setw(18) << "GB/s" << endl;
    cout << string(103, '-') << endl;
    
    for (const auto& cfg : configs) {
      sycl::half* d_output = sycl::malloc_device<sycl::half>(cfg.outputElements, q);
      sycl::half* d_input = sycl::malloc_device<sycl::half>(cfg.inputElements, q);
      sycl::half* d_encoding = sycl::malloc_device<sycl::half>(cfg.encodingElements, q);
      
      vector<sycl::half> h_input(cfg.inputElements);
      vector<sycl::half> h_encoding(cfg.encodingElements);
      
      srand(42);
      for (int i = 0; i < cfg.inputElements; i++) h_input[i] = sycl::half((float)(rand() % 100) / 100.0f);
      for (int i = 0; i < cfg.encodingElements; i++) h_encoding[i] = sycl::half((float)(rand() % 100) / 100.0f);
      
      q.memcpy(d_input, h_input.data(), cfg.inputElements * sizeof(sycl::half)).wait();
      q.memcpy(d_encoding, h_encoding.data(), cfg.encodingElements * sizeof(sycl::half)).wait();
      
      for (int i = 0; i < 3; i++) {
        inputPreprocessForAttentionBody(d_output, d_input, d_encoding, cfg.N, cfg.input_size, cfg.encoding_size, false, q);
      }
      
      vector<double> times;
      for (int iter = 0; iter < 10; iter++) {
        auto start = chrono::high_resolution_clock::now();
        inputPreprocessForAttentionBody(d_output, d_input, d_encoding, cfg.N, cfg.input_size, cfg.encoding_size, false, q);
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double, milli>(end - start).count());
      }
      
      double avg_time = 0;
      for (double t : times) avg_time += t;
      avg_time /= times.size();
      
      double flops = 1.0 * cfg.outputElements;
      double gflops = flops / (avg_time * 1e-3) / 1e9;
      
      double bytes = (cfg.inputElements + cfg.encodingElements + cfg.outputElements) * sizeof(sycl::half);
      double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
      
      cout << setw(10) << cfg.N << setw(15) << cfg.input_size << setw(15) << cfg.encoding_size
           << setw(15) << cfg.outputC << setw(15) << fixed << setprecision(3) << avg_time
           << setw(15) << setprecision(2) << gflops
           << setw(18) << setprecision(2) << bandwidth << endl;
      
      sycl::free(d_output, q);
      sycl::free(d_input, q);
      sycl::free(d_encoding, q);
    }
    
    cout << endl << "Test completed!" << endl;
    return 0;
  } catch (sycl::exception const &e) {
    cerr << "SYCL Exception: " << e.what() << endl;
    return 1;
  } catch (exception const &e) {
    cerr << "Exception: " << e.what() << endl;
    return 1;
  }
}
