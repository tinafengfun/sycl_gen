/*
  CUDA Test harness for winograd_input_transform kernel
  This file provides host-side code to test the CUDA kernel
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>

namespace lczero {
namespace cudnn_backend {

// Forward declarations (from the kernel file)
template <typename T, bool nhcw>
void InputTransform(int N, int C, T* transformed_input, const T* input, cudaStream_t stream);

// Test configuration
struct TestConfig {
  int N;      // batch size
  int C;      // channels
  bool nhcw;  // layout format
  std::string data_type;  // "float" or "half"
  std::string test_type;  // "random", "boundary", "ones", "zeros"
};

// Generate test data
template<typename T>
void generate_test_data(T* data, int size, const std::string& test_type) {
  if (test_type == "ones") {
    for (int i = 0; i < size; i++) data[i] = T(1.0);
  } else if (test_type == "zeros") {
    for (int i = 0; i < size; i++) data[i] = T(0.0);
  } else if (test_type == "sequential") {
    for (int i = 0; i < size; i++) data[i] = T(i % 100) / T(100.0);
  } else if (test_type == "boundary") {
    // Test boundary values
    T values[] = {T(0.0), T(1.0), T(-1.0), 
                  std::numeric_limits<T>::max(),
                  std::numeric_limits<T>::min(),
                  std::numeric_limits<T>::epsilon()};
    for (int i = 0; i < size; i++) {
      data[i] = values[i % 6];
    }
  } else {
    // Random uniform [-1, 1]
    srand(42);  // Fixed seed for reproducibility
    for (int i = 0; i < size; i++) {
      float r = (float(rand()) / RAND_MAX) * 2.0f - 1.0f;
      data[i] = T(r);
    }
  }
}

// Save data to binary file
template<typename T>
void save_data(const T* data, int size, const std::string& filename) {
  std::ofstream file(filename, std::ios::binary);
  file.write(reinterpret_cast<const char*>(data), size * sizeof(T));
  file.close();
}

// Load data from binary file
template<typename T>
void load_data(T* data, int size, const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  file.read(reinterpret_cast<char*>(data), size * sizeof(T));
  file.close();
}

// Check CUDA errors
#define CHECK_CUDA(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
      return 1; \
    } \
  } while(0)

// Run CUDA kernel test
template<typename T>
int run_test(const TestConfig& config, const std::string& input_file, 
             const std::string& output_file) {
  // Calculate sizes
  int input_size = config.N * config.C * 8 * 8;
  int output_size = config.N * config.C * 6 * 6 * 4;  // 4 tiles per board
  
  // Allocate host memory
  std::vector<T> h_input(input_size);
  std::vector<T> h_output(output_size);
  
  // Load or generate input data
  if (!input_file.empty()) {
    load_data(h_input.data(), input_size, input_file);
    std::cout << "Loaded input from: " << input_file << std::endl;
  } else {
    generate_test_data(h_input.data(), input_size, config.test_type);
    std::cout << "Generated " << config.test_type << " test data" << std::endl;
  }
  
  // Allocate device memory
  T* d_input;
  T* d_output;
  CHECK_CUDA(cudaMalloc(&d_input, input_size * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_output, output_size * sizeof(T)));
  
  // Copy input to device
  CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(T), cudaMemcpyHostToDevice));
  
  // Create CUDA stream
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  
  // Launch kernel
  std::cout << "Launching kernel: N=" << config.N << ", C=" << config.C << ", nhcw=" << config.nhcw << std::endl;
  
  if (config.nhcw) {
    InputTransform<T, true>(config.N, config.C, d_output, d_input, stream);
  } else {
    InputTransform<T, false>(config.N, config.C, d_output, d_input, stream);
  }
  
  CHECK_CUDA(cudaStreamSynchronize(stream));
  std::cout << "Kernel execution completed" << std::endl;
  
  // Copy output back
  CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(T), cudaMemcpyDeviceToHost));
  
  // Save output if requested
  if (!output_file.empty()) {
    save_data(h_output.data(), output_size, output_file);
    std::cout << "Saved output to: " << output_file << std::endl;
  }
  
  // Cleanup
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  
  std::cout << "Test completed successfully" << std::endl;
  return 0;
}

}  // namespace cudnn_backend
}  // namespace lczero

// Main function
int main(int argc, char* argv[]) {
  using namespace lczero::cudnn_backend;
  
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <N> <C> <float|half> [nhcw] [test_type] [input_file] [output_file]" << std::endl;
    return 1;
  }
  
  TestConfig config;
  config.N = std::stoi(argv[1]);
  config.C = std::stoi(argv[2]);
  config.data_type = argv[3];
  config.nhcw = (argc > 4 && std::string(argv[4]) == "nhcw");
  config.test_type = (argc > 5) ? argv[5] : "random";
  
  std::string input_file = (argc > 6) ? argv[6] : "";
  std::string output_file = (argc > 7) ? argv[7] : "";
  
  std::cout << "========================================" << std::endl;
  std::cout << "Winograd Input Transform Test (CUDA)" << std::endl;
  std::cout << "========================================" << std::endl;
  
  if (config.data_type == "float") {
    return run_test<float>(config, input_file, output_file);
  } else if (config.data_type == "half") {
    return run_test<half>(config, input_file, output_file);
  } else {
    std::cerr << "Unknown data type: " << config.data_type << std::endl;
    return 1;
  }
}
