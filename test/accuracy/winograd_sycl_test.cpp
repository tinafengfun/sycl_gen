/*
  Test harness for winograd_input_transform kernel accuracy testing
  This file provides host-side code to test the SYCL kernel
*/

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>

namespace lczero {
namespace sycldnn_backend {

// Forward declarations (from the kernel file)
template <typename T, bool nhcw>
void InputTransform(sycl::queue& queue, int N, int C, T* transformed_input, const T* input);

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

// Run SYCL kernel test
template<typename T>
int run_test(const TestConfig& config, const std::string& input_file, 
             const std::string& output_file) {
  try {
    // Create SYCL queue
    sycl::queue queue(sycl::default_selector_v);
    
    std::cout << "Running SYCL test on: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    
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
    T* d_input = sycl::malloc_device<T>(input_size, queue);
    T* d_output = sycl::malloc_device<T>(output_size, queue);
    
    // Copy input to device
    queue.memcpy(d_input, h_input.data(), input_size * sizeof(T));
    queue.wait();
    
    // Launch kernel
    std::cout << "Launching kernel: N=" << config.N << ", C=" << config.C << ", nhcw=" << config.nhcw << std::endl;
    
    if (config.nhcw) {
      InputTransform<T, true>(queue, config.N, config.C, d_output, d_input);
    } else {
      InputTransform<T, false>(queue, config.N, config.C, d_output, d_input);
    }
    
    queue.wait();
    std::cout << "Kernel execution completed" << std::endl;
    
    // Copy output back
    queue.memcpy(h_output.data(), d_output, output_size * sizeof(T));
    queue.wait();
    
    // Save output if requested
    if (!output_file.empty()) {
      save_data(h_output.data(), output_size, output_file);
      std::cout << "Saved output to: " << output_file << std::endl;
    }
    
    // Cleanup
    sycl::free(d_input, queue);
    sycl::free(d_output, queue);
    
    std::cout << "Test completed successfully" << std::endl;
    return 0;
    
  } catch (const sycl::exception& e) {
    std::cerr << "SYCL exception: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Standard exception: " << e.what() << std::endl;
    return 1;
  }
}

}  // namespace sycldnn_backend
}  // namespace lczero

// Main function
int main(int argc, char* argv[]) {
  using namespace lczero::sycldnn_backend;
  
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
  std::cout << "Winograd Input Transform Test (SYCL)" << std::endl;
  std::cout << "========================================" << std::endl;
  
  if (config.data_type == "float") {
    return run_test<float>(config, input_file, output_file);
  } else if (config.data_type == "half") {
    return run_test<sycl::half>(config, input_file, output_file);
  } else {
    std::cerr << "Unknown data type: " << config.data_type << std::endl;
    return 1;
  }
}
