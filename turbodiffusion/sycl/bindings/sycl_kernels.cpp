/**
 * Python bindings for TurboDiffusion SYCL kernels
 * 
 * This file uses pybind11 to expose SYCL kernels to Python.
 * It provides bindings for:
 * - RMSNorm
 * - LayerNorm  
 * - Quantization
 * - GEMM
 * 
 * Compilation:
 *   icpx -fsycl -O3 -std=c++17 sycl_kernels.cpp -shared -fPIC \
 *        $(python3 -m pybind11 --includes) \
 *        -o turbodiffusion_sycl$(python3-config --extension-suffix)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <memory>
#include <cstring>

namespace py = pybind11;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Convert numpy array to SYCL USM pointer
 */
template<typename T>
sycl::queue* get_queue_from_device(const std::string& device_name = "gpu") {
    static std::unique_ptr<sycl::queue> queue;
    if (!queue) {
        if (device_name == "gpu") {
            queue = std::make_unique<sycl::queue>(sycl::gpu_selector_v);
        } else {
            queue = std::make_unique<sycl::queue>(sycl::cpu_selector_v);
        }
        std::cout << "SYCL Queue created on: " 
                  << queue->get_device().get_info<sycl::info::device::name>() 
                  << std::endl;
    }
    return queue.get();
}

/**
 * Check if numpy array is contiguous
 */
bool is_contiguous(py::array_t<float> arr) {
    return arr.strides(0) == sizeof(float) * arr.shape(1) || arr.ndim() == 1;
}

// ============================================================================
// RMSNorm Kernel (from rmsnorm.hpp)
// ============================================================================

void rmsnorm_sycl(
    py::array_t<float> input,
    py::array_t<float> weight,
    py::array_t<float> output,
    float eps,
    int m,
    int n
) {
    // Get raw pointers
    float* input_ptr = static_cast<float*>(input.request().ptr);
    float* weight_ptr = static_cast<float*>(weight.request().ptr);
    float* output_ptr = static_cast<float*>(output.request().ptr);
    
    // Get SYCL queue
    sycl::queue* q = get_queue_from_device<float>();
    
    // Allocate device memory
    float* d_input = sycl::malloc_device<float>(m * n, *q);
    float* d_weight = sycl::malloc_device<float>(n, *q);
    float* d_output = sycl::malloc_device<float>(m * n, *q);
    
    // Copy to device
    q->memcpy(d_input, input_ptr, m * n * sizeof(float)).wait();
    q->memcpy(d_weight, weight_ptr, n * sizeof(float)).wait();
    
    // Launch kernel
    const int WG_SIZE = 256;
    q->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(WG_SIZE), h);
        
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(m * WG_SIZE), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) {
                int row = item.get_group(0);
                int lid = item.get_local_id(0);
                
                if (row >= m) return;
                
                // Compute sum of squares
                float thread_sum = 0.0f;
                for (int i = lid; i < n; i += WG_SIZE) {
                    float val = d_input[row * n + i];
                    thread_sum += val * val;
                }
                shared_mem[lid] = thread_sum;
                item.barrier();
                
                // Tree reduction
                #pragma unroll
                for (int stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        shared_mem[lid] += shared_mem[lid + stride];
                    }
                    item.barrier();
                }
                
                // Compute RMS
                float rms = sycl::sqrt(shared_mem[0] / n + eps);
                
                // Normalize and apply weight
                for (int i = lid; i < n; i += WG_SIZE) {
                    float val = d_input[row * n + i];
                    float w = d_weight[i];
                    d_output[row * n + i] = w * val / rms;
                }
            }
        );
    }).wait();
    
    // Copy back
    q->memcpy(output_ptr, d_output, m * n * sizeof(float)).wait();
    
    // Cleanup
    sycl::free(d_input, *q);
    sycl::free(d_weight, *q);
    sycl::free(d_output, *q);
}

// ============================================================================
// LayerNorm Kernel (from layernorm.hpp)
// ============================================================================

void layernorm_sycl(
    py::array_t<float> input,
    py::array_t<float> gamma,
    py::array_t<float> beta,
    py::array_t<float> output,
    float eps,
    int m,
    int n
) {
    float* input_ptr = static_cast<float*>(input.request().ptr);
    float* gamma_ptr = static_cast<float*>(gamma.request().ptr);
    float* beta_ptr = static_cast<float*>(beta.request().ptr);
    float* output_ptr = static_cast<float*>(output.request().ptr);
    
    sycl::queue* q = get_queue_from_device<float>();
    
    float* d_input = sycl::malloc_device<float>(m * n, *q);
    float* d_gamma = sycl::malloc_device<float>(n, *q);
    float* d_beta = sycl::malloc_device<float>(n, *q);
    float* d_output = sycl::malloc_device<float>(m * n, *q);
    
    q->memcpy(d_input, input_ptr, m * n * sizeof(float)).wait();
    q->memcpy(d_gamma, gamma_ptr, n * sizeof(float)).wait();
    q->memcpy(d_beta, beta_ptr, n * sizeof(float)).wait();
    
    const int WG_SIZE = 256;
    q->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(WG_SIZE * 2), h);
        
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(m * WG_SIZE), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) {
                int row = item.get_group(0);
                int lid = item.get_local_id(0);
                
                if (row >= m) return;
                
                float* shared_sum = shared_mem.get_pointer();
                float* shared_sum_sq = shared_mem.get_pointer() + WG_SIZE;
                
                // Compute partial sums
                float thread_sum = 0.0f;
                float thread_sum_sq = 0.0f;
                
                for (int i = lid; i < n; i += WG_SIZE) {
                    float val = d_input[row * n + i];
                    thread_sum += val;
                    thread_sum_sq += val * val;
                }
                
                shared_sum[lid] = thread_sum;
                shared_sum_sq[lid] = thread_sum_sq;
                item.barrier();
                
                // Tree reduction
                #pragma unroll
                for (int stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        shared_sum[lid] += shared_sum[lid + stride];
                        shared_sum_sq[lid] += shared_sum_sq[lid + stride];
                    }
                    item.barrier();
                }
                
                // Compute mean and variance
                float mean = shared_sum[0] / n;
                float var = (shared_sum_sq[0] / n) - (mean * mean);
                float inv_std = sycl::rsqrt(var + eps);
                
                // Normalize
                for (int i = lid; i < n; i += WG_SIZE) {
                    float val = d_input[row * n + i];
                    float normalized = (val - mean) * inv_std;
                    d_output[row * n + i] = d_gamma[i] * normalized + d_beta[i];
                }
            }
        );
    }).wait();
    
    q->memcpy(output_ptr, d_output, m * n * sizeof(float)).wait();
    
    sycl::free(d_input, *q);
    sycl::free(d_gamma, *q);
    sycl::free(d_beta, *q);
    sycl::free(d_output, *q);
}

// ============================================================================
// Quantization Kernel (from quantize.hpp)
// ============================================================================

void quantize_sycl(
    py::array_t<float> input,
    py::array_t<float> scale,
    py::array_t<int8_t> output,
    int m,
    int n
) {
    float* input_ptr = static_cast<float*>(input.request().ptr);
    float* scale_ptr = static_cast<float*>(scale.request().ptr);
    int8_t* output_ptr = static_cast<int8_t*>(output.request().ptr);
    
    sycl::queue* q = get_queue_from_device<float>();
    
    float* d_input = sycl::malloc_device<float>(m * n, *q);
    float* d_scale = sycl::malloc_device<float>(m, *q);
    int8_t* d_output = sycl::malloc_device<int8_t>(m * n, *q);
    
    q->memcpy(d_input, input_ptr, m * n * sizeof(float)).wait();
    q->memcpy(d_scale, scale_ptr, m * sizeof(float)).wait();
    
    q->parallel_for(sycl::range<1>(m * n), [=](sycl::item<1> item) {
        int idx = item.get_id(0);
        int row = idx / n;
        
        float val = d_input[idx];
        float scaled = val / d_scale[row];
        scaled = sycl::clamp(scaled, -127.0f, 127.0f);
        d_output[idx] = static_cast<int8_t>(sycl::rint(scaled));
    }).wait();
    
    q->memcpy(output_ptr, d_output, m * n * sizeof(int8_t)).wait();
    
    sycl::free(d_input, *q);
    sycl::free(d_scale, *q);
    sycl::free(d_output, *q);
}

// ============================================================================
// GEMM Kernel (from gemm.hpp) - INT8 version
// ============================================================================

void gemm_int8_sycl(
    py::array_t<int8_t> A,
    py::array_t<int8_t> B,
    py::array_t<float> C,
    int M,
    int N,
    int K
) {
    int8_t* A_ptr = static_cast<int8_t*>(A.request().ptr);
    int8_t* B_ptr = static_cast<int8_t*>(B.request().ptr);
    float* C_ptr = static_cast<float*>(C.request().ptr);
    
    sycl::queue* q = get_queue_from_device<float>();
    
    int8_t* d_A = sycl::malloc_device<int8_t>(M * K, *q);
    int8_t* d_B = sycl::malloc_device<int8_t>(K * N, *q);
    float* d_C = sycl::malloc_device<float>(M * N, *q);
    
    q->memcpy(d_A, A_ptr, M * K * sizeof(int8_t)).wait();
    q->memcpy(d_B, B_ptr, K * N * sizeof(int8_t)).wait();
    
    // Single-thread-per-row for small M
    q->parallel_for(sycl::range<1>(M), [=](sycl::item<1> item) {
        int m = item.get_id(0);
        if (m >= M) return;
        
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int k = 0; k < K; k++) {
                float a = static_cast<float>(d_A[m * K + k]);
                float b = static_cast<float>(d_B[k * N + n]);
                sum += a * b;
            }
            d_C[m * N + n] = sum;
        }
    }).wait();
    
    q->memcpy(C_ptr, d_C, M * N * sizeof(float)).wait();
    
    sycl::free(d_A, *q);
    sycl::free(d_B, *q);
    sycl::free(d_C, *q);
}

// ============================================================================
// Device Information
// ============================================================================

py::dict get_device_info() {
    py::dict info;
    
    try {
        sycl::queue q(sycl::gpu_selector_v);
        auto device = q.get_device();
        
        info["name"] = device.get_info<sycl::info::device::name>();
        info["vendor"] = device.get_info<sycl::info::device::vendor>();
        info["version"] = device.get_info<sycl::info::device::version>();
        info["driver_version"] = device.get_info<sycl::info::device::driver_version>();
        info["max_compute_units"] = device.get_info<sycl::info::device::max_compute_units>();
        info["max_work_group_size"] = device.get_info<sycl::info::device::max_work_group_size>();
        info["global_mem_size"] = device.get_info<sycl::info::device::global_mem_size>();
        info["local_mem_size"] = device.get_info<sycl::info::device::local_mem_size>();
        info["available"] = true;
    } catch (const sycl::exception& e) {
        info["available"] = false;
        info["error"] = e.what();
    }
    
    return info;
}

// ============================================================================
// Module Definition
// ============================================================================

PYBIND11_MODULE(turbodiffusion_sycl, m) {
    m.doc() = R"doc(
        TurboDiffusion SYCL Kernels
        ===========================
        
        Python bindings for SYCL kernels on Intel GPUs.
        
        Provides optimized kernels for:
        - RMSNorm: Root Mean Square Normalization
        - LayerNorm: Layer Normalization
        - Quantization: FP32 to INT8 conversion
        - GEMM: Matrix multiplication (INT8 input, FP32 output)
        
        Usage:
            import turbodiffusion_sycl as tds
            
            # Get device info
            info = tds.get_device_info()
            print(f"Device: {info['name']}")
            
            # Run RMSNorm
            tds.rmsnorm(input, weight, output, eps, m, n)
            
            # Run LayerNorm
            tds.layernorm(input, gamma, beta, output, eps, m, n)
            
            # Run Quantization
            tds.quantize(input, scale, output, m, n)
            
            # Run GEMM
            tds.gemm_int8(A, B, C, M, N, K)
    )doc";
    
    // Device information
    m.def("get_device_info", &get_device_info, 
          "Get information about the SYCL device");
    
    // RMSNorm
    m.def("rmsnorm", &rmsnorm_sycl,
          "RMSNorm normalization",
          py::arg("input"),
          py::arg("weight"),
          py::arg("output"),
          py::arg("eps"),
          py::arg("m"),
          py::arg("n"));
    
    // LayerNorm
    m.def("layernorm", &layernorm_sycl,
          "LayerNorm normalization",
          py::arg("input"),
          py::arg("gamma"),
          py::arg("beta"),
          py::arg("output"),
          py::arg("eps"),
          py::arg("m"),
          py::arg("n"));
    
    // Quantization
    m.def("quantize", &quantize_sycl,
          "Quantize FP32 to INT8",
          py::arg("input"),
          py::arg("scale"),
          py::arg("output"),
          py::arg("m"),
          py::arg("n"));
    
    // GEMM
    m.def("gemm_int8", &gemm_int8_sycl,
          "GEMM with INT8 inputs",
          py::arg("A"),
          py::arg("B"),
          py::arg("C"),
          py::arg("M"),
          py::arg("N"),
          py::arg("K"));
    
    // Version
    m.attr("__version__") = "0.1.0";
}
