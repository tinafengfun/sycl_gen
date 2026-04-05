/**
 * TurboDiffusion SYCL Custom Operators
 * 
 * PyTorch C++ extension providing optimized SYCL kernels for Wan2.1.
 * This eliminates Python overhead by registering custom operators that
 * directly call SYCL kernels from C++.
 * 
 * Accuracy Guarantee:
 * - Same algorithm as hook-based implementation
 * - Same epsilon values (RMSNorm: 1e-7, LayerNorm: 1e-5)
 * - Same SYCL kernel code
 * 
 * Build:
 *   python setup.py build_ext --inplace
 */

#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>

// Global SYCL queue (initialized on first use)
sycl::queue& get_sycl_queue() {
    static sycl::queue q(sycl::gpu_selector_v);
    static bool initialized = false;
    if (!initialized) {
        std::cout << "[SYCL] Using device: " 
                  << q.get_device().get_info<sycl::info::device::name>() 
                  << std::endl;
        initialized = true;
    }
    return q;
}

// ============================================================================
// RMSNorm Custom Operator
// ============================================================================

torch::Tensor rmsnorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    double eps
) {
    // Validate inputs
    TORCH_CHECK(input.device().is_xpu(), "Input must be on XPU");
    TORCH_CHECK(weight.device().is_xpu(), "Weight must be on XPU");
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");
    TORCH_CHECK(input.size(-1) == weight.size(0), "Last dim of input must match weight");
    
    // Get dimensions
    int64_t m = 1;
    for (int i = 0; i < input.dim() - 1; ++i) {
        m *= input.size(i);
    }
    int64_t n = input.size(-1);
    
    // Create output tensor
    auto output = torch::empty_like(input);
    
    // Get raw pointers
    float* input_ptr = input.data_ptr<float>();
    float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    // Get SYCL queue
    sycl::queue& q = get_sycl_queue();
    
    // Check if tensors are already in USM shared memory
    // If using XPU tensors with USM support, we can access directly
    bool use_usm = input.is_xpu() && weight.is_xpu() && output.is_xpu();
    
    if (use_usm) {
        // Direct USM access - no memcpy needed
        float* input_ptr = input.data_ptr<float>();
        float* weight_ptr = weight.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();
        
        // Launch kernel directly on USM pointers
        const int WG_SIZE = 256;
        q.submit([&](sycl::handler& h) {
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
                        float val = input_ptr[row * n + i];
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
                        float val = input_ptr[row * n + i];
                        float w = weight_ptr[i];
                        output_ptr[row * n + i] = w * val / rms;
                }
            }
        );
    });  // Async - no .wait()
    
    // Return output immediately - kernel runs async
    // PyTorch will synchronize when tensor is accessed
    return output;
    } else {
        // Fallback to device memory with explicit copy
        // Allocate device memory
        float* d_input = sycl::malloc_device<float>(m * n, q);
        float* d_weight = sycl::malloc_device<float>(n, q);
        float* d_output = sycl::malloc_device<float>(m * n, q);
        
        // Copy to device
        q.memcpy(d_input, input_ptr, m * n * sizeof(float)).wait();
        q.memcpy(d_weight, weight_ptr, n * sizeof(float)).wait();
        
        // Launch kernel
        const int WG_SIZE = 256;
        q.submit([&](sycl::handler& h) {
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
        q.memcpy(output_ptr, d_output, m * n * sizeof(float)).wait();
        
        // Cleanup
        sycl::free(d_input, q);
        sycl::free(d_weight, q);
        sycl::free(d_output, q);
    }
    
    return output;
}

// ============================================================================
// LayerNorm Custom Operator
// ============================================================================

torch::Tensor layernorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    // Validate inputs
    TORCH_CHECK(input.device().is_xpu(), "Input must be on XPU");
    TORCH_CHECK(weight.device().is_xpu(), "Weight must be on XPU");
    TORCH_CHECK(bias.device().is_xpu(), "Bias must be on XPU");
    TORCH_CHECK(input.dim() >= 2, "Input must be at least 2D");
    TORCH_CHECK(input.size(-1) == weight.size(0), "Last dim mismatch");
    
    // Get dimensions
    int64_t m = 1;
    for (int i = 0; i < input.dim() - 1; ++i) {
        m *= input.size(i);
    }
    int64_t n = input.size(-1);
    
    // Create output tensor
    auto output = torch::empty_like(input);
    
    // Get raw pointers
    float* input_ptr = input.data_ptr<float>();
    float* weight_ptr = weight.data_ptr<float>();
    float* bias_ptr = bias.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    // Get SYCL queue
    sycl::queue& q = get_sycl_queue();
    
    // Allocate device memory
    float* d_input = sycl::malloc_device<float>(m * n, q);
    float* d_weight = sycl::malloc_device<float>(n, q);
    float* d_bias = sycl::malloc_device<float>(n, q);
    float* d_output = sycl::malloc_device<float>(m * n, q);
    
    // Copy to device
    q.memcpy(d_input, input_ptr, m * n * sizeof(float)).wait();
    q.memcpy(d_weight, weight_ptr, n * sizeof(float)).wait();
    q.memcpy(d_bias, bias_ptr, n * sizeof(float)).wait();
    
    // Launch kernel
    const int WG_SIZE = 256;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> shared_mem(sycl::range<1>(WG_SIZE * 2), h);
        
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(m * WG_SIZE), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) {
                int row = item.get_group(0);
                int lid = item.get_local_id(0);
                
                if (row >= m) return;
                
                float* shared_sum = shared_mem.get_multi_ptr<sycl::access::decorated::no>().get();
                float* shared_sum_sq = shared_mem.get_multi_ptr<sycl::access::decorated::no>().get() + WG_SIZE;
                
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
                    d_output[row * n + i] = d_weight[i] * normalized + d_bias[i];
                }
            }
        );
    }).wait();
    
    // Copy back
    q.memcpy(output_ptr, d_output, m * n * sizeof(float)).wait();
    
    // Cleanup
    sycl::free(d_input, q);
    sycl::free(d_weight, q);
    sycl::free(d_bias, q);
    sycl::free(d_output, q);
    
    return output;
}

// ============================================================================
// Module Definition - DISABLED (moved to sycl_ops_main.cpp)
// ============================================================================

// PYBIND11_MODULE is now defined in sycl_ops_main.cpp
// This file contains only the implementation

// End of file
