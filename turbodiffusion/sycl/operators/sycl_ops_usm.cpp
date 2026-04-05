/**
 * TurboDiffusion SYCL Custom Operators with USM Optimization
 * 
 * Uses Unified Shared Memory (USM) for zero-copy data transfer between
 * PyTorch XPU tensors and SYCL kernels.
 * 
 * Key optimizations:
 * 1. USM shared memory - no explicit memcpy
 * 2. Memory pool - reuse allocations
 * 3. Async execution - overlap computation and memory operations
 */

#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <memory>
#include <unordered_map>

// Global SYCL queue
static sycl::queue& get_sycl_queue() {
    static sycl::queue q(sycl::gpu_selector_v);
    static bool initialized = false;
    if (!initialized) {
        std::cout << "[SYCL-USM] Device: " 
                  << q.get_device().get_info<sycl::info::device::name>() 
                  << std::endl;
        initialized = true;
    }
    return q;
}

// USM Memory Pool
class USMMemoryPool {
public:
    static USMMemoryPool& get_instance() {
        static USMMemoryPool instance;
        return instance;
    }
    
    // Allocate USM shared memory
    float* allocate(size_t size) {
        auto& q = get_sycl_queue();
        
        // Check if we have a suitable block in pool
        auto key = std::make_pair(size, q.get_context());
        auto it = pool_.find(key);
        if (it != pool_.end() && !it->second.empty()) {
            float* ptr = it->second.back();
            it->second.pop_back();
            return ptr;
        }
        
        // Allocate new USM shared memory
        float* ptr = sycl::malloc_shared<float>(size, q);
        allocations_++;
        return ptr;
    }
    
    // Return memory to pool
    void deallocate(float* ptr, size_t size) {
        if (!ptr) return;
        
        auto& q = get_sycl_queue();
        auto key = std::make_pair(size, q.get_context());
        pool_[key].push_back(ptr);
    }
    
    // Cleanup all pooled memory
    void cleanup() {
        auto& q = get_sycl_queue();
        for (auto& [key, vec] : pool_) {
            for (float* ptr : vec) {
                sycl::free(ptr, q);
            }
        }
        pool_.clear();
        std::cout << "[USM Pool] Cleanup complete. Total allocations: " << allocations_ << std::endl;
    }
    
    ~USMMemoryPool() {
        cleanup();
    }
    
private:
    USMMemoryPool() = default;
    std::unordered_map<std::pair<size_t, sycl::context>, std::vector<float*>, 
                       boost::hash<std::pair<size_t, sycl::context>>> pool_;
    size_t allocations_ = 0;
};

// ============================================================================
// RMSNorm with USM
// ============================================================================

torch::Tensor rmsnorm_forward_usm(
    torch::Tensor input,
    torch::Tensor weight,
    double eps
) {
    // Validate inputs
    TORCH_CHECK(input.device().is_xpu(), "Input must be on XPU");
    TORCH_CHECK(weight.device().is_xpu(), "Weight must be on XPU");
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
    
    // Get raw pointers (USM shared memory)
    float* input_ptr = input.data_ptr<float>();
    float* weight_ptr = weight.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    // Get SYCL queue
    sycl::queue& q = get_sycl_queue();
    
    // Launch kernel directly on USM pointers - NO MEMCPY NEEDED!
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
    });  // Note: No .wait() here - async execution
    
    return output;
}

// ============================================================================
// LayerNorm with USM
// ============================================================================

torch::Tensor layernorm_forward_usm(
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
    
    // Get raw pointers (USM shared memory)
    float* input_ptr = input.data_ptr<float>();
    float* weight_ptr = weight.data_ptr<float>();
    float* bias_ptr = bias.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    // Get SYCL queue
    sycl::queue& q = get_sycl_queue();
    
    // Launch kernel directly on USM pointers
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
                    float val = input_ptr[row * n + i];
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
                    float val = input_ptr[row * n + i];
                    float normalized = (val - mean) * inv_std;
                    output_ptr[row * n + i] = weight_ptr[i] * normalized + bias_ptr[i];
                }
            }
        );
    });  // Async - no .wait()
    
    return output;
}

// ============================================================================
// Module Definition
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TurboDiffusion SYCL Custom Operators with USM";
    
    m.def("rmsnorm_forward", &rmsnorm_forward_usm,
          "RMSNorm forward with USM optimization",
          py::arg("input"),
          py::arg("weight"),
          py::arg("eps") = 1e-7);
    
    m.def("layernorm_forward", &layernorm_forward_usm,
          "LayerNorm forward with USM optimization",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("eps") = 1e-5);
    
    m.def("cleanup_usm_pool", []() {
        USMMemoryPool::get_instance().cleanup();
    }, "Cleanup USM memory pool");
}
