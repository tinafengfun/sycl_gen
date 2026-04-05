/**
 * TurboDiffusion SYCL Operators - Unified Module
 * 
 * This is the main entry point that binds all SYCL operators to PyTorch.
 * Include this file in setup.py as the only source with PYBIND11_MODULE.
 */

#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <cmath>

// BF16 type alias
using bf16 = sycl::ext::oneapi::bfloat16;

// Global SYCL queue
static sycl::queue& get_sycl_queue() {
    static sycl::queue q(sycl::gpu_selector_v);
    static bool initialized = false;
    if (!initialized) {
        std::cout << "[TurboDiffusion-SYCL] Using device: " 
                  << q.get_device().get_info<sycl::info::device::name>() 
                  << std::endl;
        initialized = true;
    }
    return q;
}

// ============================================================================
// Forward declarations from other modules
// ============================================================================

// Flash Attention
class FlashAttentionKernel {
public:
    static void forward(
        sycl::queue& queue,
        const bf16* query,
        const bf16* key,
        const bf16* value,
        bf16* output,
        int batch_size,
        int num_heads_q,
        int num_heads_kv,
        int seq_len_q,
        int seq_len_kv,
        int head_dim,
        float softmax_scale,
        bool causal
    );
};

torch::Tensor flash_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::optional<torch::Tensor> attn_mask,
    float softmax_scale
);

torch::Tensor flash_attention_forward_varlen(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cu_seqlens,
    int max_seqlen,
    float softmax_scale
);

// RMSNorm
torch::Tensor rmsnorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    double eps
);

// LayerNorm
torch::Tensor layernorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
);

// Sparse Attention
torch::Tensor sparse_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor lut,
    int topk,
    int block_q,
    int block_k
);

// ============================================================================
// Module Definition - Single entry point
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TurboDiffusion SYCL Custom Operators - Unified Module";
    
    // Flash Attention
    m.def("flash_attention_forward", &flash_attention_forward,
          "Flash Attention forward pass (fixed length)",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("attn_mask") = nullptr,
          py::arg("softmax_scale") = 0.0f);
    
    m.def("flash_attention_forward_varlen", &flash_attention_forward_varlen,
          "Flash Attention forward pass (variable length)",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("cu_seqlens"),
          py::arg("max_seqlen"),
          py::arg("softmax_scale") = 0.0f);
    
    // Normalization
    m.def("rmsnorm_forward", &rmsnorm_forward,
          "RMSNorm forward pass using SYCL",
          py::arg("input"),
          py::arg("weight"),
          py::arg("eps") = 1e-7);
    
    m.def("layernorm_forward", &layernorm_forward,
          "LayerNorm forward pass using SYCL",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("eps") = 1e-5);
    
    // Sparse Attention
    m.def("sparse_attention_forward", &sparse_attention_forward,
          "Sparse Attention forward pass",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("lut"),
          py::arg("topk"),
          py::arg("block_q") = 64,
          py::arg("block_k") = 64);
}
