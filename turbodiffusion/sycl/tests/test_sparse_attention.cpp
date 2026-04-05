#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include "../operators/sparse_attention_kernels.hpp"

using namespace turbodiffusion;

// Helper function to compare floating point arrays
bool compare_arrays(const float* a, const float* b, size_t size, float tolerance = 1e-3) {
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Reference PyTorch-like implementation for forward pass
void reference_forward(
    const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
    const std::vector<int>& LUT,
    std::vector<float>& O_ref, std::vector<float>& LSE_ref,
    int B, int H, int M, int N, int D, int topk,
    int BLOCK_M, int BLOCK_N, float softmax_scale
) {
    int M_BLOCKS = (M + BLOCK_M - 1) / BLOCK_M;
    int N_BLOCKS = (N + BLOCK_N - 1) / BLOCK_N;
    
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            size_t bh_offset = ((size_t)b * H + h) * M * D;
            size_t lut_bh_offset = ((size_t)b * H + h) * M_BLOCKS * topk;
            size_t lse_bh_offset = ((size_t)b * H + h) * M;
            
            for (int m = 0; m < M; ++m) {
                // Compute Q @ K^T for sparse blocks
                std::vector<float> qk_scores;
                std::vector<int> key_indices;
                
                int m_block = m / BLOCK_M;
                size_t lut_offset = lut_bh_offset + m_block * topk;
                
                for (int t = 0; t < topk; ++t) {
                    int n_block = LUT[lut_offset + t];
                    for (int n = 0; n < BLOCK_N; ++n) {
                        int key_idx = n_block * BLOCK_N + n;
                        if (key_idx >= N) continue;
                        
                        float dot = 0.0f;
                        for (int d = 0; d < D; ++d) {
                            dot += Q[bh_offset + m * D + d] * 
                                   K[((size_t)b * H + h) * N * D + key_idx * D + d];
                        }
                        qk_scores.push_back(dot * softmax_scale);
                        key_indices.push_back(key_idx);
                    }
                }
                
                // Softmax
                float max_score = -INFINITY;
                for (float score : qk_scores) {
                    max_score = std::max(max_score, score);
                }
                
                float sum_exp = 0.0f;
                for (float& score : qk_scores) {
                    score = std::exp(score - max_score);
                    sum_exp += score;
                }
                
                for (float& score : qk_scores) {
                    score /= sum_exp;
                }
                
                // Apply to values
                for (int d = 0; d < D; ++d) {
                    float val = 0.0f;
                    for (size_t i = 0; i < qk_scores.size(); ++i) {
                        val += qk_scores[i] * 
                               V[((size_t)b * H + h) * N * D + key_indices[i] * D + d];
                    }
                    O_ref[bh_offset + m * D + d] = val;
                }
                
                LSE_ref[lse_bh_offset + m] = max_score + std::log(sum_exp);
            }
        }
    }
}

// Test forward pass
bool test_forward_pass() {
    std::cout << "\n=== Testing Forward Pass ===" << std::endl;
    
    // Small test case
    const int B = 1, H = 1, M = 64, N = 64, D = 64;
    const int BLOCK_M = 64, BLOCK_N = 64, topk = 1;
    const float softmax_scale = 1.0f / std::sqrt(D);
    
    // Create SYCL queue
    sycl::queue queue(sycl::default_selector_v);
    std::cout << "Using device: " 
              << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    
    // Allocate host memory
    size_t qkv_size = B * H * M * D;
    size_t lut_size = B * H * 1 * topk;  // M_BLOCKS = 1
    size_t lse_size = B * H * M;
    
    std::vector<bfloat16> Q_host(qkv_size), K_host(qkv_size), V_host(qkv_size);
    std::vector<int> LUT_host(lut_size);
    std::vector<bfloat16> O_host(qkv_size);
    std::vector<float> LSE_host(lse_size);
    std::vector<float> M_host(lse_size);
    
    // Initialize with random values
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    for (size_t i = 0; i < qkv_size; ++i) {
        Q_host[i] = bfloat16(dist(gen));
        K_host[i] = bfloat16(dist(gen));
        V_host[i] = bfloat16(dist(gen));
    }
    
    // Initialize LUT with simple diagonal pattern
    for (size_t i = 0; i < lut_size; ++i) {
        LUT_host[i] = 0;  // Only attend to first block
    }
    
    // Allocate device memory
    bfloat16 *Q_dev, *K_dev, *V_dev, *O_dev;
    int* LUT_dev;
    float *LSE_dev, *M_dev;
    
    Q_dev = sycl::malloc_device<bfloat16>(qkv_size, queue);
    K_dev = sycl::malloc_device<bfloat16>(qkv_size, queue);
    V_dev = sycl::malloc_device<bfloat16>(qkv_size, queue);
    O_dev = sycl::malloc_device<bfloat16>(qkv_size, queue);
    LUT_dev = sycl::malloc_device<int>(lut_size, queue);
    LSE_dev = sycl::malloc_device<float>(lse_size, queue);
    M_dev = sycl::malloc_device<float>(lse_size, queue);
    
    // Copy to device
    queue.memcpy(Q_dev, Q_host.data(), qkv_size * sizeof(bfloat16));
    queue.memcpy(K_dev, K_host.data(), qkv_size * sizeof(bfloat16));
    queue.memcpy(V_dev, V_host.data(), qkv_size * sizeof(bfloat16));
    queue.memcpy(LUT_dev, LUT_host.data(), lut_size * sizeof(int));
    queue.wait();
    
    // Launch kernel
    sparse_attn_fwd_kernel(
        queue,
        Q_dev, K_dev, V_dev,
        softmax_scale,
        LUT_dev,
        O_dev, LSE_dev, M_dev,
        B, H, M, N, D,
        BLOCK_M, BLOCK_N, topk
    );
    queue.wait();
    
    // Copy back
    queue.memcpy(O_host.data(), O_dev, qkv_size * sizeof(bfloat16));
    queue.memcpy(LSE_host.data(), LSE_dev, lse_size * sizeof(float));
    queue.wait();
    
    // Compute reference
    std::vector<float> Q_float(qkv_size), K_float(qkv_size), V_float(qkv_size);
    for (size_t i = 0; i < qkv_size; ++i) {
        Q_float[i] = static_cast<float>(Q_host[i]);
        K_float[i] = static_cast<float>(K_host[i]);
        V_float[i] = static_cast<float>(V_host[i]);
    }
    
    std::vector<float> O_ref(qkv_size);
    std::vector<float> LSE_ref(lse_size);
    
    reference_forward(
        Q_float, K_float, V_float, LUT_host,
        O_ref, LSE_ref,
        B, H, M, N, D, topk,
        BLOCK_M, BLOCK_N, softmax_scale
    );
    
    // Compare results
    std::vector<float> O_float(qkv_size);
    for (size_t i = 0; i < qkv_size; ++i) {
        O_float[i] = static_cast<float>(O_host[i]);
    }
    
    bool output_match = compare_arrays(O_float.data(), O_ref.data(), qkv_size, 1e-2);
    bool lse_match = compare_arrays(LSE_host.data(), LSE_ref.data(), lse_size, 1e-2);
    
    // Cleanup
    sycl::free(Q_dev, queue);
    sycl::free(K_dev, queue);
    sycl::free(V_dev, queue);
    sycl::free(O_dev, queue);
    sycl::free(LUT_dev, queue);
    sycl::free(LSE_dev, queue);
    sycl::free(M_dev, queue);
    
    std::cout << "Output match: " << (output_match ? "PASS" : "FAIL") << std::endl;
    std::cout << "LSE match: " << (lse_match ? "PASS" : "FAIL") << std::endl;
    
    return output_match && lse_match;
}

// Test backward preprocess
bool test_backward_preprocess() {
    std::cout << "\n=== Testing Backward Preprocess ===" << std::endl;
    
    const int B = 1, H = 1, L = 64, D = 64;
    const int BLOCK_M = 64;
    
    sycl::queue queue(sycl::default_selector_v);
    
    size_t qkv_size = B * H * L * D;
    size_t delta_size = B * H * L;
    
    std::vector<bfloat16> O_host(qkv_size), dO_host(qkv_size);
    std::vector<float> Delta_host(delta_size);
    
    // Initialize
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    for (size_t i = 0; i < qkv_size; ++i) {
        O_host[i] = bfloat16(dist(gen));
        dO_host[i] = bfloat16(dist(gen));
    }
    
    bfloat16 *O_dev, *dO_dev;
    float* Delta_dev;
    
    O_dev = sycl::malloc_device<bfloat16>(qkv_size, queue);
    dO_dev = sycl::malloc_device<bfloat16>(qkv_size, queue);
    Delta_dev = sycl::malloc_device<float>(delta_size, queue);
    
    queue.memcpy(O_dev, O_host.data(), qkv_size * sizeof(bfloat16));
    queue.memcpy(dO_dev, dO_host.data(), qkv_size * sizeof(bfloat16));
    queue.wait();
    
    sparse_attn_bwd_preprocess_kernel(
        queue,
        O_dev, dO_dev,
        Delta_dev,
        B, H, L, D, BLOCK_M
    );
    queue.wait();
    
    queue.memcpy(Delta_host.data(), Delta_dev, delta_size * sizeof(float));
    queue.wait();
    
    // Verify
    std::vector<float> Delta_ref(delta_size);
    for (int i = 0; i < L; ++i) {
        float sum = 0.0f;
        for (int d = 0; d < D; ++d) {
            sum += static_cast<float>(O_host[i * D + d]) * 
                   static_cast<float>(dO_host[i * D + d]);
        }
        Delta_ref[i] = sum;
    }
    
    bool match = compare_arrays(Delta_host.data(), Delta_ref.data(), delta_size, 1e-3);
    
    sycl::free(O_dev, queue);
    sycl::free(dO_dev, queue);
    sycl::free(Delta_dev, queue);
    
    std::cout << "Delta match: " << (match ? "PASS" : "FAIL") << std::endl;
    
    return match;
}

int main() {
    std::cout << "Sparse Attention SYCL Kernel Tests" << std::endl;
    std::cout << "===================================" << std::endl;
    
    try {
        bool forward_pass = test_forward_pass();
        bool backward_preprocess = test_backward_preprocess();
        
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Forward Pass: " << (forward_pass ? "PASS" : "FAIL") << std::endl;
        std::cout << "Backward Preprocess: " << (backward_preprocess ? "PASS" : "FAIL") << std::endl;
        
        return (forward_pass && backward_preprocess) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}