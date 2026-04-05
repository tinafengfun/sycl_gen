#include "sparse_attention_kernels.hpp"
#include <cmath>
#include <algorithm>

namespace turbodiffusion {

// Helper constants
constexpr float LN2_INV = 1.4426950408889634f;  // 1 / ln(2)

// ============================================================================
// Forward Pass Kernel
// ============================================================================
template<int BLOCK_M, int BLOCK_N, int D>
void sparse_attn_fwd_impl(
    sycl::queue& queue,
    const bfloat16* Q, const bfloat16* K, const bfloat16* V,
    float softmax_scale,
    const int* LUT,
    bfloat16* O, float* LSE,
    int B, int H, int M_len, int N_len, int topk
) {
    // Calculate M_BLOCKS
    int M_BLOCKS = (M_len + BLOCK_M - 1) / BLOCK_M;
    int N_BLOCKS = (N_len + BLOCK_N - 1) / BLOCK_N;
    
    // Launch kernel with 2D grid: [M_BLOCKS, B * H]
    sycl::range<2> global_range(M_BLOCKS, B * H);
    sycl::range<2> local_range(1, 1);
    
    queue.parallel_for(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> item) {
        int idx_m = item.get_group(0);      // Query block index
        int idx_bh = item.get_group(1);     // Batch*Head index
        
        // Compute base offsets
        size_t qkv_offset = static_cast<size_t>(idx_bh) * M_len * D;
        size_t lut_offset = (static_cast<size_t>(idx_bh) * M_BLOCKS + idx_m) * topk;
        size_t lse_offset = static_cast<size_t>(idx_bh) * M_len;
        
        // Iterate over rows in this query block
        for (int row = 0; row < BLOCK_M; ++row) {
            int offs_m = idx_m * BLOCK_M + row;
            if (offs_m >= M_len) break;
            
            // Initialize running statistics for online softmax
            float m_i = -INFINITY;  // Running max
            float l_i = 0.0f;       // Running sum
            float o_acc[D];         // Output accumulator
            #pragma unroll
            for (int d = 0; d < D; ++d) o_acc[d] = 0.0f;
            
            // Load Q row
            float q_row[D];
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                q_row[d] = static_cast<float>(Q[qkv_offset + offs_m * D + d]);
            }
            
            // Iterate over key blocks from LUT
            for (int block_idx = 0; block_idx < topk; ++block_idx) {
                int idx_n = LUT[lut_offset + block_idx];
                int offs_n_start = idx_n * BLOCK_N;
                
                // Compute Q @ K^T for this key block
                float qk_scores[BLOCK_N];
                #pragma unroll
                for (int n = 0; n < BLOCK_N; ++n) {
                    if (offs_n_start + n >= N_len) {
                        qk_scores[n] = -INFINITY;
                        continue;
                    }
                    
                    float dot = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < D; ++d) {
                        float k_val = static_cast<float>(K[qkv_offset + (offs_n_start + n) * D + d]);
                        dot += q_row[d] * k_val;
                    }
                    // Apply scaling and log2 conversion
                    qk_scores[n] = dot * softmax_scale * LN2_INV;
                }
                
                // Online softmax update
                float local_m = -INFINITY;
                #pragma unroll
                for (int n = 0; n < BLOCK_N; ++n) {
                    if (offs_n_start + n < N_len) {
                        local_m = sycl::max(local_m, qk_scores[n]);
                    }
                }
                
                float new_m = sycl::max(m_i, local_m);
                
                // Compute exp2(qk - new_m)
                float exp_qk[BLOCK_N];
                float l_ij = 0.0f;
                #pragma unroll
                for (int n = 0; n < BLOCK_N; ++n) {
                    if (offs_n_start + n < N_len) {
                        exp_qk[n] = sycl::exp2(qk_scores[n] - new_m);
                        l_ij += exp_qk[n];
                    } else {
                        exp_qk[n] = 0.0f;
                    }
                }
                
                float alpha = sycl::exp2(m_i - new_m);
                
                // Scale previous output and accumulate
                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    o_acc[d] *= alpha;
                    
                    // Accumulate weighted values
                    #pragma unroll
                    for (int n = 0; n < BLOCK_N; ++n) {
                        if (offs_n_start + n < N_len) {
                            float v_val = static_cast<float>(V[qkv_offset + (offs_n_start + n) * D + d]);
                            o_acc[d] += exp_qk[n] * v_val;
                        }
                    }
                }
                
                l_i = l_i * alpha + l_ij;
                m_i = new_m;
            }
            
            // Normalize output
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                o_acc[d] /= l_i;
                O[qkv_offset + offs_m * D + d] = static_cast<bfloat16>(o_acc[d]);
            }
            
            // Store log-sum-exp: m_i + log2(l_i)
            LSE[lse_offset + offs_m] = m_i + sycl::log2(l_i);
        }
    });
}

void sparse_attn_fwd_kernel(
    sycl::queue& queue,
    const bfloat16* Q, const bfloat16* K, const bfloat16* V,
    float softmax_scale,
    const int* LUT,
    bfloat16* O, float* L, float* M,
    int B, int H, int M_len, int N_len, int D,
    int BLOCK_M, int BLOCK_N, int topk
) {
    // Note: L and M are both mapped to LSE (log-sum-exp) for forward pass
    // Dispatch based on BLOCK_M and D
    if (BLOCK_M == 64 && D == 64) {
        sparse_attn_fwd_impl<64, 64, 64>(queue, Q, K, V, softmax_scale, LUT, O, L, B, H, M_len, N_len, topk);
    } else if (BLOCK_M == 64 && D == 128) {
        sparse_attn_fwd_impl<64, 64, 128>(queue, Q, K, V, softmax_scale, LUT, O, L, B, H, M_len, N_len, topk);
    } else if (BLOCK_M == 128 && D == 64) {
        sparse_attn_fwd_impl<128, 64, 64>(queue, Q, K, V, softmax_scale, LUT, O, L, B, H, M_len, N_len, topk);
    } else if (BLOCK_M == 128 && D == 128) {
        sparse_attn_fwd_impl<128, 64, 128>(queue, Q, K, V, softmax_scale, LUT, O, L, B, H, M_len, N_len, topk);
    } else {
        // Default fallback
        sparse_attn_fwd_impl<64, 64, 64>(queue, Q, K, V, softmax_scale, LUT, O, L, B, H, M_len, N_len, topk);
    }
}

// ============================================================================
// Backward Preprocess Kernel
// ============================================================================
template<int BLOCK_M, int D>
void sparse_attn_bwd_preprocess_impl(
    sycl::queue& queue,
    const bfloat16* O, const bfloat16* dO,
    float* Delta,
    int B, int H, int L_len
) {
    int M_BLOCKS = (L_len + BLOCK_M - 1) / BLOCK_M;
    
    sycl::range<2> global_range(M_BLOCKS, B * H);
    sycl::range<2> local_range(1, 1);
    
    queue.parallel_for(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> item) {
        int idx_m = item.get_group(0);
        int idx_bh = item.get_group(1);
        
        size_t qkv_offset = static_cast<size_t>(idx_bh) * L_len * D;
        size_t delta_offset = static_cast<size_t>(idx_bh) * L_len;
        
        for (int row = 0; row < BLOCK_M; ++row) {
            int offs_m = idx_m * BLOCK_M + row;
            if (offs_m >= L_len) break;
            
            // Compute delta_s = sum(o_s * do_s, axis=1)
            float delta_val = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                float o_val = static_cast<float>(O[qkv_offset + offs_m * D + d]);
                float do_val = static_cast<float>(dO[qkv_offset + offs_m * D + d]);
                delta_val += o_val * do_val;
            }
            
            Delta[delta_offset + offs_m] = delta_val;
        }
    });
}

void sparse_attn_bwd_preprocess_kernel(
    sycl::queue& queue,
    const bfloat16* O, const bfloat16* dO,
    float* Delta,
    int B, int H, int L, int D,
    int BLOCK_M
) {
    if (BLOCK_M == 64 && D == 64) {
        sparse_attn_bwd_preprocess_impl<64, 64>(queue, O, dO, Delta, B, H, L);
    } else if (BLOCK_M == 64 && D == 128) {
        sparse_attn_bwd_preprocess_impl<64, 128>(queue, O, dO, Delta, B, H, L);
    } else if (BLOCK_M == 128 && D == 64) {
        sparse_attn_bwd_preprocess_impl<128, 64>(queue, O, dO, Delta, B, H, L);
    } else if (BLOCK_M == 128 && D == 128) {
        sparse_attn_bwd_preprocess_impl<128, 128>(queue, O, dO, Delta, B, H, L);
    } else {
        sparse_attn_bwd_preprocess_impl<64, 64>(queue, O, dO, Delta, B, H, L);
    }
}

// ============================================================================
// Backward dQ Kernel
// ============================================================================
template<int BLOCK_M, int BLOCK_N, int D>
void sparse_attn_bwd_dq_impl(
    sycl::queue& queue,
    const bfloat16* Q, const bfloat16* K, const bfloat16* V,
    const bfloat16* O, const bfloat16* dO,
    const int* LUT,
    const float* Delta, const float* LSE, const float* M,
    bfloat16* dQ,
    float qk_scale,
    int B, int H, int M_len, int N_len, int topk
) {
    int M_BLOCKS = (M_len + BLOCK_M - 1) / BLOCK_M;
    
    sycl::range<2> global_range(M_BLOCKS, B * H);
    sycl::range<2> local_range(1, 1);
    
    queue.parallel_for(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> item) {
        int idx_m = item.get_group(0);
        int idx_bh = item.get_group(1);
        
        size_t qkv_offset = static_cast<size_t>(idx_bh) * M_len * D;
        size_t lut_offset = (static_cast<size_t>(idx_bh) * M_BLOCKS + idx_m) * topk;
        size_t lse_offset = static_cast<size_t>(idx_bh) * M_len;
        
        for (int row = 0; row < BLOCK_M; ++row) {
            int offs_m = idx_m * BLOCK_M + row;
            if (offs_m >= M_len) break;
            
            // Load persistent data
            float q_row[D];
            float do_row[D];
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                q_row[d] = static_cast<float>(Q[qkv_offset + offs_m * D + d]);
                do_row[d] = static_cast<float>(dO[qkv_offset + offs_m * D + d]);
            }
            
            float delta_s = Delta[lse_offset + offs_m];
            float lse = LSE[lse_offset + offs_m];
            
            // Initialize dQ accumulator
            float dq_acc[D];
            #pragma unroll
            for (int d = 0; d < D; ++d) dq_acc[d] = 0.0f;
            
            // Iterate over key blocks from LUT
            for (int block_idx = 0; block_idx < topk; ++block_idx) {
                int idx_n = LUT[lut_offset + block_idx];
                int offs_n_start = idx_n * BLOCK_N;
                
                // Load K and V tiles
                float k_tile[BLOCK_N][D];
                float v_tile[BLOCK_N][D];
                #pragma unroll
                for (int n = 0; n < BLOCK_N; ++n) {
                    if (offs_n_start + n < N_len) {
                        #pragma unroll
                        for (int d = 0; d < D; ++d) {
                            k_tile[n][d] = static_cast<float>(K[qkv_offset + (offs_n_start + n) * D + d]);
                            v_tile[n][d] = static_cast<float>(V[qkv_offset + (offs_n_start + n) * D + d]);
                        }
                    } else {
                        #pragma unroll
                        for (int d = 0; d < D; ++d) {
                            k_tile[n][d] = 0.0f;
                            v_tile[n][d] = 0.0f;
                        }
                    }
                }
                
                // Recompute P = softmax(Q @ K^T)
                float p[BLOCK_N];
                #pragma unroll
                for (int n = 0; n < BLOCK_N; ++n) {
                    if (offs_n_start + n < N_len) {
                        float dot = 0.0f;
                        #pragma unroll
                        for (int d = 0; d < D; ++d) {
                            dot += q_row[d] * k_tile[n][d];
                        }
                        float qk = dot * qk_scale * LN2_INV;
                        p[n] = sycl::exp2(qk - lse);
                    } else {
                        p[n] = 0.0f;
                    }
                }
                
                // Compute dP = dO @ V^T
                float dp[BLOCK_N];
                #pragma unroll
                for (int n = 0; n < BLOCK_N; ++n) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < D; ++d) {
                        dot += do_row[d] * v_tile[n][d];
                    }
                    dp[n] = dot;
                }
                
                // Compute dS = P * (dP - delta)
                float ds[BLOCK_N];
                #pragma unroll
                for (int n = 0; n < BLOCK_N; ++n) {
                    ds[n] = p[n] * (dp[n] - delta_s);
                }
                
                // Accumulate dQ = dS @ K
                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    #pragma unroll
                    for (int n = 0; n < BLOCK_N; ++n) {
                        dq_acc[d] += ds[n] * k_tile[n][d];
                    }
                }
            }
            
            // Scale and store dQ
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                dq_acc[d] *= qk_scale;
                dQ[qkv_offset + offs_m * D + d] = static_cast<bfloat16>(dq_acc[d]);
            }
        }
    });
}

void sparse_attn_bwd_dq_kernel(
    sycl::queue& queue,
    const bfloat16* Q, const bfloat16* K, const bfloat16* V,
    const bfloat16* O, const bfloat16* dO,
    const int* LUT,
    const float* Delta, const float* L, const float* M,
    bfloat16* dQ,
    float qk_scale,
    int B, int H, int M_len, int N_len, int D,
    int BLOCK_M, int BLOCK_N, int topk
) {
    if (BLOCK_M == 64 && D == 64) {
        sparse_attn_bwd_dq_impl<64, 64, 64>(queue, Q, K, V, O, dO, LUT, Delta, L, M, dQ, qk_scale, B, H, M_len, N_len, topk);
    } else if (BLOCK_M == 64 && D == 128) {
        sparse_attn_bwd_dq_impl<64, 64, 128>(queue, Q, K, V, O, dO, LUT, Delta, L, M, dQ, qk_scale, B, H, M_len, N_len, topk);
    } else if (BLOCK_M == 128 && D == 64) {
        sparse_attn_bwd_dq_impl<128, 64, 64>(queue, Q, K, V, O, dO, LUT, Delta, L, M, dQ, qk_scale, B, H, M_len, N_len, topk);
    } else if (BLOCK_M == 128 && D == 128) {
        sparse_attn_bwd_dq_impl<128, 64, 128>(queue, Q, K, V, O, dO, LUT, Delta, L, M, dQ, qk_scale, B, H, M_len, N_len, topk);
    } else {
        sparse_attn_bwd_dq_impl<64, 64, 64>(queue, Q, K, V, O, dO, LUT, Delta, L, M, dQ, qk_scale, B, H, M_len, N_len, topk);
    }
}

// ============================================================================
// Backward dK/dV Kernel
// ============================================================================
template<int BLOCK_M, int BLOCK_N, int D, int BLOCK_SLICE_FACTOR>
void sparse_attn_bwd_dkdv_impl(
    sycl::queue& queue,
    const bfloat16* Q, const bfloat16* K, const bfloat16* V,
    const bfloat16* dO,
    const int8_t* KBID,
    const float* Delta, const float* LSE, const float* M,
    bfloat16* dK, bfloat16* dV,
    float qk_scale,
    int B, int H, int M_len, int N_len
) {
    constexpr int BLOCK_M2 = BLOCK_M / BLOCK_SLICE_FACTOR;
    int N_BLOCKS = (N_len + BLOCK_N - 1) / BLOCK_N;
    int M_BLOCKS = (M_len + BLOCK_M - 1) / BLOCK_M;
    
    sycl::range<2> global_range(N_BLOCKS, B * H);
    sycl::range<2> local_range(1, 1);
    
    queue.parallel_for(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> item) {
        int idx_n = item.get_group(0);
        int idx_bh = item.get_group(1);
        
        size_t qkv_offset = static_cast<size_t>(idx_bh) * M_len * D;
        size_t kbid_offset = static_cast<size_t>(idx_bh) * M_BLOCKS * N_BLOCKS;
        size_t lse_offset = static_cast<size_t>(idx_bh) * M_len;
        
        // Iterate over columns (key block rows)
        for (int col = 0; col < BLOCK_N; ++col) {
            int offs_n = idx_n * BLOCK_N + col;
            if (offs_n >= N_len) break;
            
            // Load persistent K and V
            float k_val[D];
            float v_val[D];
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                k_val[d] = static_cast<float>(K[qkv_offset + offs_n * D + d]);
                v_val[d] = static_cast<float>(V[qkv_offset + offs_n * D + d]);
            }
            
            // Initialize accumulators
            float dk_acc[D];
            float dv_acc[D];
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                dk_acc[d] = 0.0f;
                dv_acc[d] = 0.0f;
            }
            
            // Iterate over query blocks
            for (int idx_m_block = 0; idx_m_block < M_BLOCKS; ++idx_m_block) {
                // Check if this query block uses this key block
                int kbid_idx = idx_m_block;
                int8_t kbid = KBID[kbid_offset + kbid_idx * N_BLOCKS + idx_n];
                
                if (kbid != 1) continue;
                
                // Process this query block
                for (int slice = 0; slice < BLOCK_SLICE_FACTOR; ++slice) {
                    int idx_m = idx_m_block * BLOCK_M + slice * BLOCK_M2;
                    
                    // Load Q slice and related data
                    float q_slice[BLOCK_M2][D];
                    float lse_slice[BLOCK_M2];
                    float delta_slice[BLOCK_M2];
                    float do_slice[BLOCK_M2][D];
                    
                    #pragma unroll
                    for (int m = 0; m < BLOCK_M2; ++m) {
                        int global_m = idx_m + m;
                        if (global_m < M_len) {
                            #pragma unroll
                            for (int d = 0; d < D; ++d) {
                                q_slice[m][d] = static_cast<float>(Q[qkv_offset + global_m * D + d]);
                                do_slice[m][d] = static_cast<float>(dO[qkv_offset + global_m * D + d]);
                            }
                            lse_slice[m] = LSE[lse_offset + global_m];
                            delta_slice[m] = Delta[lse_offset + global_m];
                        } else {
                            #pragma unroll
                            for (int d = 0; d < D; ++d) {
                                q_slice[m][d] = 0.0f;
                                do_slice[m][d] = 0.0f;
                            }
                            lse_slice[m] = INFINITY;
                            delta_slice[m] = 0.0f;
                        }
                    }
                    
                    // Compute P^T = softmax(K @ Q^T)
                    float pt[BLOCK_M2];
                    #pragma unroll
                    for (int m = 0; m < BLOCK_M2; ++m) {
                        float dot = 0.0f;
                        #pragma unroll
                        for (int d = 0; d < D; ++d) {
                            dot += k_val[d] * q_slice[m][d];
                        }
                        float qk = dot * qk_scale * LN2_INV;
                        if (offs_n < N_len && (idx_m + m) < M_len) {
                            pt[m] = sycl::exp2(qk - lse_slice[m]);
                        } else {
                            pt[m] = 0.0f;
                        }
                    }
                    
                    // Compute dV += P^T @ dO
                    #pragma unroll
                    for (int d = 0; d < D; ++d) {
                        #pragma unroll
                        for (int m = 0; m < BLOCK_M2; ++m) {
                            dv_acc[d] += pt[m] * do_slice[m][d];
                        }
                    }
                    
                    // Compute dP^T = V @ dO^T
                    float dpt[BLOCK_M2];
                    #pragma unroll
                    for (int m = 0; m < BLOCK_M2; ++m) {
                        float dot = 0.0f;
                        #pragma unroll
                        for (int d = 0; d < D; ++d) {
                            dot += v_val[d] * do_slice[m][d];
                        }
                        dpt[m] = dot;
                    }
                    
                    // Compute dS^T = P^T * (dP^T - delta)
                    float dst[BLOCK_M2];
                    #pragma unroll
                    for (int m = 0; m < BLOCK_M2; ++m) {
                        dst[m] = pt[m] * (dpt[m] - delta_slice[m]);
                    }
                    
                    // Compute dK += dS^T @ Q
                    #pragma unroll
                    for (int d = 0; d < D; ++d) {
                        #pragma unroll
                        for (int m = 0; m < BLOCK_M2; ++m) {
                            dk_acc[d] += dst[m] * q_slice[m][d];
                        }
                    }
                }
            }
            
            // Scale dK and store results
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                dk_acc[d] *= qk_scale;
                dK[qkv_offset + offs_n * D + d] = static_cast<bfloat16>(dk_acc[d]);
                dV[qkv_offset + offs_n * D + d] = static_cast<bfloat16>(dv_acc[d]);
            }
        }
    });
}

void sparse_attn_bwd_dkdv_kernel(
    sycl::queue& queue,
    const bfloat16* Q, const bfloat16* K, const bfloat16* V,
    const bfloat16* dO,
    const int8_t* KBID,
    const float* Delta, const float* L, const float* M,
    bfloat16* dK, bfloat16* dV,
    float qk_scale,
    int B, int H, int M_len, int N_len, int D,
    int BLOCK_M, int BLOCK_N, int BLOCK_SLICE_FACTOR
) {
    // BLOCK_SLICE_FACTOR is typically BLOCK_M // 64
    if (BLOCK_M == 64 && D == 64 && BLOCK_SLICE_FACTOR == 1) {
        sparse_attn_bwd_dkdv_impl<64, 64, 64, 1>(queue, Q, K, V, dO, KBID, Delta, L, M, dK, dV, qk_scale, B, H, M_len, N_len);
    } else if (BLOCK_M == 64 && D == 128 && BLOCK_SLICE_FACTOR == 1) {
        sparse_attn_bwd_dkdv_impl<64, 64, 128, 1>(queue, Q, K, V, dO, KBID, Delta, L, M, dK, dV, qk_scale, B, H, M_len, N_len);
    } else if (BLOCK_M == 128 && D == 64 && BLOCK_SLICE_FACTOR == 2) {
        sparse_attn_bwd_dkdv_impl<128, 64, 64, 2>(queue, Q, K, V, dO, KBID, Delta, L, M, dK, dV, qk_scale, B, H, M_len, N_len);
    } else if (BLOCK_M == 128 && D == 128 && BLOCK_SLICE_FACTOR == 2) {
        sparse_attn_bwd_dkdv_impl<128, 64, 128, 2>(queue, Q, K, V, dO, KBID, Delta, L, M, dK, dV, qk_scale, B, H, M_len, N_len);
    } else {
        // Default fallback to 64, 64, 64, 1
        sparse_attn_bwd_dkdv_impl<64, 64, 64, 1>(queue, Q, K, V, dO, KBID, Delta, L, M, dK, dV, qk_scale, B, H, M_len, N_len);
    }
}

} // namespace turbodiffusion

// ============================================================================
// PyTorch Bindings
// ============================================================================

#include <torch/extension.h>

// Global SYCL queue (defined in sycl_ops.cpp)
extern sycl::queue& get_sycl_queue();

using bf16 = sycl::ext::oneapi::bfloat16;
using bfloat16 = sycl::ext::oneapi::bfloat16;

/**
 * Sparse Attention forward pass - PyTorch wrapper
 */
torch::Tensor sparse_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor lut,
    int topk,
    int block_q,
    int block_k
) {
    // Validate inputs
    TORCH_CHECK(query.device().is_xpu(), "Query must be on XPU");
    TORCH_CHECK(key.device().is_xpu(), "Key must be on XPU");
    TORCH_CHECK(value.device().is_xpu(), "Value must be on XPU");
    TORCH_CHECK(lut.device().is_xpu(), "LUT must be on XPU");
    
    // Get dimensions
    int B = query.size(0);
    int L = query.size(1);
    int H = query.size(2);
    int D = query.size(3);
    
    // Create output tensor
    auto output = torch::empty_like(query);
    
    // Create LSE (log-sum-exp) tensor
    auto lse = torch::empty({B * H, L}, torch::dtype(torch::kFloat32).device(query.device()));
    
    // Get SYCL queue
    sycl::queue& q = get_sycl_queue();
    
    // Get pointers
    bf16* q_ptr = reinterpret_cast<bf16*>(query.data_ptr<at::BFloat16>());
    bf16* k_ptr = reinterpret_cast<bf16*>(key.data_ptr<at::BFloat16>());
    bf16* v_ptr = reinterpret_cast<bf16*>(value.data_ptr<at::BFloat16>());
    bf16* o_ptr = reinterpret_cast<bf16*>(output.data_ptr<at::BFloat16>());
    float* lse_ptr = lse.data_ptr<float>();
    int* lut_ptr = lut.data_ptr<int32_t>();
    
    // Compute softmax scale
    float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
    
    // For simplicity, call the kernel directly with fixed block size
    // In production, you'd want to select optimal block sizes based on dimensions
    if (D == 64) {
        turbodiffusion::sparse_attn_fwd_impl<64, 64, 64>(
            q, q_ptr, k_ptr, v_ptr, softmax_scale, lut_ptr, o_ptr, lse_ptr,
            B, H, L, L, topk
        );
    } else if (D == 128) {
        turbodiffusion::sparse_attn_fwd_impl<64, 64, 128>(
            q, q_ptr, k_ptr, v_ptr, softmax_scale, lut_ptr, o_ptr, lse_ptr,
            B, H, L, L, topk
        );
    } else {
        // Default to 64 for other dimensions
        turbodiffusion::sparse_attn_fwd_impl<64, 64, 64>(
            q, q_ptr, k_ptr, v_ptr, softmax_scale, lut_ptr, o_ptr, lse_ptr,
            B, H, L, L, topk
        );
    }
    
    return output;
}