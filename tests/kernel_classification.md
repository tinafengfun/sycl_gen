# Kernel Classification for Batch Optimization

## Classification Criteria
- **Type A (Element-wise)**: Simple point-wise operations, memory bandwidth bound
- **Type B (Winograd/Spatial)**: Spatial transformations, medium compute intensity  
- **Type C (Reduction)**: Pooling, softmax, reduction operations
- **Type D (Matrix)**: GEMM, attention, SE layer - MUST use XMX

## Kernel List (28 Unique Kernels)

### Type A - Element-wise (6 kernels)
- Round 1 only, expect <10% improvement between rounds
1. test_add_vectors
2. test_add_vectors_hnc_nhc
3. test_add_bias_batched
4. test_add_bias_nchw
5. test_expand_planes_fp16_nhwc
6. test_expand_planes_nchw
7. test_expand_planes_nhwc
8. test_global_scale
9. test_global_scale_fp16_nhwc
10. test_copy_type_converted

### Type B - Winograd/Spatial (4 kernels)
- Rounds 1-3, expect 40-60% gains
1. test_winograd_filter_transform
2. test_winograd_input_transform
3. test_winograd_input
4. test_winograd_output_relu_input

### Type C - Reduction (6 kernels)
- Single-thread per output optimal, expect 50-70% gains
1. test_global_avg_pool_nhwc_fp16
2. test_global_avg_pool_real
3. test_softmax_real
4. test_softmax_v0
5. test_softmax_v1
6. test_layer_norm
7. test_hard_batch_norm

### Type D - Matrix/XMX (8 kernels)
- MUST use XMX joint_matrix, expect 100+ TFLOPS
1. test_se_layer_nhwc
2. test_hard_fused_kernel
3. test_nchw_to_nhwc
4. test_policy_map
5. test_gemm_aot
6. test_gemm_large
7. test_gemm_onednn
8. test_winograd_real

## Optimization Strategy

### Type A: Quick Pass (Round 1 only)
- Focus: Vectorized loads/stores, proper work-group size
- Stop if speedup < 15%

### Type B: Full Optimization (Rounds 1-2)
- Focus: Tile optimization, SLM usage
- Test multiple tile sizes (8x8, 16x16, 32x32)

### Type C: Single-thread Focus (Rounds 1-2)
- Focus: Single-thread per output element
- Reduce atomic operations

### Type D: XMX Mandatory (Rounds 1-3)
- Focus: joint_matrix API, 8x16x16 tiles
- AOT compilation required
- Subgroup size 16 mandatory

## Execution Priority
1. **Type D first** (highest impact, TFLOPS level)
2. **Type C second** (60% gains proven)
3. **Type B third** (proven gains)
4. **Type A last** (minimal gains, quick pass)

## Batch Execution Plan

### Batch 1: Type D Matrix Kernels (8 kernels)
Status: READY TO START
Expected: 100+ TFLOPS with XMX
Priority: HIGH

### Batch 2: Type C Reduction Kernels (7 kernels)
Status: PENDING
Expected: 50-70% speedup
Priority: HIGH

### Batch 3: Type B Winograd Kernels (4 kernels)
Status: PENDING
Expected: 40-60% speedup
Priority: MEDIUM

### Batch 4: Type A Element-wise Kernels (10 kernels)
Status: PENDING
Expected: <15% speedup
Priority: LOW

## Notes
- Already tested: test_add_vectors (Type A), test_winograd_filter_transform (Type B), test_global_avg_pool_nhwc_fp16 (Type C)
- XMX variants (*_xmx.cpp) are already optimized, can skip or use as reference
- Total unique kernels to optimize: 28
