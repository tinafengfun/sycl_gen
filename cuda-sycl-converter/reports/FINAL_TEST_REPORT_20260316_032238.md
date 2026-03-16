# CUDA to SYCL Kernel Conversion Test Results

**Test Date:** 2026-03-16 03:22:38

## Summary

- **Total Kernels:** 28
- **Passed:** 0 (0.0%)
- **Failed:** 28 (100.0%)
- **Execution Time:** 192.8s (3.2 min)

## Detailed Results

| Kernel | Status | MAE | Max Error | CUDA(s) | SYCL(s) |
|--------|--------|-----|-----------|---------|---------|
| add_bias_batched | ❌ COMPARE_FAILED | - | - | 3.4 | 3.5 |
| add_bias_nchw | ❌ COMPARE_FAILED | - | - | 3.2 | 3.5 |
| add_vectors | ❌ COMPARE_FAILED | - | - | 3.2 | 3.5 |
| add_vectors_hnc_nhc | ❌ COMPARE_FAILED | - | - | 3.1 | 3.5 |
| batch_norm | ❌ COMPARE_FAILED | - | - | 3.3 | 3.7 |
| copy_type_converted | ❌ COMPARE_FAILED | - | - | 3.6 | 3.5 |
| expand_planes_nchw | ❌ COMPARE_FAILED | - | - | 3.2 | 3.5 |
| expand_planes_nhwc | ❌ COMPARE_FAILED | - | - | 3.2 | 3.6 |
| gen_offset_pointers | ❌ COMPARE_FAILED | - | - | 3.1 | 3.3 |
| global_avg_pool | ❌ COMPARE_FAILED | - | - | 3.6 | 3.5 |
| global_avg_pool_nhwc_fp16 | ❌ COMPARE_FAILED | - | - | 3.7 | 3.5 |
| global_scale | ❌ COMPARE_FAILED | - | - | 3.3 | 3.7 |
| global_scale_fp16_nhwc | ❌ COMPARE_FAILED | - | - | 3.5 | 3.7 |
| input_gating | ❌ COMPARE_FAILED | - | - | 3.3 | 3.6 |
| layer_norm | ❌ COMPARE_FAILED | - | - | 3.3 | 3.4 |
| nchw_to_nhwc | ❌ COMPARE_FAILED | - | - | 3.3 | 3.5 |
| output_input_transform_fp16_shmem | ❌ COMPARE_FAILED | - | - | 3.5 | 3.6 |
| policy_map | ❌ COMPARE_FAILED | - | - | 3.5 | 3.5 |
| preprocess_attention_body | ❌ COMPARE_FAILED | - | - | 3.4 | 3.6 |
| promotion_logits | ❌ COMPARE_FAILED | - | - | 3.3 | 3.3 |
| se_layer_nhwc | ❌ COMPARE_FAILED | - | - | 3.4 | 3.6 |
| softmax | ❌ COMPARE_FAILED | - | - | 3.3 | 3.7 |
| softmax_opt_64 | ❌ COMPARE_FAILED | - | - | 3.4 | 3.7 |
| winograd_filter_transform | ❌ COMPARE_FAILED | - | - | 3.4 | 3.5 |
| winograd_input_transform | ❌ COMPARE_FAILED | - | - | 3.2 | 3.5 |
| winograd_output_relu_input | ❌ COMPARE_FAILED | - | - | 3.3 | 3.5 |
| winograd_output_se_relu_input | ❌ COMPARE_FAILED | - | - | 3.3 | 3.5 |
| winograd_output_transform | ❌ COMPARE_FAILED | - | - | 3.4 | 3.5 |