# CUDA to SYCL Kernel Conversion Test Results

**Test Date:** 2026-03-16 03:28:03

## Summary

- **Total Kernels:** 28
- **Passed:** 25 (89.3%)
- **Failed:** 3 (10.7%)
- **Execution Time:** 193.7s (3.2 min)

## Detailed Results

| Kernel | Status | MAE | Max Error | CUDA(s) | SYCL(s) |
|--------|--------|-----|-----------|---------|---------|
| add_bias_batched | ✅ PASS | 9.65e-09 | 5.96e-08 | 3.4 | 3.6 |
| add_bias_nchw | ✅ PASS | 1.66e-08 | 1.19e-07 | 3.5 | 3.5 |
| add_vectors | ✅ PASS | 1.36e-07 | 1.91e-06 | 3.3 | 3.5 |
| add_vectors_hnc_nhc | ✅ PASS | 2.26e-08 | 1.19e-07 | 3.3 | 3.6 |
| batch_norm | ✅ PASS | 1.92e-08 | 2.38e-07 | 3.2 | 3.7 |
| copy_type_converted | ✅ PASS | - | - | 3.4 | 3.5 |
| expand_planes_nchw | ✅ PASS | - | - | 3.5 | 3.5 |
| expand_planes_nhwc | ✅ PASS | - | - | 3.3 | 4.0 |
| gen_offset_pointers | ✅ PASS | - | - | 3.1 | 3.3 |
| global_avg_pool | ✅ PASS | 2.24e-08 | 5.96e-08 | 3.3 | 3.5 |
| global_avg_pool_nhwc_fp16 | ✅ PASS | 8.55e-05 | 2.33e-04 | 3.6 | 3.6 |
| global_scale | ✅ PASS | 5.73e-09 | 2.98e-08 | 3.3 | 3.6 |
| global_scale_fp16_nhwc | ✅ PASS | - | - | 3.6 | 3.7 |
| input_gating | ❌ FAILED | 5.53e-01 | 1.24e+00 | 3.3 | 3.6 |
| layer_norm | ✅ PASS | 1.48e-07 | 7.15e-07 | 3.2 | 3.4 |
| nchw_to_nhwc | ✅ PASS | 1.21e-08 | 5.96e-08 | 3.5 | 3.5 |
| output_input_transform_fp16_shmem | ✅ PASS | - | - | 3.5 | 3.5 |
| policy_map | ✅ PASS | 1.21e-08 | 5.96e-08 | 3.4 | 3.5 |
| preprocess_attention_body | ✅ PASS | 1.24e-08 | 5.96e-08 | 3.4 | 3.6 |
| promotion_logits | ✅ PASS | 2.50e-06 | 1.14e-05 | 3.3 | 3.3 |
| se_layer_nhwc | ❌ FAILED | 1.62e-04 | 8.89e-04 | 3.5 | 3.6 |
| softmax | ✅ PASS | 1.32e-09 | 3.73e-09 | 3.2 | 3.7 |
| softmax_opt_64 | ❌ FAILED | 1.46e-02 | 2.39e-02 | 3.3 | 3.7 |
| winograd_filter_transform | ✅ PASS | 1.21e-08 | 5.96e-08 | 3.4 | 3.5 |
| winograd_input_transform | ✅ PASS | 1.21e-08 | 5.96e-08 | 3.4 | 3.5 |
| winograd_output_relu_input | ✅ PASS | 8.91e-09 | 5.96e-08 | 3.3 | 3.5 |
| winograd_output_se_relu_input | ✅ PASS | 8.91e-09 | 5.96e-08 | 3.2 | 3.5 |
| winograd_output_transform | ✅ PASS | 1.13e-08 | 1.19e-07 | 3.4 | 3.5 |