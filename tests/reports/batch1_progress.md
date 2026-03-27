# Batch 1: Type D Matrix Kernels - Progress Report

## Kernel 1: test_se_layer_nhwc ✅

### Results Summary

| Version | N=64 | N=128 | N=256 | Best?
|---------|------|-------|-------|-------|
| V0 (baseline) | 0.94 GFLOPS | 1.17 GFLOPS | 1.17 GFLOPS | ❌ |
| V1 (unroll 64) | 3.18 GFLOPS | 3.30 GFLOPS | 3.34 GFLOPS | ❌ |
| V2 (single-wi) | **7.54 GFLOPS** | **16.34 GFLOPS** | **21.10 GFLOPS** | ✅ |
| V3 (XMX) | 1.66 GFLOPS | 1.68 GFLOPS | 1.75 GFLOPS | ❌ |

### Key Findings

1. **V2 Single Work-Item per Batch is OPTIMAL**
   - Achieves 18x speedup over baseline at N=256
   - 3.4x faster than XMX version
   
2. **XMX Not Effective for Small Matrices**
   - SE layer uses C=128, se_K=64 (too small for XMX efficiency)
   - XMX requires matrices >256x256 for peak performance
   - Overhead of XMX tile management exceeds benefits

3. **Optimization Strategy Validated**
   - Single-thread per output pattern works best for this architecture
   - Similar to global_avg_pool results (V2 was 60% faster)

### Recommendation
- **Use V2 (single work-item per batch)** for SE layer
- Skip XMX for small matrix operations (<256 dimensions)
- Apply single-thread pattern to other Type D kernels with small matrices

## Next: Kernel 2 - test_hard_fused_kernel

This kernel likely has larger matrix operations suitable for XMX.

## Progress
- ✅ 1/8 Type D kernels optimized
- ⏳ 7/8 Type D kernels remaining
- ⏳ 20/28 total kernels remaining

## Files
- Baseline: `test_se_layer_nhwc.cpp`
- XMX version: `test_se_layer_nhwc_xmx.cpp`
- Results: `se_layer_nhwc_results.csv`, `se_layer_nhwc_xmx_results.csv`
