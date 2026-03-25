# GPU Kernel Benchmark Results

**Platform:** Intel Graphics [0xe211] (Battlemage G21)  
**Test Date:** 2026-03-24  
**Total Kernels:** 23  
**Test Configurations:** 69+ version combinations  
**Status:** ✅ Complete

---

## Executive Summary

Comprehensive performance testing of all 23 SYCL kernels from the LCZero neural network backend on Intel Battlemage G21 GPU.

### Peak Performances Achieved

| Metric | Value | Kernel |
|--------|-------|--------|
| **Peak GFLOPS** | 767 GFLOPS | winograd_output_relu_input |
| **Best Filter Transform** | 446 GFLOPS | winograd_filter_transform |
| **Best Memory Bandwidth** | 338 GB/s | copy_type_converted |
| **Best Speedup** | +1331% | se_layer_nhwc (single-thread vs multi-thread) |
| **Optimal WG Size** | 128 | 9/10 element-wise kernels |

---

## All 23 Kernels - Performance Summary

### Category 1: Winograd Transforms (4 kernels)

| Kernel | Type | Peak GFLOPS | Best Config | Key Finding |
|--------|------|-------------|-------------|-------------|
| [winograd_output_relu_input](results/winograd_output_relu_input_results.csv) | Fused Transform | **767** | V0 (1D, WG=256) | Simple 1D best for ReLU fusion |
| [winograd_filter_transform](results/winograd_filter_transform_results.csv) | Filter Transform | **446** | V0 (1D, WG=256) | 1D beats 2D by 35% |
| [winograd_real](results/winograd_real_results.csv) | Output Transform | 156 | V0 (3D, 16×4×4) | 3D essential for spatial data |
| [winograd_input_transform](results/winograd_input_transform_results.csv) | Input Transform | 85 | V1 (3D+unroll) | Loop unroll gives +9% |

**Optimization Insights:**
- Simple fused operations (ReLU) prefer 1D work-groups
- Filter transforms (compact data) should use 1D, not 2D/3D
- Spatial transforms (input/output) benefit from 3D topology

---

### Category 2: Element-wise Operations (10 kernels)

| Kernel | Type | Peak Perf | Best WG | Notes |
|--------|------|-----------|---------|-------|
| [global_scale](results/global_scale_results.csv) | Element-wise | 200 GFLOPS | V1 (128) | Consistently optimal |
| [add_bias_batched](results/add_bias_batched_results.csv) | Bias Add | 32.43 GFLOPS | V2 (128, unroll) | Grid-stride +51% |
| [add_bias_nchw](results/add_bias_nchw_results.csv) | Bias Add | 26.82 GFLOPS | V2 (128, unroll) | Grid-stride +15% |
| [add_vectors](results/add_vectors_results.csv) | Vector Add | 4.29 GFLOPS | V1 (128) | WG=128 optimal |
| [copy_type_converted](results/copy_type_converted_results.csv) | Memory Copy | 338 GB/s | V1 (128) | Memory bandwidth bound |
| [global_scale_fp16_nhwc](results/global_scale_fp16_nhwc_results.csv) | FP16 Scale | 117 GB/s | V0/V1 (128/256) | Similar to FP32 |
| [expand_planes_nhwc](results/expand_planes_nhwc_results.csv) | Data Expansion | 160 GB/s | V1 (128) | Best for small sizes |
| [expand_planes_nchw](results/expand_planes_nchw_results.csv) | Data Expansion | 20 GB/s | V2 (4x/thread) | Multi-element/thread +5% |
| [expand_planes_fp16_nhwc](results/expand_planes_fp16_nhwc_results.csv) | FP16 Expansion | 5 GB/s | V2 (128) | Limited by indexing |
| [add_vectors_hnc_nhc](results/add_vectors_hnc_nhc_results.csv) | Layout Transform | 1.13 GFLOPS | V0 (256) | Problem-size dependent |

**Optimization Insights:**
- WG=128 consistently wins for element-wise (9/10 kernels)
- FP16 achieves similar compute to FP32, ~50% bandwidth
- Multi-element processing (4x/thread) benefits expansion kernels

---

### Category 3: Normalization (2 kernels)

| Kernel | Type | Peak GFLOPS | Best Version | Key Technique |
|--------|------|-------------|--------------|---------------|
| [hard_batch_norm](results/hard_batch_norm_results.csv) | Batch Norm | 145 GFLOPS | V1 | Loop unrolling |
| [layer_norm](results/layer_norm_results.csv) | Layer Norm | 16.86 GFLOPS | V1 (128, unroll) | Loop unroll +82% |

**Optimization Insights:**
- Loop unrolling consistently helps normalization kernels
- WG=128 optimal for layer_norm
- Channel-wise operations benefit from aggressive unrolling

---

### Category 4: Reduction Operations (3 kernels)

| Kernel | Type | Peak GFLOPS | Best Config | Technique |
|--------|------|-------------|-------------|-----------|
| [global_avg_pool_real](results/global_avg_pool_real_results.csv) | Avg Pool | 62.54 GFLOPS | V2 (256) | Vectorization +10% |
| [global_avg_pool_nhwc_fp16](results/global_avg_pool_nhwc_fp16_results.csv) | FP16 Pool | 57 GFLOPS | V2 (single-thread) | Single-thread best |
| [softmax_real](results/softmax_real_results.csv) | Softmax | 10.98 GFLOPS | V0 (256) | Baseline optimal |

**Optimization Insights:**
- Single-thread mode beats collaborative reduction for FP16
- Vectorization helps FP32 reduction kernels
- WG=256 provides good balance for reduction operations

---

### Category 5: Complex Fused (2 kernels)

| Kernel | Type | Peak GFLOPS | Best Config | Speedup |
|--------|------|-------------|-------------|---------|
| [se_layer_nhwc](results/se_layer_nhwc_results.csv) | SE Layer | 20.60 GFLOPS | V2 (single-thread) | **+1331%** vs V0 |
| [hard_fused_kernel](results/hard_fused_kernel_results.csv) | Fused Winograd | 87.35 GFLOPS | V1 (unroll) | +447% vs baseline |

**Optimization Insights:**
- **Single-thread mode is transformative** for complex kernels
- Avoid multi-thread collaboration with barriers (99% slowdown)
- Aggressive loop unrolling essential for fused operations

---

### Category 6: Layout/Gather (2 kernels)

| Kernel | Type | Peak Bandwidth | Best Config | Notes |
|--------|------|----------------|-------------|-------|
| [nchw_to_nhwc](results/nchw_to_nhwc_results.csv) | Layout Transform | 232 GB/s | V1 (128, medium) | Size-dependent |
| [policy_map](results/policy_map_results.csv) | Gather | 30 GB/s | V2 (128) | Indexing limited |

**Optimization Insights:**
- Layout transforms are memory-bandwidth limited
- Indirect indexing (gather) prevents coalesced access (~30 GB/s)
- Optimal WG size depends on problem size

---

## Test Methodology

### Configuration Matrix

Each kernel tested with:
- **3 versions:** V0 (baseline), V1 (optimized), V2 (alternative)
- **4-5 problem sizes:** Small to large workloads
- **50-100 iterations:** For stable timing
- **Warmup:** 5 iterations before measurement

### Performance Metrics

**GFLOPS Calculation:**
```
GFLOPS = (Total Operations) / (Time in seconds) / 1e9
```

**Bandwidth Calculation:**
```
Bandwidth (GB/s) = (Bytes Read + Bytes Written) / Time / 1e9
```

### Hardware Specs

- **GPU:** Intel Graphics [0xe211]
- **Architecture:** Xe2 (Battlemage G21)
- **Sub-group:** 16 (fixed)
- **SLM:** 128 KB per work-group
- **Max Work-group:** 1024 threads
- **Compiler:** icpx -fsycl -O3

---

## Key Optimization Findings

### 1. Work-Group Size Selection

| Kernel Type | Optimal WG | Verification |
|-------------|-----------|--------------|
| Element-wise | 128 | 9/10 kernels |
| Reduction | 256 | 2/3 kernels |
| Spatial (3D) | 256 (16×4×4) | 3/3 kernels |
| Filter/Matrix | 256 (1D) | 2/2 kernels |

### 2. Anti-Patterns Confirmed

| Anti-Pattern | Impact | Example Kernel |
|--------------|--------|----------------|
| Multi-thread collaboration | **-99%** | se_layer_nhwc |
| 2D for compact data | **-35%** | winograd_filter_transform |
| 1D for spatial data | **-45%** | winograd transforms |

### 3. Loop Unrolling Effectiveness

| Kernel Category | Unroll Benefit |
|-----------------|----------------|
| Complex nested loops | +9% to +447% |
| Simple element-wise | Minimal |
| Reduction | +10% to +82% |

---

## Raw Data Files

All 23 result files in [results/](results/) directory:

```
results/
├── add_bias_batched_results.csv
├── add_bias_nchw_results.csv
├── add_vectors_hnc_nhc_results.csv
├── add_vectors_results.csv
├── copy_type_converted_results.csv
├── expand_planes_fp16_nhwc_results.csv
├── expand_planes_nchw_results.csv
├── expand_planes_nhwc_results.csv
├── global_avg_pool_nhwc_fp16_results.csv
├── global_avg_pool_real_results.csv
├── global_scale_fp16_nhwc_results.csv
├── global_scale_results.csv
├── hard_batch_norm_results.csv
├── hard_fused_kernel_results.csv
├── layer_norm_results.csv
├── nchw_to_nhwc_results.csv
├── policy_map_results.csv
├── se_layer_nhwc_results.csv
├── softmax_real_results.csv
├── winograd_filter_transform_results.csv
├── winograd_input_transform_results.csv
├── winograd_output_relu_input_results.csv
└── winograd_real_results.csv
```

---

## Visualization

Performance charts available in [charts/](charts/) directory:
- Optimization speedup comparison
- Work-group size impact
- Multi-dimensional analysis

---

## References

- [Skill: Intel GPU E211 Optimizer](../.opencode/skills/intel-gpu-e211-optimizer/SKILL.md)
- [Skill: BMG B60 Optimizer](../.opencode/skills/bmg-b60-optimizer/SKILL.md)
- [Documentation](../docs/)

---

**Generated:** 2026-03-24  
**Total Kernels:** 23/23 (100% coverage)  
**Status:** Complete
