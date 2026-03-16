# Winograd Convolution SYCL Implementation Skill

## Overview

This skill documents the implementation of Winograd F(4x4, 3x3) convolution kernels for SYCL backend in lc0, including the transformation from CUDA and layout compatibility fixes.

## Background

Winograd convolution reduces the number of multiply-accumulate operations for small filters (3x3) by transforming input, filter, and output into a different space.

### Transformation Flow

```
Input (8x8) -> InputTransform -> 4x(6x6) tiles
Filter (3x3) -> FilterTransform -> 6x6 transformed
4x(6x6) tiles + 6x6 filter -> GEMM -> 4x(6x6) output tiles
4x(6x6) output tiles -> OutputTransform -> Output (8x8)
```

## Critical Layout Compatibility

### FilterTransform Output Layout

**CUDA Layout: HWCK** (Hardware layout)
```
Index: tile * C * K + c * K + k
where tile = i * 6 + j (0-35 for 6x6 transformed filter)
```

**Original SYCL Bug: KCRS** (Incorrect)
```
Index: (k * C + c) * 36 + tile  // WRONG - incompatible with CUDA GEMM
```

**Fixed SYCL: HWCK** (Correct)
```cpp
// In FilterTransform kernel (sycl_kernels.cc)
int tile = i * 6 + j;
int out_idx = tile * C_ * K_ + c * K_ + k;  // HWCK layout
transformed_filter_[out_idx] = transformed_tile[i * 6 + j];
```

### Data Layout Transformations

| Stage | Input Layout | Output Layout | Notes |
|-------|--------------|---------------|-------|
| FilterTransform | KCRS (3x3) | HWCK (6x6xCxK) | For GEMM compatibility |
| InputTransform | NCHW (8x8) | HWNC (6x6x4NxC) | 4 tiles per board |
| GEMM | HWNC @ HWCK | HWNC | Matrix multiplication |
| OutputTransform | HWNC | NCHW (8x8) | Final output |

## Implementation Details

### 1. FilterTransform Kernel

```cpp
template <typename T>
struct FilterTransformKernel {
  void operator()(sycl::nd_item<1> item) const {
    int tid = item.get_global_id(0);
    if (tid >= elements_) return;

    int c = tid % C_;
    int k = tid / C_;

    // Read 3x3 filter (NCHW: k*C*9 + c*9)
    T filter_tile[9];
    int filter_offset = k * C_ * 9 + c * 9;
    for (int s = 0; s < 3; s++) {
      for (int r = 0; r < 3; r++) {
        filter_tile[s * 3 + r] = filter_[filter_offset + s * 3 + r];
      }
    }

    // Transform: G * filter * G^T
    T transformed_tile[36];
    FilterTransform4x4(transformed_tile, filter_tile);

    // Write in HWCK layout (CUDA compatible)
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 6; j++) {
        int tile = i * 6 + j;
        int out_idx = tile * C_ * K_ + c * K_ + k;
        transformed_filter_[out_idx] = transformed_tile[i * 6 + j];
      }
    }
  }
};
```

### 2. InputTransform Kernel

Transforms 8x8 input boards into 4x6x6 tiles:
- Top-left, top-right, bottom-left, bottom-right

**Critical Fix: HWNC Layout with N dimension**

```cpp
// CORRECT: Include N dimension in stride calculation
int shln = n * 4 + tile_index;  // 4 tiles per board
int idx = y * 6 * (4 * N) * C + x * (4 * N) * C + shln * C + c;
//       [h][  w  ][  n*4  ][c]  = HWNC layout
```

**Common Bug (N=1 only)**:
```cpp
// WRONG: Missing N dimension - only works for batch=1
int idx = y * 6 * 4 * C + x * 4 * C + shln * C + c;  // [6][6][4][C]
```

### 3. OutputTransform Kernel

Transforms 4x6x6 tiles back to 8x8 output with bias and activation:

```cpp
template <typename T, bool use_se, ActivationFunction activation,
          bool use_bias, bool use_skip, bool skipInput_nhwc, bool output_nhwc>
void OutputTransform(...)
```

### 4. Fused OutputInputTransform

Combines OutputTransform + InputTransform for residual blocks:
- Reduces memory bandwidth
- Keeps data in registers between transforms

## Transformation Matrices

### G (6x3) - Filter Transform
```
1/4     0       0
-1/6   -1/6    -1/6
-1/6    1/6    -1/12
1/12    1/6     1/6
1/24   -1/12    0
0       0       1
```

### B^T (6x6) - Input Transform
```
4   0  -5   0   1   0
0  -4  -4   1   1   0
0   4  -4  -1   1   0
0  -2  -1   2   1   0
0   2  -1  -2   1   0
0   4   0  -5   0   1
```

### A^T (4x6) - Output Transform
```
1   1   1   1   1   0
0   1  -1   2  -2   0
0   1   1   4   4   0
0   1  -1   8  -8   1
```

## Testing

### Build Tests in Docker

```bash
# Build specific test
docker exec lsv-container bash -c "cd /intel/tianfeng/lc0/src/neural/backends/sycl && ./build_test.sh -t test_filter_transform --clean"

# Run test
docker exec lsv-container bash -c "cd /intel/tianfeng/lc0/src/neural/backends/sycl/build && ./test_filter_transform"
```

### CPU Reference Implementation

```cpp
// HWCK layout reference for testing
void FilterTransform_ref(float* out, const float* filter, int K, int C) {
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      float temp_tile[36];
      FilterTransform4x4_ref(temp_tile, &filter[(k*C+c)*9]);

      // Write in HWCK layout
      for (int tile = 0; tile < 36; ++tile) {
        int out_idx = tile * C * K + c * K + k;
        out[out_idx] = temp_tile[tile];
      }
    }
  }
}
```

### Numerical Accuracy

| Data Type | Max Error | Mean Error | Status |
|-----------|-----------|------------|--------|
| float     | 0         | 0          | ✅ Exact match with CPU reference |
| half      | < 1e-2    | < 1e-3     | Within tolerance |

### Batch Size Testing

| N (batch) | C (channels) | K (filters) | Result |
|-----------|--------------|-------------|--------|
| 1         | 32           | 32          | ✅ PASS (max error=0) |
| 4         | 64           | 64          | ✅ PASS (max error=0) |
| 8         | 256          | 256         | ✅ PASS (max error=0) |
| 16        | 256          | 256         | ✅ PASS (max error=0) |

**Note:** After InputTransform index fix (2025-02-02), all batch sizes produce exact numerical match with CPU reference.

## Common Pitfalls

### 1. Layout Mismatch

**Symptom:** GEMM produces garbage output, all zeros, or NaN
**Fix:** Ensure FilterTransform outputs HWCK layout matching CUDA

### 2. Index Calculation Error (Critical)

**Symptom:** Works for batch=1 but fails for batch>1; numerical errors scale with batch size
**Root Cause:** Missing N dimension in stride calculation

**Wrong (N=1 only):**
```cpp
// Only works when N=1!
transformed[h * 6 * 4 * C + w * 4 * C + shln * C + c]
```

**Correct (any N):**
```cpp
// Works for any batch size
transformed[h * 6 * (4 * N) * C + w * (4 * N) * C + shln * C + c]
```

### 3. Tile Position Errors

**Symptom:** Output has artifacts at tile boundaries
**Fix:** Check 4-tile offsets (n*4, n*4+1, n*4+2, n*4+3)

### 4. E2E Numerical Validation

**Recommended:** Always validate against CPU reference, not just CUDA reference data
```cpp
// CPU reference for InputTransform (HWNC layout)
void InputTransform_ref(float* out, const float* in, int N, int C) {
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int tile = 0; tile < 4; tile++) {
                int shln = n * 4 + tile;
                for (int y = 0; y < 6; y++)
                    for (int x = 0; x < 6; x++)
                        out[y * 6 * (4*N) * C + x * (4*N) * C + shln * C + c] = ...;
            }
        }
    }
}
```

## Integration with lc0

### Backend Registration

```cpp
// In backend registration
REGISTER_NETWORK("sycl", CreateSyclNetwork, 90)
```

### Layer Usage

```cpp
// Residual block with Winograd
conv1->Eval(...);           // Standard convolution
winograd_input->Eval(...);   // InputTransform
gemm->Eval(...);             // SYCL-BLAS GEMM
winograd_output->Eval(...);  // OutputTransform with SE
```

## Performance Tuning

### Current Optimizations
- Vectorized memory access (4 elements/thread)
- 2D work-groups for better locality
- Template specialization for compile-time constants

### Future Optimizations
- Subgroup reductions for SE layer
- Shared memory caching for input tiles
- Fused kernels (Output+Input transform)

## Files Modified

| File | Changes |
|------|---------|
| `sycl_kernels.cc` | FilterTransform HWCK layout, InputTransform N-dimension fix, SE-OutputTransform, OutputInputTransform |
| `test_filter_transform.cc` | Updated reference to HWCK layout |
| `test_winograd.cc` | Updated reference to HWCK layout |
| `test_winograd_e2e.cc` | **NEW** End-to-end numerical validation with CPU reference |
| `build_test.sh` | Added test_winograd_e2e to build list |

## Bug Fix History

### 2025-02-02: InputTransform Index Bug

**Problem:** InputTransform only worked correctly for batch size N=1. For N>1, produced incorrect results due to missing N dimension in output stride calculation.

**Impact:** Numerical errors increased with batch size (e.g., N=4 had max error ~50, N=8 had max error ~59).

**Fix:** Changed index calculation from:
```cpp
// WRONG: [6][6][4][C] - ignores batch dimension
transformed[h * 6 * 4 * C + w * 4 * C + shln * C + c]
```
to:
```cpp
// CORRECT: [6][6][N*4][C] - includes batch dimension
transformed[h * 6 * (4 * N) * C + w * (4 * N) * C + shln * C + c]
```

**Verification:** After fix, all batch sizes (N=1,4,8,16) produce max error = 0 (exact match with CPU reference).

## E2E Testing Guide

### Quick Validation

```bash
# Build and run E2E test
docker exec lsv-container bash -c "cd /intel/tianfeng/lc0/src/neural/backends/sycl && ./build_test.sh -t test_winograd_e2e --clean"
docker exec lsv-container bash -c "cd /intel/tianfeng/lc0/src/neural/backends/sycl/build && ./test_winograd_e2e"

# Expected output:
# === Testing N=1 C=32 K=32 ===
#   FilterTransform: PASS (max error=0)
#   InputTransform: PASS (max error=0)
# === Testing N=4 C=64 K=64 ===
#   FilterTransform: PASS (max error=0)
#   InputTransform: PASS (max error=0)
```

### CPU Reference Implementation Pattern

```cpp
// Pure CPU reference (layout-agnostic validation)
void WinogradConvolution_ref(
    float* output,
    const float* input,
    const float* filter,
    int N, int C, int K) {

    // 1. Transform filters (NCHW -> HWCK)
    std::vector<float> t_filter(36 * C * K);
    FilterTransform_ref(t_filter.data(), filter, K, C);

    // 2. Transform input (NCHW -> HWNC)
    std::vector<float> t_input(36 * N * 4 * C);
    InputTransform_ref(t_input.data(), input, N, C);

    // 3. GEMM (36 independent matmuls)
    std::vector<float> t_output(36 * N * 4 * K);
    for (int tile = 0; tile < 36; tile++)
        for (int n4 = 0; n4 < N * 4; n4++)
            for (int k = 0; k < K; k++) {
                float sum = 0;
                for (int c = 0; c < C; c++) {
                    float a = t_input[tile * (N*4) * C + n4 * C + c];
                    float b = t_filter[tile * C * K + c * K + k];
                    sum += a * b;
                }
                t_output[tile * (N*4) * K + n4 * K + k] = sum;
            }

    // 4. Output transform (HWNC -> NCHW)
    OutputTransform_ref(output, t_output.data(), N, K);
}
```

### Key Takeaway

Always validate SYCL kernels against **CPU reference implementation** using the same layout:
1. Generate random test data
2. Run SYCL kernel
3. Run CPU reference with same data
4. Compare with tolerance (float: 1e-4, half: 1e-2)
5. Test multiple batch sizes (N=1, 4, 8, 16) to catch stride bugs

## References

- CUDA implementation: `src/neural/backends/cuda/winograd_helper.inc`
- Original paper: "Fast Algorithms for Convolutional Neural Networks" (Lavin & Gray, 2016)
- SYCL 2020 spec: https://www.khronos.org/registry/SYCL/
- E2E Test: `test_winograd_e2e.cc` - Numerical validation against CPU reference
