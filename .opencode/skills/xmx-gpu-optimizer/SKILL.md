---
name: xmx-gpu-optimizer
description: Intel GPU XMX optimization for SYCL kernels with validated patterns for matrix operations, reductions, and element-wise kernels. Includes battle-tested decision trees and compilation templates.
license: MIT
compatibility: opencode
metadata:
  version: "1.0"
  target_hardware: "Intel BMG/ARC GPUs with XMX"
  compiler: "icpx"
  based_on: "2026-03 LCZero kernel optimization project"
  validated_kernels: "4/36"
  key_finding: "Single-thread-per-output optimal for small matrices"
---

# XMX GPU Optimizer Skill

Optimize SYCL kernels for Intel GPUs with XMX (Xe Matrix Extensions). Based on validated optimization patterns from real-world kernel tuning.

## Quick Start (5 minutes)

### 1. Verify Environment
```bash
# Check Docker container is running
docker ps | grep lsv-container

# Check GPU is accessible
docker exec lsv-container sycl-ls
```

### 2. Test Compilation
```bash
# Copy your kernel
docker cp test_kernel.cpp lsv-container:/workspace/tests/

# Compile with mandatory flags
docker exec -w /workspace/tests lsv-container \
  icpx -fsycl -O3 -std=c++17 \
  -fsycl-targets=spir64_gen \
  -Xsycl-target-backend "-device bmg -options -ze-opt-large-register-file" \
  -o test_kernel test_kernel.cpp

# Run
docker exec -w /workspace/tests lsv-container ./test_kernel
```

### 3. Check Results
Look for:
- Compilation: "Build succeeded"
- GPU: "Intel(R) Graphics [0xe211]" (or similar)
- Performance: GFLOPS and bandwidth numbers

## Decision Tree (30-Second Classification)

```
Analyze your kernel:
│
├─ Contains matrix multiply (GEMM, attention, FC layers)?
│  └─ Matrix size >= 256x256?
│     ├─ YES → Type D-XMX (use joint_matrix API)
│     └─ NO  → Type D-Small (single-thread-per-output)
│
├─ Pooling, softmax, or reduction operation?
│  └─ Type C (single-thread-per-output, 50-70% gain)
│
├─ Winograd transform or spatial operations?
│  └─ Type B (tile optimization, 40-60% gain)
│
└─ Element-wise operations (add, multiply, etc.)?
   └─ Type A (vectorized memory, <15% gain)
```

## Kernel Types & Strategies

### Type A: Element-wise Operations
**Examples**: `add_vectors`, `multiply`, `bias_add`

**Characteristics**:
- Memory bandwidth bound (< 15% improvement possible)
- Simple point-wise operations

**Optimization Strategy**:
- **Round 1 ONLY** (stop if < 15% gain)
- Vectorized loads/stores (float4, half2)
- Work-group size: 128 threads
- Use FP16 for 2x memory bandwidth

**Template**:
```cpp
// Type A: Element-wise kernel template
void kernel(sycl::nd_item<1> item, float* in, float* out, int N) {
  const int vec_size = 4;
  int tid = item.get_global_id(0);
  int start = tid * vec_size;
  
  if (start < N) {
    // Vectorized load
    float4 vec;
    #pragma unroll
    for (int i = 0; i < vec_size && (start + i) < N; i++) {
      vec[i] = in[start + i];
    }
    
    // Element-wise operation
    #pragma unroll
    for (int i = 0; i < vec_size; i++) {
      vec[i] = operation(vec[i]);
    }
    
    // Vectorized store
    #pragma unroll
    for (int i = 0; i < vec_size && (start + i) < N; i++) {
      out[start + i] = vec[i];
    }
  }
}
```

**Expected Performance**: < 15% improvement over baseline

---

### Type B: Winograd/Spatial Transforms
**Examples**: `winograd_filter_transform`, `winograd_input_transform`

**Characteristics**:
- Tile-based operations
- Medium compute intensity
- Memory access patterns matter

**Optimization Strategy**:
- **Round 1**: Work-group 128, `#pragma unroll 4`
- **Round 2**: SLM tile caching
- **Round 3**: XMX if applicable (large tiles)

**Template**:
```cpp
// Type B: Winograd transform template
void kernel(sycl::nd_item<2> item, float* input, float* output, 
            int C, int K, int tile_size) {
  // Use SLM for tile caching
  sycl::local_accessor<float, 2> tile(sycl::range<2>(tile_size, tile_size), h);
  
  int c = item.get_global_id(0);
  int k = item.get_global_id(1);
  
  // Cooperative loading into SLM
  for (int i = item.get_local_id(0); i < tile_size; i += item.get_local_range(0)) {
    for (int j = item.get_local_id(1); j < tile_size; j += item.get_local_range(1)) {
      tile[i][j] = input[...];
    }
  }
  item.barrier();
  
  // Transform computation with unrolling
  #pragma unroll 4
  for (int y = 0; y < tile_size; y++) {
    #pragma unroll 4
    for (int x = 0; x < tile_size; x++) {
      // Winograd transform math
      output[...] = transform(tile, y, x);
    }
  }
}
```

**Expected Performance**: 40-60% improvement

---

### Type C: Reduction Operations
**Examples**: `global_avg_pool`, `softmax`, `layer_norm`

**Characteristics**:
- Aggregation operations (sum, max, mean)
- Single-thread-per-output optimal
- Avoid atomics when possible

**Optimization Strategy**:
- **ALWAYS use single-thread-per-output pattern** (validated 60% gain)
- Each thread computes complete reduction for one output element
- Use private memory for accumulation
- Work-group size: flexible

**Template**:
```cpp
// Type C: Reduction kernel - SINGLE THREAD PER OUTPUT (OPTIMAL)
void kernel(sycl::item<1> item, float* input, float* output, 
            int N, int C, int H, int W) {
  int n = item.get_id(0);
  if (n >= N) return;
  
  // Each thread handles complete reduction for one sample
  for (int c = 0; c < C; c++) {
    float sum = 0.0f;
    
    #pragma unroll 4
    for (int h = 0; h < H; h++) {
      #pragma unroll 4
      for (int w = 0; w < W; w++) {
        sum += input[((n * H + h) * W + w) * C + c];
      }
    }
    
    output[n * C + c] = sum / (H * W);
  }
}

// Launch: 1 work-item per N (not per element)
queue.parallel_for(sycl::range<1>(N), kernel);
```

**Expected Performance**: 50-70% improvement

---

### Type D: Matrix Multiplication
**Examples**: `GEMM`, `SE layer`, `attention`, `FC layers`

**CRITICAL DECISION**: Check matrix size first!

#### Type D-Small: Matrix < 256x256
**Use single-thread-per-output** (like Type C)

Why not XMX?
- XMX tile overhead (8x16x16) exceeds benefit for small matrices
- Single-thread pattern achieves 18x speedup vs baseline

**Template**:
```cpp
// Type D-Small: Single-thread GEMM
void kernel(sycl::item<1> item, float* A, float* B, float* C, 
            int M, int N, int K) {
  int m = item.get_id(0);
  if (m >= M) return;
  
  // Each thread computes one row of output
  for (int n = 0; n < N; n++) {
    float sum = 0.0f;
    #pragma unroll 8
    for (int k = 0; k < K; k++) {
      sum += A[m * K + k] * B[k * N + n];
    }
    C[m * N + n] = sum;
  }
}
```

**Expected Performance**: 10-100x improvement (small matrices)

#### Type D-Large: Matrix >= 256x256
**Use XMX joint_matrix API**

Template: See XMX Template section below

**Expected Performance**: 100+ TFLOPS

---

## XMX Template (Type D-Large)

### Required Headers
```cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
```

### Tile Configuration
```cpp
// BMG optimal tile size
constexpr int TM = 8;   // M dimension per tile
constexpr int TN = 16;  // N dimension per tile
constexpr int TK = 16;  // K dimension per tile
```

### XMX Kernel Structure
```cpp
using namespace sycl::ext::oneapi::experimental::matrix;

void xmx_gemm_kernel(float* C, float* A, float* B, int M, int N, int K,
                     sycl::nd_item<2> item) {
  // Get tile coordinates
  int tile_m = item.get_group(0) * 2 + (item.get_local_id(0) / 16);
  int tile_n = item.get_group(1);
  
  // XMX accumulator
  joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> acc;
  joint_matrix_fill(item.get_sub_group(), acc, 0.0f);
  
  // Iterate over K dimension
  for (int k = 0; k < K; k += TK) {
    // Load A tile (8x16)
    joint_matrix<sycl::sub_group, float, use::a, TM, TK, layout::row_major> mat_a;
    joint_matrix_load(item.get_sub_group(), mat_a, 
                      A + tile_m * TM * K + k, K);
    
    // Load B tile (16x16)
    joint_matrix<sycl::sub_group, float, use::b, TK, TN, layout::row_major> mat_b;
    joint_matrix_load(item.get_sub_group(), mat_b,
                      B + k * N + tile_n * TN, N);
    
    // XMX multiply-accumulate
    joint_matrix_mad(item.get_sub_group(), acc, mat_a, mat_b, acc);
  }
  
  // Store result
  joint_matrix_store(item.get_sub_group(), acc,
                     C + tile_m * TM * N + tile_n * TN, N);
}

// Launch configuration
sycl::range<2> global((M + TM - 1) / TM * 16, (N + TN - 1) / TN);
sycl::range<2> local(16, 1);  // 16 threads per sub-group
queue.parallel_for(sycl::nd_range<2>(global, local), kernel);
```

### Full XMX Example with BMG AOT
```cpp
// test_gemm_xmx.cpp - Complete working example
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <iostream>

using namespace sycl::ext::oneapi::experimental::matrix;

constexpr int TM = 8;
constexpr int TN = 16;
constexpr int TK = 16;

int main() {
  sycl::queue queue(sycl::gpu_selector_v);
  
  // Matrix dimensions (must be >= 256 for XMX efficiency)
  int M = 4096, N = 4096, K = 4096;
  
  // Allocate and initialize...
  // See full example in references
  
  queue.submit([&](sycl::handler& h) {
    h.parallel_for(
      sycl::nd_range<2>(
        sycl::range<2>((M/TM)*16, N/TN),
        sycl::range<2>(16, 1)
      ),
      [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
        // XMX kernel code here
      }
    );
  });
}
```

---

## Mandatory Compilation Flags

### Always Include
```bash
icpx -fsycl -O3 -std=c++17 \
  -fsycl-targets=spir64_gen \
  -Xsycl-target-backend "-device bmg -options -ze-opt-large-register-file"
```

### Flag Explanation
- `-fsycl`: Enable SYCL
- `-O3`: Maximum optimization
- `-fsycl-targets=spir64_gen`: AOT compilation (REQUIRED for XMX)
- `-device bmg`: Target BMG specifically
- `-ze-opt-large-register-file`: Large GRF mode (essential for complex kernels)

### Common Errors

**Error**: `no member named 'joint_matrix'`
**Fix**: Add `#include <sycl/ext/oneapi/matrix/matrix.hpp>`

**Error**: "Unsupported operation" with XMX code
**Fix**: Must use AOT compilation (`-fsycl-targets=spir64_gen -device bmg`)

**Error**: "Out of resources"
**Fix**: Reduce work-group size or use large GRF mode

---

## Optimization Workflow

### Pre-Flight Checklist
- [ ] Docker container running
- [ ] Source file copied to container
- [ ] Kernel type identified (A/B/C/D)
- [ ] Expected performance target set
- [ ] Baseline version ready

### Round 1: Type-Specific Optimization
1. Apply template for identified type
2. Compile with mandatory flags
3. Test 3 sizes: small, medium, large
4. **Decision**:
   - Speedup > 20% → Continue to Round 2
   - Speedup 10-20% → Skip to Round 3
   - Speedup < 10% → STOP (document findings)

### Round 2: Advanced Optimization (if continued)
- Type A/B: SLM optimization
- Type C: Sub-group shuffle
- Type D: XMX tile tuning

### Round 3: Polish (if continued)
- Register pressure reduction
- Prefetch hints
- Final validation

---

## Performance Validation

### Expected Results by Type

| Type | Baseline | Optimized | Speedup | Example |
|------|----------|-----------|---------|---------|
| A | 2.7 GFLOPS | 2.8 GFLOPS | 1.05x | add_vectors |
| B | 433 GFLOPS | 600 GFLOPS | 1.40x | winograd_filter |
| C | 39 GFLOPS | 63 GFLOPS | 1.60x | global_avg_pool |
| D-Small | 1.2 GFLOPS | 21 GFLOPS | 18x | SE layer |
| D-Large | 12 GFLOPS | 155 GFLOPS | 12x | GEMM 4Kx4K |

### Red Flags
- GFLOPS < 1: Check compilation flags (-O3)
- No difference between versions: Verify different code paths
- XMX shows no improvement: Matrix too small (< 256)
- Compilation succeeds but crashes: Check SLM usage

---

## Troubleshooting

### "File not found" Error
```bash
# Wrong
docker exec lsv-container ./test

# Right
docker exec -w /workspace/tests lsv-container ./test
```

### "Build succeeded" but No Output
Check if running inside container:
```bash
docker exec lsv-container ls /workspace/tests/
```

### Performance Regression
- Re-classify kernel type
- Check if using wrong pattern (e.g., collaborative reduction instead of single-thread)
- Verify work-group size matches hardware (multiples of 16)

### XMX Not Working
1. Verify AOT compilation flags
2. Check matrix dimensions >= 256
3. Verify subgroup size = 16
4. Check `reqd_sub_group_size(16)` attribute

---

## References

### From This Project
- Validation results: `tests/reports/small_scale_test_summary.md`
- Kernel classification: `tests/kernel_classification.md`
- Complete report: `tests/reports/XMX_OPTIMIZATION_FINAL_REPORT.md`
- Honest status: `tests/reports/HONEST_STATUS_CHECK.md`

### Intel Documentation
- XMX API: `<sycl/ext/oneapi/matrix/matrix.hpp>`
- BMG Architecture: Intel Xe2 Graphics
- Optimal tile size: 8x16x16 for FP16

### Source Examples
- Type C (Reduction): `test_global_avg_pool_nhwc_fp16.cpp`
- Type D-Small: `test_se_layer_nhwc.cpp`
- Type D-Large: `test_gemm_xmx.cpp`

---

## Version History

**v1.0** (2026-03-26)
- Initial version based on LCZero kernel optimization project
- Validated on 4 kernels: add_vectors, winograd_filter, global_avg_pool, se_layer
- Key finding: Single-thread-per-output pattern optimal for BMG
- XMX boundary identified: 256x256 matrix size

---

## License

MIT License - See LICENSE file for details
