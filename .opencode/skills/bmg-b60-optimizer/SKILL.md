---
name: bmg-b60-optimizer
description: Intel BMG B60 GPU optimization guide for SYCL kernels
license: MIT
compatibility: opencode
metadata:
  architecture: Xe2 (Battlemage)
  device: "B60"
  sub_group_size: 16
  slm_size: 262144  # 256 KB
  max_work_group: 1024
  max_registers_per_thread: 255
  compiler_flags: "-fsycl -O3"
  version: "3.0"
---

# Intel BMG B60 GPU Optimization Guide

Optimization guide for SYCL kernels on Intel BMG B60 (Battlemage) architecture.

## Architecture Overview

```
Intel BMG B60 - Xe2/BMG
├── Sub-group: 16 (fixed)
├── Work-group: 64-1024 threads
├── SLM: 256 KB per work-group
├── Registers: 255 per thread max
└── Max threads/SM: 1024
```

### Key Characteristics

- **SIMD Width**: 16 (sub-group)
- **Memory**: Unified memory architecture
- **Cache**: L1 per compute unit, shared L2
- **Precision**: Native FP32, FP16, INT8 support

---

## Memory Hierarchy Optimization

### Global Memory Access Patterns

**Coalesced Access (Required for Performance)**

```cpp
// GOOD: Consecutive threads access consecutive addresses
// Warp (16 threads) reads 64 bytes in one transaction
int idx = item.get_global_id(0);
float val = input[idx];  // Thread N reads address N

// BAD: Strided access (each thread hits different cache line)
int idx = item.get_global_id(0) * 256;  // High stride
float val = input[idx];  // 16 separate transactions
```

**Access Pattern Guidelines:**
- Minimum transaction: 32 bytes
- Optimal: 64 bytes (full sub-group, FP32)
- Align data to 64-byte boundaries
- Avoid strided access (stride > 1)

### Vectorized Memory Access

Vectorized loads/stores improve bandwidth utilization:

**FP32 Vectorization (load 4 elements):**
```cpp
// Process 4 elements per thread
const int vec_size = 4;
int tid = item.get_global_id(0);
int start_idx = tid * vec_size;

// Load 4 consecutive floats
float4 vec;
#pragma unroll
for (int i = 0; i < vec_size; i++) {
    vec[i] = input[start_idx + i];
}

// Process vectorized data
#pragma unroll
for (int i = 0; i < vec_size; i++) {
    vec[i] = activate(vec[i]);
}

// Store vectorized
#pragma unroll
for (int i = 0; i < vec_size; i++) {
    output[start_idx + i] = vec[i];
}
```

**FP16 Vectorization:**
```cpp
// Load 2 half/float16 elements at once
using sycl::half;
const sycl::half2* vec_input = reinterpret_cast<const sycl::half2*>(input);
sycl::half2 v = vec_input[tid];

// Convert to float for computation
float v0 = static_cast<float>(v[0]);
float v1 = static_cast<float>(v[1]);

// Process...

// Store back
sycl::half2 result;
result[0] = static_cast<sycl::half>(v0);
result[1] = static_cast<sycl::half>(v1);
vec_output[tid] = result;
```

### SLM (Shared Local Memory) Configuration

SLM is 256 KB per work-group:

```cpp
// Calculate SLM usage
// Each work-item: 64 floats = 256 bytes
// WG=256: 256 * 256 = 64 KB (fits comfortably)
sycl::local_accessor<float, 1> local_mem(256, h);

// Access pattern matters:
// GOOD: Consecutive access
float val = local_mem[item.get_local_id(0)];

// BAD: Bank conflicts (if banks = 16)
// Stride-16 hits same bank
float val = local_mem[item.get_local_id(0) * 16];  // Conflict!
```

**Bank Conflict Avoidance:**
```cpp
// Pad array to avoid conflicts
// For 2D: add +1 to inner dimension
sycl::local_accessor<float, 2> local_mem(sycl::range<2>(32, 33), h);
// Now local_mem[row][col] hits different banks
```

---

## Sub-group Optimizations

### Shuffle Operations

Fastest data sharing within sub-group (no SLM needed):

```cpp
// Reduction using shuffles
float val = input[idx];
sycl::sub_group sg = item.get_sub_group();

#pragma unroll
for (int offset = 8; offset > 0; offset >>= 1) {
    val += sycl::shift_group_left(sg, val, offset);
}
// Now val contains sum of all 16 lanes

// Broadcast from lane 0
float broadcast_val = sycl::group_broadcast(sg, val, 0);

// Butterfly shuffle for max
float max_val = sycl::shift_group_left(sg, val, 8);
val = sycl::max(val, max_val);
// ... repeat for 4, 2, 1
```

### Sub-group Reduction

```cpp
// Built-in reduction (SYCL 2020)
sycl::sub_group sg = item.get_sub_group();
float sum = sycl::reduce(sg, local_val, sycl::plus<float>());
float min_val = sycl::reduce(sg, local_val, sycl::minimum<float>());
float max_val = sycl::reduce(sg, local_val, sycl::maximum<float>());
```

---

## Occupancy and Work-Group Tuning

### Occupancy Calculation

```
Occupancy = Active Warps / Max Warps per Compute Unit

Factors affecting occupancy:
1. Work-group size: 1024 max
2. SLM usage: 128 KB max
3. Register usage: 255 per thread max

Example:
- WG size = 256 threads
- SLM = 32 KB
- Registers = 64 per thread
- Occupancy = Limited by SLM: 128KB/32KB = 4 WGs per CU
```

### Work-Group Size Selection

| Kernel Type | WG Size | Reasoning |
|-------------|---------|-----------|
| Element-wise | 128 | High occupancy, simple access |
| Element-wise (large) | 256 | More parallelism |
| Reduction | 256 | Balance parallelism/reduction steps |
| Reduction (large) | 512-1024 | Fewer levels |
| 2D Spatial | 256 (16×16) | Square tiles |
| 3D Spatial | 256 (16×4×4) | Match dimensions |
| Matrix/Compact | 256 | Register pressure |

**Selection Strategy:**
1. Start with 128 for element-wise operations
2. Start with 256 for compute-intensive kernels
3. Test 64 if high register pressure
4. Test 512 if low occupancy with 256

---

## Precision and Data Types

### FP32 (Standard)

```cpp
// Default precision
float val = input[idx];
val = activate(val);
output[idx] = val;
```

### FP16 (Half Precision)

Trade-offs:
- **Pros**: 2x memory bandwidth, 2x cache capacity
- **Cons**: Limited range, precision loss in accumulation
- **Best for**: Memory-bound kernels, inference

```cpp
using sycl::half;

// Load as half, compute as float
half h_val = input[idx];
float val = static_cast<float>(h_val);

// Compute in FP32
val = complex_math(val);

// Store as half
output[idx] = static_cast<half>(val);
```

**FP16 Accumulation Pattern:**
```cpp
// DON'T accumulate in FP16 (precision loss)
half sum = 0;  // BAD
for (...) sum += input[i];  // Loses precision

// DO accumulate in FP32
float sum = 0.0f;  // GOOD
for (...) sum += static_cast<float>(input[i]);
output[0] = static_cast<half>(sum);
```

### BF16 (Brain Float)

Trade-offs:
- **Pros**: Same range as FP32, more precision than FP16
- **Cons**: Not universally supported
- **Best for**: Training, when FP16 overflows

```cpp
// SYCL doesn't have native BF16, use uint16_t
// Convert manually or use Intel extensions
```

### Mixed Precision Pattern

```cpp
// Input/output: FP16
// Computation: FP32

float acc = 0.0f;
#pragma unroll
for (int i = 0; i < N; i++) {
    float val = static_cast<float>(input[i]);  // FP16 -> FP32
    acc += val * val;  // Compute in FP32
}
output[0] = static_cast<sycl::half>(acc);  // FP32 -> FP16
```

---

## Code Templates

### Template 1: Element-wise Operation

```cpp
// Generic element-wise kernel
// Usage: any pointwise operation (activation, scale, etc.)

template<typename T, typename Op>
void elementwise_kernel(T* output, const T* input, int N, Op op, sycl::queue& q) {
    const int wg_size = 128;  // Tuned for Xe2
    
    q.parallel_for(
        sycl::nd_range<1>(sycl::range<1>((N + wg_size - 1) / wg_size * wg_size),
                          sycl::range<1>(wg_size)),
        [=](sycl::nd_item<1> item) {
            int idx = item.get_global_id(0);
            if (idx < N) {
                output[idx] = op(input[idx]);
            }
        }
    );
}

// Example usage:
// elementwise_kernel(output, input, N, [](float x) { return x > 0 ? x : 0; }, q);
```

### Template 2: Reduction (Sum)

```cpp
// Two-stage reduction: global -> local -> atomic
// Usage: sum, avg, max, min

template<typename T>
void reduction_kernel(T* output, const T* input, int N, sycl::queue& q) {
    const int wg_size = 256;
    int num_wgs = (N + wg_size - 1) / wg_size;
    
    // Temporary buffer for partial sums
    T* partial = sycl::malloc_device<T>(num_wgs, q);
    
    // Stage 1: Reduce to partial sums
    q.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_wgs * wg_size),
                          sycl::range<1>(wg_size)),
        [=](sycl::nd_item<1> item) {
            int tid = item.get_global_id(0);
            int lid = item.get_local_id(0);
            
            // Local reduction
            sycl::local_accessor<T, 1> local_sum(wg_size, item);
            
            T sum = 0;
            for (int i = tid; i < N; i += item.get_global_range(0)) {
                sum += input[i];
            }
            local_sum[lid] = sum;
            
            // Tree reduction in local memory
            item.barrier();
            #pragma unroll
            for (int offset = wg_size / 2; offset > 0; offset >>= 1) {
                if (lid < offset) {
                    local_sum[lid] += local_sum[lid + offset];
                }
                item.barrier();
            }
            
            // Write partial result
            if (lid == 0) {
                partial[item.get_group(0)] = local_sum[0];
            }
        }
    );
    
    // Stage 2: Final reduction (can use single work-group or atomic)
    q.parallel_for(sycl::range<1>(1), [=](sycl::item<1>) {
        T total = 0;
        for (int i = 0; i < num_wgs; i++) {
            total += partial[i];
        }
        output[0] = total;
    });
    
    sycl::free(partial, q);
}
```

### Template 3: 2D Spatial Operation

```cpp
// 2D convolution, pooling, or similar
// Uses tiling in SLM

template<typename T, int TILE_H, int TILE_W>
void spatial_2d_kernel(T* output, const T* input, 
                       int H, int W, int C, sycl::queue& q) {
    // Work-group processes one tile
    sycl::range<2> wg(TILE_H, TILE_W);
    sycl::range<2> global((H + TILE_H - 1) / TILE_H * TILE_H,
                          (W + TILE_W - 1) / TILE_W * TILE_W);
    
    q.parallel_for(
        sycl::nd_range<2>(global, wg),
        [=](sycl::nd_item<2> item) {
            int h = item.get_global_id(0);
            int w = item.get_global_id(1);
            
            if (h >= H || w >= W) return;
            
            // Example: simple 3x3 filter
            T sum = 0;
            #pragma unroll
            for (int dh = -1; dh <= 1; dh++) {
                #pragma unroll
                for (int dw = -1; dw <= 1; dw++) {
                    int hh = sycl::clamp(h + dh, 0, H - 1);
                    int ww = sycl::clamp(w + dw, 0, W - 1);
                    sum += input[(hh * W + ww) * C + item.get_global_id(2)];
                }
            }
            output[(h * W + w) * C + item.get_global_id(2)] = sum / 9;
        }
    );
}

// Usage: spatial_2d_kernel<float, 16, 16>(out, in, H, W, C, q);
```

### Template 4: Matrix Multiplication (Naive)

```cpp
// Simple GEMM for small matrices
// For large matrices, use optimized libraries

template<typename T>
void gemm_kernel(T* C, const T* A, const T* B, 
                 int M, int N, int K, sycl::queue& q) {
    // Each work-item computes one element of C
    sycl::range<2> wg(16, 16);  // 256 threads
    sycl::range<2> global((M + 15) / 16 * 16, (N + 15) / 16 * 16);
    
    q.parallel_for(
        sycl::nd_range<2>(global, wg),
        [=](sycl::nd_item<2> item) {
            int m = item.get_global_id(0);
            int n = item.get_global_id(1);
            
            if (m >= M || n >= N) return;
            
            T sum = 0;
            #pragma unroll 8
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    );
}
```

### Template 5: Batch Normalization

```cpp
// Per-channel batch normalization
// N = batch, C = channels, H*W = spatial

template<typename T>
void batch_norm_kernel(T* output, const T* input,
                       const T* gamma, const T* beta,
                       const T* mean, const T* var,
                       int N, int C, int HW, T epsilon, sycl::queue& q) {
    const int wg_size = 256;
    int total = N * C * HW;
    
    q.parallel_for(
        sycl::nd_range<1>((total + wg_size - 1) / wg_size * wg_size,
                          wg_size),
        [=](sycl::nd_item<1> item) {
            int idx = item.get_global_id(0);
            if (idx >= total) return;
            
            int c = (idx / HW) % C;  // Channel index
            
            T x = input[idx];
            T norm = (x - mean[c]) / sycl::sqrt(var[c] + epsilon);
            output[idx] = norm * gamma[c] + beta[c];
        }
    );
}
```

### Template 6: Attention (Simplified)

```cpp
// Simplified single-head attention
// Q, K, V: [seq_len, head_dim]
// Output: [seq_len, head_dim]

template<typename T, int HEAD_DIM>
void attention_kernel(T* output, const T* Q, const T* K, const T* V,
                      int seq_len, sycl::queue& q) {
    // Each work-item processes one query position
    q.parallel_for(sycl::range<1>(seq_len), [=](sycl::item<1> item) {
        int q_idx = item.get_id(0);
        
        // Compute Q·K^T for this query
        float attn_scores[128];  // Max seq_len assumed
        float row_max = -INFINITY;
        
        #pragma unroll 4
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            float score = 0;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                score += static_cast<float>(Q[q_idx * HEAD_DIM + d]) *
                         static_cast<float>(K[k_idx * HEAD_DIM + d]);
            }
            score /= sycl::sqrt(static_cast<float>(HEAD_DIM));
            attn_scores[k_idx] = score;
            row_max = sycl::max(row_max, score);
        }
        
        // Softmax
        float row_sum = 0;
        #pragma unroll 4
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            attn_scores[k_idx] = sycl::exp(attn_scores[k_idx] - row_max);
            row_sum += attn_scores[k_idx];
        }
        
        // Weighted sum of V
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            float out_val = 0;
            #pragma unroll 4
            for (int k_idx = 0; k_idx < seq_len; k_idx++) {
                out_val += attn_scores[k_idx] * 
                          static_cast<float>(V[k_idx * HEAD_DIM + d]);
            }
            output[q_idx * HEAD_DIM + d] = static_cast<T>(out_val / row_sum);
        }
    });
}
```

### Template 7: Data Format Conversion

```cpp
// NCHW to NHWC conversion
// Common layout transformation

template<typename T>
void nchw_to_nhwc_kernel(T* output, const T* input,
                         int N, int C, int H, int W, sycl::queue& q) {
    const int wg_size = 256;
    int total = N * C * H * W;
    
    q.parallel_for(
        sycl::nd_range<1>((total + wg_size - 1) / wg_size * wg_size,
                          wg_size),
        [=](sycl::nd_item<1> item) {
            int idx = item.get_global_id(0);
            if (idx >= total) return;
            
            // Decode NCHW index
            int w = idx % W;
            int h = (idx / W) % H;
            int c = (idx / (W * H)) % C;
            int n = idx / (W * H * C);
            
            // NCHW input index
            int src_idx = ((n * C + c) * H + h) * W + w;
            
            // NHWC output index
            int dst_idx = ((n * H + h) * W + w) * C + c;
            
            output[dst_idx] = input[src_idx];
        }
    );
}
```

### Template 8: Fused Operations

```cpp
// Fused bias-add + activation
// Common in neural networks

template<typename T, typename Activation>
void fused_bias_activation_kernel(T* output, const T* input, const T* bias,
                                   int N, int C, Activation act, sycl::queue& q) {
    // N = batch * height * width (total elements)
    // C = channels (bias size)
    
    const int wg_size = 128;
    q.parallel_for(
        sycl::nd_range<1>((N + wg_size - 1) / wg_size * wg_size,
                          wg_size),
        [=](sycl::nd_item<1> item) {
            int idx = item.get_global_id(0);
            if (idx >= N) return;
            
            int c = idx % C;  // Channel index
            T val = input[idx] + bias[c];
            output[idx] = act(val);
        }
    );
}

// Usage:
// fused_bias_activation_kernel(out, in, bias, N, C, 
//     [](float x) { return x > 0 ? x : 0; }, q);  // ReLU
```

---

## Profiling and Debugging

### Basic Timing

```cpp
// Simple kernel timing
sycl::event evt = q.submit([&](sycl::handler& h) {
    h.parallel_for(...);
});
evt.wait();

auto start = evt.get_profiling_info<sycl::info::event_profiling::command_start>();
auto end = evt.get_profiling_info<sycl::info::event_profiling::command_end>();
double ms = (end - start) / 1e6;
```

### Performance Metrics

Track these metrics:
1. **Kernel time** - Total execution time
2. **Memory bandwidth** - Bytes transferred / time
3. **Compute utilization** - Actual vs theoretical GFLOPS
4. **Occupancy** - Active warps / max warps

### Common Issues

**Low Occupancy**
- Cause: Too much SLM or registers
- Fix: Reduce WG size or simplify kernel

**Memory Bound**
- Cause: Low arithmetic intensity
- Fix: Fuse operations, use vectorized loads

**Bank Conflicts**
- Cause: Strided SLM access
- Fix: Pad arrays, change access pattern

**Warp Divergence**
- Cause: Conditionals within warp
- Fix: Restructure to uniform branches

---

## Compiler Flags

```bash
# Basic optimization
icpx -fsycl -O2 -std=c++17 kernel.cpp -o kernel

# Aggressive optimization
icpx -fsycl -O3 -std=c++17 \
    -ffast-math \
    -funroll-loops \
    kernel.cpp -o kernel

# Ahead-of-time compilation
icpx -fsycl -O3 \
    -fsycl-targets=spir64 \
    kernel.cpp -o kernel

# Debug info
icpx -fsycl -O2 -g -lineinfo kernel.cpp -o kernel
```

---

## Best Practices Summary

1. **Memory First**: Ensure coalesced access before other optimizations
2. **Work-group Size**: Start with 128, profile alternatives
3. **SLM**: Use for data reuse, watch bank conflicts
4. **Registers**: Prefer for thread-private data, limit to 64-128 per thread
5. **Precision**: FP32 compute, lower precision for memory-bound kernels
6. **Vectorization**: Load/store 2-4 elements per thread when possible
7. **Reduction**: Use tree-based, avoid atomics
8. **Profile**: Measure before optimizing
9. **Fuse**: Combine kernels to reduce memory traffic
10. **Test**: Benchmark 3+ configurations

---

## Decision Tree

```
Kernel Type?
├── Element-wise (pointwise)
│   └── WG=128, 1D, coalesced access
├── Reduction
│   └── WG=256, tree reduction, SLM
├── Spatial (2D/3D)
│   ├── 2D: WG=(16,16), tile in SLM
│   └── 3D: WG=(16,4,4), match dims
├── Matrix/Compact
│   └── WG=256, 1D (avoid 2D/3D)
└── Fused Complex
    └── Single-thread per output, minimize barriers

Memory Bound?
├── Yes: Vectorize loads (2x-4x), FP16
└── No: Focus on compute efficiency

Occupancy Low?
├── Yes: Reduce WG size or SLM usage
└── No: Good, check other bottlenecks
```

---

**Version**: 3.0
**Target**: Intel Xe2 (Battlemage G21)
**Last Updated**: 2026
