# Intel BMG B60 GPGPU Specifications

**Hardware Generation**: BMG (Xe2 Architecture)
**Target Market**: Data center GPU for AI/HPC workloads

---

## Architecture Overview

| Component | Specification (BMG / Xe2) | Optimization Implication |
|-----------|--------------------------|--------------------------|
| **Compute Capability** | Xe2 Architecture (BMG) | Use AOT compilation with `--target` matching BMG |
| **Sub-Group Size** | 16-32 (default: 16) | Design kernels assuming 16-wide sub-groups |
| **Xe Matrix Extensions (XMX / DPAS)** | FP16/BF16: 8M×16N×16K (SYS depth: 8) / 8K (depth: 4) | Use `sycl::ext::oneapi::experimental::matrix` for DPAS operations |
| **Local Memory (SLM)** | 256 KB per XeCore (L1) | Tile data to fit in SLM per work-group |
| **L2 Cache** | 18 MB | Tile intermediate results to stay in L2 |
| **Memory Bandwidth** | ~500 GB/s (HBM2e) | Target: 80-90% of peak through coalesced access |
| **Max Threads / EU** | 7-8 threads concurrently | Max work-group size: 512-1024 threads |
| **Sub-Group Usage** | Reductions, shuffle, cross-lane ops | Use `sub_group::reduce`, `sub_group::shuffle` for efficiency |
| **Registers (GRF)** | 128 GRF (small mode) / 256 GRF (large mode) | Use large mode for register-heavy kernels |

---

## Detailed Specifications

### Sub-Group Configuration

```cpp
// Query sub-group size at runtime
sycl::queue queue;
size_t sg_size = queue.get_device().get_info<sycl::info::device::sub_group_sizes>().front();
// Returns: 16 for BMG B60
```

**Optimization Implications:**
- Design kernels assuming 16-wide sub-groups
- Use `sycl::sub_group::reduce<T, 16>` for explicit size
- Load 16 elements per work-item for optimal vectorization
- Align data structures to 16-element boundaries

### XMX Matrix Extensions (DPAS)

BMG supports XMX (Xe Matrix Extensions) with DPAS instructions:

| Data Type | Matrix Size | SYS Depth | Performance |
|-----------|-------------|-----------|--------------|
| FP16 / BF16 | 8M×16N×16K | 8 (standard) | 2 TFLOPS/EU |
| FP16 / BF16 | 8M×16N×16K | 4 (alt) | 1.5 TFLOPS/EU |
| TF32 | 4M×4N×4K | 8 | Available for mixed precision |

**Code Example:**
```cpp
using namespace sycl::ext::oneapi::experimental::matrix;

// DPAS matrix multiplication (8x16x16)
constexpr size_t M = 8, N = 16, K = 16;
using a_type = sycl::ext::oneapi::experimental::matrix::precision::tf32;
using b_type = sycl::ext::oneapi::experimental::matrix::precision::tf32;
using c_type = sycl::ext::oneapi::experimental::matrix::precision::tf32;

// Use joint_matrix for DPAS operations
sycl::ext::oneapi::experimental::matrix::joint_matrix<a_type, M, K, A_layout> a_mdx;
```

### Local Memory (SLM)

- **Size**: 256 KB per XeCore (shared within work-group)
- **Access Pattern**: Bank conflicts occur with stride = power of 2
- **Optimization**: Use padding or transpose to avoid bank conflicts

**Best Practices:**
```cpp
// GOOD: Access with stride that avoids bank conflicts
constexpr int TILE_SIZE = 256;
sycl::local_accessor<float, 1> local_mem(TILE_SIZE + 8, h);  // Pad to avoid conflicts

// BAD: Direct index causing conflicts
local_mem[local_id * 16];  // 16-way bank conflict possible
```

### Memory Bandwidth Optimization

| Access Pattern | Bandwidth Utilization | Optimization |
|---------------|----------------------|--------------|
| Coalesced 64-byte | 90-95% | ✅ Target: consecutive threads access consecutive addresses |
| Non-coalesced | 10-30% | ❌ Avoid: random access patterns |
| Vectorized (16-wide) | 85-90% | ✅ Use `sycl::vec<float, 16>` |

**Optimization Target:** Achieve >80% of peak bandwidth (~400 GB/s)

### GRF Mode Selection

| Mode | Registers | Use Case | Selection |
|------|-----------|----------|-----------|
| Small (128 GRF) | 128 registers per work-item | Memory-bandwidth bound kernels | Default |
| Large (256 GRF) | 256 registers per work-item | Compute-heavy kernels | Set via compile flag |

**Compile-time selection:**
```bash
# For large GRF mode
-I -Xclang -fsycl-device-code-size=256KB
```

### Work-Group Sizing

| Configuration | Threads | EU Utilization | Use Case |
|---------------|---------|----------------|----------|
| Small | 64-128 | 50-70% | Memory-bound with high register pressure |
| Medium | 256-512 | 80-95% | **Recommended** for most kernels |
| Large | 1024 | 90-100% | Compute-bound with low register pressure |

**Optimization Strategy:**
```cpp
// For BMG B60, use work-group size of 256-512
constexpr int WORK_GROUP_SIZE = 256;  // Good balance
sycl::nd_range<1>(sycl::range{N}, sycl::range{WORK_GROUP_SIZE});
```

---

## Performance Optimization Guidelines

### 1. Memory Access Optimization

**Target**: >80% bandwidth utilization

```cpp
// Vectorized load (16-wide for BMG)
using vec16_t = sycl::vec<float, 16>;

// Load coalesced
vec16_t data;
data.load(0, input_ptr + global_id * 16);

// Process entire vector
#pragma unroll
for (int i = 0; i < 16; ++i) {
    data[i] = compute(data[i]);
}
```

### 2. Sub-Group Operations

**High-efficiency cross-lane operations:**

```cpp
sycl::sub_group sg = it.get_sub_group();
int lane_id = sg.get_local_id()[0];  // 0-15 for BMG

// Shuffle for data exchange
float value = sg.shuffle_down(1);

// Reduction (faster than work-group barrier)
float sum = sg.reduce(local_value, sycl::plus<>());

// Broadcast
float shared = sg.broadcast(value, 0);
```

### 3. DPAS/XMX Matrix Operations

**For matrix multiplication and convolution:**

```cpp
// Use joint_matrix for DPAS acceleration
constexpr size_t M = 8, N = 16, K = 16;
using joint_matrix_t = sycl::ext::oneapi::experimental::matrix::joint_matrix<
    float, M, N, sycl::ext::oneapi::experimental::matrix::layout::row_major,
    sycl::ext::oneapi::experimental::matrix::use::a>;

joint_matrix_t a_mdx, b_mdx, c_mdx;
// Load, compute, store using DPAS instructions
```

### 4. Register Pressure Management

**Target**: Stay within 128 GRF (default) or 256 GRF (large mode)

```cpp
// Monitor register usage
// - Minimize local variables in kernel
// - Use float4/int4 for packed data
// - Avoid large local arrays

// GOOD: Packed data
sycl::vec<float, 4> data;  // Uses 4 registers

// BAD: Large array
float local_data[256];  // May spill to memory
```

---

## Optimization Checklist

For BMG B60 kernels, ensure:

- [ ] Work-group size: 256-512 threads
- [ ] Sub-group size: 16 lanes
- [ ] Memory access: 64-byte aligned, coalesced
- [ ] Vector width: 16 for compute-heavy kernels
- [ ] SLM usage: Stay within 256 KB per work-group
- [ ] L2 cache: Tile data to fit in 18 MB
- [ ] Register pressure: <128 GRF (default) or enable large mode
- [ ] Bandwidth target: >80% of peak (~400 GB/s)
- [ ] Use DPAS for matrix operations when applicable
- [ ] Use sub-group operations instead of work-group barriers where possible

---

## Reference Implementations

### SIMD Width Selection for BMG

```cpp
// Query optimal SIMD width at runtime
sycl::device device = queue.get_device();
size_t preferred_vec = device.get_info<sycl::info::device::native_vector_width_float>();

// For BMG B60, typically returns 8 or 16
constexpr int VEC_WIDTH = preferred_vec;
using vec_t = sycl::vec<float, VEC_WIDTH>;
```

### Local Memory Configuration

```cpp
// Calculate optimal SLM usage per work-group
constexpr int SLM_SIZE = 256 * 1024;  // 256 KB
constexpr int TILE_SIZE = SLM_SIZE / sizeof(float) / WORK_GROUP_SIZE;
// TILE_SIZE ~256 floats per work-item for 256 threads
```

---

## Related Documentation

- Intel oneAPI GPU Optimization Guide 2024.2
- Intel GPU Architecture Specifications
- SYCL Programming Guide for Intel GPUs

---

**Last Updated**: 2026-03-19
**Source**: Knowledge base synthesis from Intel documentation and optimization entries
