---
name: intel-gpu-e211-optimizer
description: Intel Graphics [0xe211] GPU optimization skill for SYCL kernels. Optimized for current test environment with 128KB SLM, sub-group sizes 16/32, and 4-wide vectorization.
license: MIT
compatibility: opencode
metadata:
  hardware: Intel Graphics [0xe211]
  architecture: Xe (Gen12.7)
  sub_group_sizes: [16, 32]
  slm_size: 131072  # 128 KB
  max_work_group: 1024
  native_vector_width: 4
  compute_units: 160
  compiler_flags: "-fsycl -O2"
  type: optimization
  version: "1.0"
---

## What I do

针对 Intel Graphics [0xe211] GPU 提供优化的 SYCL 内核优化服务，基于 Phase 0 验证结果。

### 与 BMG B60 的差异

| 参数 | 当前 GPU (0xe211) | BMG B60 (目标) | 优化策略调整 |
|------|-------------------|----------------|--------------|
| **SLM** | 128 KB | 256 KB | 使用 128 KB 上限 |
| **Sub-group** | 16, 32 | 16 | 两者都支持，推荐 16 |
| **Vector Width** | 4 (native) | 16 (optimal) | 使用 4-wide，代码兼容 16 |
| **CUs** | 160 | ~128 | 更多并行单元 |

### 核心优化领域

1. **Work-Group 优化 (512-1024)**
   - 推荐: WG=512 (平衡)
   - 最大: WG=1024 (高并行)
   - 基于 Phase 0 验证结果

2. **Sub-Group 优化 (16/32)**
   - 两者性能相当
   - 推荐 SG=16 (BMG 兼容)
   - 使用 `sycl::reqd_sub_group_size(16)`

3. **内存层次优化**
   - SLM (128 KB): 保守使用
   - 避免 256 KB+ 的 tile sizes
   - Bank conflict 避免 (padding)

4. **向量化 (4-wide)**
   - Native width: 4
   - 使用 `sycl::vec<float, 4>`
   - 代码可升级到 16-wide for BMG

## Optimization Categories

### 1. Work-Group Configuration

**Phase 0 验证结果:**
```
Best: WG=1024 (1.01-1.06x speedup)
Safe: WG=512 (consistent performance)
Baseline: WG=256
```

**推荐模板:**
```cpp
// 对于当前 GPU (0xe211)
constexpr int WORK_GROUP_SIZE = 512;  // 推荐
// 或
constexpr int WORK_GROUP_SIZE = 1024;  // 最大并行

sycl::nd_range<1> nd_range(
    sycl::range<1>((N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE * WORK_GROUP_SIZE),
    sycl::range<1>(WORK_GROUP_SIZE)
);
```

**适用场景:**
- WG=256: 内存带宽受限，寄存器压力高
- WG=512: **推荐**，大多数 kernel 的最佳选择
- WG=1024: 计算密集，低寄存器压力

### 2. Sub-Group Operations

**验证结果:** SG=16 和 SG=32 性能相当，推荐 SG=16

**标准模板:**
```cpp
// 使用 SG=16 (BMG 兼容)
queue.parallel_for(
    sycl::nd_range<1>(global_size, WORK_GROUP_SIZE),
    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        sycl::sub_group sg = item.get_sub_group();
        int lane_id = sg.get_local_id()[0];  // 0-15
        
        // Shuffle
        float value = sg.shuffle_down(1);
        
        // Reduction via permute (sg.reduce 可能不可用)
        float val = input[i];
        for (int offset = 16 / 2; offset > 0; offset /= 2) {
            float tmp = sycl::permute_group_by_xor(sg, val, offset);
            val += tmp;
        }
        
        // Broadcast
        float shared = sg.broadcast(value, 0);
    }
);
```

**注意:** 当前 GPU 的 SYCL 版本可能不支持 `sg.reduce()`，使用 `permute_group_by_xor` 代替。

### 3. SLM (Shared Local Memory) 优化

**⚠️ 关键限制: 128 KB (vs BMG 256 KB)**

**安全 Tile 大小:**
```cpp
// ✅ 安全 (≤128 KB)
constexpr int TILE_SIZE = 128;  // 128x128 float = 64 KB
constexpr int PADDING = 8;      // 避免 bank conflicts
sycl::local_accessor<float, 2> local_mem(
    sycl::range<2>(TILE_SIZE + PADDING, TILE_SIZE + PADDING), h
);

// ❌ 危险 (>128 KB)
// constexpr int TILE_SIZE = 256;  // 256x256 float = 256 KB - 超出限制!
```

**Bank Conflict 避免:**
```cpp
// GOOD: 添加 padding
constexpr int TILE_M = 128;
sycl::local_accessor<float, 2> a_tile(TILE_M + 8, TILE_K, h);

// 协作加载
for (int i = local_id; i < TILE_M * TILE_K; i += WORK_GROUP_SIZE) {
    int row = i / TILE_K;
    int col = i % TILE_K;
    a_tile[row][col] = A[...];  // 避免 stride=2^n
}
```

### 4. Vectorized Memory Access (4-wide)

**当前 GPU Native: 4-wide**

**优化代码:**
```cpp
// 4-wide 向量化 (匹配 native width)
using vec4_t = sycl::vec<float, 4>;

// 合并加载
int base_idx = global_id * 4;
if (base_idx + 4 <= n) {
    vec4_t data = *reinterpret_cast<const vec4_t*>(input_ptr + base_idx);
    
    // 向量化计算
    data = data * 2.0f + 1.0f;
    
    // 存储
    *reinterpret_cast<vec4_t*>(output_ptr + base_idx) = data;
}
```

**BMG 升级路径:**
```cpp
// 条件编译支持 BMG
#ifdef BMG_B60_MODE
    using vec_t = sycl::vec<float, 16>;  // BMG optimal
#else
    using vec_t = sycl::vec<float, 4>;   // Current GPU native
#endif
```

### 5. Memory Access Optimization

**目标: >80% 带宽利用率**

| Access Pattern | Bandwidth | Optimization |
|---------------|-----------|--------------|
| Coalesced 64-byte | 85-90% | ✅ 连续线程访问连续地址 |
| Vectorized (4-wide) | 80-85% | ✅ 使用 vec4 |
| Non-coalesced | 10-30% | ❌ 避免随机访问 |

**最佳实践:**
```cpp
// GOOD: 合并访问
int global_id = item.get_global_id(0);
float4 data = input[global_id];  // 16-byte aligned

// BAD: 分散访问
int global_id = item.get_global_id(0);
float data = input[global_id * 1024];  // 大 stride
```

## Optimization Checklist

### 当前 GPU (0xe211) 优化检查表

- [ ] Work-group size: 512 (推荐) 或 1024 (最大)
- [ ] Sub-group size: 16 (BMG 兼容) 或 32
- [ ] SLM usage: **≤ 128 KB** (关键限制!)
- [ ] Vector width: 4 (native), 代码兼容 16
- [ ] Memory access: 64-byte aligned, coalesced
- [ ] Bank conflicts: 添加 padding 避免
- [ ] L2 cache: Tile data for reuse
- [ ] Register pressure: Monitor for spillage

### BMG B60 迁移检查表

- [ ] SLM 扩展到 256 KB
- [ ] Vector width 升级到 16
- [ ] 验证 SG=16 性能
- [ ] XMX DPAS 集成
- [ ] 重新调优 WG size

## Performance Targets

### 当前 GPU (0xe211)

| Metric | Target | Notes |
|--------|--------|-------|
| Memory Bandwidth | >80% peak | ~80-90 GB/s estimated |
| EU Utilization | >80% | 160 CUs available |
| SLM Efficiency | 100% | No bank conflicts |
| Vector Efficiency | 100% | Use all 4 lanes |
| Kernel Occupancy | >90% | WG size optimization |

## Compiler Flags

### 基础编译 (当前 GPU)
```bash
icpx -fsycl -O2 -std=c++17 kernel.cpp -o kernel
```

### AOT 编译 (推荐)
```bash
# 检测当前 GPU
icpx -fsycl -O3 -std=c++17 \
  -fsycl-targets=spir64 \
  kernel.cpp -o kernel
```

### Large GRF 模式
```bash
icpx -fsycl -O3 \
  -Xclang -fsycl-device-code-size=256KB \
  kernel.cpp -o kernel_large_grf
```

### BMG B60 AOT (未来验证)
```bash
icpx -fsycl -O3 \
  -fsycl-targets=spir64_gen-unknown-unknown-sycldevice \
  -Xs "-device bmg" \
  -DBMG_B60_MODE \
  kernel.cpp -o kernel_bmg
```

## Code Examples

### Example 1: Vector Add (Optimized)

```cpp
#include <sycl/sycl.hpp>

void vector_add_optimized(sycl::queue& q, float* c, const float* a, const float* b, int n) {
    constexpr int WG_SIZE = 512;  // Optimized for 0xe211
    
    q.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>((n + WG_SIZE - 1) / WG_SIZE * WG_SIZE),
            sycl::range<1>(WG_SIZE)
        ),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
            int i = item.get_global_id(0);
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        }
    );
}
```

### Example 2: Matrix Multiplication with SLM

```cpp
// Safe for 128 KB SLM
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 64;
// Total: 3 * 64 * 64 * 4 bytes = 48 KB < 128 KB ✅

sycl::local_accessor<float, 2> a_tile(sycl::range<2>(TILE_M, TILE_K), h);
sycl::local_accessor<float, 2> b_tile(sycl::range<2>(TILE_K, TILE_N), h);
```

### Example 3: Reduction with Sub-group

```cpp
float local_sum = compute_local(item);
auto sg = item.get_sub_group();

// Manual reduction (sg.reduce may not be available)
for (int offset = 16 / 2; offset > 0; offset /= 2) {
    float tmp = sycl::permute_group_by_xor(sg, local_sum, offset);
    local_sum += tmp;
}
```

## When to use me

1. **Testing on current GPU**: Optimize for 0xe211 test environment
2. **BMG B60 preparation**: Write forward-compatible code
3. **SLM-sensitive kernels**: Stay within 128 KB limit
4. **Performance debugging**: Identify architecture-specific bottlenecks
5. **Migration planning**: Prepare code for BMG B60 upgrade

## Differences from BMG B60 Optimizer

| Aspect | This Skill (0xe211) | BMG B60 Skill |
|--------|---------------------|---------------|
| SLM Size | 128 KB | 256 KB |
| Vector Width | 4 (native) | 16 (native) |
| Sub-group | 16, 32 | 16 |
| XMX DPAS | ❌ Not available | ✅ Available |
| Target | Test/Development | Production |
| Code Compatibility | Forward to BMG | Optimal on BMG |

## Related Skills

- `bmg-b60-optimizer`: Production BMG B60 optimizations
- `b60-sycl-builder`: SYCL compilation in B60 container

## References

- Phase 0 Validation Report: `performance_optimization/04_results/reports/phase0_validation_report.md`
- Intel Graphics [0xe211] Specifications
- SYCL 2020 Specification
- Phase 0 Test Results:
  - `wg_size_sweep_results.txt`
  - `sg_size_test_results.txt`
  - `vector_width_test_results.txt`

---

**Last Updated**: 2026-03-19
**Version**: 1.0
**Target Hardware**: Intel Graphics [0xe211] (Test Environment)
**Future Target**: Intel BMG B60 (Production)