---
name: bmg-b60-optimizer
description: Intel BMG B60 GPU optimization skill for SYCL kernels. Provides architecture-specific optimizations including sub-group tuning, XMX matrix extensions, SLM/L2 cache management, and work-group sizing.
license: MIT
compatibility: opencode
metadata:
  hardware: Intel BMG B60 GPGPU
  architecture: Xe2
  sub_group_size: 16
  slm_size: 262144  # 256 KB
  l2_cache: 18874368  # 18 MB
  max_work_group: 1024
  compiler_flags: "-fsycl -O2"
  type: optimization
  version: "1.0"
---

## What I do

针对 Intel BMG B60 GPU 提供专业的 SYCL 内核优化服务，基于 Xe2 架构特性进行代码调优。

### 核心优化领域

1. **Sub-Group 优化 (16-wide)**
   - 默认 sub-group size: 16 lanes
   - 优化 sub-group shuffle/reduce 操作
   - 避免 bank conflicts

2. **XMX 矩阵扩展 (DPAS)**
   - 8M×16N×16K FP16/BF16 矩阵乘法
   - joint_matrix API 集成
   - 2 TFLOPS/EU 峰值性能

3. **内存层次优化**
   - SLM (256 KB): Work-group 内数据复用
   - L2 Cache (18 MB): 中间结果缓存
   - HBM2e (~500 GB/s): 合并访问模式

4. **Work-Group 调优**
   - 推荐大小: 256-512 threads
   - 最大大小: 1024 threads
   - EU 利用率目标: >90%

## Optimization Categories

### 1. Sub-Group Operations (高效跨lane操作)

**标准模板:**
```cpp
// 获取 sub-group
sycl::sub_group sg = it.get_sub_group();
int lane_id = sg.get_local_id()[0];  // 0-15

// Shuffle (数据交换)
float value = sg.shuffle_down(1);

// Reduction (比 work-group barrier 更快)
float sum = sg.reduce(local_value, sycl::plus<>());

// Broadcast
float shared = sg.broadcast(value, 0);
```

**适用场景:**
- 规约操作 (reduce)
- 数据广播 (broadcast)
- 跨 lane 数据交换 (shuffle)

### 2. Vectorized Memory Access (向量化加载)

**BMG B60 优化代码:**
```cpp
// 16-wide 向量化 (匹配 BMG sub-group)
using vec16_t = sycl::vec<float, 16>;

// 合并加载
vec16_t data;
data.load(0, input_ptr + global_id * 16);

// 处理整个向量
#pragma unroll
for (int i = 0; i < 16; ++i) {
    data[i] = compute(data[i]);
}

// 存储
result_vec.store(0, output_ptr + global_id * 16);
```

**性能目标:** >80% 峰值带宽 (~400 GB/s)

### 3. SLM (Shared Local Memory) 优化

**配置模板:**
```cpp
// 避免 bank conflicts: 添加 padding
constexpr int TILE_SIZE = 256;
constexpr int PADDING = 8;
sycl::local_accessor<float, 1> local_mem(TILE_SIZE + PADDING, h);

// 访问模式: 连续线程访问连续地址
int local_id = it.get_local_id(0);
local_mem[local_id] = global_data[global_id];  // Good

// 避免: local_mem[local_id * 16]  // 可能导致 bank conflicts
```

**容量限制:** 256 KB per XeCore

### 4. XMX Matrix Extensions (DPAS)

**矩阵乘法模板:**
```cpp
#include <sycl/ext/oneapi/experimental/matrix.hpp>

using namespace sycl::ext::oneapi::experimental::matrix;

// BMG 最优配置: 8x16x16
constexpr size_t M = 8, N = 16, K = 16;

// 矩阵类型定义
using a_layout = layout::row_major;
using b_layout = layout::col_major;
using c_layout = layout::row_major;

// joint_matrix 声明
joint_matrix<sycl::half, M, K, a_layout, use::a> a_mdx;
joint_matrix<sycl::half, K, N, b_layout, use::b> b_mdx;
joint_matrix<float, M, N, c_layout, use::accumulator> c_mdx;

// 加载、计算、存储
load_matrix_sync(a_mdx, a_ptr + a_offset, K);
load_matrix_sync(b_mdx, b_ptr + b_offset, N);
load_matrix_sync(c_mdx, c_ptr + c_offset, N);

// DPAS 计算
joint_matrix_mad(a_mdx, b_mdx, c_mdx);

// 存储结果
store_matrix_sync(c_ptr + c_offset, c_mdx, N);
```

**支持数据类型:**
- FP16/BF16: 8M×16N×16K, SYS depth 8
- TF32: 4M×4N×4K, SYS depth 8

### 5. Work-Group 配置

**推荐模板:**
```cpp
// BMG B60 推荐配置
constexpr int WORK_GROUP_SIZE = 256;  // 平衡选择
constexpr int ITEMS_PER_THREAD = 16;  // 匹配 sub-group

sycl::nd_range<1> nd_range(
    sycl::range<1>(total_items),
    sycl::range<1>(WORK_GROUP_SIZE)
);

queue.parallel_for(nd_range, [=](sycl::nd_item<1> it) {
    size_t global_id = it.get_global_id(0);
    size_t local_id = it.get_local_id(0);
    sycl::sub_group sg = it.get_sub_group();
    
    // 每个 work-item 处理 16 个元素
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        // 处理逻辑
    }
});
```

**配置表:**

| Work-Group Size | EU Utilization | Use Case |
|----------------|----------------|----------|
| 64-128 | 50-70% | Memory-bound, high register pressure |
| **256-512** | **80-95%** | **Recommended for most kernels** |
| 1024 | 90-100% | Compute-bound, low register pressure |

### 6. GRF (General Register File) 模式

**编译时选择:**
```bash
# Small mode (默认, 128 GRF) - 内存带宽受限
-fsycl

# Large mode (256 GRF) - 计算密集型
-fsycl -Xclang -fsycl-device-code-size=256KB
```

**选择建议:**
- 内存带宽受限: 小模式 (更多线程并发)
- 计算密集型: 大模式 (更多寄存器)

## Optimization Checklist

优化前检查:

- [ ] Work-group size: 256-512 threads
- [ ] Sub-group size: 16 lanes (BMG默认)
- [ ] Memory access: 64-byte aligned, coalesced
- [ ] Vector width: 16 for compute-heavy kernels
- [ ] SLM usage: < 256 KB per work-group
- [ ] L2 cache: Tile data to fit in 18 MB
- [ ] Register pressure: < 128 GRF (默认) 或启用大模式
- [ ] Bandwidth target: > 80% peak (~400 GB/s)
- [ ] DPAS: 矩阵运算使用 XMX 扩展
- [ ] Sub-group ops: 替代 work-group barriers

## API Reference

### Query Device Info

```cpp
sycl::queue queue;
sycl::device device = queue.get_device();

// Sub-group sizes
auto sg_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
size_t sg_size = sg_sizes.front();  // BMG: 16

// Local memory size
size_t slm_size = device.get_info<sycl::info::device::local_mem_size>();
// BMG: 262144 (256 KB)

// Max work-group size
size_t max_wg = device.get_info<sycl::info::device::max_work_group_size>();
// BMG: 1024

// Preferred vector width
size_t vec_width = device.get_info<sycl::info::device::native_vector_width_float>();
// BMG: 8 or 16
```

### Common Optimized Patterns

#### Pattern 1: Parallel Reduction
```cpp
// 使用 sub-group reduce 替代 barrier
float local_sum = compute_local(item);
float group_sum = sg.reduce(local_sum, sycl::plus<>());
```

#### Pattern 2: Matrix Multiply Tile
```cpp
// SLM tiling for matrix A
constexpr int TILE_M = 64;
constexpr int TILE_K = 64;
sycl::local_accessor<float, 2> a_tile(TILE_M, TILE_K, h);

// 协作加载到 SLM
for (int i = local_id; i < TILE_M * TILE_K; i += WORK_GROUP_SIZE) {
    int row = i / TILE_K;
    int col = i % TILE_K;
    a_tile[row][col] = A[...];
}
it.barrier(sycl::access::fence_space::local_space);
```

#### Pattern 3: Element-wise with Vectorization
```cpp
// 16-wide processing
constexpr int VEC = 16;
using vec_t = sycl::vec<float, VEC>;

size_t base_idx = global_id * VEC;
vec_t input;
input.load(0, &data[base_idx]);

// 向量化计算
vec_t output = func(input);

output.store(0, &result[base_idx]);
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Memory Bandwidth | > 400 GB/s | 80% of peak |
| EU Utilization | > 90% | 7-8 threads/EU |
| Sub-group Efficiency | 100% | Use all 16 lanes |
| SLM Bank Conflicts | 0 | Proper padding |
| L2 Cache Hit Rate | > 90% | Data locality |
| DPAS Utilization | > 80% | XMX pipeline |

## Compiler Flags

### 基础编译
```bash
icpx -fsycl -O2 -std=c++17 kernel.cpp -o kernel
```

### BMG 优化编译
```bash
icpx -fsycl -O3 -std=c++17 \
  -fsycl-targets=spir64 \
  -Xclang -fsycl-device-code-split=per_kernel \
  kernel.cpp -o kernel
```

### Large GRF 模式
```bash
icpx -fsycl -O3 \
  -Xclang -fsycl-device-code-size=256KB \
  kernel.cpp -o kernel
```

### AOT 编译 (BMG)
```bash
icpx -fsycl -O3 \
  -fsycl-targets=spir64_gen-unknown-unknown-sycldevice \
  -Xs "-device bmg" \
  kernel.cpp -o kernel
```

## When to use me

1. **新内核开发**: 从 BMg 优化模板开始
2. **性能调优**: 分析并应用架构特定优化
3. **代码审查**: 检查是否符合 BMG 最佳实践
4. **移植 CUDA**: CUDA to SYCL + BMG 优化
5. **性能分析**: 识别瓶颈并提供优化建议

## Limitations

- 需要 oneAPI 2024.2+ 以支持最新 XMX 特性
- DPAS 仅支持特定数据类型 (FP16/BF16/TF32)
- Large GRF 模式会减少并发线程数
- Sub-group size 16 是 BMG 固定值

## Related Skills

- `b60-sycl-builder`: SYCL 编译和测试
- `remote-cuda-builder`: CUDA 参考实现构建

## References

- Intel oneAPI GPU Optimization Guide 2024.2
- Intel GPU Architecture Specifications
- SYCL 2020 Specification
- Intel XMX Matrix Extensions

---

**Last Updated**: 2026-03-19
**Version**: 1.0
**Target Hardware**: Intel BMG B60 (Xe2 Architecture)