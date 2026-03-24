---
name: intel-gpu-e211-optimizer
description: Intel Graphics [0xe211] (Battlemage G21) GPU optimization skill for SYCL kernels. Provides architecture-specific guidance for work-group tuning, loop unrolling, and memory optimization.
license: MIT
compatibility: opencode
metadata:
  hardware: Intel Graphics [0xe211] (Battlemage G21)
  architecture: Xe2 (BMG)
  sub_group_size: 16
  slm_size: 131072  # 128 KB
  max_work_group: 1024
  compiler_flags: "-fsycl -O2"
  type: optimization
  version: "2.0"
---

## What I do

针对 Intel Graphics [0xe211] (Battlemage G21) GPU 提供 SYCL kernel 优化指导。

### 架构特性

```
Intel Graphics [0xe211] - Battlemage G21
├── Sub-group: 16 (固定值)
├── Work-group: 推荐 64-512
├── SLM: 128 KB
├── 理论带宽: ~700 GB/s
└── 最大 Work-group: 1024
```

---

## 核心优化策略

### 1. Loop Unrolling（循环展开）

**适用场景：**
- 嵌套循环层数较多的 kernel
- 循环次数固定的场景

**实施方法：**
```cpp
// 基础：展开最内层循环
#pragma unroll
for (int i = 0; i < 8; i++) {
    sum += data[i];
}

// 进阶：展开多层循环
#pragma unroll
for (int y = 0; y < 8; y++) {
    #pragma unroll
    for (int x = 0; x < 8; x++) {
        process(y, x);
    }
}

// 指定展开次数
#pragma unroll 4
for (int i = 0; i < N; i++) { ... }
```

**注意：**
- 对复杂嵌套循环效果最佳
- 简单循环提升有限
- 可能增加编译时间和代码体积

---

### 2. Work-Group 大小选择

**推荐测试范围：** 64, 128, 256, 512

**按 Kernel 类型推荐：**

```cpp
// Element-wise 操作
// 推荐：128-256
sycl::range<1> wg(256);

// Reduction 操作
// 推荐：256-512
sycl::range<1> wg(256);

// 3D 空间操作 (conv, winograd)
// 推荐：匹配数据维度
sycl::range<3> wg(16, 4, 4);  // 256 total
```

**关键原则：**
- 没有 universal optimal size
- 每个 kernel 单独测试
- 记录最优配置

---

### 3. Memory Access 优化

**Coalesced Access（合并访问）：**
```cpp
// ✅ 正确：连续线程访问连续地址
int idx = item.get_global_id(0);
float val = input[idx];

// ❌ 错误：Stride 访问
int idx = item.get_global_id(0) * 1024;
float val = input[idx];
```

**3D 数据局部性：**
```cpp
// ✅ 推荐：3D work-group 保持空间局部性
int c = item.get_global_id(0);  // Channel
int h = item.get_global_id(1);  // Height
int w = item.get_global_id(2);  // Width

// ❌ 避免：1D 展开损失局部性
int idx = item.get_global_id(0);
```

---

### 4. Sub-group 操作

**必须使用 SG=16：**
```cpp
h.parallel_for(
    sycl::nd_range<1>(global_size, local_size),
    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        sycl::sub_group sg = item.get_sub_group();
        // ...
    }
);
```

**Sub-group 操作：**
```cpp
// Shuffle
float value = sg.shuffle_down(1);

// Reduction
float sum = sg.reduce(local_value, sycl::plus<>());

// Broadcast
float shared = sg.broadcast(value, 0);
```

---

## 避免的陷阱

### ❌ 陷阱 1: Multi-thread 协作

**避免：**
```cpp
// 不要这样做
for (int k = tid; k < C; k += threads) { ... }
item.barrier();
for (int k = tid; k < C; k += threads) { ... }
item.barrier();
```

**原因：**
- 频繁同步开销
- Local memory bank conflict
- 线程负载不均衡

**替代方案：**
```cpp
// 推荐：单线程完成所有计算
int idx = item.get_global_id(0);
// 独立完成所有工作
```

### ❌ 陷阱 2: 过度向量化

**BMG 上向量化效果有限：**
- Sub-group=16 限制向量化收益
- Remainder 处理增加分支
- 编译器自动向量化通常足够

**建议：**
- 优先保证内存访问模式正确
- 仅在简单 element-wise 尝试手动向量化
- 实测验证再采用

---

## Kernel 类型专项指南

### Element-wise 操作

**特征：** add_vectors, activation, bias_add, global_scale

**配置：**
```cpp
// Work-group: 128 (consistently optimal)
// 每个线程处理 1 个元素
// 无需 local memory
// Loop unroll for complex element-wise ops

sycl::range<1> wg(128);  // Start with 128
h.parallel_for(
    sycl::nd_range<1>(DivUp(N, 128) * 128, wg),
    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        int idx = item.get_global_id(0);
        if (idx < N) {
            // For simple ops: direct access
            // For complex ops (activation): use unroll
            output[idx] = activate(input[idx]);
        }
    }
);
```

**实测数据：**
- add_vectors: WG=128 achieves 4.29 GFLOPS (vs 3.98 with WG=256)
- global_scale: WG=128 achieves 200 GFLOPS (consistently best)

### Reduction 操作

**特征：** global_avg_pool, softmax

**配置：**
```cpp
// Work-group: 256-512
// 使用 local memory 存储中间结果
// Two-stage reduction

sycl::range<1> wg(256);
sycl::local_accessor<float, 1> local_mem(wg, h);

h.parallel_for(
    sycl::nd_range<1>(..., wg),
    [=](sycl::nd_item<1> item) {
        // Stage 1: Load and first-level reduce
        float sum = 0;
        for (int i = gid; i < N; i += global_range) {
            sum += input[i];
        }
        local_mem[lid] = sum;
        item.barrier();
        
        // Stage 2: Tree reduction
        #pragma unroll
        for (int offset = wg / 2; offset > 0; offset >>= 1) {
            if (lid < offset) {
                local_mem[lid] += local_mem[lid + offset];
            }
            item.barrier();
        }
    }
);
```

### 2D/3D 卷积相关

**特征：** winograd_transform, convolution

**配置：**
```cpp
// 3D work-group for input/output transforms (spatial data)
// 1D work-group for filter transforms (compact data)
// 保持空间局部性
// 使用 register 存储 tile

// For input/output (spatial): 3D
sycl::range<3> global(blocks_c * 16, blocks_h * 4, blocks_w * 4);
sycl::range<3> local(16, 4, 4);

// For filter transform: 1D is often better
sycl::range<1> global(DivUp(C * K, 256) * 256);
sycl::range<1> local(256);

h.parallel_for(
    sycl::nd_range<3>(global, local),
    [=](sycl::nd_item<3> item) {
        int c = item.get_global_id(0);
        int h = item.get_global_id(1);
        int w = item.get_global_id(2);
        
        // Load tile to registers
        float tile[6][6];
        #pragma unroll
        for (int y = 0; y < 6; y++) {
            #pragma unroll
            for (int x = 0; x < 6; x++) {
                tile[y][x] = input[...];
            }
        }
        
        // Compute transform
        // ...
    }
);
```

**关键发现：**
- **winograd_input/output_transform**: 3D topology essential (156 GFLOPS)
- **winograd_filter_transform**: 1D work-group better than 2D (446 GFLOPS vs 287)
- Don't blindly apply 2D/3D - depends on data access pattern

### Layout Transform 操作

**特征：** nchw_to_nhwc, add_vectors_hnc_nhc

**配置：**
```cpp
// Memory-bound: optimize for bandwidth
// WG size depends on problem size:
//   - Small sizes (N=4): WG=256
//   - Medium sizes (N=16): WG=128 (232 GB/s)
//   - Large sizes (N=32): WG=256
// Grid-stride loop provides minimal benefit

sycl::range<1> wg(128);  // Start here, test 256 if needed
queue.parallel_for(
    sycl::nd_range<1>(DivUp(N*C*H*W, 128) * 128, wg),
    [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid >= total) return;
        
        // Decode index (expensive!)
        int tmp = tid;
        int c = tmp % C; tmp /= C;
        int w = tmp % W; tmp /= W;
        int h = tmp % H;
        int n = tmp / H;
        
        output[tid] = input[((n*C + c)*H + h)*W + w];
    }
);
```

**关键发现：**
- **nchw_to_nhwc**: WG=128 optimal for medium sizes (232 GB/s)
- Index decode overhead dominates - keep it simple
- Grid-stride loops add overhead for memory-bound kernels

### 复杂融合 Kernel

**特征：** winograd + SE + relu + input

**配置：**
```cpp
// 单线程处理完整计算单元
// 避免 thread 间同步
// 激进循环展开

#pragma unroll
for (int h = 0; h < 8; h += 4) {
    #pragma unroll
    for (int w = 0; w < 8; w += 4) {
        #pragma unroll
        for (int y = 0; y < 6; y++) {
            #pragma unroll
            for (int x = 0; x < 6; x++) {
                // 处理逻辑
            }
        }
    }
}
```

**关键原则：**
1. 单个线程完成所有计算，避免同步
2. 激进展开所有嵌套循环
3. 使用 register 存储临时数组
4. 不要使用 multi-thread 协作

---

## 优化检查清单

### 编译前检查

- [ ] 所有循环都有 `#pragma unroll`
- [ ] 使用了 `[[sycl::reqd_sub_group_size(16)]]`
- [ ] 内存访问连续 (coalesced)
- [ ] Work-group 大小在 64-512 范围内
- [ ] 无不必要的 `item.barrier()`
- [ ] 无 multi-thread 协作模式

### 运行时检查

- [ ] GFLOPS 在合理范围
- [ ] 带宽利用率正常
- [ ] 结果数值正确

---

## Compiler Flags

### 推荐编译选项

```bash
# 标准优化
icpx -fsycl -O2 -std=c++17 kernel.cpp -o kernel

# 激进优化
icpx -fsycl -O3 -std=c++17 \
    -ffast-math \
    -funroll-loops \
    kernel.cpp -o kernel

# AOT 编译
icpx -fsycl -O3 \
    -fsycl-targets=spir64 \
    kernel.cpp -o kernel
```

---

## When to use me

1. **开发新 kernel** - 从优化的模板开始
2. **性能调优** - 应用架构特定的优化策略
3. **代码审查** - 检查是否符合 BMG 最佳实践
4. **性能分析** - 识别瓶颈并提供优化建议

---

## Key Takeaways

### 三大原则

1. **Loop Unrolling 是高效优化手段**
   - 对复杂嵌套循环尤其有效
   - 简单添加 `#pragma unroll`

2. **没有 Universal Optimal WG Size**
   - 每个 kernel 需要单独测试
   - 记录最优配置

3. **避免 Multi-thread 协作**
   - 可能导致严重性能下降
   - 优先使用单线程处理完整数据单元

---

**Version:** 2.2  
**Target:** Intel Graphics [0xe211] (Battlemage G21)

---

## New Findings (v2.2)

### Work-Group Size Selection Refined

Based on 15-kernel test data:

| Kernel Type | Recommended WG | Performance |
|-------------|---------------|-------------|
| Simple element-wise | 128 | +5-8% vs 256 |
| Complex element-wise | 128 | Consistently optimal |
| Layout transform | Size-dependent | Test both 128, 256 |
| Filter transform | 256 | Simpler is better |
| Spatial transforms | 16×4×4 3D | Essential |

### Anti-Pattern Refined

**❌ Avoid 2D/3D for compact data:**
- winograd_filter_transform: 2D (16×8) is 35% slower than 1D
- Only use multi-D for spatial data with locality

**✅ Prefer WG=128 for element-wise:**
- Tested across 4 kernels (add_vectors, global_scale, add_bias_*)
- 128 consistently outperforms 256 by 5-15%
