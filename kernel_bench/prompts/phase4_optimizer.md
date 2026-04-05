# Phase X: Kernel Optimization Guide
# 基于LCZero 28核优化项目的实战经验

## Overview / 概述

本指南基于Intel BMG B60 GPU上对28个LCZero CUDA核进行SYCL优化的实战经验，涵盖5轮优化过程和关键发现。

## Real-World Results / 实战结果

| 指标 | 数值 |
|------|------|
| **测试核总数** | 28个 |
| **峰值性能** | 1094 GFLOPS (layer_norm) |
| **最佳提升** | +179% (softmax: 26→73 GFLOPS) |
| **次佳提升** | +56% (batch_norm: 70→109 GFLOPS) |
| **已优化核** | 2/28 (7%) |
| **已最优核** | 26/28 (93%) |

**关键发现**: 93%的核已经是最优状态，只有特定类型的核有优化空间。

---

## 5-Round Optimization Process / 5轮优化流程

### Round 1: Baseline Profiling / 基线分析

**目标**: 建立性能基线，识别优化潜力

**步骤**:
1. 编译并运行原始SYCL代码
2. 记录以下指标：
   - 执行时间 (ms)
   - 计算性能 (GFLOPS)
   - 内存带宽利用率 (%)
   - Occupancy (%)

**决策点**:
```
内存带宽利用率 > 70%?
├── YES: 核可能已经是最优状态
│   └── 进入Round 2验证
└── NO: 存在优化空间
    └── 检查内存合并访问模式
```

**输出要求**:
```
Kernel: {kernel_name}
Baseline Performance: {X} GFLOPS
Memory Bandwidth: {Y}%
Occupancy: {Z}%
Status: [HIGH_POTENTIAL | MEDIUM_POTENTIAL | ALREADY_OPTIMAL]
```

---

### Round 2: Memory Optimization / 内存优化

**目标**: 优化内存访问模式

**检查清单**:
- [ ] **内存合并访问**: 验证threadIdx.x对应连续内存地址
- [ ] **工作组大小**: 测试128, 256, 512
- [ ] **向量化加载**: 尝试加载2-4个元素
- [ ] **FP16精度**: 如果内存受限，尝试FP16

**代码模式**:

```cpp
// ✅ 好的内存访问模式
int idx = item.get_global_id(0);
float4 vec = reinterpret_cast<const float4*>(input)[idx];

// ❌ 坏的内存访问模式
int idx = item.get_global_id(0) * 256;  // 高步长
float val = input[idx];  // 16个独立事务
```

**验证指标**:
- 内存事务数减少
- 带宽利用率提升
- 性能改善 > 10%

**决策点**:
```
性能提升 > 20%?
├── YES: 继续Round 3
├── NO (10-20%): 继续Round 3
└── NO (<10%): 
    ├── 内存带宽 > 70%? → 核已最优，停止
    └── 内存带宽 < 70%? → 检查合并访问
```

---

### Round 3: SLM Optimization / SLM优化

**目标**: 使用SLM缓存小参数，减少全局内存访问

**适用条件**:
- 核有小的、频繁重用的参数（< 100KB）
- 参数被多个线程重复读取
- 典型场景：batch_norm, softmax, layer_norm

**代码模板**:

```cpp
// batch_norm: 缓存5个参数 per channel
q.parallel_for(..., [=](sycl::nd_item<1> item) {
    int lid = item.get_local_id(0);
    sycl::local_accessor<float, 1> local_params(C * 5, item);
    
    // 1. 协作加载参数到SLM（只需一次）
    if (lid < C) {
        local_params[lid * 5 + 0] = mean[lid];
        local_params[lid * 5 + 1] = var[lid];
        local_params[lid * 5 + 2] = gamma[lid];
        local_params[lid * 5 + 3] = beta[lid];
        local_params[lid * 5 + 4] = scale[lid];
    }
    item.barrier();
    
    // 2. 从SLM快速访问（重复使用）
    int idx = item.get_global_id(0);
    int c = (idx / HW) % C;
    
    float m = local_params[c * 5 + 0];  // SLM访问
    float v = local_params[c * 5 + 1];  // SLM访问
    // ... 其他参数
    
    output[idx] = (input[idx] - m) / sycl::sqrt(v + epsilon);
});
```

**关键要点**:
1. **SLM大小**: BMG有256KB，但应保持< 64KB以维持高occupancy
2. **Bank冲突**: Xe2有16个bank，避免步长为16的访问
3. **Barrier位置**: 只在数据同步点放置barrier

**实际结果**:
- batch_norm: +56% (70→109 GFLOPS)
- softmax: +179% (26→73 GFLOPS)

**决策点**:
```
核有小的、可重用的参数?
├── YES: 尝试SLM缓存
│   └── 提升 > 20%? → 保留优化
│   └── 提升 < 20%? → 回退到Round 2版本
└── NO: 跳过SLM优化
    └── 核可能是element-wise，保持Round 2优化
```

---

### Round 4: Cooperative Algorithms / 协作算法

**目标**: 使用sub-group shuffle优化reduction操作

**适用场景**:
- Reduction: softmax, global_avg_pool, sum
- Scan/prefix sum
- 需要cross-thread通信的场景

**代码模板**:

```cpp
// softmax: 三阶段协作reduction
q.parallel_for(..., [=](sycl::nd_item<2> item) {
    sycl::sub_group sg = item.get_sub_group();
    int row = item.get_group(0);
    int lid = item.get_local_id(0);
    
    // Phase 1: 使用shuffle找max
    float local_max = -INFINITY;
    for (int i = lid; i < cols; i += wg_size) {
        local_max = sycl::max(local_max, input[row * cols + i]);
    }
    // Sub-group reduction (16 lanes)
    local_max = sycl::reduce(sg, local_max, sycl::maximum<float>());
    
    // Phase 2: 计算exp(x-max)和sum
    float sum = 0;
    for (int i = lid; i < cols; i += wg_size) {
        float val = sycl::exp(input[row * cols + i] - local_max);
        sum += val;
        output[row * cols + i] = val;
    }
    sum = sycl::reduce(sg, sum, sycl::plus<float>());
    
    // Phase 3: 广播sum并归一化
    float total_sum = sycl::group_broadcast(sg, sum, 0);
    for (int i = lid; i < cols; i += wg_size) {
        output[row * cols + i] /= total_sum;
    }
});
```

**关键要点**:
1. **Sub-group大小**: Xe2固定为16
2. **Shuffle vs SLM**: Shuffle更快，但只适用于16线程内
3. **多sub-group**: 需要SLM在sub-group间通信

**实际结果**:
- softmax: +179% (使用cooperative reduction)
- global_avg_pool: +60% (使用single-thread per output)

---

### Round 5: Grid Configuration / 网格配置

**目标**: 优化网格维度以获得最佳occupancy

**关键发现**: Grid配置比算法更重要！

**案例**: layer_norm
```cpp
// ❌ 2D grid: 1094 → 33 GFLOPS (97%性能损失!)
sycl::range<2> global(N, C);
sycl::range<2> local(1, C);

// ✅ 1D flattened grid: 1094 GFLOPS (峰值)
sycl::range<1> global(N * wg_size);
sycl::range<1> local(wg_size);
```

**指导原则**:
1. **尽可能使用1D grid**: 更好的occupancy和调度
2. **工作组大小**: 128或256是甜点
3. **全局大小**: 至少覆盖所有数据，向上取整到wg_size倍数

**验证步骤**:
1. 测试不同grid配置
2. 记录每种配置的GFLOPS
3. 选择最佳配置

---

## Anti-Patterns / 反模式（不要这样做）

### 1. 破坏内存合并访问
```cpp
// ❌ 不要改变原始访问模式
int plane = idx % planes;  // 改变了顺序！
int pos = idx / planes;

// ✅ 保持CUDA的原始模式
int plane = idx / 64;
int pos = idx % 64;
```
**后果**: expand_planes性能下降10%

### 2. 对Element-wise核使用SLM
```cpp
// ❌ Element-wise不需要SLM
q.parallel_for(..., [=](sycl::nd_item<1> item) {
    sycl::local_accessor<float, 1> local(256, item);  // 不需要！
    item.barrier();  // 不必要的barrier
    output[idx] = op(input[idx]);
});
```
**后果**: 20-50%性能下降

### 3. 错误的网格维度
```cpp
// ❌ 2D grid限制并行度
sycl::range<2> global(batch, channels);

// ✅ 1D grid最大化并行度
sycl::range<1> global(batch * channels * wg_size);
```
**后果**: layer_norm从1094降到33 GFLOPS

### 4. 过度优化
```cpp
// ❌ 不必要的复杂化
// 原始代码已经是最优，不要"改进"
```
**后果**: 26/28核已经是原始最优状态

---

## Optimization Decision Tree / 优化决策树

```
开始优化
│
├─ Round 1: 基线分析
│   ├─ 性能 > 50%峰值? → 可能已最优
│   └─ 内存带宽 > 70%? → 内存受限，检查合并访问
│
├─ Round 2: 内存优化
│   ├─ 工作组大小: 测试128, 256
│   ├─ 向量化: 尝试float4加载
│   └─ FP16: 如果内存受限
│   └─ 提升 < 10% 且 内存带宽 > 70%? → 停止，核已最优
│
├─ Round 3: SLM优化（仅适用特定核）
│   ├─ 有小的、重用参数? (< 100KB)
│   │   ├─ batch_norm: 缓存mean/var/gamma/beta
│   │   ├─ softmax: 缓存max和sum
│   │   └─ layer_norm: 缓存scale/shift
│   └─ 无重用参数? → 跳过SLM
│
├─ Round 4: 协作算法（Reduction核）
│   ├─ 是reduction/scan操作?
│   │   ├─ 使用sub-group shuffle
│   │   └─ 使用SLM在sub-group间通信
│   └─ 不是reduction? → 跳过
│
└─ Round 5: 网格配置
    ├─ 当前是2D/3D grid?
    │   └─ 尝试flatten到1D
    └─ 测试不同wg_size: 128, 256, 512
```

---

## When to Stop / 何时停止

**立即停止的情况**:
1. 性能达到>70%理论峰值
2. 内存带宽利用率>90%
3. 连续两轮优化提升<5%
4. 出现性能回退(regression)

**继续优化的情况**:
1. 性能<50%峰值且内存带宽<50%
2. 有明显的内存合并问题
3. 有未使用的SLM优化机会

---

## Performance Checklist / 性能检查清单

每轮优化后检查：

- [ ] **编译成功**: 无编译错误
- [ ] **数值正确**: 最大误差 < 1e-5
- [ ] **性能提升**: 相比上一轮有改善
- [ ] **内存带宽**: 记录并比较
- [ ] **Occupancy**: >50%为良好
- [ ] **无回归**: 不低于原始版本

---

## Example Optimization Log / 优化日志示例

```
=== Kernel: batch_norm ===

Round 1 (Baseline):
  Performance: 70 GFLOPS
  Memory Bandwidth: 35%
  Occupancy: 65%
  Status: HIGH_POTENTIAL

Round 2 (Memory):
  Changes: WG=256, vectorized load
  Performance: 85 GFLOPS (+21%)
  Memory Bandwidth: 42%
  Decision: Continue to Round 3

Round 3 (SLM):
  Changes: Cache 5 parameters per channel (20KB SLM)
  Performance: 109 GFLOPS (+56% total)
  Memory Bandwidth: 55%
  Occupancy: 70%
  Decision: SUCCESS - Significant improvement

Round 4 (Skipped):
  Not a reduction kernel

Round 5 (Grid):
  Tested 1D vs 2D: No improvement
  Final Performance: 109 GFLOPS

=== FINAL RESULT ===
Baseline: 70 GFLOPS
Optimized: 109 GFLOPS
Improvement: +56%
Techniques: SLM parameter caching
Status: SUCCESS
```

---

## Best Practices Summary / 最佳实践总结

1. **先分析再优化**: 70%时间用于分析，30%用于优化
2. **保留原始代码**: 作为baseline和fallback
3. **一次一个优化**: 隔离变量，准确测量影响
4. **内存合并第一**: 这是最重要的优化
5. **SLM用于参数**: 不是用于element-wise数据
6. **网格配置关键**: 有时比算法更重要
7. **93%核已最优**: LCZero代码质量很高，不要盲目优化
8. **测试准确性**: 每次优化后验证数值正确性
9. **记录所有结果**: 包括失败的尝试
10. **知道何时停止**: 不要过度优化

---

## Tools and Commands / 工具和命令

### 编译命令
```bash
# SYCL编译
icpx -fsycl -O3 -std=c++17 \
    -fsycl-targets=spir64_gen \
    -Xsycl-target-backend "-device bmg" \
    kernel.cpp -o kernel

# 性能分析
icpx -fsycl -O3 -fsycl-enable-profiling \
    kernel.cpp -o kernel
```

### 性能测试脚本
```bash
#!/bin/bash
# test_performance.sh

KERNEL=$1
REPEAT=50

echo "Testing $KERNEL..."
for i in $(seq 1 $REPEAT); do
    ./$KERNEL >> results.txt
done

# 计算平均GFLOPS
awk '{sum+=$1} END {print "Average: " sum/NR " GFLOPS"}' results.txt
```

---

**Version**: 1.0
**Based on**: LCZero 28-kernel optimization project
**Last Updated**: 2026-03-31
**Target**: Intel BMG B60 GPU (Xe2 Architecture)