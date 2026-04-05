# RMSNorm CUDA → SYCL 转换分析

## 源文件
TurboDiffusion/turbodiffusion/ops/norm/rmsnorm.hpp

## 核心结构

### 1. 模板参数 (行8-15)
```cpp
template <
  class InputDtype_,      // 输入类型
  class OutputDtype_,     // 输出类型
  class WeightDtype_,     // weight类型
  int MaxHiddenSize_,     // 最大hidden维度
  int NumThrPerCta_,      // 每个CTA线程数
  bool IsEven             // 是否对齐
>
```
**SYCL转换**: 保持模板参数不变

### 2. operator()主函数 (行52-87)

**CUDA代码流程**:
1. 获取block/thread索引 (行54-56)
2. 加载输入数据 (行60-61)
3. RMS归约计算 (行64)
4. 加载weight (行67-69)
5. 归一化计算 (行72-74)
6. 存储结果 (行77-85)

**SYCL转换映射**:

| CUDA | SYCL | 行号 |
|------|------|------|
| `blockIdx.x` | `item.get_group(0)` | 54 |
| `threadIdx.x` | `item.get_local_id(0)` | 56 |
| `CUTLASS_DEVICE` | 移除 | 52 |
| `Loader<...>` | 自定义加载逻辑 | 60-61 |
| `_reduce_square()` | 手动实现归约 | 64 |
| `Saver<...>` | 自定义存储逻辑 | 84-85 |

### 3. _reduce_square函数 (行90-114)

**CUDA实现关键点**:
- 每个线程计算局部平方和 (行93-96)
- Warp内shuffle归约 (行99-101)
- Shared memory atomic累加 (行107-109)

**SYCL转换**:
```cpp
// CUDA
for (int i = 16; i >= 1; i >>= 1) {
    sum_square += __shfl_down_sync(0xFFFFFFFF, sum_square, i);
}
atomicAdd((float*)shared_data, sum_square);

// SYCL
sycl::sub_group sg = item.get_sub_group();
for (int i = 8; i >= 1; i >>= 1) {  // B60: 16 lanes
    sum_square += sycl::shift_group_right(sg, sum_square, i);
}
sycl::atomic_ref<float, ...> atomic_ref(...);
atomic_ref.fetch_add(sum_square);
```

### 4. 启动函数 (行125-147)

**CUDA**: 使用launch_kernel封装
**SYCL**: 直接使用queue.parallel_for

## 关键差异

### Warp/Sub-group大小
- CUDA: 32 threads
- B60 SYCL: 16 lanes (固定)

### 归约实现
- CUDA: warp shuffle + shared atomic
- SYCL: sub-group shuffle + shared atomic (类似)

### 内存访问
- CUDA: 使用Loader/Saver模板类
- SYCL: 直接指针访问，需要手动向量化

## 实现策略

1. **简化版本**: 先实现基础功能，不使用Loader/Saver模板
2. **直接使用指针**: 替换模板类为直接内存访问
3. **保持算法一致**: 确保数值计算与CUDA完全一致
4. **使用sub-group**: warp操作改为sub-group操作