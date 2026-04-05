# TurboDiffusion CUDA-to-SYCL 迁移计划

## 项目概况

**项目**: TurboDiffusion - Video Diffusion 加速框架
**目标**: 将 FP16 CUDA 算子迁移到 SYCL，支持 Intel GPU
**代码规模**: ~1,810 行 CUDA C++ 代码
**算子数量**: 4 个核心算子

## 算子清单

### 1. LayerNorm (P0)
- **文件**: `ops/norm/layernorm.hpp` (202行)
- **功能**: 层归一化，支持 Affine 和 Bias
- **复杂度**: ⭐⭐⭐⭐
- **关键特性**:
  - 模板化实现
  - Warp shuffle 归约
  - Shared memory atomic 操作
  - BOOL_SWITCH/CONFIG_SWITCH 配置

### 2. RMSNorm (P0)
- **文件**: `ops/norm/rmsnorm.hpp`
- **功能**: RMS 归一化
- **复杂度**: ⭐⭐⭐⭐
- **关键特性**:
  - 类似 LayerNorm 结构
  - RMS 计算替代 variance
  - 权重缩放

### 3. Quantization (P1)
- **文件**: `ops/quant/quant.hpp`
- **功能**: FP16/BF16 → INT8 量化
- **复杂度**: ⭐⭐⭐
- **关键特性**:
  - Block-wise AMMAX 归约
  - Warp shuffle + atomicMax
  - CUTLASS 数值转换

### 4. GEMM (P2)
- **文件**: `ops/gemm/kernel.hpp` (522行)
- **功能**: INT8 矩阵乘法
- **复杂度**: ⭐⭐⭐⭐⭐
- **关键特性**:
  - CUTLASS/CuTe 框架
  - 多级流水线 (Stage=3)
  - Swizzle 优化
  - 异步拷贝 (cp.async)

## 迁移时间表

| 阶段 | 内容 | 时间 | 依赖 |
|------|------|------|------|
| Phase 1 | 基础设施 | 1-2天 | 无 |
| Phase 2 | LayerNorm | 2-3天 | Phase 1 |
| Phase 2 | RMSNorm | 1-2天 | Phase 1 |
| Phase 3 | Quantization | 2-3天 | Phase 1 |
| Phase 4 | GEMM | 5-7天 | Phase 1 |
| Phase 5 | 测试验证 | 2-3天 | Phase 2-4 |
| **总计** | | **13-20天** | |

## 技术映射

### 线程模型
```cpp
// CUDA
dim3 grid(m);
dim3 block(NumThrPerCta);
kernel<<<grid, block, ShmSize, stream>>>(params);

// SYCL
queue.parallel_for(
  sycl::nd_range<1>(sycl::range<1>(m * NumThrPerCta), 
                    sycl::range<1>(NumThrPerCta)),
  [=](sycl::nd_item<1> item) {
    kernel(params, item);
  }
);
```

### Warp 操作
```cpp
// CUDA warp shuffle
__shfl_down_sync(0xFFFFFFFF, value, i);

// SYCL sub-group shuffle
auto sg = item.get_sub_group();
sycl::shift_group_left(sg, value, i);
```

### Shared Memory
```cpp
// CUDA
__shared__ char shared_data[ShmSize];

// SYCL
sycl::local_accessor<char, 1> shared_data(
  sycl::range<1>(ShmSize), h);
```

### 原子操作
```cpp
// CUDA
atomicAdd((float*)shared_data, sum);

// SYCL
sycl::atomic_ref<float, sycl::memory_order_relaxed,
                 sycl::memory_scope_work_group> 
  atomic_ref(*(float*)shared_data);
atomic_ref.fetch_add(sum);
```

## 迁移步骤

### Phase 1: 基础设施

1. **创建 SYCL 目录结构**
   ```
   turbodiffusion/ops/sycl/
   ├── common/
   │   ├── common.hpp
   │   ├── launch.hpp
   │   ├── load.hpp
   │   └── store.hpp
   ├── gemm/
   │   ├── gemm.dp.cpp
   │   └── kernel.hpp
   ├── norm/
   │   ├── layernorm.dp.cpp
   │   ├── layernorm.hpp
   │   ├── rmsnorm.dp.cpp
   │   └── rmsnorm.hpp
   ├── quant/
   │   ├── quant.dp.cpp
   │   └── quant.hpp
   └── bindings.cpp
   ```

2. **适配通用头文件**
   - 替换 CUDA 类型 (dim3, cudaStream_t)
   - 添加 SYCL 头文件 (<sycl/sycl.hpp>)
   - 定义统一宏

3. **修改构建系统**
   - 更新 setup.py 支持 SYCL 编译
   - 添加 Intel oneAPI 支持

### Phase 2-4: 算子迁移

对每个算子执行：
1. 分析 CUDA 实现
2. 转换为 SYCL functor/lambda
3. 替换 warp 操作为 sub-group 操作
4. 适配 shared memory
5. 更新 Python 绑定
6. 编写单元测试
7. 性能基准测试

## 测试策略

### 单元测试
- 数值正确性 (SYCL vs PyTorch)
- 形状和 stride 处理
- 边界情况 (空张量、极端尺寸)
- 数据类型测试 (FP16/BF16/FP32)

### 性能测试
- SYCL vs CUDA 性能对比
- 内存带宽测量
- 扩展性测试

### 集成测试
- 端到端模型推理
- Quantization + GEMM 流水线
- Norm + Activation 组合

## 成功指标

- ✅ 所有算子编译通过
- ✅ 数值精度与 CUDA 一致 (误差 < 1e-4)
- ✅ 性能达到 CUDA 的 80%+
- ✅ 通过全部单元测试

## 风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| XMX 性能不达预期 | 中 | 高 | oneMKL fallback 方案 |
| PyTorch XPU API 变更 | 低 | 中 | 封装适配层 |
| 数值精度问题 | 中 | 高 | 增加中间计算精度 |
| 编译时间过长 | 高 | 低 | 增量编译 + ccache |

## 下一步行动

1. ✅ 完成迁移计划制定
2. 🔄 搭建 SYCL 编译环境
3. ⏳ 迁移 LayerNorm (P0)
4. ⏳ 迁移 RMSNorm (P0)
5. ⏳ 迁移 Quantization (P1)
6. ⏳ 迁移 GEMM (P2)
7. ⏳ 全面测试验证

## 参考资料

- 源码: `turbodiffusion/ops/`
- 迁移指南: `.opencode/plans/turbodiffusion_migration_guide.md`
- SYCL 规范: https://www.khronos.org/sycl/
- Intel oneAPI: https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html
