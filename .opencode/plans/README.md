# TurboDiffusion CUDA-to-SYCL 迁移工作区

## 📚 文档索引

1. **[turbodiffusion_migration_guide.md](turbodiffusion_migration_guide.md)** - 快速入门指南和通用 Prompt 模板
2. **[turbodiffusion_migration_plan.md](turbodiffusion_migration_plan.md)** - 详细迁移计划和技术规范
3. **本文件** - 使用说明和最佳实践

---

## 🚀 快速开始

### 方式一：使用通用 Prompt 模板

1. 打开 `turbodiffusion_migration_guide.md`
2. 复制 **通用 Prompt 模板** 部分
3. 替换占位符（如 `[算子名称]`、`[目录]`）
4. 运行 Prompt 开始迁移

**示例 - 迁移 LayerNorm：**

```
请迁移 TurboDiffusion 的 LayerNorm CUDA 算子到 SYCL 实现。

源文件位置：
- Header: turbodiffusion/ops/norm/layernorm.hpp
- Implementation: turbodiffusion/ops/norm/layernorm.cu

要求：
1. 分析 CUDA 实现，识别所有 CUDA 特定功能
2. 使用 SYCL 等效功能替换：
   - __global__ → queue.parallel_for
   - __shared__ → sycl::local_accessor
   - __syncthreads() → item.barrier()
   - __shfl_down_sync → sycl::shift_group_right
   - atomicAdd/atomicMax → sycl::atomic_ref
3. 保持模板配置和编译时选项
4. 创建 PyTorch XPU 绑定
5. 编写单元测试和性能基准

参考现有的 layernorm.hpp 实现模式。
```

### 方式二：请求完整迁移计划

```
请为 TurboDiffusion 项目制定完整的 CUDA-to-SYCL 算子迁移计划：
1. 分析现有的 4 个 CUDA 算子实现
2. 制定详细的迁移时间表（含优先级）
3. 为每个算子提供技术映射指导
4. 创建测试策略和验证标准
5. 提供风险评估和缓解措施
```

### 方式三：特定任务请求

**仅迁移特定算子：**
```
请仅迁移 RMSNorm 算子，包含：
- SYCL header 和实现文件
- Python 绑定更新
- 单元测试
- 性能基准
```

**仅创建测试：**
```
请为已迁移的 SYCL LayerNorm 创建完整的测试套件：
1. 数值正确性测试（多形状、多数据类型）
2. 性能基准测试（与 CUDA 对比）
3. 边界情况测试
```

---

## 📋 Prompt 模板变量说明

| 占位符 | 说明 | 示例 |
|--------|------|------|
| `[算子名称]` | 算子的英文名称 | LayerNorm, RMSNorm, Quantization, GEMM |
| `[目录]` | 算子所在子目录 | norm, quant, gemm |
| `[算子]` | 算子文件名（小写） | layernorm, rmsnorm, quant, gemm |

---

## 🎯 迁移优先级

推荐使用以下优先级顺序：

1. **P0 - LayerNorm** (2-3天)
   - 基础算子，结构清晰
   - 可作为其他算子的模板

2. **P0 - RMSNorm** (1-2天)
   - 类似 LayerNorm，可复用模式
   - 依赖 LayerNorm 经验

3. **P1 - Quantization** (2-3天)
   - 中等复杂度
   - 关键路径算子

4. **P2 - GEMM** (5-7天)
   - 最高复杂度
   - 依赖 XMX 优化

---

## 🛠️ 技术要点

### CUDA-to-SYCL 核心映射

| CUDA | SYCL | 说明 |
|------|------|------|
| `blockIdx.x` | `item.get_group(0)` | Block 索引 |
| `threadIdx.x` | `item.get_local_id(0)` | Thread 索引 |
| `blockDim.x` | `item.get_local_range(0)` | Block 大小 |
| `gridDim.x` | `item.get_group_range(0)` | Grid 大小 |
| `__syncthreads()` | `item.barrier(...)` | 线程同步 |
| `__shared__` | `sycl::local_accessor` | 共享内存 |
| `__shfl_down_sync` | `sycl::shift_group_right` | Warp shuffle |
| `atomicAdd` | `sycl::atomic_ref.fetch_add` | 原子加 |

### 关键代码模式

**1. Kernel 转换：**
```cpp
// CUDA
__global__ void kernel(params) { ... }
kernel<<<grid, block>>>(params);

// SYCL
queue.parallel_for(
  sycl::nd_range<1>(global_size, local_size),
  [=](sycl::nd_item<1> item) {
    // kernel body
  }
);
```

**2. Shared Memory：**
```cpp
// CUDA
__shared__ float smem[256];

// SYCL
sycl::local_accessor<float, 1> smem(sycl::range<1>(256), h);
```

**3. Warp Shuffle：**
```cpp
// CUDA
float val = __shfl_down_sync(0xFFFFFFFF, value, offset);

// SYCL
auto sg = item.get_sub_group();
float val = sycl::shift_group_right(sg, value, offset);
```

---

## ✅ 检查清单

每个算子迁移完成后，确认：

- [ ] SYCL Header 文件创建
- [ ] SYCL Implementation 文件创建
- [ ] Python 绑定更新
- [ ] 单元测试通过（数值正确性）
- [ ] 性能测试通过（≥80% CUDA）
- [ ] 边界情况测试通过
- [ ] 代码审查完成
- [ ] 文档更新

---

## 📊 项目统计

- **总代码量**: ~1,810 行 CUDA C++
- **算子数量**: 4 个
- **预计总时间**: 13-20 天
- **测试覆盖率目标**: 100%

---

## 🔗 相关文件

### 源码位置
```
turbodiffusion/ops/
├── common/
│   ├── common.hpp      # 通用定义
│   ├── launch.hpp      # Kernel 启动
│   ├── load.hpp        # 数据加载
│   └── store.hpp       # 数据存储
├── gemm/
│   ├── gemm.cu         # GEMM 实现
│   ├── kernel.hpp      # GEMM kernel
│   ├── launch.hpp      # GEMM 启动
│   └── utils.hpp       # GEMM 工具
├── norm/
│   ├── layernorm.cu    # LayerNorm 实现
│   ├── layernorm.hpp   # LayerNorm kernel
│   ├── rmsnorm.cu      # RMSNorm 实现
│   └── rmsnorm.hpp     # RMSNorm kernel
└── quant/
    ├── quant.cu        # Quantization 实现
    └── quant.hpp       # Quantization kernel
```

### 文档位置
```
.opencode/plans/
├── README.md                           # 本文件
├── turbodiffusion_migration_guide.md   # 快速指南
└── turbodiffusion_migration_plan.md    # 详细计划
```

---

## 💡 最佳实践

### 1. 渐进式迁移
- 先迁移简单算子（LayerNorm）
- 积累经验后再迁移复杂算子（GEMM）
- 每个算子完整测试后再进行下一个

### 2. 保持兼容性
- 使用条件编译支持 CUDA/SYCL 双后端
- 保持 Python API 不变
- 原有 CUDA 代码不删除，添加 SYCL 分支

### 3. 测试驱动
- 先编写测试，再实现功能
- 每个算子必须有单元测试和基准测试
- 使用 CI/CD 自动化测试

### 4. 性能优先
- 关注内存访问模式
- 优化 work-group 大小
- 使用 XMX (Intel Matrix Extensions) 加速 GEMM

---

## 🆘 故障排除

### 常见问题

**Q: SYCL 编译失败**
- 检查 Intel oneAPI 环境是否激活
- 确认 `icpx` 编译器可用
- 检查 SYCL 目标设备是否正确

**Q: 数值精度不符**
- 增加中间计算精度（使用 FP32）
- 检查 atomic 操作的内存序
- 验证 sub-group shuffle 的实现

**Q: 性能不达标**
- 分析 memory bandwidth
- 检查 bank conflict
- 优化 work-group 大小
- 考虑使用 oneMKL 替代手动实现

**Q: PyTorch XPU 绑定问题**
- 确认 PyTorch XPU 版本
- 检查 `at::xpu::getCurrentXPUStream()` 可用性
- 验证设备类型检测逻辑

---

## 📞 支持

如有问题：
1. 查看详细计划文档
2. 参考 SYCL 规范
3. 检查 Intel oneAPI 文档
4. 在 Intel GPU 硬件上测试

---

## 📝 更新日志

- **2024-XX-XX**: 创建迁移计划和 Prompt 模板
- **未来**: 更新各算子迁移进度

---

**准备好了吗？开始使用 [turbodiffusion_migration_guide.md](turbodiffusion_migration_guide.md) 中的 Prompt 模板开始迁移吧！**
