# TurboDiffusion SYCL Migration - Quick Start Guide

## 通用 Prompt 模板

使用以下模板来迁移任何 CUDA 算子到 SYCL：

```
请迁移 TurboDiffusion 的 [算子名称] CUDA 算子到 SYCL 实现。

源文件位置：
- Header: turbodiffusion/ops/[目录]/[算子].hpp
- Implementation: turbodiffusion/ops/[目录]/[算子].cu

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

交付物：
- SYCL Header (.hpp)
- SYCL Implementation (.dp.cpp)
- Python bindings
- Unit tests
- Benchmark tests

参考：turbodiffusion/ops/norm/layernorm.hpp 的实现模式
```

## 算子列表

| 算子 | 文件 | 复杂度 | 优先级 | 状态 |
|------|------|--------|--------|------|
| LayerNorm | ops/norm/layernorm.hpp | ⭐⭐⭐⭐ | P0 | 待迁移 |
| RMSNorm | ops/norm/rmsnorm.hpp | ⭐⭐⭐⭐ | P0 | 待迁移 |
| Quantization | ops/quant/quant.hpp | ⭐⭐⭐ | P1 | 待迁移 |
| GEMM | ops/gemm/kernel.hpp | ⭐⭐⭐⭐⭐ | P2 | 待迁移 |

## CUDA-to-SYCL 速查表

| CUDA | SYCL |
|------|------|
| `blockIdx.x` | `item.get_group(0)` |
| `threadIdx.x` | `item.get_local_id(0)` |
| `blockDim.x` | `item.get_local_range(0)` |
| `__syncthreads()` | `item.barrier(sycl::access::fence_space::local_space)` |
| `__shfl_down_sync(mask, val, offset)` | `sycl::shift_group_right(sg, val, offset)` |
| `atomicAdd(addr, val)` | `sycl::atomic_ref(...).fetch_add(val)` |

## 使用步骤

1. **选择算子**：从上方列表选择要迁移的算子
2. **复制 Prompt**：复制通用 Prompt 模板
3. **替换占位符**：将 `[算子名称]` 和 `[目录]` 替换为实际值
4. **执行迁移**：运行 Prompt 开始迁移
5. **验证测试**：确保通过所有测试和性能基准

## 测试要求

- 数值正确性：FP32 rtol=1e-4, FP16 rtol=1e-3
- 性能目标：SYCL >= 80% CUDA 性能
- 边界情况：空张量、非对齐尺寸、极端值

## 文件位置

- CUDA 源码：`turbodiffusion/ops/`
- SYCL 目标：`turbodiffusion/ops/sycl/`
- 测试代码：`tests/unit/` 和 `tests/performance/`
