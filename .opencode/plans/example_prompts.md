# TurboDiffusion 迁移 Prompt 示例

本文档提供可直接使用的 Prompt 示例，复制后即可运行。

---

## 示例 1: LayerNorm 迁移

```
请迁移 TurboDiffusion 的 LayerNorm CUDA 算子到 SYCL 实现。

## 源文件
- Header: turbodiffusion/ops/norm/layernorm.hpp
- Implementation: turbodiffusion/ops/norm/layernorm.cu
- Common: turbodiffusion/ops/common/common.hpp, launch.hpp, load.hpp, store.hpp

## 要求

### 1. 功能分析
LayerNorm 算子实现层归一化，支持：
- 模板配置：InputDtype, OutputDtype, WeightDtype
- Affine 选项：是否使用权重缩放
- Bias 选项：是否使用偏置
- 配置切换：BOOL_SWITCH 和 CONFIG_SWITCH

### 2. CUDA 特性识别
当前 CUDA 实现使用：
- __global__ kernel (通过 launch.hpp 启动)
- __shared__ char shared_data[32] 用于归约
- __shfl_down_sync 进行 warp 级归约
- atomicAdd 进行 block 级归约
- __syncthreads() 同步
- Template 特化 (BOOL_SWITCH, CONFIG_SWITCH)

### 3. SYCL 转换要求
- 将 LayerNorm class 转换为 SYCL functor
- 使用 sycl::nd_item 替代 CUDA thread/block 索引
- 使用 sycl::local_accessor<char, 1> 替代 __shared__
- 使用 sycl::shift_group_right 替代 __shfl_down_sync
- 使用 sycl::atomic_ref 替代 atomicAdd
- 使用 item.barrier() 替代 __syncthreads()
- 保持模板参数和 CONFIG_SWITCH 逻辑

### 4. 文件结构
创建以下文件：
- turbodiffusion/ops/sycl/norm/layernorm.hpp
- turbodiffusion/ops/sycl/norm/layernorm.dp.cpp

### 5. Python 绑定
更新 bindings.cpp，添加：
- layer_norm_sycl 函数
- 条件编译支持 (#ifdef USE_SYCL)
- XPU stream 获取: at::xpu::getCurrentXPUStream().queue()

### 6. 测试
创建测试文件：
- tests/unit/test_layernorm_sycl.py
  - 测试多种形状：(16,128), (256,512), (1024,2048)
  - 测试多种数据类型：FP32, FP16, BF16
  - 测试 Affine/Bias 组合
  - 与 PyTorch F.layer_norm 对比
  - 容差：FP32 rtol=1e-4, FP16 rtol=1e-3

### 7. 性能基准
创建基准文件：
- tests/performance/benchmark_layernorm.py
  - 对比 SYCL vs CUDA 性能
  - 测试多种 batch size 和 hidden size
  - 目标：SYCL >= 80% CUDA 性能

## 参考实现
参考 turbodiffusion/ops/norm/layernorm.hpp 中的实现模式。
```

---

## 示例 2: RMSNorm 迁移

```
请迁移 TurboDiffusion 的 RMSNorm CUDA 算子到 SYCL 实现。

## 源文件
- Header: turbodiffusion/ops/norm/rmsnorm.hpp
- Implementation: turbodiffusion/ops/norm/rmsnorm.cu

## 要求

### 1. 功能分析
RMSNorm 算子实现 RMS 归一化：
- 计算 RMS = sqrt(mean(x^2) + eps)
- 支持权重缩放
- 比 LayerNorm 简单（无需 mean 计算）

### 2. CUDA 特性
与 LayerNorm 类似，但更简单：
- 只需一次归约（平方和）
- 无需 mean 计算
- 其他 CUDA 特性与 LayerNorm 相同

### 3. SYCL 转换
参考 LayerNorm 的转换模式，但简化：
- 只实现 _reduce_square 归约
- 移除 mean 计算相关代码
- 保持权重加载逻辑

### 4. 文件结构
- turbodiffusion/ops/sycl/norm/rmsnorm.hpp
- turbodiffusion/ops/sycl/norm/rmsnorm.dp.cpp

### 5. 测试
- tests/unit/test_rmsnorm_sycl.py
- 参考 test_layernorm_sycl.py 但移除 bias 测试

## 注意
RMSNorm 与 LayerNorm 结构高度相似，可以先完成 LayerNorm，然后复用大部分模式。
```

---

## 示例 3: Quantization 迁移

```
请迁移 TurboDiffusion 的 Quantization CUDA 算子到 SYCL 实现。

## 源文件
- Header: turbodiffusion/ops/quant/quant.hpp
- Implementation: turbodiffusion/ops/quant/quant.cu

## 要求

### 1. 功能分析
Quantization 算子实现 FP16/BF16 → INT8 量化：
- Block-wise AMMAX 归约（128x128 blocks）
- 计算 scale = amax / 128
- 量化公式：x_q = round(x * scale)
- 输出：INT8 tensor + scale tensor

### 2. CUDA 特性
- __shfl_xor_sync 进行 warp 级 amax 归约
- atomicMax 进行 block 级 amax 归约
- CUTLASS NumericConverter 进行类型转换
- cp_async 异步拷贝（GEMM 相关）

### 3. SYCL 转换
- 将 Quantization class 转换为 SYCL functor
- 使用 sycl::shift_group_left 替代 __shfl_xor_sync
- 使用 sycl::atomic_ref::fetch_max 替代 atomicMax
- 使用 SYCL 内置类型转换替代 CUTLASS converter
- 处理 FP16/BF16 类型转换

### 4. 文件结构
- turbodiffusion/ops/sycl/quant/quant.hpp
- turbodiffusion/ops/sycl/quant/quant.dp.cpp

### 5. 特殊考虑
- AMMAX 归约逻辑需要仔细验证
- INT8 范围验证：[-128, 127]
- Scale tensor 形状：(num_blocks_m, num_blocks_n)

### 6. 测试
- tests/unit/test_quantization_sycl.py
  - 测试形状：多种矩阵尺寸
  - 测试数据类型：FP16, BF16
  - 验证输出范围在 INT8 内
  - 验证 dequantization 误差
```

---

## 示例 4: GEMM 迁移

```
请迁移 TurboDiffusion 的 GEMM CUDA 算子到 SYCL 实现。

## 源文件
- Header: turbodiffusion/ops/gemm/kernel.hpp (522行)
- Implementation: turbodiffusion/ops/gemm/gemm.cu
- Utils: turbodiffusion/ops/gemm/utils.hpp, launch.hpp

## 要求

### 1. 功能分析
GEMM 算子实现 INT8 矩阵乘法：
- C = A × B^T，其中 A,B 是 INT8，C 是 FP16/BF16
- 使用 block-wise scale：C = A × B^T × scale_a × scale_b
- 复杂的流水线实现 (Stage=3)
- Swizzle 优化提高内存访问效率

### 2. CUDA 特性（复杂）
- CUTLASS/CuTe 框架
- 多级流水线：Global -> Shared -> Register
- cp_async 异步拷贝
- MMA (Matrix Multiply Accumulate) 指令
- Swizzle memory layout
- Complex epilogue for dequantization

### 3. 迁移策略
GEMM 是最复杂的算子，有两种方案：

#### 方案 A: 使用 oneMKL（推荐）
```cpp
#include <oneapi/mkl.hpp>

// 使用 oneMKL 的 gemm_int8
oneapi::mkl::blas::column_major::gemm(
  queue, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc
);
// 后处理：应用 scale
```

#### 方案 B: 手动 XMX 实现
使用 SYCL joint_matrix 扩展实现 XMX tile-based GEMM：
```cpp
using namespace sycl::ext::oneapi::experimental::matrix;

joint_matrix<sub_group, half, use::accumulator, 8, 16> acc;
joint_matrix_mad(sub_group, acc, mat_a, mat_b, acc);
```

### 4. 推荐做法
1. 先实现 oneMKL 版本作为 baseline
2. 性能达标即可，不一定要手动 XMX
3. 如果 oneMKL 不满足性能，再考虑手动实现

### 5. 文件结构
- turbodiffusion/ops/sycl/gemm/gemm.hpp
- turbodiffusion/ops/sycl/gemm/gemm.dp.cpp

### 6. 测试
- tests/unit/test_gemm_sycl.py
  - 测试多种 M,N,K 组合
  - 测试 FP16 和 BF16 输出
  - 验证 INT8 × INT8 → FP16 正确性
  - 边界情况：非对齐尺寸

### 7. 性能基准
- tests/performance/benchmark_gemm.py
  - 这是最关键的基准
  - 测试不同矩阵形状的性能
  - 目标：达到 CUDA 版本的 80%

## 警告
GEMM 实现非常复杂（522行），不建议手动实现完整版本。优先使用 oneMKL。
```

---

## 示例 5: 完整迁移计划请求

```
请为 TurboDiffusion 项目制定完整的 CUDA-to-SYCL 算子迁移计划。

## 项目背景
- TurboDiffusion 是视频生成加速框架
- 当前有 4 个 CUDA 算子需要迁移
- 目标平台：Intel GPU (BMG/ARC)
- 总代码量：~1,810 行 CUDA C++

## 需要的内容

### 1. 算子分析
详细分析每个算子：
- LayerNorm (202行)
- RMSNorm
- Quantization
- GEMM (522行)

### 2. 迁移时间表
- 优先级排序 (P0/P1/P2)
- 每个算子的时间估算
- 依赖关系
- 总时间估算

### 3. 技术映射
- CUDA-to-SYCL 映射表
- 代码转换示例
- 关键技术点说明

### 4. 测试策略
- 单元测试计划
- 性能基准计划
- 集成测试计划
- 验证标准

### 5. 风险评估
- 技术风险
- 性能风险
- 缓解措施

### 6. 交付物清单
- 源代码文件
- 测试文件
- 文档
- CI/CD 配置

## 输出格式
请以结构化文档形式输出，包含：
- 执行摘要
- 详细计划
- 技术规范
- 检查清单

## 参考
现有文档：
- turbodiffusion/ops/ 目录下的源码
- TurboDiffusion/SYCL_MIGRATION_PLAN.md
```

---

## 示例 6: 仅创建测试

```
请为迁移后的 SYCL LayerNorm 算子创建完整的测试套件。

## 测试范围

### 1. 单元测试 (tests/unit/test_layernorm_sycl.py)

#### 数值正确性测试
- 测试形状：(1,128), (16,128), (256,512), (1024,2048), (4096,8192)
- 测试数据类型：torch.float32, torch.float16, torch.bfloat16
- 测试配置：
  - elementwise_affine=True/False
  - bias=True/False (仅当 affine=True)
- 与 PyTorch F.layer_norm 对比
- 容差标准：
  - FP32: rtol=1e-4, atol=1e-5
  - FP16/BF16: rtol=1e-3, atol=1e-3

#### 边界情况测试
- 空张量 (0, 128)
- 极小值：x * 1e-6
- 极大值：x * 1e3
- 全零输入
- 单元素 batch (1, N)
- 非对齐维度 (127, 127)

#### 数据生成策略
- 随机数据（固定 seed）
- 正态分布
- 均匀分布
- 边界值

### 2. 性能基准 (tests/performance/benchmark_layernorm.py)

#### 对比测试
- SYCL vs CUDA 性能
- SYCL vs PyTorch native 性能

#### 测试矩阵
- Batch sizes: 16, 64, 256, 1024, 4096
- Hidden sizes: 128, 512, 1024, 2048, 4096, 8192

#### 指标
- 执行时间 (ms)
- 内存带宽 (GB/s)
- 相对加速比

#### 目标
- SYCL 性能 >= 80% CUDA 性能
- 内存带宽 >= 60% 峰值带宽

### 3. 测试基础设施

#### Fixtures (conftest.py)
- device fixture (xpu, cuda, cpu)
- dtype fixture (fp32, fp16, bf16)
- shape fixture
- tolerance fixture

#### 工具函数
- 数据生成器
- 精度比较工具
- 性能测量工具

### 4. CI/CD 集成
- GitHub Actions workflow
- 自动运行测试
- 性能回归检测
- 报告生成

## 文件输出
1. tests/unit/test_layernorm_sycl.py
2. tests/performance/benchmark_layernorm.py
3. tests/conftest.py (更新)
4. .github/workflows/test-sycl.yml
5. tests/utils/ (数据生成器和工具)

## 参考
- 现有测试：turbodiffusion/rcm/networks/wan2pt1_jvp_test.py
- PyTorch testing 最佳实践
```

---

## 使用建议

1. **复制示例**：复制对应的示例 Prompt
2. **调整细节**：根据实际需要调整参数和路径
3. **逐步执行**：建议按优先级顺序执行 (LayerNorm → RMSNorm → Quantization → GEMM)
4. **验证结果**：每个算子完成后验证测试和性能
5. **迭代优化**：根据测试结果调整实现

---

**提示**：将这些示例保存为模板，每次迁移新算子时复制并修改即可。
