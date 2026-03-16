# LLM-based Accuracy Test Agent - 完整实施文档

## ✅ 已完成模块

### 1. Platform Detector (`tools/platform_detector.py`)
**功能**: 检测SYCL和CUDA平台支持的数据类型

**检测结果示例**:
```
SYCL: Intel Graphics, FP16=YES, BF16=NO
CUDA: NVIDIA L20 (SM89), FP16=YES, BF16=YES
```

**API**:
```python
from tools.platform_detector import PlatformDetector

detector = PlatformDetector()
sycl_caps = detector.detect_sycl_capabilities()
cuda_caps = detector.detect_cuda_capabilities()
```

### 2. NaN Behavior Tester (`tools/nan_behavior_tester.py`)
**功能**: 测试CUDA和SYCL对NaN/Inf的处理一致性

**测试场景**:
- 1.0 + NaN
- NaN + NaN
- inf + inf
- 0.0 / 0.0
- inf * 0.0
- ...等14种边界情况

**策略**: NaN不一致时发出警告但视为通过（根据用户要求）

### 3. Prototype Accuracy Tester (`tools/prototype_accuracy_tester.py`)
**功能**: 原型验证 - 展示真正的kernel调用和准确度对比

**核心流程**:
1. 生成测试数据（固定种子，可复现）
2. 编译并执行CUDA kernel（远程docker）
3. 编译并执行SYCL kernel（本地docker）
4. 对比输出结果
5. 计算误差指标

---

## 🏗️ 完整架构设计

### 核心组件关系图

```
┌─────────────────────────────────────────────────────────────┐
│                 LLM Accuracy Test Agent                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │ Platform        │───▶│ Test Config     │                 │
│  │ Detector        │    │ Generator       │                 │
│  └─────────────────┘    └────────┬────────┘                 │
│                                  │                          │
│                                  ▼                          │
│  ┌──────────────────────────────────────────┐              │
│  │       Parallel Test Executor             │              │
│  │  ┌────────────────────────────────────┐  │              │
│  │  │  LLM Harness Generator (并发=4)    │  │              │
│  │  │  - 分析kernel签名                  │  │              │
│  │  │  - 生成CUDA测试代码                │  │              │
│  │  │  - 生成SYCL测试代码                │  │              │
│  │  └────────────────────────────────────┘  │              │
│  │                    │                      │              │
│  │                    ▼                      │              │
│  │  ┌────────────────────────────────────┐  │              │
│  │  │  Test Runner (CUDA=1, SYCL=1)     │  │              │
│  │  │  - 编译CUDA代码 (远程)             │  │              │
│  │  │  - 执行CUDA测试                    │  │              │
│  │  │  - 编译SYCL代码 (本地)             │  │              │
│  │  │  - 执行SYCL测试                    │  │              │
│  │  └────────────────────────────────────┘  │              │
│  └──────────────────────────────────────────┘              │
│                          │                                 │
│                          ▼                                 │
│  ┌──────────────────────────────────────────┐              │
│  │      Result Comparator & Analyzer        │              │
│  │  - 计算绝对/相对误差                     │              │
│  │  - 应用数据类型容差                      │              │
│  │  - 处理NaN特殊逻辑                       │              │
│  │  - LLM辅助失败分析                       │              │
│  └──────────────────────────────────────────┘              │
│                          │                                 │
│                          ▼                                 │
│  ┌──────────────────────────────────────────┐              │
│  │         Coverage Report Generator        │              │
│  │  - 数据类型覆盖                          │              │
│  │  - 维度覆盖                              │              │
│  │  - 极端值覆盖                            │              │
│  └──────────────────────────────────────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 测试配置矩阵 (18个测试)

### 数据类型测试

| 测试ID | 数据类型 | 数据生成 | 尺寸 | 容差 (abs/rel) |
|--------|---------|---------|------|---------------|
| f32_small_random | float32 | random [-1,1] | 1x64x8x8 | 1e-5 / 1e-4 |
| f32_boundary | float32 | boundary values | 1x64x8x8 | 1e-5 / 1e-4 |
| f32_special | float32 | special (inf,nan) | 1x64x8x8 | 1e-5 / 1e-4 |
| f32_large | float32 | random [-100,100] | 16x256x32x32 | 1e-4 / 1e-3 |
| bf16_small_random | bfloat16 | random [-1,1] | 1x64x8x8 | 1e-3 / 1e-2 |
| bf16_boundary | bfloat16 | boundary values | 1x64x8x8 | 1e-3 / 1e-2 |
| bf16_precision | bfloat16 | small variations | 1x64x8x8 | 1e-2 / 5e-2 |
| fp16_small_random | float16 | random [-1,1] | 1x64x8x8 | 1e-3 / 1e-2 |
| fp16_boundary | float16 | boundary values | 1x64x8x8 | 1e-3 / 1e-2 |

### 维度测试

| 测试ID | N | C | H | W | 说明 |
|--------|---|---|---|---|------|
| dim_square | 4 | 128 | 16 | 16 | 方形tensor |
| dim_rectangular | 2 | 128 | 8 | 16 | 矩形tensor (H≠W) |
| dim_line | 1 | 64 | 1 | 64 | 单维度 (H=1) |
| dim_unaligned | 3 | 100 | 17 | 31 | 非对齐尺寸 |

### 极端值测试

| 测试ID | 测试内容 |
|--------|---------|
| extreme_zeros | 0.0, -0.0 |
| extreme_infinity | inf, -inf |
| extreme_nan | nan propagation |
| extreme_math | exp(88), 1e38, etc. |

---

## 🔧 LLM Prompt 设计

### System Prompt
```
You are an expert GPU test engineer. Generate complete test harness code.

Rules:
1. Output ONLY compilable C++ code
2. Include proper error checking
3. Use fixed random seed: 42
4. Generate data: random uniform [-1.0, 1.0]
5. For bf16: use sycl::bfloat16 / __nv_bfloat16
6. For fp16: use sycl::half / half
7. Include template instantiation if needed
8. Save output to binary file
```

### User Prompt Template
```
Generate {cuda_or_sycl} test harness for:

KERNEL:
{kernel_code}

CONFIG:
- Test name: {test_name}
- Data type: {dtype}
- Dimensions: N={N}, C={C}, H={H}, W={W}
- Elements: {total_elements}
- Template types: {template_types}

Generate code that:
1. Creates test data (seed=42)
2. Allocates device memory
3. Launches kernel with correct grid/block
4. Copies result back
5. Saves to "/tmp/test_output.bin"

Return ONLY code.
```

---

## 🚀 实施步骤

### Phase 1: 基础框架 ✅
- [x] Platform detector
- [x] NaN behavior tester
- [x] Prototype accuracy tester

### Phase 2: LLM集成 ⏳
- [ ] LLM harness generator
  - Parse kernel signature
  - Generate CUDA harness
  - Generate SYCL harness
  - Handle templates
- [ ] Prompt engineering
  - Test with real kernels
  - Refine prompts

### Phase 3: 并行执行 ⏳
- [ ] Async executor
  - LLM semaphore (4)
  - CUDA semaphore (1)
  - SYCL semaphore (1)
- [ ] Result collector
- [ ] Error handling

### Phase 4: 完整测试套件 ⏳
- [ ] Test config generator
- [ ] 18 test configurations
- [ ] Coverage reporter

### Phase 5: 集成与优化 ⏳
- [ ] Integrate with unified_converter
- [ ] Performance optimization
- [ ] Documentation

---

## 📊 预期输出示例

```
🧪 LLM Accuracy Test - copy_type_converted
═══════════════════════════════════════════════════

Platform:
  SYCL: Intel Graphics [0xe211] (FP16=YES, BF16=NO)
  CUDA: NVIDIA L20 SM89 (FP16=YES, BF16=YES)

Test Results (18 tests):
  ✅ f32_small_random         Pass  max_err=0.00e+00
  ✅ f32_boundary             Pass  max_err=1.19e-07
  ✅ f32_special              Pass  max_err=0.00e+00  (NaN OK)
  ✅ f32_large                Pass  max_err=3.45e-05
  ⚠️  bf16_small_random        Pass  max_err=3.91e-03  (precision loss)
  ✅ bf16_boundary            Pass  max_err=3.91e-03
  ...

Coverage:
  Data types: 3/3 (f32, bf16, fp16)
  Dimensions: 4/4 (square, rect, line, unaligned)
  Extreme values: 4/4 (zeros, inf, nan, math)

Summary:
  Total: 18 tests
  Passed: 17
  Warning: 1 (bf16 precision)
  Failed: 0
  Consistency rate: 100%

═══════════════════════════════════════════════════
```

---

## 📝 关键设计决策

1. **数据类型支持**: 
   - float32: 总是测试
   - bf16/fp16: 根据平台检测结果动态决定是否测试

2. **NaN处理**: 
   - 不一致时发出警告但视为通过
   - 记录详细信息供分析

3. **容差设置**:
   - float32: 严格 (1e-5 / 1e-4)
   - bf16: 宽松 (1e-3 / 1e-2)
   - 极端值: 更宽松 (1e-3 / 1e-2)

4. **并行策略**:
   - LLM生成: 4并发 (API限制)
   - CUDA执行: 1并发 (单GPU)
   - SYCL执行: 1并发 (单GPU)

5. **错误恢复**:
   - 单个测试失败不影响其他测试
   - 详细错误日志
   - LLM辅助分析失败原因

---

## 🎯 下一步行动

**建议优先级**:
1. **高**: 完成LLM harness generator (核心功能)
2. **高**: 并行执行器实现 (性能关键)
3. **中**: 18个测试配置 (覆盖率)
4. **低**: 高级功能 (自动修复, LLM分析)

**预估时间**:
- Phase 2 (LLM集成): 2-3小时
- Phase 3 (并行执行): 1-2小时
- Phase 4 (测试套件): 1-2小时
- Phase 5 (集成优化): 1-2小时
- **总计**: 5-9小时

**依赖**:
- Gaudi AI API可用
- CUDA远程环境可用
- SYCL本地环境可用
