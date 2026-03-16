# CUDA-to-SYCL 自动转换工作流规划文档

**文档版本**: 1.0  
**创建日期**: 2026-02-28  
**目标**: 自动化将LCZero CUDA kernel转换为SYCL kernel  
**适用范围**: Agent自动执行系统

---

## 1. 项目概述

### 1.1 目标范围
- **源文件**: 30个CUDA kernel文件 (`.cu`)
- **目标**: 生成等价的SYCL实现 (`.dp.cpp`)
- **验证**: 100%编译通过 + 数值精度验证

### 1.2 数据来源
```
源目录: /home/intel/tianfeng/opencode_bench/kernel_dataset/cuda/
目标目录: /home/intel/tianfeng/opencode_bench/kernel_dataset/sycl/
索引文件: /home/intel/tianfeng/opencode_bench/kernel_dataset/index.json
```

### 1.3 Kernel复杂度分级

| 级别 | Kernel数量 | 特征 | 示例 |
|------|-----------|------|------|
| Level 1 | 8 | 基础向量/矩阵操作 | add_vectors, copy_type_converted |
| Level 2 | 12 | 使用shared memory和warp操作 | softmax, batch_norm, layer_norm |
| Level 3 | 8 | 复杂算法(Winograd, Attention) | winograd_*, se_layer_nhwc |
| Level 4 | 2 | NVIDIA专有库(CUTLASS) | fused_mha_cutlass |

---

## 2. 工作流架构

### 2.1 整体流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     CUDA-to-SYCL Conversion Pipeline             │
└─────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
┌───────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ Phase 1:      │      │ Phase 2:        │      │ Phase 3:        │
│ 环境准备      │ ───► │ CUDA分析        │ ───► │ SYCL生成        │
│ (基础设施)    │      │ (算法理解)      │      │ (代码生成)      │
└───────────────┘      └─────────────────┘      └─────────────────┘
                                                        │
        ┌───────────────────────────────────────────────┼───────────┐
        ▼                                               ▼           ▼
┌───────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ Phase 4:      │      │ Phase 5:        │      │ Phase 6:        │
│ 编译验证      │ ◄─── │ 测试生成        │ ◄─── │ 数据验证        │
│ (语法检查)    │      │ (用例生成)      │      │ (精度检查)      │
└───────────────┘      └─────────────────┘      └─────────────────┘
        │
        ▼
┌───────────────┐
│ Phase 7:      │
│ 文档与交付    │
└───────────────┘
```

### 2.2 目录结构

```
kernel_dataset/
├── cuda/                          # 源CUDA kernel文件
│   ├── add_vectors_kernel.cu
│   ├── softmax_kernel.cu
│   └── ... (30 files)
├── sycl/                          # 生成的SYCL kernel文件
│   └── [待生成] 30个.dp.cpp文件
├── tests/                         # 测试框架
│   ├── include/
│   │   ├── test_framework.h      # 统一测试接口
│   │   └── data_generator.h      # 测试数据生成
│   ├── cuda_wrappers/            # CUDA包装器
│   ├── sycl_wrappers/            # SYCL包装器
│   └── test_cases/               # 测试用例
│       ├── level1/               # 基础kernel测试
│       ├── level2/               # 中等复杂度测试
│       └── level3/               # 复杂kernel测试
├── tools/                         # 自动化工具
│   ├── cuda_analyzer.py          # CUDA代码分析器
│   ├── sycl_generator.py         # SYCL代码生成器
│   ├── build_validator.py        # 编译验证工具
│   ├── test_generator.py         # 测试用例生成器
│   ├── accuracy_checker.py       # 精度验证工具
│   └── workflow_runner.py        # 主工作流控制器
├── plan/                          # 规划文档
│   └── WORKFLOW.md               # 本文档
└── index.json                     # Kernel元数据索引
```

---

## 3. 各阶段详细规范

### Phase 1: 环境准备 (Environment Setup)

**目标**: 建立完整的基础设施

**任务清单**:
1. [ ] 创建目录结构 (sycl/, tests/, tools/)
2. [ ] 检查编译器环境
   - Intel oneAPI (icpx)
   - NVIDIA CUDA (nvcc)
   - 验证SYCL后端支持
3. [ ] 创建基础配置文件
   - `tools/config.yaml`: 编译器路径、标志位
   - `tools/mappings.json`: CUDA-SYCL映射规则

**输出产物**:
- 完整目录结构
- 环境检查报告
- 配置文件模板

---

### Phase 2: CUDA算法分析 (CUDA Analysis)

**目标**: 深度理解CUDA kernel实现

**输入**: `.cu`文件
**输出**: 分析JSON报告

**分析维度**:

| 分析项 | 说明 | 重要性 |
|--------|------|--------|
| Kernel签名 | 参数类型、模板参数 | 高 |
| Grid/Block结构 | 线程组织方式(1D/2D/3D) | 高 |
| Shared Memory | 分配大小、使用模式 | 高 |
| 同步点 | `__syncthreads()`位置 | 高 |
| Warp操作 | `__shfl_*`, warp reduce | 中 |
| Atomic操作 | `atomicAdd`, `atomicMax` | 中 |
| 内存访问模式 | Coalesced/Scattered | 中 |
| 数学函数 | `exp`, `log`, `sqrt`等 | 低 |

**分析输出格式**:
```json
{
  "kernel_id": "add_vectors",
  "complexity_level": 1,
  "grid_dim": "1D",
  "block_size": "256",
  "shared_memory": "0 bytes",
  "synchronization_points": 0,
  "warp_operations": [],
  "atomic_operations": [],
  "recommended_strategy": "direct_mapping",
  "potential_issues": [],
  "notes": "Simple element-wise operation"
}
```

**Agent指令**:
```
对每个.cu文件:
  1. 解析AST提取kernel函数
  2. 识别CUDA特定关键字和模式
  3. 评估复杂度并分类
  4. 生成分析报告JSON
  5. 标记需要特殊处理的case
```

---

### Phase 3: SYCL代码生成 (SYCL Generation)

**目标**: 生成符合规范的SYCL代码

**策略**: 规则转换 + AI增强

#### 3.1 基础映射规则

| CUDA | SYCL | 说明 |
|------|------|------|
| `__global__` | 移除 | 使用lambda或functor |
| `__shared__` | `sycl::local_accessor` | Shared memory |
| `__syncthreads()` | `item.barrier()` | 线程同步 |
| `__threadfence()` | `item.barrier(sycl::access::fence_space::global_space)` | 全局内存fence |
| `threadIdx.x` | `item.get_local_id(0)` | 线程本地ID |
| `blockIdx.x` | `item.get_group(0)` | Block ID |
| `blockDim.x` | `item.get_local_range(0)` | Block大小 |
| `gridDim.x` | `item.get_group_range(0)` | Grid大小 |
| `warpSize` | `32` (或查询) | Warp大小 |
| `__shfl_xor_sync` | `sycl::shift_group_left/right` | Warp shuffle |
| `atomicAdd` | `sycl::atomic_ref::fetch_add` | 原子加 |
| `atomicMax` | `sycl::atomic_ref::fetch_max` | 原子最大 |
| `__ldg` | `sycl::ext::oneapi::ldg` | 只读缓存加载 |

#### 3.2 代码模板规范

**必须包含的要素**:
1. GPL License头 (从源文件复制)
2. 标准头文件: `sycl/sycl.hpp`
3. 命名空间: `lczero::sycldnn_backend`
4. 常量定义 (从AGENTS.md复制)
5. Helper函数 (`DivUp`等)
6. Kernel文档注释

**模板结构**:
```cpp
/*
  [GPL License Header - 从源文件复制]
*/

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/bfloat16.hpp>

namespace lczero {
namespace sycldnn_backend {

// 标准常量 (复制自AGENTS.md)
constexpr int kNumOutputPolicy = 1858;
constexpr int kMaxResBlockFusingChannels = 384;
// ... 其他常量

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// ActivationFunction enum (从源文件或AGENTS.md复制)
enum ActivationFunction {
  ACTIVATION_NONE,
  // ...
};

// Kernel: {kernel_id}
// Description: {description}
// Category: {category}
template <typename T>
void {kernel_name}_kernel(sycl::queue& queue, /* 参数 */) {
  queue.parallel_for(
    sycl::nd_range<1>(global_size, local_size),
    [=](sycl::nd_item<1> item) {
      // 转换后的kernel逻辑
    }
  );
}

// 模板实例化
// ...

} // namespace sycldnn_backend
} // namespace lczero
```

#### 3.3 特殊情况处理

**Warp-level Primitives**:
```cpp
// CUDA
__device__ __forceinline__ float warpReduce(float x) {
  for (int mask = 16; mask > 0; mask >>= 1)
    x += __shfl_xor_sync(0xFFFFFFFF, x, mask);
  return x;
}

// SYCL替代方案
float warpReduce(sycl::sub_group sg, float x) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    x += sycl::shift_group_left(sg, x, offset);
  }
  return x;
}
```

**Atomic Float Max**:
```cpp
// CUDA有内置atomicMax(float*)
// SYCL需要自定义
void atomicMaxFloat(sycl::atomic_ref<int, ...> addr, float val) {
  // 使用bitwise转换实现
}
```

**Agent指令**:
```
对每个分析后的kernel:
  1. 应用基础映射规则
  2. 检测并处理特殊模式
  3. 应用代码模板
  4. 验证语法完整性
  5. 输出到sycl/目录
```

---

### Phase 4: 编译验证 (Build Validation)

**目标**: 确保生成的SYCL代码可编译

**验证流程**:
```
对每个.dp.cpp文件:
  1. 语法检查: icpx -fsycl -c <file>
  2. 若失败:
     - 分类错误类型
     - 尝试自动修复
     - 记录需要人工干预的case
  3. 生成编译报告
```

**错误处理策略**:

| 错误类型 | 自动修复策略 | 人工介入 |
|---------|-------------|---------|
| 缺失头文件 | 自动添加 | 否 |
| 类型不匹配 | 显式类型转换 | 否 |
| 未定义函数 | 查找并添加定义 | 可能 |
| SYCL特性不支持 | 标记并跳过 | 是 |
| 语法错误 | 尝试基于模式修复 | 可能 |

**编译矩阵**:
```yaml
compilers:
  - name: icpx
    flags: -fsycl -O2 -std=c++17
    targets:
      - host
      - intel_gpu
      - nvidia_gpu
  - name: clang++
    flags: -fsycl -O2
    targets:
      - host
```

**输出产物**:
- 编译日志
- 错误报告
- 通过率统计

---

### Phase 5: 测试生成 (Test Generation)

**目标**: 生成全面的测试用例

**测试类型**:

#### 5.1 单元测试结构
```cpp
TEST(KernelName, BasicFunctionality) {
  // 1. 准备输入数据
  std::vector<float> input = generate_random_data(size);
  
  // 2. 运行CUDA kernel
  std::vector<float> cuda_output = run_cuda_kernel(input);
  
  // 3. 运行SYCL kernel
  std::vector<float> sycl_output = run_sycl_kernel(input);
  
  // 4. 比较结果
  EXPECT_TRUE(compare_tensors(cuda_output, sycl_output));
}
```

#### 5.2 测试用例生成策略

| 测试类型 | 生成方法 | 覆盖目标 |
|---------|---------|---------|
| 边界值 | 0, 1, -1, min, max | 边界条件 |
| 随机数据 | 多种分布(uniform, normal) | 一般情况 |
| 小规模 | size=1, 16, 32, 64 | 基础功能 |
| 大规模 | size=1024, 65536 | 性能特征 |
| 特殊值 | inf, nan, subnormal | 鲁棒性 |
| 多维度 | 各种N,C,H,W组合 | 参数兼容性 |

#### 5.3 精度验证标准

```cpp
// 数值比较函数
bool compare_values(float cuda_val, float sycl_val) {
  const float abs_tol = 1e-5f;
  const float rel_tol = 1e-4f;
  
  float abs_diff = std::abs(cuda_val - sycl_val);
  float rel_diff = abs_diff / (std::abs(cuda_val) + 1e-8f);
  
  return (abs_diff <= abs_tol) || (rel_diff <= rel_tol);
}
```

**Agent指令**:
```
对每个kernel:
  1. 分析参数范围和类型
  2. 生成至少5类测试用例
  3. 创建测试驱动代码
  4. 输出到tests/test_cases/
```

---

### Phase 6: 数据准确性验证 (Accuracy Validation)

**目标**: 验证CUDA和SYCL输出数值一致

**验证流程**:
```
对每个测试用例:
  1. 初始化相同随机种子
  2. 生成相同输入数据
  3. 运行CUDA kernel获取baseline
  4. 运行SYCL kernel获取结果
  5. 逐元素比较
  6. 统计通过/失败率
  7. 生成验证报告
```

**通过率标准**:
- 100%通过: 完全匹配
- ≥99.9%通过: 可接受(记录差异)
- <99.9%通过: 需要调查

**差异分析方法**:
```python
def analyze_differences(cuda_out, sycl_out):
    diff = np.abs(cuda_out - sycl_out)
    
    # 统计信息
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # 差异分布
    hist, bins = np.histogram(diff, bins=100)
    
    # 定位问题区域
    error_indices = np.where(diff > threshold)
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'error_count': len(error_indices[0]),
        'error_rate': len(error_indices[0]) / len(cuda_out)
    }
```

**输出产物**:
- 验证报告 (每个kernel)
- 差异分析 (如有失败)
- 总体通过率统计

---

### Phase 7: 文档与交付 (Documentation)

**目标**: 生成完整的项目文档

**文档清单**:

| 文档 | 内容 | 位置 |
|------|------|------|
| WORKFLOW.md | 本工作流文档 | plan/ |
| CONVERSION_GUIDE.md | CUDA-to-SYCL转换指南 | docs/ |
| API_MAPPING.md | API对照表 | docs/ |
| TEST_REPORT.md | 测试报告 | docs/ |
| STATUS.md | 转换状态总览 | 根目录 |

**index.json更新规范**:
```json
{
  "id": "kernel_name",
  "name": "Kernel Name",
  "has_sycl_mapping": true,
  "sycl": {
    "file": "sycl/kernel_name_kernel.dp.cpp",
    "verified": true,
    "compilation_status": "success",
    "test_coverage": 95.5,
    "last_updated": "2026-02-28"
  },
  "notes": "Successfully converted and verified"
}
```

---

## 4. Agent执行指令

### 4.1 主控制器

```python
# workflow_runner.py 伪代码

class CudaToSyclWorkflow:
    def __init__(self):
        self.phases = [
            EnvironmentSetupPhase(),
            CudaAnalysisPhase(),
            SyclGenerationPhase(),
            BuildValidationPhase(),
            TestGenerationPhase(),
            AccuracyValidationPhase(),
            DocumentationPhase()
        ]
    
    def run(self, kernel_filter=None):
        # 1. 环境准备
        self.phases[0].execute()
        
        # 2-6. 对每个kernel执行pipeline
        kernels = self.get_kernels(kernel_filter)
        
        for kernel in kernels:
            try:
                # 分析
                analysis = self.phases[1].analyze(kernel)
                
                # 生成
                sycl_code = self.phases[2].generate(kernel, analysis)
                
                # 编译验证
                if not self.phases[3].validate(sycl_code):
                    self.handle_compile_error(kernel)
                    continue
                
                # 生成测试
                tests = self.phases[4].generate(kernel)
                
                # 数据验证
                if not self.phases[5].validate(kernel, tests):
                    self.handle_accuracy_error(kernel)
                    continue
                
                # 标记成功
                self.mark_success(kernel)
                
            except Exception as e:
                self.log_error(kernel, e)
        
        # 7. 生成文档
        self.phases[6].generate()
```

### 4.2 批量执行策略

**按优先级分批**:

```
Batch 1: Level 1 kernels (基础操作)
  - add_vectors_kernel
  - add_bias_batched_kernel
  - copy_type_converted_kernel
  - nchw_to_nhwc_kernel
  - add_vectors_hnc_nhc_kernel
  - add_bias_nchw_kernel
  - expand_planes_nhwc_kernel
  - expand_planes_nchw_kernel

Batch 2: Level 2 kernels (归一化/池化)
  - batch_norm_kernel
  - layer_norm_kernel
  - global_scale_kernel
  - global_scale_fp16_nhwc_kernel
  - global_avg_pool_kernel
  - global_avg_pool_nhwc_fp16_kernel
  - softmax_kernel
  - softmax_opt_64_kernel

Batch 3: Level 2+ kernels (Attention相关)
  - policy_map_kernel
  - promotion_logits_kernel
  - preprocess_attention_body_kernel
  - input_gating_kernel
  - gen_offset_pointers_kernel
  - se_layer_nhwc_kernel

Batch 4: Level 3 kernels (Winograd)
  - winograd_filter_transform_kernel
  - winograd_input_transform_kernel
  - winograd_output_transform_kernel
  - winograd_output_se_relu_input_kernel
  - winograd_output_relu_input_kernel
  - output_input_transform_fp16_shmem_kernel

Batch 5: Special cases
  - fused_mha_cutlass_kernel (CUDA only,标记为无SYCL映射)
```

### 4.3 错误处理与恢复

**检查点机制**:
- 每个phase完成后保存状态
- 支持从断点恢复
- 记录已完成的kernel

**错误分类**:
```python
class ErrorType(Enum):
    COMPILATION_ERROR = "compilation"
    ACCURACY_MISMATCH = "accuracy"
    UNSUPPORTED_FEATURE = "unsupported"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

# 处理策略
error_handlers = {
    ErrorType.COMPILATION_ERROR: auto_fix_or_skip,
    ErrorType.ACCURACY_MISMATCH: investigate_and_tune,
    ErrorType.UNSUPPORTED_FEATURE: mark_manual_conversion,
    ErrorType.TIMEOUT: retry_with_limits,
}
```

---

## 5. 工具链规范

### 5.1 工具清单

| 工具名 | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `cuda_analyzer.py` | 分析CUDA代码 | .cu文件 | JSON分析报告 |
| `sycl_generator.py` | 生成SYCL代码 | JSON分析 | .dp.cpp文件 |
| `build_validator.py` | 编译验证 | .dp.cpp文件 | 编译报告 |
| `test_generator.py` | 生成测试用例 | kernel元数据 | 测试代码 |
| `accuracy_checker.py` | 数值验证 | 可执行文件 | 验证报告 |
| `workflow_runner.py` | 主控制器 | 配置 | 完整结果 |

### 5.2 配置规范

**config.yaml**:
```yaml
environment:
  cuda_compiler: /usr/local/cuda/bin/nvcc
  sycl_compiler: /opt/intel/oneapi/compiler/latest/bin/icpx
  
compilation:
  cuda_flags: -O2 -arch=sm_70
  sycl_flags: -fsycl -O2 -std=c++17
  
validation:
  abs_tolerance: 1.0e-5
  rel_tolerance: 1.0e-4
  min_pass_rate: 99.9
  
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## 6. 质量指标

### 6.1 成功标准

| 指标 | 目标值 | 测量方法 |
|------|--------|---------|
| 编译通过率 | 100% (29/30) | 成功编译的kernel数 |
| 数值验证通过率 | ≥99.9% | 测试通过的assertion比例 |
| 代码覆盖率 | ≥90% | 行覆盖率 |
| 文档完整性 | 100% | 每个kernel有对应文档 |

### 6.2 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 复杂kernel转换错误 | 中 | 高 | 分阶段验证,先简单后复杂 |
| 编译器不支持SYCL特性 | 低 | 中 | 使用标准SYCL,避免扩展 |
| 数值精度问题 | 中 | 高 | 严格的epsilon-based比较 |
| FP16处理差异 | 中 | 中 | 单独测试FP16路径 |
| 时间超支 | 中 | 低 | 按优先级分批,关键路径优先 |

---

## 7. 时间估算

| Phase | 时间估算 | 关键产出 |
|-------|---------|---------|
| Phase 1: 环境准备 | 0.5天 | 基础设施就绪 |
| Phase 2: CUDA分析 | 1天 | 30个分析报告 |
| Phase 3: SYCL生成 | 2-3天 | 30个.dp.cpp文件 |
| Phase 4: 编译验证 | 1天 | 编译报告 |
| Phase 5: 测试生成 | 1-2天 | 测试套件 |
| Phase 6: 数据验证 | 2-3天 | 验证报告 |
| Phase 7: 文档交付 | 0.5天 | 完整文档 |
| **总计** | **8-11天** | 完整SYCL库 |

**并行优化**:
- Batch 1-4可独立执行
- 每个kernel的pipeline可并行
- 预计实际时间: **5-7天** (4线程并行)

---

## 8. 附录

### 8.1 CUDA-SYCL速查表

详见 `docs/API_MAPPING.md`

### 8.2 参考资源

- **SYCL 2020规范**: https://www.khronos.org/registry/SYCL/
- **Intel oneAPI DPC++**: https://intel.github.io/llvm-docs/
- **CUDA-to-SYCL迁移指南**: https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-to-dpcpp-migration.html

### 8.3 代码风格检查清单

- [ ] GPL License头完整
- [ ] 命名空间正确 (`lczero::sycldnn_backend`)
- [ ] 缩进2空格
- [ ] 行长度≤80字符
- [ ] 常量使用`kCamelCase`
- [ ] 函数使用`snake_case`
- [ ] 包含所有必要的头文件
- [ ] 模板实例化完整

---

**文档结束**

*本文档作为Agent执行CUDA-to-SYCL转换任务的输入规范。执行前请确认所有前置条件已满足。*
