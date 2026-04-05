# TurboDiffusion CUDA-to-SYCL 转换任务说明

## 1. 任务概述

将 **TurboDiffusion Video Diffusion加速框架** 从 CUDA 迁移到 **Intel BMG B60 GPU (Xe2架构)** 的 SYCL 实现。

**项目规模**:
- 代码总量: ~1,920行 CUDA C++
- 核心算子: 4个 (LayerNorm, RMSNorm, Quantization, GEMM)
- 数据类型: BF16, FP32 (关键要求)
- 优先级: LayerNorm (P0) → RMSNorm (P0) → Quantization (P1) → GEMM (P2)

**关键约束**:
- 使用 Intel oneAPI DPC++/C++ Compiler
- 目标架构: BMG B60 (Xe2), Sub-group size = 16
- 支持 BF16 ↔ FP32 混合精度计算
- 数值误差 < 1e-4
- 性能目标: 达到 CUDA 版本的 80%+

---

## 2. 输入文件清单

### 2.1 P0 优先级 (首先实现)

#### LayerNorm (`ops/norm/layernorm.hpp` + `ops/norm/layernorm.cu`)
- **文件路径**: `/home/intel/tianfeng/opencode_bench/TurboDiffusion/turbodiffusion/ops/norm/`
- **代码行数**: 202行 (hpp) + 相关cu文件
- **核心特性**:
  - 模板化: `InputDtype, OutputDtype, WeightDtype, Affine, Bias, MaxHiddenSize, NumThrPerCta, IsEven`
  - Warp shuffle归约: `__shfl_down_sync`
  - Shared memory atomic: `atomicAdd`
  - 向量化加载: `Loader<InputDtype, 1, MaxHiddenSize, NumThrPerCta, true, IsEven>`
  - 配置切换: `BOOL_SWITCH`, `CONFIG_SWITCH`

#### RMSNorm (`ops/norm/rmsnorm.hpp` + `ops/norm/rmsnorm.cu`)
- **文件路径**: `/home/intel/tianfeng/opencode_bench/TurboDiffusion/turbodiffusion/ops/norm/`
- **代码行数**: 147行 (hpp)
- **核心特性**:
  - 与LayerNorm高度相似，但不计算mean
  - 只计算RMS: `sqrtf(_reduce_square(x, shared_data) / params.n + params.eps)`
  - 相同的warp shuffle和shared memory模式

### 2.2 P1 优先级 (其次实现)

#### Quantization (`ops/quant/quant.hpp` + `ops/quant/quant.cu`)
- **文件路径**: `/home/intel/tianfeng/opencode_bench/TurboDiffusion/turbodiffusion/ops/quant/`
- **代码行数**: 195行 (hpp)
- **核心特性**:
  - FP16/BF16 → INT8 量化
  - Block-wise AMMAX归约
  - Warp shuffle XOR: `__shfl_xor_sync`
  - atomicMax shared memory操作
  - CUTLASS numeric conversion

### 2.3 P2 优先级 (最后实现)

#### GEMM (`ops/gemm/kernel.hpp` + `ops/gemm/gemm.cu`)
- **文件路径**: `/home/intel/tianfeng/opencode_bench/TurboDiffusion/turbodiffusion/ops/gemm/`
- **代码行数**: 522行 (kernel.hpp)
- **核心特性**:
  - CUTLASS/CuTe框架（高难度）
  - 多级流水线 Stage=3
  - 异步拷贝 `cp.async`
  - INT8输入，BF16/FP16输出

### 2.4 通用依赖文件

```
ops/common/
├── load.hpp      # 向量化数据加载器
├── store.hpp     # 向量化数据存储器
├── launch.hpp    # CUDA kernel启动封装
└── common.hpp    # 通用定义和工具
```

---

## 3. CUDA-to-SYCL 转换映射表

### 3.1 线程模型转换

| CUDA | SYCL |
|------|------|
| `dim3 grid(m);` | `sycl::range<1> global(m * NumThrPerCta);` |
| `dim3 block(NumThrPerCta);` | `sycl::range<1> local(NumThrPerCta);` |
| `kernel<<<grid, block, ShmSize, stream>>>(params);` | `queue.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> item) { kernel(params, item, shared_data); });` |
| `blockIdx.x` | `item.get_group(0)` |
| `threadIdx.x` | `item.get_local_id(0)` |
| `blockDim.x` | `item.get_local_range(0)` |
| `gridDim.x` | `item.get_group_range(0)` |

### 3.2 Warp/Sub-group操作转换

| CUDA | SYCL (Xe2, sub-group size=16) |
|------|-------------------------------|
| `__shfl_down_sync(0xFFFFFFFF, value, offset)` | `sycl::shift_group_right(sg, value, offset)` |
| `__shfl_xor_sync(0xffffffff, value, offset, 32)` | `sycl::select_from_group(sg, value, item.get_sub_group().get_local_linear_id() ^ offset)` |
| `__syncwarp()` | `item.barrier(sycl::access::fence_space::local_space)` |
| Warp size 32 | Sub-group size **16** (B60固定) |

**注意**: B60 GPU sub-group size为16，不是32。CUDA代码中通常假设32-wide warp，需要调整:
- 原CUDA: `for (int i = 16; i >= 1; i >>= 1)` (5 iterations for 32-wide)
- 新SYCL: `for (int i = 8; i >= 1; i >>= 1)` (4 iterations for 16-wide)

### 3.3 Shared Memory转换

| CUDA | SYCL |
|------|------|
| `__shared__ char shared_data[ShmSize];` | `sycl::local_accessor<char, 1> shared_data(sycl::range<1>(ShmSize), h);` |
| `__shared__ float warp_sums[32];` | `sycl::local_accessor<float, 1> warp_sums(sycl::range<1>(32), h);` |
| `__syncthreads();` | `item.barrier(sycl::access::fence_space::local_space);` |

### 3.4 原子操作转换

| CUDA | SYCL |
|------|------|
| `atomicAdd((float*)shared_data, sum);` | `sycl::atomic_ref<float, sycl::memory_order_relaxed, sycl::memory_scope_work_group> atomic_ref(*(float*)shared_data); atomic_ref.fetch_add(sum);` |
| `atomicMax((uint32_t*)smem_ptr, value);` | `sycl::atomic_ref<uint32_t, sycl::memory_order_relaxed, sycl::memory_scope_work_group> atomic_ref(*(uint32_t*)smem_ptr); atomic_ref.fetch_max(value);` |

### 3.5 数据类型转换

| CUDA | SYCL |
|------|------|
| `__half` / `half` | `sycl::half` |
| `__nv_bfloat16` | `sycl::ext::oneapi::bfloat16` |
| `float` | `float` (same) |
| `int8_t` | `int8_t` (same) |

### 3.6 CUTLASS宏转换

| CUDA/CUTLASS | SYCL |
|--------------|------|
| `CUTLASS_DEVICE` | 移除 (SYCL使用lambda或functor) |
| `CUTLASS_PRAGMA_UNROLL` | `#pragma unroll` 或 `[[intel::loop_unroll]]` |

---

## 4. B60 GPU 特定优化 (基于LCZero经验)

### 4.1 BF16/FP32 混合精度模式

所有计算应在FP32中进行，输入/输出使用BF16:

```cpp
// 输入: BF16 → FP32
using bfloat16 = sycl::ext::oneapi::bfloat16;
bfloat16 input_val = input_ptr[idx];
float x = static_cast<float>(input_val);

// 计算在FP32...
float result = compute_in_fp32(x);

// 输出: FP32 → BF16
output_ptr[idx] = static_cast<bfloat16>(result);
```

### 4.2 SLM (Shared Local Memory) 缓存优化

对于小参数 (weight/bias)，缓存到SLM:

```cpp
// 如果C < 100KB (SLM预算)
if (params.n < 25600) {  // 100KB / sizeof(float)
  sycl::local_accessor<float, 1> local_weight(
    sycl::range<1>(params.n), h);
  
  // 协作加载weight到SLM
  for (int i = tidx; i < params.n; i += NumThrPerCta) {
    local_weight[i] = static_cast<float>(weight_ptr[i]);
  }
  item.barrier(sycl::access::fence_space::local_space);
  
  // 快速SLM访问
  float w = local_weight[c];
}
```

### 4.3 Sub-group 大小注解

B60固定sub-group size = 16，必须显式标注:

```cpp
q.parallel_for(
  sycl::nd_range<1>(global_size, local_size),
  [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
    auto sg = item.get_sub_group();
    // 使用16-lane sub-group操作
  }
);
```

### 4.4 Grid 配置优化

```cpp
// ❌ 避免2D grid限制并行度
sycl::range<2> global(cdiv(m, BlockSize), cdiv(n, BlockSize));

// ✅ 使用1D flattened grid最大化并行度
sycl::range<1> global(m * n * wg_size / (BlockSize * BlockSize));
```

### 4.5 Work-group 大小选择

基于LCZero在B60上的测试:
- LayerNorm/RMSNorm: 256-512 threads per CTA
- Quantization: 128-256 threads per CTA
- 优先使用256 (平衡寄存器使用和occupancy)

---

## 5. 输出文件结构

对于每个算子，生成以下文件:

```
turbodiffusion/
└── ops/
    └── norm/
        ├── layernorm.hpp          # SYCL头文件 (原CUDA hpp转换)
        ├── layernorm.dp.cpp       # SYCL实现 (kernel代码)
        ├── layernorm_binding.cpp  # pybind11绑定
        └── test_layernorm.py      # Python单元测试
    
    └── quant/
        ├── quant.hpp
        ├── quant.dp.cpp
        ├── quant_binding.cpp
        └── test_quant.py
```

### 5.1 SYCL头文件模板 (`layernorm.hpp`)

```cpp
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

namespace turbodiffusion {

template <
  class InputDtype_,
  class OutputDtype_,
  class WeightDtype_,
  bool Affine_,
  bool Bias_,
  int MaxHiddenSize_,
  int NumThrPerCta_,
  bool IsEven
>
class LayerNorm {
public:
  using InputDtype = InputDtype_;
  using OutputDtype = OutputDtype_;
  using WeightDtype = WeightDtype_;
  static constexpr int NumThrPerCta = NumThrPerCta_;
  static constexpr int MaxHiddenSize = MaxHiddenSize_;
  static constexpr bool Affine = Affine_;
  static constexpr bool Bias = Bias_;
  
  static constexpr size_t ShmSize = 32;
  static constexpr int NumElementPerThread = MaxHiddenSize / NumThrPerCta;
  
  static_assert(MaxHiddenSize % NumThrPerCta == 0);
  
  struct Params {
    const void* Iptr;
    const void* Wptr;
    const void* Bptr;
    void* Optr;
    float eps;
    int64_t m;
    int64_t n;
  };
  
  // Kernel launcher
  static void launch(
    sycl::queue& q,
    const Params& params,
    int64_t m,
    int64_t n
  );
  
private:
  // Kernel implementation as functor or lambda
  static void kernel(
    const Params params,
    sycl::nd_item<1> item,
    sycl::local_accessor<char, 1> shared_data
  );
  
  static float reduce_sum(float* reg, sycl::nd_item<1> item, 
                          sycl::local_accessor<char, 1> shared_data);
  static float reduce_square(float* reg, sycl::nd_item<1> item,
                             sycl::local_accessor<char, 1> shared_data);
};

// Convenience function
template <class InputDtype, class OutputDtype, class WeightDtype,
          bool Affine, bool Bias, int MaxHiddenSize, int NumThrPerCta>
bool layernorm(sycl::queue& q,
               const void* Iptr, const void* Wptr, const void* Bptr,
               void* Optr, float eps, int64_t m, int64_t n);

} // namespace turbodiffusion
```

### 5.2 SYCL实现模板 (`layernorm.dp.cpp`)

```cpp
#include "layernorm.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

namespace turbodiffusion {

using bfloat16 = sycl::ext::oneapi::bfloat16;

template <class InputDtype, class OutputDtype, class WeightDtype,
          bool Affine, bool Bias, int MaxHiddenSize, int NumThrPerCta, bool IsEven>
void LayerNorm<InputDtype, OutputDtype, WeightDtype, Affine, Bias, 
               MaxHiddenSize, NumThrPerCta, IsEven>::launch(
    sycl::queue& q,
    const Params& params,
    int64_t m,
    int64_t n
) {
  sycl::range<1> global(m * NumThrPerCta);
  sycl::range<1> local(NumThrPerCta);
  
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<char, 1> shared_data(sycl::range<1>(ShmSize), h);
    
    h.parallel_for(
      sycl::nd_range<1>(global, local),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        kernel(params, item, shared_data);
      }
    );
  });
}

template <class InputDtype, class OutputDtype, class WeightDtype,
          bool Affine, bool Bias, int MaxHiddenSize, int NumThrPerCta, bool IsEven>
void LayerNorm<InputDtype, OutputDtype, WeightDtype, Affine, Bias,
               MaxHiddenSize, NumThrPerCta, IsEven>::kernel(
    const Params params,
    sycl::nd_item<1> item,
    sycl::local_accessor<char, 1> shared_data
) {
  const int blk_m = item.get_group(0);
  const int tidx = item.get_local_id(0);
  float x[NumElementPerThread];
  
  // Load data (implement or adapt Loader)
  // TODO: Adapt Loader class for SYCL
  
  // Mean reduction
  float u = reduce_sum(x, item, shared_data) / params.n;
  
  #pragma unroll
  for (int i = 0; i < NumElementPerThread; ++i) {
    x[i] -= u;
  }
  
  item.barrier(sycl::access::fence_space::local_space);
  
  // Variance reduction
  float v = sycl::sqrt(reduce_square(x, item, shared_data) / params.n + params.eps);
  
  #pragma unroll
  for (int i = 0; i < NumElementPerThread; ++i) {
    x[i] /= v;
  }
  
  // Affine transformation (if enabled)
  if constexpr (Affine) {
    // TODO: Load weight and bias, apply transformation
  }
  
  // Store result
  // TODO: Adapt Saver class for SYCL
}

template <class InputDtype, class OutputDtype, class WeightDtype,
          bool Affine, bool Bias, int MaxHiddenSize, int NumThrPerCta, bool IsEven>
float LayerNorm<InputDtype, OutputDtype, WeightDtype, Affine, Bias,
                MaxHiddenSize, NumThrPerCta, IsEven>::reduce_sum(
    float* reg,
    sycl::nd_item<1> item,
    sycl::local_accessor<char, 1> shared_data
) {
  float sum = 0.0f;
  
  #pragma unroll
  for (int i = 0; i < NumElementPerThread; ++i) {
    sum += reg[i];
  }
  
  // Sub-group reduction (16-wide)
  auto sg = item.get_sub_group();
  #pragma unroll
  for (int i = 8; i >= 1; i >>= 1) {
    sum += sycl::shift_group_right(sg, sum, i);
  }
  
  // Initialize shared memory
  if (item.get_local_id(0) == 0) {
    *reinterpret_cast<float*>(shared_data.get_pointer().get()) = 0.0f;
  }
  item.barrier(sycl::access::fence_space::local_space);
  
  // Atomic add to shared memory
  if (item.get_sub_group().get_local_linear_id() == 0) {
    sycl::atomic_ref<float, sycl::memory_order_relaxed,
                     sycl::memory_scope_work_group>
      atomic_ref(*reinterpret_cast<float*>(shared_data.get_pointer().get()));
    atomic_ref.fetch_add(sum);
  }
  
  item.barrier(sycl::access::fence_space::local_space);
  
  return *reinterpret_cast<float*>(shared_data.get_pointer().get());
}

// Explicit template instantiations
// TODO: Add instantiations for common types

} // namespace turbodiffusion
```

### 5.3 Pybind11绑定模板 (`layernorm_binding.cpp`)

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sycl/sycl.hpp>
#include "layernorm.hpp"

namespace py = pybind11;
using namespace turbodiffusion;

class SYCLQueue {
public:
  sycl::queue q;
  SYCLQueue() : q(sycl::gpu_selector_v) {}
};

py::array_t<float> layernorm_py(
    SYCLQueue& queue_wrapper,
    py::array_t<float> input,
    py::array_t<float> weight,
    py::array_t<float> bias,
    float eps
) {
  auto buf_in = input.request();
  auto buf_w = weight.request();
  auto buf_b = bias.request();
  
  int64_t m = buf_in.shape[0];
  int64_t n = buf_in.shape[1];
  
  auto result = py::array_t<float>({m, n});
  auto buf_out = result.request();
  
  // Configure and launch kernel
  // TODO: Implement
  
  return result;
}

PYBIND11_MODULE(turbodiffusion_ops, m) {
  py::class_<SYCLQueue>(m, "SYCLQueue")
    .def(py::init<>());
  
  m.def("layernorm", &layernorm_py, "LayerNorm forward",
        py::arg("queue"),
        py::arg("input"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("eps") = 1e-5f);
}
```

### 5.4 Python单元测试模板 (`test_layernorm.py`)

```python
import unittest
import numpy as np
import torch
import turbodiffusion_ops as ops

class TestLayerNorm(unittest.TestCase):
  def setUp(self):
    self.queue = ops.SYCLQueue()
    self.batch_size = 32
    self.hidden_size = 768
    self.eps = 1e-5
    
  def test_layernorm_fp32(self):
    """Test FP32 LayerNorm accuracy"""
    # Generate test data
    x = np.random.randn(self.batch_size, self.hidden_size).astype(np.float32)
    weight = np.random.randn(self.hidden_size).astype(np.float32)
    bias = np.random.randn(self.hidden_size).astype(np.float32)
    
    # Reference implementation (PyTorch)
    x_torch = torch.from_numpy(x)
    weight_torch = torch.from_numpy(weight)
    bias_torch = torch.from_numpy(bias)
    
    ln_torch = torch.nn.functional.layer_norm(
      x_torch, (self.hidden_size,),
      weight=weight_torch, bias=bias_torch,
      eps=self.eps
    )
    
    # SYCL implementation
    result = ops.layernorm(self.queue, x, weight, bias, self.eps)
    
    # Compare
    max_diff = np.max(np.abs(result - ln_torch.numpy()))
    self.assertLess(max_diff, 1e-4, f"Max diff {max_diff} exceeds threshold")
    
  def test_layernorm_bf16(self):
    """Test BF16 LayerNorm accuracy"""
    # TODO: Implement BF16 test
    pass
    
  def test_layernorm_performance(self):
    """Benchmark performance vs PyTorch CUDA"""
    # TODO: Implement performance benchmark
    pass

if __name__ == '__main__':
  unittest.main()
```

---

## 6. 转换清单 (按算子)

### 6.1 LayerNorm转换清单

#### Phase 1: 基础转换
- [ ] 替换 `#include <cuda_runtime.h>` → `#include <sycl/sycl.hpp>`
- [ ] 添加 `#include <sycl/ext/oneapi/bfloat16.hpp>`
- [ ] 移除 `CUTLASS_DEVICE` 宏
- [ ] 替换 `__syncthreads()` → `item.barrier()`
- [ ] 转换线程索引: `blockIdx.x` → `item.get_group(0)`
- [ ] 转换shared memory: `__shared__` → `sycl::local_accessor`

#### Phase 2: 归约操作转换
- [ ] 替换 warp shuffle: `__shfl_down_sync` → `sycl::shift_group_right`
- [ ] 调整sub-group大小: 32-wide → 16-wide (B60)
- [ ] 替换 atomicAdd: CUDA atomic → `sycl::atomic_ref`
- [ ] 更新归约循环: `for (int i = 16; ...)` → `for (int i = 8; ...)`

#### Phase 3: 数据加载/存储适配
- [ ] 转换 `Loader` 类 (向量化加载)
- [ ] 转换 `Saver` 类 (向量化存储)
- [ ] 添加BF16支持: `sycl::ext::oneapi::bfloat16`
- [ ] 实现数据类型转换: BF16↔FP32

#### Phase 4: 优化
- [ ] 添加 `[[sycl::reqd_sub_group_size(16)]]` 注解
- [ ] 实现SLM缓存 (对于小weight)
- [ ] 优化grid配置 (1D flattened)
- [ ] 测试不同work-group大小 (256 vs 512)

### 6.2 RMSNorm转换清单
- [ ] (与LayerNorm相同，但不计算mean)
- [ ] 复用LayerNorm的reduce_square实现
- [ ] 简化kernel逻辑 (跳过mean计算)

### 6.3 Quantization转换清单
- [ ] warp shuffle XOR: `__shfl_xor_sync` → `sycl::select_from_group`
- [ ] atomicMax: CUDA → `sycl::atomic_ref::fetch_max`
- [ ] Numeric conversion: CUTLASS → SYCL built-in
- [ ] Block-wise AMMAX归约适配

---

## 7. 编译和构建要求

### 7.1 编译命令

```bash
# 使用Intel oneAPI DPC++ Compiler
source /opt/intel/oneapi/setvars.sh

# 编译SYCL kernel
dpcpp -fsycl -fsycl-targets=spir64_gen \
  -Xsycl-target-backend=spir64_gen \
  "-device bmg -options '-ze-opt-large-register-file'" \
  -O3 -o layernorm.dp.o -c layernorm.dp.cpp

# 编译Python绑定
dpcpp -fsycl -shared -fPIC \
  $(python3 -m pybind11 --includes) \
  -o turbodiffusion_ops$(python3-config --extension-suffix) \
  layernorm_binding.cpp layernorm.dp.o
```

### 7.2 CMakeLists.txt 模板

```cmake
cmake_minimum_required(VERSION 3.16)
project(TurboDiffusionSYCL)

set(CMAKE_CXX_COMPILER dpcpp)
set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
find_package(IntelSYCL REQUIRED)

add_sycl_library(turbodiffusion_ops SHARED
  ops/norm/layernorm.dp.cpp
  ops/norm/rmsnorm.dp.cpp
  ops/quant/quant.dp.cpp
)

target_compile_options(turbodiffusion_ops PRIVATE
  -O3
  -fsycl
  -fsycl-targets=spir64_gen
)

pybind11_add_module(turbodiffusion_py
  ops/bindings/pybind_module.cpp
)

target_link_libraries(turbodiffusion_py PRIVATE turbodiffusion_ops)
```

---

## 8. 测试要求

### 8.1 功能测试

每个算子必须通过:
1. **数值精度测试**: 与PyTorch CUDA参考实现对比
   - FP32: max_diff < 1e-5
   - BF16: max_diff < 1e-3
   
2. **边界条件测试**:
   - 极小batch size (m=1)
   - 极大hidden size (n=8192)
   - 非对齐数据

3. **随机性测试**:
   - 多次运行结果一致性
   - 不同随机种子稳定性

### 8.2 性能基准

1. **吞吐量测试**:
   - 测量GB/s (内存带宽利用率)
   - 测量TFLOPS (计算利用率)

2. **与CUDA对比**:
   - 性能 ≥ CUDA的80%
   - 如果 < 80%，需分析瓶颈并提供优化建议

3. **Scaling测试**:
   - 不同batch size (1, 16, 32, 64, 128)
   - 不同hidden size (256, 512, 768, 1024, 2048)

### 8.3 性能分析工具

```bash
# Intel VTune Profiler
vtune -collect gpu-hotspots -result-dir vtune_results ./test_layernorm

# Intel Advisor
advisor --collect=roofline --profile-gpu -- ./test_layernorm

# oneAPI Performance Profiling
ZE_ENABLE_TRACING=1 ./test_layernorm  # Generate traces
```

---

## 9. 验收标准

### 9.1 必需项 (Must Have)

- [ ] 所有P0算子 (LayerNorm, RMSNorm) 编译通过
- [ ] 所有P0算子数值测试通过 (FP32 diff < 1e-5, BF16 diff < 1e-3)
- [ ] Python绑定可用 (`import turbodiffusion_ops` 成功)
- [ ] 基础性能基准文档化

### 9.2 期望项 (Should Have)

- [ ] P1算子 (Quantization) 完成转换和测试
- [ ] 性能达到CUDA的80%+
- [ ] 完整的单元测试覆盖率 (>80%)
- [ ] 性能分析报告

### 9.3 加分项 (Nice to Have)

- [ ] P2算子 (GEMM) 部分实现
- [ ] 性能超过CUDA
- [ ] 自动tuning脚本 (grid size, wg size)
- [ ] 与PyTorch集成 (torch.autograd.Function)

---

## 10. 常见问题与解决方案

### Q1: Sub-group大小不匹配
**问题**: CUDA假设32-wide warp，SYCL B60是16-wide  
**解决**: 使用 `[[sycl::reqd_sub_group_size(16)]]` 并调整shuffle循环

### Q2: Shared memory atomics性能差
**问题**: SYCL atomic_ref可能比CUDA atomic慢  
**解决**: 使用sub-group reduction替代atomic，或尽量减少atomic操作次数

### Q3: BF16支持
**问题**: SYCL BF16需要特定头文件和类型  
**解决**: 使用 `sycl::ext::oneapi::bfloat16`，注意显式类型转换

### Q4: CUTLASS依赖
**问题**: CUTLASS是NVIDIA特有  
**解决**: 使用oneMKL或手动实现等效功能

### Q5: Grid大小限制
**问题**: CUDA 2D grid在SYCL中可能不是最优  
**解决**: 使用1D flattened grid最大化并行度

---

## 11. 参考资源

### 11.1 Intel文档
- [oneAPI DPC++ Programming Guide](https://spec.oneapi.io/oneapi-spec.pdf)
- [Intel GPU Optimization Guide](https://www.intel.com/content/www/us/en/developer/articles/guide/optimize-gpu-applications.html)
- [Xe Architecture Guide](https://www.intel.com/content/www/us/en/products/docs/accelerators/gaudi.html)

### 11.2 类似项目
- LCZero SYCL后端: 本项目已有经验积累
- IPEX (Intel Extension for PyTorch)
- oneDNN / oneMKL

### 11.3 测试数据
- [Hugging Face Diffusion Models](https://huggingface.co/models)
- Synthetic data generators (见test/目录)

---

## 12. 交付物清单

1. **源代码**:
   - [ ] SYCL头文件 (.hpp)
   - [ ] SYCL实现 (.dp.cpp)
   - [ ] Python绑定 (.cpp)

2. **测试代码**:
   - [ ] Python单元测试 (.py)
   - [ ] C++基准测试 (.cpp)
   - [ ] 性能对比脚本

3. **文档**:
   - [ ] API文档
   - [ ] 性能报告
   - [ ] 已知问题列表

4. **构建脚本**:
   - [ ] CMakeLists.txt
   - [ ] setup.py (Python包)
   - [ ] 编译说明 (README_BUILD.md)

---

## 总结

本prompt提供了一个完整的TurboDiffusion CUDA-to-SYCL转换指南，包含:
- 详细的技术映射表
- B60 GPU特定优化建议
- 完整的代码模板
- 测试和验收标准

实施时应遵循:
1. **先P0后P1P2**: 优先LayerNorm和RMSNorm
2. **保持数值精度**: 所有计算在FP32，I/O用BF16
3. **优化Sub-group**: 牢记B60是16-wide
4. **充分测试**: 功能+性能双重验证

祝转换顺利！
