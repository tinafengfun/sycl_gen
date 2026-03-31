---
name: b60-sycl-builder
description: 在本地B60 docker容器(lsv-container)中进行SYCL kernel编译和测试。自动处理目录创建、代码同步、编译执行和结果回传。
license: MIT
compatibility: opencode
metadata:
  environment: b60
  container: lsv-container
  compiler: icpx
  workspace: /workspace
  type: build-and-test
  version: "1.1"
---

## What I do

在B60环境的docker容器内进行SYCL kernel的编译和测试，自动处理所有目录创建和同步。

### 核心功能

1. **自动目录初始化**
   - 本地自动创建 `results/b60/` 和 `scripts/b60/`
   - 容器内自动创建 `/workspace/kernel_dataset/sycl/`
   - 如 `/workspace` 不存在，自动创建

2. **健壮的文件同步**
   - 先创建目标目录结构
   - 使用 `docker cp` 复制文件
   - 验证文件是否正确同步
   - 显示同步的文件数量

3. **编译脚本生成**
   - 每个kernel生成独立脚本
   - 包含源文件存在性检查
   - 脚本位置: `scripts/b60/build_<kernel>_YYYYMMDD_HHMMSS.sh`
   - 自动生成目录: `build_sycl/`, `results/b60/`

4. **错误处理和验证**
   - 检查Docker是否安装
   - 检查容器是否运行
   - 验证源文件存在
   - 验证编译输出
   - 详细的错误日志

5. **结果回传**
   - 编译日志: `results/b60/compile_<kernel>_YYYYMMDD_HHMMSS.log`
   - 编译产物: `results/b60/build_sycl/<kernel>.o`
   - 状态文件: `.build_status.json`

## 目录结构 (自动创建)

```
本地 workspace/
├── results/
│   └── b60/
│       ├── build_sycl/         # 编译产物
│       ├── compile_*.log       # 编译日志
│       ├── batch_status_*.jsonl # 批量编译状态
│       └── summary_*.json      # 汇总报告
├── scripts/
│   └── b60/
│       └── build_*_*.sh        # 生成的编译脚本
└── .build_status.json          # 构建状态

容器内 /workspace/
├── kernel_dataset/sycl/        # 同步的代码
├── build_sycl/                 # 编译输出
└── results/b60/                # 容器内日志
```

## Commands

### 编译单个文件

**流程**:
1. 检查Docker和容器状态
2. 在容器内创建工作目录（如不存在）
3. 生成编译脚本
4. 同步代码到容器
5. 在容器内执行编译
6. 回传编译产物和日志
7. 更新状态文件

**生成的编译脚本示例**:
```bash
#!/bin/bash
# SYCL Build Script
set -e
set -x

# 验证源文件存在
if [ ! -f "/workspace/kernel_dataset/sycl/<kernel>.dp.cpp" ]; then
    echo "[ERROR] Source file not found"
    exit 1
fi

# 创建输出目录
mkdir -p /workspace/build_sycl
mkdir -p /workspace/results/b60

# 编译
icpx -fsycl -O2 -std=c++17 \
  -c "/workspace/kernel_dataset/sycl/<kernel>.dp.cpp" \
  -o "/workspace/build_sycl/<kernel>.o"

# 验证输出
if [ -f "/workspace/build_sycl/<kernel>.o" ]; then
    echo "Compilation successful!"
else
    echo "[WARNING] Output file not found!"
    exit 1
fi
```

### 批量编译

**执行流程**:
```bash
# 查找所有 .dp.cpp 文件
for kernel in kernel_dataset/sycl/*.dp.cpp; do
    # 编译单个文件
    compile_single $kernel
    
    # 记录状态到 batch_status_YYYYMMDD_HHMMSS.jsonl
    echo '{"kernel":"...","status":"...","time":"..."}' >> batch_status.jsonl
done

# 生成汇总报告 summary_YYYYMMDD_HHMMSS.json
```

## Error Handling

### 自动修复的问题

| 问题 | 自动处理 | 说明 |
|------|---------|------|
| /workspace 不存在 | ✅ | 自动创建 |
| kernel_dataset/sycl/ 不存在 | ✅ | 自动创建 |
| results/b60/ 不存在 | ✅ | 自动创建 |
| scripts/b60/ 不存在 | ✅ | 自动创建 |
| build_sycl/ 不存在 | ✅ | 自动创建 |

### 错误分类

| 错误类型 | 检测方式 | 调试建议 |
|---------|---------|---------|
| Docker未安装 | `which docker` | 安装Docker |
| 容器未运行 | `docker ps` | `docker start lsv-container` |
| 源文件不存在 | 本地检查 | 检查文件路径 |
| 同步失败 | docker cp 退出码 | 检查容器状态 |
| 编译失败 | icpx 退出码 | 查看编译日志 |
| 输出文件缺失 | 文件存在性检查 | 检查编译错误 |

## Output Format

### 编译日志 (`results/b60/compile_<kernel>_YYYYMMDD_HHMMSS.log`)

```
Exit code: 0
=== STDOUT ===
=== SYCL Compilation Start ===
Timestamp: 20260303_092517
Kernel: <kernel_name>
Source: /workspace/kernel_dataset/sycl/<kernel>.dp.cpp
...
=== STDERR ===
+ echo '=== SYCL Compilation Start ==='
...
```

### 状态文件 (`.build_status.json`)

```json
{
  "metadata": {
    "last_updated": "2026-03-03T09:25:18.097312"
  },
  "environments": {
    "b60": {
      "type": "local",
      "container": "lsv-container",
      "compiler": "icpx",
      "kernels": {
        "<kernel_name>": {
          "status": "success",
          "last_build": "20260303_092517",
          "source_file": "kernel_dataset/sycl/<kernel>.dp.cpp",
          "log_file": "results/b60/compile_<kernel>_20260303_092517.log",
          "script_file": "scripts/b60/build_<kernel>_20260303_092517.sh",
          "duration_seconds": 0.1
        }
      },
      "statistics": {
        "total": 1,
        "success": 1,
        "failed": 0
      }
    }
  }
}
```

## When to use me

1. **编译验证**: 验证SYCL代码语法正确性
2. **批量构建**: 编译整个sycl目录
3. **错误调试**: 获取详细编译错误信息
4. **CI集成**: 生成结构化构建报告

## Prerequisites

- Docker守护进程运行中
- 容器 `lsv-container` 已启动
- 本地磁盘空间充足（保留历史日志）

**环境检查命令**:
```bash
# 检查Docker
docker ps | grep lsv-container

# 检查编译器
docker exec lsv-container which icpx

# 自动检查
./tools/b60_sycl_builder.sh check
```

## 已知问题和解决

### 问题1: 容器内 /workspace 不存在
**解决**: 工具自动检测并创建

### 问题2: docker cp 同步失败
**解决**: 先创建目标目录，再复制文件，最后验证同步

### 问题3: 编译成功但输出文件缺失
**解决**: 脚本内添加输出文件存在性检查

## Optimization Guide (Based on LCZero Project Experience)

基于28个kernel的实际优化经验总结的优化策略和陷阱。

### ✅ Proven Optimization Patterns (验证有效的优化模式)

#### Pattern 1: SLM Parameter Caching (SLM参数缓存) - Impact: +30-60%

**适用场景**: 有per-channel参数的kernel (batch_norm, softmax, layer_norm)

**实现示例**:
```cpp
template <typename T>
void kernelWithSLM(T* output, const T* input, const float* params, int C, ...) {
  const int block_size = 256;
  const int slm_size = 256;  // Cache up to 256 channels
  
  queue.submit([&](sycl::handler& cgh) {
    // SLM allocation
    sycl::local_accessor<float, 1> slm_params(slm_size, cgh);
    
    cgh.parallel_for(
      sycl::nd_range<1>(grid_size * block_size, block_size),
      [=](sycl::nd_item<1> item) {
        const int tid = item.get_local_id(0);
        
        // Cooperative loading into SLM
        for (int i = tid; i < C && i < slm_size; i += block_size) {
          slm_params[i] = params[i];
        }
        item.barrier(sycl::access::fence_space::local_space);
        
        // Use cached parameters
        int ch = index % C;
        float param = (ch < slm_size) ? slm_params[ch] : params[ch];
        // ... computation
      });
  });
}
```

**实际结果**: batch_norm 70→109 GFLOPS (+56%)

#### Pattern 2: Sub-group Reduction (子群组归约) - Impact: +10-20%

**适用场景**: 归约操作 (mean, sum, max, softmax)

**实现**:
```cpp
inline float warpReduce(float x, sycl::sub_group sg) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += sycl::permute_group_by_xor(sg, x, offset);
  }
  return x;
}

// Usage in kernel
auto sg = item.get_sub_group();
float sum = warpReduce(local_sum, sg);
```

**优势**:
- 避免使用SLM进行warp内通信
- 比shared memory更快
- 在Intel B60/Xe2 GPUs上效果很好

#### Pattern 3: Vectorized Memory Access (向量化内存访问) - Conditional

**使用条件**: 只有当coalescing被保持时才使用

**正确示例** (保持coalescing):
```cpp
// Each thread loads consecutive elements
uint16_t pair = *reinterpret_cast<const uint16_t*>(&input[idx]);
sycl::half* vals = reinterpret_cast<sycl::half*>(&pair);
// vals[0] and vals[1] are consecutive in memory
```

**错误示例** (破坏coalescing):
```cpp
// DON'T: Thread i loads element i*2, not consecutive
int idx = tid * 2;  // Bad for coalescing!
```

### ❌ Anti-Patterns (无效或有害的优化)

#### Anti-Pattern 1: 改变内存访问模式

**为什么失败**:
- LCZero kernel已经有optimal coalescing
- 任何偏离都会降低带宽
- Thread-to-data mapping已经仔细调优

**实例**:
```cpp
// Original (optimal): 770 GFLOPS
// "Optimized" (worse): 658 GFLOPS
// Processing 8 elements per thread broke coalescing
```

#### Anti-Pattern 2: 不注意的向量化

**layer_norm的教训**:
- Grid configuration至关重要
- 3D grids很难正确复制
- 结果: 我们的版本33 GFLOPS vs 原版1094 GFLOPS

#### Anti-Pattern 3: 过多的线程发散

**问题**:
```cpp
// DON'T: Divergent branches within warp
if (threadIdx.x % 2 == 0) {
  // Path A
} else {
  // Path B
}
```

**解决方案**: 使用算术操作或predication
```cpp
// Better: Arithmetic selection
result = condition ? valueA : valueB;
```

### 🔧 CUDA→SYCL Conversion Reference

| CUDA Construct | SYCL Equivalent | Notes |
|----------------|-----------------|-------|
| `__shared__` | `sycl::local_accessor<T, N>` | Must declare in submit lambda |
| `__syncthreads()` | `item.barrier(sycl::access::fence_space::local_space)` | Work-group scope |
| `threadIdx.x` | `item.get_local_id(0)` | 0-indexed |
| `blockIdx.x` | `item.get_group(0)` | Grid position |
| `blockDim.x` | `item.get_local_range(0)` | Block size |
| `gridDim.x` | `item.get_group_range(0)` | Grid size |
| `warpReduce()` | `sycl::permute_group_by_xor` | Sub-group shuffle |
| `__shfl_xor()` | `sycl::permute_group_by_xor` | Same functionality |
| `cudaStream_t` | `sycl::queue` | Command queue |
| `cudaMemcpy` | `queue.memcpy` | Async, need `.wait()` |
| `<<<grid, block>>>` | `sycl::nd_range` | Parallel dispatch |
| `nullptr` | `(const T*)nullptr` | **Must cast!** |

### ⚠️ Common Pitfalls & Solutions

#### Pitfall 1: nullptr Type Mismatch
**Error**:
```cpp
kernel(..., nullptr, ...);  // ERROR: template deduction fails
```

**Solution**:
```cpp
kernel(..., (const sycl::half*)nullptr, ...);  // OK
```

#### Pitfall 2: Missing Barrier After SLM Write
**Problem**:
```cpp
slm_data[tid] = value;
// Missing barrier!
float other = slm_data[other_tid];  // Race condition!
```

**Solution**:
```cpp
slm_data[tid] = value;
item.barrier(sycl::access::fence_space::local_space);
float other = slm_data[other_tid];  // Safe
```

#### Pitfall 3: Grid Configuration Issues
**Problem**: 1D grids don't work well for reduction

**Solution**: Match original CUDA grid dimensions exactly
- 3D grids: `sycl::nd_range<3>`
- Careful thread-to-data mapping
- Validate with small test cases

#### Pitfall 4: Memory Alignment
**Requirement**: Vectorized loads need alignment

**Solution**:
```cpp
// Ensure C is multiple of vector size
if (C % 32 != 0) throw runtime_error("Alignment required");

// Use aligned types
sycl::uint4 vec = *reinterpret_cast<const sycl::uint4*>(ptr);
```

### 📊 Kernel-Specific Optimization Strategies

#### Type A: Parameter-Heavy Kernels (batch_norm, softmax, layer_norm)
**Strategy**: SLM caching
**Expected Gain**: +30-60%
**Key**: Cache per-channel parameters

#### Type B: Memory-Bound Element-wise (expand_planes, copy)
**Strategy**: Keep original, already optimal
**Expected Gain**: 0%
**Key**: Don't break coalescing

#### Type C: Reduction Kernels (global_avg_pool, softmax)
**Strategy**: Sub-group reduction + SLM
**Expected Gain**: +20-40%
**Key**: Minimize global memory round-trips

#### Type D: Matrix Operations (Winograd, Attention)
**Strategy**: XMX (joint_matrix)
**Expected Gain**: +50-200%
**Key**: Use Intel Xe Matrix Extensions

### 🎯 Performance Targets (Intel B60)

| Kernel Type | Baseline | Optimized | Method |
|-------------|----------|-----------|---------|
| batch_norm | 70 GFLOPS | 109 GFLOPS | SLM caching |
| layer_norm | 1094 GFLOPS | 1094 GFLOPS | Already optimal |
| expand_planes | 770 GFLOPS | 770 GFLOPS | Already optimal |
| softmax | 44 GFLOPS | 80 GFLOPS (est) | SLM + sub-group |
| Winograd | 984 GFLOPS | 1100 GFLOPS (est) | XMX |

### Quick Reference: Optimization Checklist

Before optimizing a kernel, check:

- [ ] Is memory access already coalesced? (Don't change if yes)
- [ ] Are there per-channel/per-element parameters? (SLM opportunity)
- [ ] Are there reduction operations? (Sub-group opportunity)
- [ ] Are there matrix multiplications? (XMX opportunity)
- [ ] Is the grid layout complex? (Match original exactly)
- [ ] Did I add proper barriers after SLM writes?
- [ ] Did I cast nullptr to correct type?
- [ ] Did I test with multiple sizes?

## 改进日志

- **v1.1** (2026-03-03): 改进版
  - 自动创建所有缺失目录
  - 改进文件同步逻辑（先创建目录再复制）
  - 添加文件存在性验证
  - 改进错误处理和日志记录
  - 添加路径安全检查
  - 改进状态文件更新（异常处理）

- **v2.0** (2026-03-30): Complete 5-Round Optimization Project
  - **Round 1**: Baseline testing - 28/28 kernels (100% coverage)
  - **Round 2**: Memory optimization - demonstrated patterns
  - **Round 3**: SLM optimization - 2 kernels improved (+56%, +179%)
  - **Round 4**: XMX optimization - documented (not implemented)
  - **Round 5**: Final polish - comprehensive documentation
  - **Results**: batch_norm +56% (70→109 GFLOPS), softmax +179% (26→73 GFLOPS)
  - **Key Finding**: SLM caching provides real benefits for parameter-heavy kernels
  - **Peak Performance**: 1094 GFLOPS (layer_norm)
  - **Best Optimization**: SLM Parameter Caching Pattern
