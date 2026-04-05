---
name: sycl-builder
description: Build and test SYCL kernels in Docker containers. Supports B60 local container with automatic directory sync, compilation, and result collection.
disable-model-invocation: true
allowed-tools: Bash
argument-hint: "[kernel_file] [test_command]"
---

# SYCL Builder

Build and test SYCL kernels in the B60 Docker container (lsv-container_2026_3).

## Configuration

- **Container:** lsv-container_2026_3
- **Base Image:** intel/llm-scaler-vllm:0.14.0-b8.1
- **Compiler:** icpx (Intel DPC++/C++ 2025.3)
- **Proxy:** https_proxy=http://child-prc.intel.com:912

### 目录映射
```
宿主机                                       → 容器内       用途
───────────────────────────────────────────────────────────────
/home/intel/tianfeng/opencode_bench          → /sandbox     代码目录 (持久)
/tmp/intel/                                  → /workspace   编译输出/临时目录
```

- **源码**放在 `/sandbox/`，编辑在宿主机同步可见
- **编译产物**输出到 `/workspace/build/`
- **不要**在 `/sandbox/` 内做 build

## Usage

```bash
/sycl-builder <kernel_file> [test_command]
```

### Build a single kernel
```bash
/sycl-builder kernel_bench/kernel_dataset/sycl/softmax_kernel.dp.cpp
```

### Build and run a test
```bash
/sycl-builder kernel_bench/tests/test_softmax_r1.cpp "./test_softmax"
```

## What This Skill Does

1. **Directory Check** - Verifies kernel file exists
2. **Auto-sync** - Files in `/sandbox/` are already visible to container
3. **Compile** - Two-step AOT compilation (see below)
4. **Execute** - Runs compiled binary if test_command provided
5. **Result Collection** - Captures output and error logs

## Environment Setup (每次 exec 进入后)

```bash
source /opt/intel/oneapi/compiler/latest/env/vars.sh
icpx --version
sycl-ls
```

> 建议写入 `~/.bashrc`: `echo 'source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null' >> ~/.bashrc`

## Compilation

### Quick compile (日常开发推荐)

```bash
docker exec lsv-container_2026_3 bash -c 'source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null && \
icpx -fsycl -fsycl-targets=spir64_gen \
  -Xs "-device bmg" \
  -std=c++17 -O2 \
  /sandbox/<source.cpp> -o /workspace/build/<output>'
```

### Full compile (参考 torch-xpu-ops 全部优化选项)

```bash
docker exec lsv-container_2026_3 bash -c 'source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null && \
icpx -fsycl \
  -fsycl-targets=spir64_gen \
  -fsycl-unnamed-lambda \
  -sycl-std=2020 \
  -std=c++17 -O2 \
  -fno-fast-math -fma -no-ftz \
  -Xs "-device bmg \
       -options -cl-intel-enable-auto-large-GRF-mode \
       -options -cl-fp32-correctly-rounded-divide-sqrt \
       -options -cl-intel-greater-than-4GB-buffer-required \
       -options -cl-poison-unsupported-fp64-kernels" \
  --offload-compress \
  /sandbox/<source.cpp> -o /workspace/build/<output>'
```

### Debug compile

```bash
docker exec lsv-container_2026_3 bash -c 'source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null && \
icpx -fsycl -fsycl-targets=spir64_gen \
  -Xs "-device bmg" \
  -std=c++17 -g -O0 \
  -Rno-debug-disables-optimization \
  /sandbox/<source.cpp> -o /workspace/build/<output>_dbg'
```

### JIT fallback (快速迭代，跳过 AOT)

```bash
docker exec lsv-container_2026_3 bash -c 'source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null && \
icpx -fsycl -std=c++17 -O2 \
  /sandbox/<source.cpp> -o /workspace/build/<output>'
```

### Multi-file compile

```bash
# Step compile
docker exec lsv-container_2026_3 bash -c 'source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null && \
icpx -fsycl -fsycl-targets=spir64_gen -std=c++17 -O2 -c /sandbox/a.cpp -o /workspace/build/a.o && \
icpx -fsycl -fsycl-targets=spir64_gen -std=c++17 -O2 -c /sandbox/b.cpp -o /workspace/build/b.o && \
# Link
icpx -fsycl -fsycl-targets=spir64_gen \
  -Xs "-device bmg -options -cl-intel-enable-auto-large-GRF-mode" \
  --offload-compress \
  /workspace/build/a.o /workspace/build/b.o -o /workspace/build/app'
```

## Compiler Flags Reference

### SYCL Basics
| Flag | Purpose |
|------|---------|
| `-fsycl` | Enable SYCL |
| `-fsycl-targets=spir64_gen` | Target Intel GPU (AOT) |
| `-sycl-std=2020` | SYCL 2020 standard |
| `-fsycl-unnamed-lambda` | Support anonymous lambda in kernels (**必须**，我们的 kernel 用 lambda 写法) |
| `-O2` | Optimization level |
| `-std=c++17` | C++ standard (`-std=c++20` 用 SYCL-TLA 时) |

### Numerical Precision
| Flag | Purpose |
|------|---------|
| `-fno-fast-math` | 禁用 fast-math 全家桶 |
| `-fma` | 启用 FMA 指令 (性能+精度) |
| `-fhonor-nans` | Preserve NaN |
| `-fhonor-infinities` | Preserve Infinity |
| `-fno-associative-math` | Disable reassociation (确定性 FP) |
| `-fno-approx-func` | 用标准数学函数 |
| `-no-ftz` | Don't flush denormals to zero |

### Device Backend (via `-Xs`)
| Flag | Purpose |
|------|---------|
| `-device bmg` | Target Battlemage GPU |
| `-cl-intel-enable-auto-large-GRF-mode` | Auto large register file (**推荐**) |
| `-cl-poison-unsupported-fp64-kernels` | FP64 不支持时报错 |
| `-cl-fp32-correctly-rounded-divide-sqrt` | FP32 精确除法/开方 |
| `-cl-intel-greater-than-4GB-buffer-required` | 支持 >4GB buffer |

### Link Optimization
| Flag | Purpose |
|------|---------|
| `--offload-compress` | 压缩设备代码减小体积 |
| `-fsycl-max-parallel-link-jobs=4` | 并行设备链接 |

## Quick Aliases (建议加入容器 ~/.bashrc)

```bash
alias bmg='f(){ source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null; icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg -options -cl-intel-enable-auto-large-GRF-mode" -std=c++17 -O2 "$@"; }; f'
# 用法: bmg /sandbox/test.cpp -o /workspace/build/test

alias bmg-dbg='f(){ source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null; icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -std=c++17 -g -O0 "$@"; }; f'

alias sycl-jit='f(){ source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null; icpx -fsycl -std=c++17 -O2 "$@"; }; f'
```

## Troubleshooting

### Container Not Running
```bash
docker start lsv-container_2026_3
```

### 编译报 `fp64 is not supported`
BMG 无原生 FP64。避免用 `double`，或加 `-fsycl-fp64-conv-emu`（性能差）。

### `PI_ERROR_DEVICE_NOT_FOUND`
确认 `--device=/dev/dri` 映射，以及 `sycl-ls` 能看到设备。

### AOT "Out of resources"
减少 work-group size 或确保 `-cl-intel-enable-auto-large-GRF-mode` 已设置。

### AOT 编译慢
只指定 `-device bmg`，不要编译多设备。日常开发可用 JIT 模式跳过 AOT。

## Implementation

Uses `b60_sycl_builder.py` for automated build workflow.
