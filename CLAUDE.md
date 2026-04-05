# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two main projects sharing Intel GPU infrastructure:

1. **kernel_bench/** — 30 CUDA kernels from LCZero chess engine converted to SYCL, with accuracy validation and performance optimization
2. **turbodiffusion/** — Video generation acceleration framework (CUDA → SYCL port)

Supporting: **sycle-tla/** (SYCL Templates for Linear Algebra, third-party library)

## Build & Compilation

```bash
# SYCL (Intel oneAPI) — quick compile (日常开发)
docker exec lsv-container_2026_3 bash -c 'source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null && \
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" \
  -fsycl-unnamed-lambda -std=c++17 -O2 \
  /sandbox/<source.cpp> -o /workspace/build/<output>'

# SYCL — full optimization compile (参考 torch-xpu-ops)
docker exec lsv-container_2026_3 bash -c 'source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null && \
icpx -fsycl -fsycl-targets=spir64_gen \
  -fsycl-unnamed-lambda -sycl-std=2020 -std=c++17 -O2 \
  -fno-fast-math -fma -no-ftz \
  -Xs "-device bmg \
       -options -cl-intel-enable-auto-large-GRF-mode \
       -options -cl-fp32-correctly-rounded-divide-sqrt \
       -options -cl-intel-greater-than-4GB-buffer-required \
       -options -cl-poison-unsupported-fp64-kernels" \
  --offload-compress \
  /sandbox/<source.cpp> -o /workspace/build/<output>'

# SYCL — JIT only (快速迭代，跳过 AOT)
docker exec lsv-container_2026_3 bash -c 'source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null && \
icpx -fsycl -std=c++17 -O2 /sandbox/<source.cpp> -o /workspace/build/<output>'

# CUDA
nvcc -O2 -arch=sm_70 <source.cu> -o <output>
```

Container: `lsv-container_2026_3` | Path: `/home/intel/tianfeng/opencode_bench` → `/sandbox`, `/tmp/intel/` → `/workspace`
Full flags reference: `compile_refer/claude_skill_bmg_b60_docker_build.md`

## Testing

Tests are standalone programs compiled individually — no unified test runner.

```bash
# Single kernel test
icpx -fsycl -O2 -std=c++17 kernel_bench/tests/test_softmax_r1.cpp -o test && ./test

# Accuracy testing (Python)
python3 kernel_bench/tools/accuracy_tester.py <kernel_id>
```

Accuracy tolerances: FP32 abs_tol=1e-5, rel_tol=1e-4; FP16 abs_tol=1e-3, rel_tol=1e-2.

## Key Directories

### kernel_bench/
- `kernel_dataset/` — Source CUDA kernels (`cuda/`) and SYCL conversions (`sycl/`), indexed by `index.json`
- `tests/` — Standalone SYCL test harnesses, accuracy framework, CUDA harnesses
- `tools/` — Python tools for conversion, accuracy testing, batch processing
- `prompts/` — Phase-based LLM agent prompts (accuracy, reporting, conversion, optimization)
- `benchmarks/` — CSV results and PNG charts from real GPU runs
- `cuda-sycl-converter/` — Dedicated accuracy test suite with Docker support
- `performance_optimization/` — Systematic 5-phase optimization framework
- `config/` — Agent pipeline configuration
- `docs/` — Optimization guides, analysis reports
- `archive/` — Historical reports, one-off scripts, old optimization rounds

### turbodiffusion/
- `original/` — Original CUDA implementation
- `sycl/` — SYCL port with Intel GPU operators, hooks, and tests
- `docs/` — Migration documentation

### skills/
- `docker-executor/` — Unified Docker execution (local Intel + remote NVIDIA)
- `sycl-builder/` — SYCL kernel build in B60 container
- `intel-gpu-optimizer/` — Comprehensive Intel GPU optimization guide (BMG B60 + G21 + XMX)
- `winograd-sycl/` — Winograd convolution specialization

## CUDA-to-SYCL Mapping

| CUDA | SYCL |
|------|------|
| `__global__` | Lambda in `queue.parallel_for()` |
| `__shared__` | `sycl::local_accessor` |
| `__syncthreads()` | `item.barrier()` |
| `threadIdx.x` | `item.get_local_id(0)` |
| `blockIdx.x` | `item.get_group(0)` |
| `blockDim.x` | `item.get_local_range(0)` |
| `__shfl_xor_sync` | `sycl::shift_group_left` via `sub_group` |
| `atomicAdd` | `sycl::atomic_ref::fetch_add` |

## Code Conventions

- **Naming**: CUDA `*_kernel.cu`, SYCL `*_kernel.dp.cpp`
- **Namespace**: `lczero::sycldnn_backend`
- **Constants**: `kCamelCase`; Functions: `snake_case`; Enums: `ALL_CAPS`
- **Style**: 2-space indent, K&R braces, 80-char lines
- **Required in SYCL files**: GPL header, `#include <sycl/sycl.hpp>`, standard constants (`kNumOutputPolicy=1858`, etc.)

## Optimization Quick Reference

1. `#pragma unroll` on nested loops — up to 4.5x speedup
2. Work-group size is kernel-dependent (test 64/128/256/512)
3. 3D topology for spatial kernels (~80% gain)
4. Avoid `item.barrier()` in loops — use single-thread-per-output
5. XMX (joint_matrix) for matrix ops >= 256x256
