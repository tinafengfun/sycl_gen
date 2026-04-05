# GPU Kernel Benchmark & TurboDiffusion SYCL Suite

Intel GPU kernel optimization benchmarks (SYCL) and TurboDiffusion video generation acceleration — all validated on real Intel Battlemage hardware.

## Repository Structure

```
opencode_bench/
├── kernel_bench/              # Project 1: CUDA-to-SYCL kernel benchmark
│   ├── kernel_dataset/        # 30 CUDA/SYCL kernels from LCZero
│   ├── tests/                 # Test harnesses (cpp + accuracy framework)
│   ├── tools/                 # Python conversion/validation tools
│   ├── prompts/               # Phase-based LLM agent prompts
│   ├── benchmarks/            # Performance results & charts
│   ├── cuda-sycl-converter/   # Automated accuracy test suite
│   ├── performance_optimization/  # Systematic optimization framework
│   ├── config/                # Agent pipeline configuration
│   ├── docs/                  # Optimization guides & reports
│   └── archive/               # Historical reports & one-off scripts
│
├── turbodiffusion/            # Project 2: TurboDiffusion SYCL port
│   ├── original/              # Original CUDA implementation
│   ├── sycl/                  # SYCL port with Intel GPU support
│   └── docs/                  # Migration docs
│
├── skills/                    # Consolidated Claude Code skills
│   ├── docker-executor/       # Unified Docker exec (local + remote)
│   ├── sycl-builder/          # SYCL kernel build in B60 container
│   ├── intel-gpu-optimizer/   # Intel GPU optimization guide (merged)
│   └── winograd-sycl/         # Winograd convolution specialization
│
├── sycle-tla/                 # SYCL Templates for Linear Algebra (3rd-party)
├── BMG_B60_SPE.md             # BMG B60 hardware specifications
├── XMX.md                     # XMX optimization reference
└── compile.config             # SYCL compiler flags
```

## Project 1: Kernel Benchmark

30 CUDA kernels from [LCZero](https://github.com/LeelaChessZero/lc0) chess engine, converted to SYCL and optimized for Intel Battlemage GPUs.

### Key Findings

| Technique | Effect | Best Case |
|-----------|--------|-----------|
| Loop Unrolling (`#pragma unroll`) | +50-446% | fused_winograd_se (4.47x) |
| 3D Work-Group Topology | +80% | winograd spatial kernels |
| Work-Group Tuning (64-512) | +10-40% | kernel-dependent |
| Vectorization | -4% to +10% | global_avg_pool |
| Multi-thread barriers | **Avoid** | 300x degradation |

### Quick Start

```bash
# Compile a SYCL kernel test
icpx -fsycl -fsycl-targets=spir64_gen \
  -Xsycl-target-backend "-device bmg -options -ze-opt-large-register-file" \
  -O2 -std=c++17 kernel_bench/tests/test_softmax_r1.cpp -o test_softmax

# Run
./test_softmax
```

### Documentation

- [GPU Optimization Guide](kernel_bench/docs/GPU_OPTIMIZATION_GUIDE.md)
- [Kernel Dataset README](kernel_bench/kernel_dataset/README.md)
- [Accuracy Test Suite](kernel_bench/cuda-sycl-converter/README.md)

## Project 2: TurboDiffusion

Video generation acceleration framework ported from CUDA to SYCL for Intel GPUs.

- [Original Implementation](turbodiffusion/original/)
- [SYCL Port](turbodiffusion/sycl/)
- [Conversion Guide](turbodiffusion/docs/conversion_prompt.md)

## Hardware Targets

| GPU | Device ID | Sub-group Size | SLM | Memory |
|-----|-----------|----------------|-----|--------|
| BMG B60 | Xe2 | 16 | 256 KB/XeCore | ~500 GB/s HBM2e |
| Battlemage G21 | 0xe211 | 16 | 128 KB | GDDR |

## Shared Configuration

- **Compiler:** Intel oneAPI 2025.1 (`icpx`)
- **SYCL Flags:** See `compile.config`
- **LLM Endpoint:** See `model.conf`

## License

- Kernel dataset: GNU GPL v3 (from LCZero)
- TurboDiffusion: See respective directories
- Optimization tools: MIT
