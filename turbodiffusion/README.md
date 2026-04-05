# TurboDiffusion SYCL

Video generation acceleration framework, ported from CUDA to SYCL for Intel GPUs (BMG B60 / Battlemage G21).

## Structure

```
turbodiffusion/
├── original/     # Original CUDA implementation (reference)
│   └── turbodiffusion/   # Python package with CUDA ops
│       ├── ops/           # CUDA kernels (GEMM, norm, quant)
│       ├── inference/     # Wan2.1/Wan2.2 inference scripts
│       └── rcm/           # rCM timestep distillation
│
├── sycl/         # SYCL port for Intel GPU
│   ├── operators/         # SYCL kernel implementations
│   ├── src/               # Optimized SYCL headers (GEMM, norm, quant)
│   ├── bindings/          # Python bindings via CMake
│   ├── hooks/             # Zero-intrusive PyTorch integration
│   ├── tests/             # Unit & integration tests (phase1-3)
│   └── scripts/           # Build, inference, and benchmark scripts
│
└── docs/         # Migration documentation
    └── conversion_prompt.md  # CUDA-to-SYCL conversion guide
```

## Quick Start

See `sycl/README_SYCL.md` for build instructions.

## Key Components

- **Operators**: LayerNorm, RMSNorm, Quantization, GEMM in SYCL
- **Flash Attention**: XMX-optimized flash attention for Intel GPU
- **Hooks**: Zero-intrusive dispatcher replaces CUDA ops at runtime
- **Models**: Supports Wan2.1-T2V-1.3B and Wan2.2-I2V
