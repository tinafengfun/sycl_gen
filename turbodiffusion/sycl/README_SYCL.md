# TurboDiffusion SYCL Implementation

## Overview

This directory contains the SYCL implementation of TurboDiffusion kernels for Intel GPUs. The implementation provides optimized attention mechanisms (Flash Attention and Sparse Attention) using SYCL for running the Wan2.1 video generation model on Intel XPU devices.

## Quick Start

### Prerequisites

- Intel oneAPI Base Toolkit (with SYCL support)
- PyTorch with Intel Extension for PyTorch (IPEX)
- Python 3.8+

### Build

```bash
cd operators
export CC=icpx CXX=icpx
python setup.py build_ext --inplace
```

### Run Inference

```bash
python scripts/infer_sycl.py \
  --model_path /intel/hf_models/WAN2.1/checkpoints/TurboWan2.1-T2V-1.3B-480P.pth \
  --prompt "A cat playing with a ball" \
  --attention sparse \
  --output output.mp4
```

### Benchmark

```bash
python scripts/benchmark_sycl.py
```

## Architecture

### Directory Structure

```
turbodiffusion-sycl/
├── configs/              # Configuration files
│   └── wan2.1_1.3B_sycl.yaml
├── operators/            # SYCL kernel implementations
│   ├── flash_attention.cpp
│   ├── sparse_attention.cpp
│   └── setup.py
├── scripts/              # Utility scripts
│   ├── infer_sycl.py     # End-to-end inference
│   ├── benchmark_sycl.py # Performance benchmarking
│   └── compare_with_cuda.py # Validation
├── bindings/             # Python bindings
└── results/              # Benchmark outputs
```

### Key Components

1. **Flash Attention SYCL**: Memory-efficient attention mechanism optimized for Intel XPU
2. **Sparse Attention SYCL**: Top-k sparse attention for faster inference with minimal quality loss
3. **LayerNorm/RMSNorm**: Optimized normalization layers using SYCL

## Configuration

The `configs/wan2.1_1.3B_sycl.yaml` file controls model and inference parameters:

```yaml
model:
  name: "Wan2.1-1.3B"
  dim: 1536
  num_heads: 12
  num_layers: 30

attention:
  type: "sparse"  # or "flash"
  topk: 0.2
  
optimization:
  use_bf16: true
  device: "xpu"
```

## Scripts Usage

### Inference Script

```bash
python scripts/infer_sycl.py \
  --model_path <path_to_checkpoint> \
  --prompt "Your text prompt here" \
  --attention [flash|sparse] \
  --topk 0.2 \
  --device xpu \
  --output video.mp4 \
  --num_frames 81 \
  --resolution 480p
```

### Benchmark Script

Measures throughput and memory usage:

```bash
python scripts/benchmark_sycl.py
```

Results are saved to `results/benchmark_sycl.json`.

### Comparison Script

Validates SYCL outputs against reference PyTorch implementation:

```bash
python scripts/compare_with_cuda.py
```

Generates a report in `results/comparison_report.md`.

## Performance Tips

1. **Use BF16**: Enable `use_bf16: true` in config for 2x memory savings
2. **Sparse Attention**: Use `topk: 0.1-0.2` for 30-50% speedup with minimal quality impact
3. **Batch Size**: Larger batch sizes better utilize XPU compute units

## Troubleshooting

### XPU Not Available

If `torch.xpu.is_available()` returns False:

```bash
# Source Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Verify IPEX installation
python -c "import intel_extension_for_pytorch as ipex; print(ipex.__version__)"
```

### Memory Issues

For large video generation:

- Reduce `--num_frames` (e.g., 41 instead of 81)
- Lower resolution to 480p
- Use sparse attention with lower topk ratio

## Development

### Adding New Kernels

1. Implement kernel in `operators/` using SYCL
2. Add Python binding in `bindings/`
3. Create wrapper in `turbodiffusion_sycl/__init__.py`
4. Update scripts to use new functionality

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Validate against reference
python scripts/compare_with_cuda.py
```

## License

This implementation follows the same license as the original TurboDiffusion project.

## References

- [TurboDiffusion](https://github.com/original/turbodiffusion)
- [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
- [Wan2.1 Paper](https://arxiv.org/abs/xxxx.xxxxx)
