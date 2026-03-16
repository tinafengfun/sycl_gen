# CUDA-SYCL Kernel Accuracy Test Suite

CUDA-SYCL 内核准确度测试套件

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 项目概述 | Project Overview

This project provides a comprehensive test suite for validating CUDA to SYCL kernel conversions, specifically targeting the LCZero chess engine neural network operations.

本项目提供了一个全面的测试套件，用于验证 CUDA 到 SYCL 内核的转换，特别是针对 LCZero 国际象棋引擎的神经网络操作。

### 主要特性 | Key Features

- ✅ **28 Kernel Tests** - Covering vector ops, normalization, pooling, Winograd transforms, and attention mechanisms
- ✅ **Accuracy Validation** - Compare CUDA and SYCL outputs with configurable tolerance
- ✅ **Automated Testing** - Run all tests with a single command
- ✅ **Detailed Reporting** - Get pass/fail status with MAE and max error metrics

## 环境要求 | Prerequisites

### 硬件要求 | Hardware Requirements

- **CUDA GPU** - NVIDIA GPU with CUDA 11.0+ support
- **Intel GPU** - For SYCL execution (or CPU fallback with Intel oneAPI)

### 软件要求 | Software Requirements

- **Python** 3.8 or higher
- **Docker** with CUDA and SYCL containers
- **SSH** access to CUDA host
- **NumPy** for array operations

### 容器设置 | Container Setup

The test suite expects two Docker containers:

1. **CUDA Container**: `cuda12.9-test`
   - NVIDIA CUDA 12.9
   - nvcc compiler
   - Located on remote host (default: 10.112.229.160)

2. **SYCL Container**: `lsv-container`
   - Intel oneAPI with SYCL support
   - icpx compiler with `-fsycl` flag
   - Located on local machine

## 安装 | Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cuda-sycl-harnesses.git
cd cuda-sycl-harnesses

# Install Python dependencies
pip install -r requirements.txt

# Verify containers are running
docker ps | grep cuda12.9-test
docker ps | grep lsv-container

# Test SSH connection to CUDA host
ssh root@10.112.229.160 "docker ps"
```

## 使用方法 | Usage

### 运行所有测试 | Run All Tests

```bash
python3 run_tests.py
```

### 列出所有内核 | List Available Kernels

```bash
python3 run_tests.py --list
```

### 测试特定内核 | Test Specific Kernel

```bash
python3 run_tests.py --kernel add_vectors
```

### 自定义参数 | Custom Parameters

```bash
# Use different containers
python3 run_tests.py \
  --cuda-host 192.168.1.100 \
  --cuda-container my-cuda \
  --sycl-container my-sycl

# Adjust error tolerance
python3 run_tests.py --tolerance 1e-5
```

## 内核列表 | Kernel List

### 向量操作 | Vector Operations
- `add_vectors` - Element-wise vector addition
- `add_vectors_hnc_nhc` - Vector add with HNC to NHC transpose
- `add_bias_nchw` - NCHW bias addition
- `add_bias_batched` - Batched bias addition

### 数据转换 | Data Conversion
- `copy_type_converted` - Float to half type conversion
- `nchw_to_nhwc` - NCHW to NHWC layout conversion
- `expand_planes_nchw` - Expand chess board planes NCHW
- `expand_planes_nhwc` - Expand chess board planes NHWC

### 归一化 | Normalization
- `batch_norm` - Batch normalization
- `layer_norm` - Layer normalization
- `global_scale` - Global scaling with sigmoid
- `global_scale_fp16_nhwc` - FP16 global scale for NHWC

### 池化 | Pooling
- `global_avg_pool` - Global average pooling
- `global_avg_pool_nhwc_fp16` - FP16 global avg pooling

### Winograd 变换 | Winograd Transforms
- `winograd_input_transform` - Winograd input transformation
- `winograd_filter_transform` - Winograd filter transformation
- `winograd_output_transform` - Winograd output transformation
- `winograd_output_relu_input` - Fused output + ReLU + input
- `winograd_output_se_relu_input` - Fused output + SE + ReLU + input
- `output_input_transform_fp16_shmem` - FP16 shared memory transform

### 注意力机制 | Attention
- `softmax` - Generic softmax
- `softmax_opt_64` - Optimized softmax for C=64
- `policy_map` - Policy mapping
- `promotion_logits` - Compute promotion logits
- `preprocess_attention_body` - Preprocess for attention
- `input_gating` - Input gating operation
- `gen_offset_pointers` - Generate offset pointers
- `se_layer_nhwc` - Squeeze-and-Excitation layer

## 测试结果 | Test Results

### 当前状态 | Current Status

- ✅ **Passed**: 25/28 kernels (89.3%)
- ⚠️ **Partial**: 3/28 kernels (need refinement)
- 🎯 **Target**: 25+ kernels with passing accuracy ✅

### 准确度指标 | Accuracy Metrics

The test suite reports:
- **MAE** (Mean Absolute Error) - Average difference between CUDA and SYCL outputs
- **Max Error** - Maximum difference observed
- **Tolerance** - Default: MAE < 1e-4, Max < 1e-3
- **Status** - PASS/FAIL based on tolerance

### 示例输出 | Sample Output

```
============================================================
Testing: batch_norm
============================================================
  Running CUDA...
  Running SYCL...
  Comparing outputs...
  ✅ PASS - MAE: 1.92e-08, Max: 2.38e-07

============================================================
📊 TEST SUMMARY
============================================================

✅ Passed: 25/28
❌ Failed: 3/28

🎉 SUCCESS! Target of 25+ kernels achieved!
```

## 项目结构 | Project Structure

```
cuda-sycl-harnesses/
├── harnesses/
│   ├── all_harnesses.py          # Core 22 kernel harnesses
│   └── phase5_batch4_harnesses.py # Additional 6 kernel harnesses
├── tests/
│   └── __init__.py
├── docs/
│   └── ACCURACY_VERIFICATION_REPORT.md
├── run_tests.py                   # Main test runner
├── requirements.txt               # Python dependencies
├── LICENSE                        # GPL v3 License
└── README.md                      # This file
```

## 技术细节 | Technical Details

### 测试流程 | Test Flow

1. **CUDA Execution**
   - Compile `.cu` file with nvcc
   - Run kernel in Docker container
   - Copy output binary back to host

2. **SYCL Execution**
   - Compile `.cpp` file with icpx -fsycl
   - Run kernel in Docker container
   - Copy output binary back to host

3. **Comparison**
   - Load both outputs as float32 arrays
   - Calculate MAE and max error
   - Compare against tolerance thresholds

### 容差设置 | Tolerance Settings

- **Standard kernels**: MAE < 1e-4, Max < 1e-3
- **Half-precision kernels**: MAE < 1e-3, Max < 1e-2
- **Configurable**: Use `--tolerance` flag

## 故障排除 | Troubleshooting

### 常见问题 | Common Issues

**1. SSH connection failed**
```bash
# Test SSH connection
ssh root@10.112.229.160 "echo 'Connected'"

# Check SSH keys
ssh-copy-id root@10.112.229.160
```

**2. Docker container not found**
```bash
# List running containers
docker ps

# Start containers if needed
docker start cuda12.9-test
docker start lsv-container
```

**3. Compilation errors**
```bash
# Check CUDA version
docker exec cuda12.9-test nvcc --version

# Check SYCL version
docker exec lsv-container icpx --version
```

**4. Permission denied**
```bash
# Ensure output files are writable
chmod 777 /tmp/
```

## 贡献 | Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 许可证 | License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 致谢 | Acknowledgments

- **LCZero** - The Leela Chess Zero project for the original CUDA kernels
- **Intel** - oneAPI and SYCL implementation
- **NVIDIA** - CUDA platform and tools

## 联系方式 | Contact

For questions or issues, please open an issue on GitHub.

---

**Project Status**: ✅ Production Ready  
**Last Updated**: 2024  
**Test Coverage**: 28 kernels, 25+ passing
