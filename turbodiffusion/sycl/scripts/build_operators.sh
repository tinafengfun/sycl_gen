#!/bin/bash
# Build script for TurboDiffusion SYCL operators

set -e

# Activate conda environment
source /home/intel/miniforge3/etc/profile.d/conda.sh
conda activate xpu

# Source Intel oneAPI (required for icpx compiler)
source /opt/intel/oneapi/setvars.sh --force

# Set compiler
export CC=icpx
export CXX=icpx

# Navigate to operators directory
cd /home/intel/tianfeng/opencode_bench/turbodiffusion-sycl/operators

# Clean previous builds
rm -rf build dist *.egg-info *.so

# Build the extension
echo "Building TurboDiffusion SYCL operators..."
python setup.py build_ext --inplace

echo "Build complete!"
ls -la *.so 2>/dev/null || echo "No .so files found"
