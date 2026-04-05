"""
Setup script for TurboDiffusion SYCL Custom Operators

Builds PyTorch C++ extensions with SYCL support for Intel GPUs.

Usage:
    CC=icpx CXX=icpx python setup.py build_ext --inplace
    
Requirements:
    - PyTorch with XPU support
    - Intel oneAPI (icpx compiler)
    - SYCL runtime
"""

import os
import sys
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Force use of Intel compiler
os.environ['CC'] = 'icpx'
os.environ['CXX'] = 'icpx'

# Get PyTorch paths
import torch
torch_include = os.path.join(os.path.dirname(torch.__file__), 'include')

# Compiler settings for Intel SYCL
extra_compile_args = [
    '-fsycl',
    '-O3',
    '-std=c++17',
    '-fPIC',
    f'-I{torch_include}',
]

extra_link_args = [
    '-fsycl',
    '-shared',
]

# Source files for the extension
# Note: sycl_ops_main.cpp contains the PYBIND11_MODULE definition
sources = [
    'sycl_ops_main.cpp',         # Main module entry point
    'sycl_ops.cpp',              # Norm operations (no PYBIND11_MODULE)
    'flash_attention_sycl.cpp',  # Flash Attention v2 (no PYBIND11_MODULE)
    'sparse_attention_sycl.cpp', # Sparse Linear Attention (no PYBIND11_MODULE)
]

setup(
    name='turbodiffusion_sycl_ops',
    version='0.1.0',
    description='TurboDiffusion SYCL Custom Operators',
    author='TurboDiffusion Team',
    ext_modules=[
        CppExtension(
            name='turbodiffusion_sycl_ops',
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
