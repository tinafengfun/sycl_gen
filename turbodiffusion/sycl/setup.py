#!/usr/bin/env python3
"""
Setup script for TurboDiffusion SYCL bindings

Usage:
    python setup.py build_ext --inplace
    python setup.py install
"""

from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import os

# Get the directory containing this file
dir_path = os.path.dirname(os.path.realpath(__file__))

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "turbodiffusion_sycl.turbodiffusion_sycl",
        sources=["bindings/sycl_kernels.cpp"],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            # Path to SYCL headers (in container)
            "/opt/intel/oneapi/compiler/latest/include",
            # Path to our source headers
            "src/norm",
            "src/quant", 
            "src/gemm",
        ],
        cxx_std=17,
        # Compiler flags for SYCL
        extra_compile_args=[
            "-fsycl",
            "-O3",
            "-std=c++17",
        ],
        # Linker flags for SYCL
        extra_link_args=[
            "-fsycl",
            "-lsycl",
            "-lOpenCL",
        ],
    ),
]

setup(
    name="turbodiffusion-sycl",
    version="0.1.0",
    author="TurboDiffusion-SYCL Team",
    description="Python bindings for TurboDiffusion SYCL kernels",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(where="bindings"),
    package_dir={"": "bindings"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "pybind11>=2.10",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
