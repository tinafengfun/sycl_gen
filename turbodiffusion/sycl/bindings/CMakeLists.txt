# CMakeLists.txt for TurboDiffusion SYCL Python Bindings
#
# This CMake configuration builds Python bindings for SYCL kernels
# using pybind11. It compiles the C++ SYCL kernels into a shared
# library that can be imported from Python.
#
# Usage:
#   mkdir build && cd build
#   cmake ..
#   make -j4
#   make install

cmake_minimum_required(VERSION 3.16)
project(turbodiffusion_sycl VERSION 0.1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)

# Find pybind11
# Try to find pybind11 installed in the system or conda environment
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(PYBIND11_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
    find_package(pybind11 REQUIRED)
    message(STATUS "Found pybind11 at: ${PYBIND11_CMAKE_DIR}")
else()
    # Fallback: try standard find
    find_package(pybind11 REQUIRED)
endif()

# Intel oneAPI SYCL configuration
# Check if we're in the B60 container
if(EXISTS "/opt/intel/oneapi/setvars.sh")
    message(STATUS "Intel oneAPI environment detected")
    # SYCL flags will be set by the compiler wrapper (icpx)
    set(SYCL_COMPILER "icpx")
else()
    message(WARNING "Intel oneAPI not found. Make sure to source setvars.sh")
    set(SYCL_COMPILER "icpx")
endif()

# Compiler settings
set(CMAKE_CXX_COMPILER ${SYCL_COMPILER})

# SYCL compilation flags (as a list, not a string)
set(SYCL_FLAGS "-fsycl" "-O3" "-std=c++17")

# Add specific flags for Intel GPU (BMG/B60)
if(DEFINED ENV{SYCL_DEVICE_FILTER})
    message(STATUS "SYCL_DEVICE_FILTER: $ENV{SYCL_DEVICE_FILTER}")
endif()

# Source files
set(SOURCES
    sycl_kernels.cpp
)

# Include directories
include_directories(
    ${Python_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../src/norm
    ${CMAKE_SOURCE_DIR}/../src/quant
    ${CMAKE_SOURCE_DIR}/../src/gemm
)

# Create the pybind11 module
pybind11_add_module(turbodiffusion_sycl ${SOURCES})

# Set compiler flags for SYCL
target_compile_options(turbodiffusion_sycl PRIVATE ${SYCL_FLAGS})

# Set link flags for SYCL
target_link_options(turbodiffusion_sycl PRIVATE ${SYCL_FLAGS})

# Link with SYCL runtime
target_link_libraries(turbodiffusion_sycl PRIVATE
    sycl
    OpenCL
)

# Set properties
set_target_properties(turbodiffusion_sycl PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    POSITION_INDEPENDENT_CODE ON
)

# Installation
set(PYTHON_PACKAGE_DIR ${CMAKE_SOURCE_DIR}/turbodiffusion_sycl)

install(TARGETS turbodiffusion_sycl
    LIBRARY DESTINATION ${PYTHON_PACKAGE_DIR}
    RUNTIME DESTINATION ${PYTHON_PACKAGE_DIR}
)

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  CMake version: ${CMAKE_VERSION}")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  Python: ${Python_EXECUTABLE}")
message(STATUS "  pybind11: ${pybind11_VERSION}")
message(STATUS "  Install directory: ${PYTHON_PACKAGE_DIR}")
message(STATUS "  SYCL flags: ${SYCL_FLAGS}")
message(STATUS "")

# Custom target for clean rebuild
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E echo "Build directory cleaned"
)

# Custom target to run tests
add_custom_target(test
    COMMAND ${Python_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/../tests/bindings -v
    DEPENDS turbodiffusion_sycl
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
)
