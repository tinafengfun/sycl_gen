---
name: test-build-in-docker
description: Build and test SYCL kernels in the lsv-container Docker environment. Use when building SYCL tests, running kernel tests, or verifying implementations on Intel GPU.
---

# SYCL Docker Build and Test Skill

This skill provides commands for building and testing SYCL kernels in the Intel oneAPI Docker container (lsv-container).

## Prerequisites

- Docker container `lsv-container` must be running
- Host directory mounted at `/intel/tianfeng/lc0/` in container
- Intel GPU available in container

## Environment Variables

```bash
SYCL_DIR="/intel/tianfeng/lc0/src/neural/backends/sycl"
BUILD_DIR="/intel/tianfeng/lc0/builddir"
```

## Common Tasks

### Build Specific Test

Build a single SYCL kernel test in Docker:

```bash
docker exec lsv-container bash -c "cd $SYCL_DIR && ./build_test.sh -t test_softmax --clean 2>&1"
```

### Run Specific Test

Execute a built test in Docker:

```bash
docker exec lsv-container bash -c "cd $SYCL_DIR/build && ./test_softmax 2>&1"
```

### Build All Tests

Build all SYCL kernel tests:

```bash
docker exec lsv-container bash -c "cd $SYCL_DIR && ./build_test.sh --clean 2>&1"
```

### Build lc0 Binary

Build the main lc0 binary with SYCL backend:

```bash
docker exec lsv-container bash -c "cd /intel/tianfeng/lc0 && ninja -C builddir/ lc0 2>&1"
```

### Test lc0 with Network

Run a quick test of the SYCL backend:

```bash
docker exec lsv-container bash -c "cd /intel/tianfeng/lc0 && echo 'go nodes 1' | ./builddir/lc0 --backend=sycl --weights=./t1-256x10-distilled-swa-2432500.pb.gz 2>&1"
```

## Workflow

1. **Verify container is running:**
   ```bash
   docker ps --filter "name=lsv-container"
   ```

2. **Build the code:**
   - For kernel tests: Use `build_test.sh` in the sycl directory
   - For lc0 binary: Use `ninja -C builddir/ lc0`

3. **Run tests:**
   - Kernel tests: Execute from `build/` directory
   - Integration tests: Run lc0 with a network file

## Available Tests

| Test Name | Description |
|-----------|-------------|
| test_softmax | Softmax normalization |
| test_expand_planes | Input plane expansion |
| test_filter_transform | Winograd filter transform |
| test_winograd | Winograd transforms |
| test_layer_norm | Layer normalization |
| test_global_avg_pool | Global average pooling |
| test_global_scale | Global scale (sigmoid gating) |
| test_add_vectors | Element-wise vector addition |
| test_add_bias_nchw | Bias add for NCHW |
| test_accuracy | Accuracy validation |

## Troubleshooting

**Container not running:**
```bash
docker start lsv-container
```

**Permission denied:**
Ensure the container is running and the user has proper permissions.

**Build errors:**
Check that the build directory is properly configured with meson.
