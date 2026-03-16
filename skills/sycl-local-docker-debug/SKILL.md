---
name: sycl-local-docker-debug
description: Debug and test SYCL kernels in local Docker container (lsv-container) using local_docker_exec.sh. Use for iterative kernel development, build verification, and GPU testing on Intel GPU.
disable-model-invocation: true
allowed-tools: Bash
argument-hint: "[command] [test_name]"
---

# SYCL Local Docker Debug Skill

## Overview

This skill provides a standardized workflow for debugging and testing SYCL kernels in the local `lsv-container` Docker environment using the `local_docker_exec.sh` script. The container has Intel oneAPI compilers and GPU drivers pre-configured.

**Key Benefits:**
- Automatic directory sync to container
- Captured output for analysis
- Consistent test environment
- No manual Docker commands needed

## Prerequisites

- Docker container `lsv-container` is running (or can be auto-started)
- `local_docker_exec.sh` script exists in project root
- Host directory structure mounted at `/intel/tianfeng/` in container
- Intel GPU available in container

## Quick Commands

### Check Environment
```bash
# Verify container status
./local_docker_exec.sh . "docker ps" container_check.log

# Check GPU in container
./local_docker_exec.sh . "sycl-ls" sycl_ls.log

# Check compiler version
./local_docker_exec.sh . "icpx --version" compiler_version.log
```

### Build and Test
```bash
# Build specific test
./local_docker_exec.sh src/neural/backends/sycl \
    "./build_test.sh -t test_softmax --clean" \
    build_softmax.log

# Run specific test
./local_docker_exec.sh src/neural/backends/sycl \
    "cd build && ./test_softmax" \
    run_softmax.log

# Build all tests
./local_docker_exec.sh src/neural/backends/sycl \
    "./build_test.sh --clean" \
    build_all.log
```

## Complete Debug Workflow

### 1. Verify Environment

```bash
# Check container is running
./local_docker_exec.sh . "echo 'Container ready'" env_check.log

# Check GPU availability
./local_docker_exec.sh . "sycl-ls" sycl_ls.log

# Check compiler
./local_docker_exec.sh . "which icpx && icpx --version" compiler_check.log
```

### 2. Iterative Kernel Development

```bash
# SYCL backend directory
SYCL_DIR="src/neural/backends/sycl"

# Step 1: Edit kernel code locally
# (Edit sycl_kernels.cc with your changes)

# Step 2: Build specific test
./local_docker_exec.sh $SYCL_DIR \
    "./build_test.sh -t test_your_kernel --clean" \
    build_iteration_1.log

# Step 3: Run test and capture output
./local_docker_exec.sh $SYCL_DIR \
    "cd build && ./test_your_kernel" \
    test_iteration_1.log

# Step 4: Check results
cat test_iteration_1.log | grep -E "(PASSED|FAILED|Error|error)"
```

### 3. Debug Build Failures

```bash
# Build with full error output
./local_docker_exec.sh src/neural/backends/sycl \
    "./build_test.sh -t test_softmax --clean 2>&1" \
    build_errors.log

# Analyze errors
grep -E "(error:|warning:|Error)" build_errors.log | head -20
```

### 4. Run All Tests

```bash
# Build all tests
./local_docker_exec.sh src/neural/backends/sycl \
    "./build_test.sh --clean" \
    build_all.log

# Run all tests sequentially
for test in test_softmax test_expand_planes test_filter_transform test_winograd test_layer_norm; do
    echo "Running $test..."
    ./local_docker_exec.sh src/neural/backends/sycl \
        "cd build && ./$test" \
        "${test}_result.log"
done

# Summary of results
for log in *_result.log; do
    echo "$log: $(grep -E 'PASSED|FAILED' $log | tail -1)"
done
```

## Available Tests

| Test Name | Description | Kernel Tested | Typical Runtime |
|-----------|-------------|---------------|-----------------|
| `test_softmax` | Softmax normalization | Softmax | ~2s |
| `test_expand_planes` | Input plane expansion | expandPlanes_NHWC/NCHW | ~2s |
| `test_filter_transform` | Winograd filter transform | FilterTransform | ~3s |
| `test_winograd` | Winograd transforms | InputTransform, OutputTransform | ~5s |
| `test_layer_norm` | Layer normalization | LayerNorm | ~3s |
| `test_global_avg_pool` | Global average pooling | globalAvgPool | ~2s |
| `test_global_scale` | Global scale operation | globalScale | ~3s |

## Common Debug Scenarios

### Scenario 1: Build Failure Investigation

```bash
# Full clean build with verbose output
./local_docker_exec.sh src/neural/backends/sycl \
    "bash -x ./build_test.sh -t test_softmax --clean 2>&1" \
    verbose_build.log

# Check for missing includes
grep -n "file not found" verbose_build.log

# Check for linking errors
grep -n "undefined reference" verbose_build.log
```

### Scenario 2: Runtime Error Analysis

```bash
# Run with GDB (if available in container)
./local_docker_exec.sh src/neural/backends/sycl \
    "cd build && gdb -batch -ex run -ex bt ./test_softmax 2>&1" \
    gdb_output.log

# Check for memory errors
./local_docker_exec.sh src/neural/backends/sycl \
    "cd build && valgrind --error-exitcode=1 ./test_softmax 2>&1" \
    valgrind.log || echo "Memory errors detected"
```

### Scenario 3: Numerical Accuracy Debugging

```bash
# Run with detailed output
./local_docker_exec.sh src/neural/backends/sycl \
    "cd build && ./test_softmax 2>&1" \
    accuracy_debug.log

# Analyze error patterns
grep -E "(Max error|Mean error)" accuracy_debug.log
```

### Scenario 4: Performance Profiling

```bash
# Build with profiling flags first
./local_docker_exec.sh src/neural/backends/sycl \
    "cd build && icpx -O2 -pg -DPROFILE ..." \
    profile_build.log

# Run test and generate profile
./local_docker_exec.sh src/neural/backends/sycl \
    "cd build && ./test_softmax && gprof ./test_softmax gmon.out > profile.txt 2>&1" \
    profile_run.log
```

## Python Script Alternative

For advanced features (JSON results, automatic parsing, timeout):

```bash
# Run with output parsing
python3 local_docker_exec.py src/neural/backends/sycl \
    "./build_test.sh -t test_softmax --clean" \
    --output build.log \
    --parse-output \
    --json-result build_analysis.json

# Check JSON results
cat build_analysis.json | python3 -m json.tool
```

## Interactive Debug Session

```bash
# Start interactive bash in container for manual debugging
./local_docker_exec.sh src/neural/backends/sycl bash

# Inside container (manually):
# cd /intel/tianfeng/test/sycl
# ls -la
# ./build_test.sh -t test_softmax
# cd build && ./test_softmax
```

## Advanced Usage

### Parallel Test Execution

```bash
# Build once
./local_docker_exec.sh src/neural/backends/sycl \
    "./build_test.sh --clean" \
    build.log

# Run tests in parallel (from host)
for test in test_softmax test_expand_planes test_layer_norm; do
    ./local_docker_exec.sh src/neural/backends/sycl \
        "cd build && ./$test" \
        "${test}_parallel.log" &
done
wait

# Collect results
for test in test_softmax test_expand_planes test_layer_norm; do
    echo "$test: $(grep -E 'Test.*PASSED|Test.*FAILED' ${test}_parallel.log)"
done
```

### Compare with CUDA Reference

```bash
# Run SYCL test
./local_docker_exec.sh src/neural/backends/sycl \
    "cd build && ./test_winograd" \
    sycl_winograd.log

# Compare output patterns
diff -u <(grep "Max error" cuda_winograd.log) <(grep "Max error" sycl_winograd.log)
```

## Troubleshooting

### Issue 1: Container Not Running

```bash
# Error: "Docker container 'lsv-container' is not running"

# Fix: Start container manually
docker start lsv-container

# Or check why it's not starting
docker logs lsv-container
```

### Issue 2: Permission Denied

```bash
# Error: Cannot write to /intel/tianfeng/test/

# Fix: Check ownership inside container
./local_docker_exec.sh . "ls -la /intel/tianfeng/" permissions.log

# Fix from host
docker exec lsv-container chown -R $(id -u):$(id -g) /intel/tianfeng/test/
```

### Issue 3: Include Path Errors

```bash
# Error: 'neural/tables/activation_function.h' file not found

# Fix: Ensure test uses correct relative path
#include "../../tables/activation_function.h"
```

### Issue 4: GPU Not Available

```bash
# Error: No device of requested type available

# Check GPU in container
./local_docker_exec.sh . "sycl-ls" gpu_check.log

# Verify /dev/dri is mounted
docker exec lsv-container ls -la /dev/dri
```

## Debug Checklist

When kernel test fails:

- [ ] Build successful (check build.log)
- [ ] Test executable exists in build/
- [ ] GPU available (sycl-ls shows Intel GPU)
- [ ] Numerical accuracy within tolerance
- [ ] No memory errors (valgrind clean)
- [ ] Compare with CPU reference implementation
- [ ] Check kernel functor signature (nd_item vs id)
- [ ] Verify work-group size limits
- [ ] Check for race conditions
- [ ] Test with both float and sycl::half

## Integration with VS Code

Add to `.vscode/tasks.json`:

```json
{
    "label": "SYCL: Build Test (Docker)",
    "type": "shell",
    "command": "./local_docker_exec.sh",
    "args": [
        "src/neural/backends/sycl",
        "./build_test.sh -t ${input:testName} --clean",
        "${input:testName}_build.log"
    ]
},
{
    "label": "SYCL: Run Test (Docker)",
    "type": "shell",
    "command": "./local_docker_exec.sh",
    "args": [
        "src/neural/backends/sycl",
        "cd build && ./${input:testName}",
        "${input:testName}_run.log"
    ]
}
```

## See Also

- `test-build-in-docker` skill - Alternative using direct docker exec
- `local-docker-exec` skill - Generic local docker execution
- `src/neural/backends/sycl/SKILL.md` - SYCL kernel implementation guide
- `CLAUDE.md` - Project context and architecture
