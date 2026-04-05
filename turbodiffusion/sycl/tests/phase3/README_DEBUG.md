# SYCL Kernel Debug Verification Scripts

These scripts help diagnose whether SYCL kernels are actually being executed during Phase 3.2 testing.

## Problem Statement

If Phase 3.2 test reports `error=0.0`, this could indicate:
1. SYCL kernel is not being called
2. SYCL kernel is failing silently and falling back to PyTorch
3. Hook mechanism is not intercepting the forward pass

## Quick Start

### Option 1: Quick Check (30 seconds)
Run this first for a fast sanity check:

```bash
python3 /workspace/turbodiffusion-sycl/tests/phase3/quick_sycl_check.py
```

**Expected output:**
```
[OK] SYCL device: Intel Graphics
[TEST] Creating unique test input...
  Input marker: 3.141590
  Weight[0]: 2.000000
[TEST] Calling SYCL kernel...
  SYCL output[0,0,0]: 1.998XXX
[RESULT] Max error: X.XXe-XX
[PASS] SYCL is producing different outputs
```

If you see `[FAIL] Error is exactly 0.0`, run the full debug script.

### Option 2: Full Debug Suite (2-3 minutes)
Run comprehensive verification:

```bash
python3 /workspace/turbodiffusion-sycl/tests/phase3/debug_sycl_execution.py
```

This runs 4 detailed tests and saves results to:
- `/workspace/turbodiffusion-sycl/tests/phase3/debug_sycl_execution_results.json`

## Test Descriptions

### Test 1: Direct SYCL Execution
- Calls SYCL kernel directly without hooks
- Verifies kernel produces unique output
- Detects if kernel is actually running

### Test 2: Hook Instrumentation
- Tests hook registration and invocation
- Tracks every hook call with markers
- Verifies hook mechanism works

### Test 3: Output Differentiation
- Compares PyTorch vs SYCL outputs
- Element-wise difference distribution
- Cosine similarity analysis

### Test 4: Fallback Detection
- Tests normal execution
- Tests disabled hook (should skip SYCL)
- Tests error injection (should trigger fallback)

## Interpreting Results

### If Quick Check Shows `error=0.0`:

**Most likely causes:**

1. **SYCL kernel not executing**
   - Check queue creation message: "SYCL Queue created on: Intel Graphics"
   - If no message, bindings may not be compiled

2. **Silent fallback**
   - Check full debug output for fallback events
   - Hook may be catching exceptions and returning original output

3. **Hook not called**
   - Check layer path is correct
   - Verify hook is enabled after registration

### Diagnostic Commands

```bash
# Check if SYCL bindings exist
ls -la /workspace/turbodiffusion-sycl/bindings/turbodiffusion_sycl/

# Check if compiled extension exists
python3 -c "import turbodiffusion_sycl as tds; print(tds.is_available())"

# Run unit test for RMSNorm
python3 /workspace/turbodiffusion-sycl/tests/unit/test_rmsnorm.py

# Check PyTorch and XPU
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'XPU: {torch.xpu.is_available()}')"
```

## Files

- `quick_sycl_check.py` - Fast sanity check
- `debug_sycl_execution.py` - Comprehensive debug suite
- `debug_sycl_execution_results.json` - Test results output
- `README_DEBUG.md` - This file

## Container Requirements

These scripts are designed to run in the B60 container with:
- Intel oneAPI SYCL compiler
- PyTorch with XPU support
- Compiled TurboDiffusion-SYCL bindings

## Exit Codes

- `0` - All checks passed, SYCL is executing correctly
- `1` - One or more checks failed, investigate debug output
