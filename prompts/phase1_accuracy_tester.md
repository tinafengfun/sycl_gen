# Phase 1: UnifiedAccuracyTester Implementation
# Phase 1: UnifiedAccuracyTester 实现

## Task Description / 任务描述

Implement a fully functional UnifiedAccuracyTester class by integrating existing accuracy testing tools into the unified_converter.py workflow.

将现有的准确度测试工具集成到 unified_converter.py 工作流程中，实现功能完整的 UnifiedAccuracyTester 类。

## Input Files / 输入文件

### Source Files to Analyze / 需要分析的源文件:

1. **tools/accuracy_tester.py** (407 lines)
   - Contains: Test data generation, CUDA/SYCL compilation, result comparison
   - Key methods to migrate:
     - `generate_test_data()` - Generate boundary/random/special test data
     - `run_cuda_test()` - Compile and execute CUDA test
     - `run_sycl_test()` - Compile and execute SYCL test  
     - `compare_results()` - Numerical comparison logic

2. **test/accuracy/run_accuracy_test.py** (353 lines)
   - Contains: Full workflow integration, parallel execution, report generation
   - Key methods to migrate:
     - `run_sycl_test()` - SYCL execution with file sync
     - `run_cuda_test()` - Remote CUDA execution
     - `compare_results()` - Detailed error analysis

3. **tools/unified_converter.py** (Target file)
   - Current: Lines 439-478 contain placeholder implementation
   - Current `test()` method only simulates testing
   - Need to replace with real execution logic

## Current Implementation (Placeholder) / 当前实现（占位符）

```python
class UnifiedAccuracyTester:
    async def test(self, kernel_id, cuda_file, sycl_file,
                  cuda_binary, sycl_binary) -> dict:
        """Execute accuracy test - CURRENTLY SIMULATED"""
        # ❌ This is just simulation, needs real implementation
        test_sizes = [64, 256, 1024]
        all_pass = True
        total_tests = len(test_sizes)
        passed_tests = 0
        
        for size in test_sizes:
            print(f"   📊 Test scale: {size}")
            # NO ACTUAL EXECUTION - just counting
            passed_tests += 1
        
        pass_rate = passed_tests / total_tests
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": pass_rate,
            "status": "PASS" if pass_rate >= 0.99 else "FAIL"
        }
```

## Required Implementation / 需要实现的功能

### 1. Update UnifiedAccuracyTester class

Replace the simulated `test()` method with real execution:

```python
class UnifiedAccuracyTester:
    def __init__(self, tracer: UnifiedTracer):
        self.tracer = tracer
        # Test configurations (from run_accuracy_test.py)
        self.test_configs = [
            {"N": 1, "C": 64, "dtype": "float", "layout": "nchw", "test_type": "random"},
            {"N": 1, "C": 64, "dtype": "float", "layout": "nchw", "test_type": "ones"},
            {"N": 1, "C": 64, "dtype": "float", "layout": "nchw", "test_type": "sequential"},
            {"N": 4, "C": 128, "dtype": "float", "layout": "nchw", "test_type": "random"},
            {"N": 1, "C": 64, "dtype": "float", "layout": "nhcw", "test_type": "random"},
        ]
    
    async def test(self, kernel_id, cuda_file, sycl_file,
                  cuda_binary, sycl_binary) -> dict:
        """Real accuracy test execution"""
        # 1. Generate test data
        # 2. Compile CUDA test harness
        # 3. Compile SYCL test harness
        # 4. Execute both in parallel
        # 5. Compare outputs
        # 6. Calculate error metrics
        # 7. Return detailed report
```

### 2. Methods to Implement / 需要实现的方法

#### Method 1: generate_test_data()
```python
def generate_test_data(self, config: dict) -> np.ndarray:
    """
    Generate test data based on configuration.
    
    Args:
        config: Dictionary with keys:
            - N: batch size
            - C: channels
            - dtype: "float" or "half"
            - test_type: "random", "ones", "sequential", "boundary"
    
    Returns:
        numpy array with test data
    """
    # Support 4 test types:
    # - "random": uniform distribution [-1, 1]
    # - "ones": all 1.0
    # - "sequential": (i % 100) / 100.0
    # - "boundary": [0.0, 1.0, -1.0, max, min, epsilon]
```

#### Method 2: compile_and_run_cuda()
```python
async def compile_and_run_cuda(self, kernel_file: str, 
                                input_data: np.ndarray,
                                config: dict) -> np.ndarray:
    """
    Compile CUDA kernel and run test.
    
    Steps:
    1. Save input data to binary file
    2. Copy files to remote CUDA environment
    3. Compile CUDA test program (nvcc)
    4. Execute with GPU support
    5. Copy output back
    6. Load and return result
    """
```

#### Method 3: compile_and_run_sycl()
```python
async def compile_and_run_sycl(self, kernel_file: str,
                                input_data: np.ndarray, 
                                config: dict) -> np.ndarray:
    """
    Compile SYCL kernel and run test.
    
    Steps:
    1. Save input data to binary file
    2. Copy files to B60 container
    3. Compile SYCL test program (icpx)
    4. Execute on Intel GPU
    5. Copy output back
    6. Load and return result
    """
```

#### Method 4: compare_outputs()
```python
def compare_outputs(self, cuda_output: np.ndarray,
                   sycl_output: np.ndarray,
                   config: dict) -> dict:
    """
    Compare CUDA and SYCL outputs.
    
    Calculate:
    - Max absolute difference
    - Max relative difference
    - Mean absolute difference
    - Mean relative difference
    - Pass/Fail status based on tolerances
    
    Tolerances:
    - float32: abs_tol=1e-5, rel_tol=1e-4
    - float16: abs_tol=1e-3, rel_tol=1e-2
    
    Returns:
        dict with comparison metrics
    """
```

### 3. Integration with Phase 4 / 集成到 Phase 4

Update `run_phase4_accuracy_test()` method in UnifiedOrchestrator:

```python
async def run_phase4_accuracy_test(self) -> dict:
    """Phase 4: Accuracy validation"""
    self.state["current_phase"] = Phase.ACCURACY
    
    # Get compiled binaries from Phase 3
    sycl_binary = f"{self.workspace}/build_sycl/{self.kernel_id}.o"
    cuda_binary = f"/tmp/{self.kernel_id}_cuda"
    
    # Run real accuracy test
    result = await self.accuracy_tester.test(
        self.kernel_id,
        self.cuda_file,
        f"kernel_dataset/sycl/{self.kernel_id}_kernel.dp.cpp",
        cuda_binary,
        sycl_binary
    )
    
    self.state["phases_completed"].append(Phase.ACCURACY)
    return result
```

### 4. Error Handling / 错误处理

Implement comprehensive error handling:

```python
try:
    cuda_result = await self.compile_and_run_cuda(...)
except Exception as e:
    self.tracer.log("AccuracyTester", "cuda_error", {"error": str(e)})
    return {"status": "FAIL", "error": f"CUDA execution failed: {e}"}

try:
    sycl_result = await self.compile_and_run_sycl(...)
except Exception as e:
    self.tracer.log("AccuracyTester", "sycl_error", {"error": str(e)})
    return {"status": "FAIL", "error": f"SYCL execution failed: {e}"}
```

### 5. Trace Integration / Trace集成

Log all key events:

```python
self.tracer.log("AccuracyTester", "test_start", {
    "kernel": kernel_id,
    "test_configs": len(self.test_configs)
})

self.tracer.log("AccuracyTester", "data_generated", {
    "config": config,
    "size": data.size
})

self.tracer.log("AccuracyTester", "cuda_complete", {
    "duration": cuda_time,
    "output_size": cuda_output.size
})

self.tracer.log("AccuracyTester", "comparison_complete", {
    "max_abs_diff": max_abs_diff,
    "max_rel_diff": max_rel_diff,
    "pass_rate": pass_rate
})
```

## Output Requirements / 输出要求

### 1. Return Value Format

```python
{
    "status": "PASS" or "FAIL",
    "total_tests": 5,
    "passed_tests": 5,
    "pass_rate": 1.0,
    "test_results": [
        {
            "config": {...},
            "max_abs_diff": 0.0,
            "max_rel_diff": 0.0,
            "mean_abs_diff": 0.0,
            "mean_rel_diff": 0.0,
            "status": "PASS"
        },
        ...
    ],
    "summary": {
        "overall_max_abs_diff": 0.0,
        "overall_max_rel_diff": 0.0,
        "avg_pass_rate": 1.0
    }
}
```

### 2. Console Output

```
🎯 Phase 4: Accuracy validation...
   🧪 Running 5 test configurations...
   
   Test 1/5: N=1, C=64, random
     ✅ Max abs diff: 0.000000e+00
     ✅ Max rel diff: 0.000000e+00
     ✅ Status: PASS
   
   Test 2/5: N=1, C=64, ones
     ✅ Max abs diff: 0.000000e+00
     ✅ Status: PASS
   
   ...
   
   ✅ Accuracy test complete: PASS
      Pass rate: 100.0%
      Overall max error: 0.000000e+00
```

## Testing Requirements / 测试要求

### Test with winograd_input_transform kernel:

```bash
python3 tools/unified_converter.py winograd_input_transform
```

### Expected Results:
- ✅ All 5 test configurations pass
- ✅ Pass rate >= 99.9%
- ✅ Max absolute error < 1e-5
- ✅ Max relative error < 1e-4
- ✅ Complete trace logs

## Code Structure / 代码结构

Maintain the existing structure, only modify UnifiedAccuracyTester class:

```python
# Keep existing:
- UnifiedOrchestrator
- Phase enum and Status enum
- ConversionResult dataclass
- UnifiedTracer
- UnifiedAnalyzer
- UnifiedConverter
- UnifiedValidator

# Modify:
- UnifiedAccuracyTester (lines 439-478)

# Update integration:
- run_phase4_accuracy_test() (lines 241-258)
```

## Success Criteria / 成功标准

1. ✅ Real CUDA execution (not simulated)
2. ✅ Real SYCL execution (not simulated)
3. ✅ Parallel execution of CUDA and SYCL
4. ✅ Numerical comparison with error metrics
5. ✅ All 5 test configurations supported
6. ✅ Pass rate calculation
7. ✅ Trace integration
8. ✅ Error handling
9. ✅ Test with winograd kernel passes

## Notes / 注意事项

1. **Keep existing interfaces**: Don't change method signatures
2. **Reuse existing code**: Copy logic from accuracy_tester.py
3. **Maintain async pattern**: All methods should be async
4. **File paths**: Use absolute paths or relative to workspace
5. **Container access**: Use docker exec for B60, SSH for CUDA
6. **Timeout**: Add timeout for compilation and execution (120s)
7. **Cleanup**: Clean up temporary files after testing

## Example Migration / 迁移示例

From accuracy_tester.py:
```python
def generate_test_data(self, test_type, size, dtype):
    if test_type == "boundary":
        values = [0.0, 1.0, -1.0, np.finfo(dtype).max, ...]
        data = np.tile(values, (size // len(values)) + 1)[:size]
    # ... more types
    return data.astype(dtype)
```

To unified_converter.py:
```python
def generate_test_data(self, config):
    dtype = np.float32 if config["dtype"] == "float" else np.float16
    size = config["N"] * config["C"] * 8 * 8
    
    if config["test_type"] == "boundary":
        values = [0.0, 1.0, -1.0, np.finfo(dtype).max, ...]
        data = np.tile(values, (size // len(values)) + 1)[:size]
    # ... adapt for config dict
    return data.astype(dtype)
```

## Deliverables / 交付物

1. ✅ `tools/unified_converter.py` (Phase 1 updated version)
2. ✅ Test execution log showing real tests (not simulated)
3. ✅ winograd kernel accuracy report with real data

---

**Start implementing now!**
