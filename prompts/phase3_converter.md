# Phase 3: UnifiedConverter Enhancement
# Phase 3: UnifiedConverter Agent Enhancement

## Task Description

Enhance UnifiedConverter to support model-based code generation using opencode AI, with fallback to rule-based conversion.

## Current Implementation (Rule-Based)

```python
class UnifiedConverter:
    async def convert(self, cuda_file: str, analysis: dict) -> str:
        """Basic rule-based conversion"""
        # Read CUDA code
        with open(cuda_file, 'r') as f:
            cuda_code = f.read()
        
        # Simple replacements
        replacements = [
            ("#include <cuda_runtime.h>", "#include <sycl/sycl.hpp>"),
            ("__device__", ""),
            ("__global__", ""),
            # ... more rules
        ]
        
        sycl_code = cuda_code
        for old, new in replacements:
            sycl_code = sycl_code.replace(old, new)
        
        return sycl_code
```

## Required Implementation

### 1. Create ModelBasedConverter Class

```python
class ModelBasedConverter:
    """CUDA-to-SYCL converter using AI model"""
    
    def __init__(self, tracer: UnifiedTracer):
        self.tracer = tracer
        self.system_prompt = self._load_system_prompt()
        self.user_prompt_template = self._load_user_prompt_template()
    
    async def convert(self, cuda_file: str, analysis: dict) -> str:
        """
        Convert CUDA to SYCL using AI model
        
        Args:
            cuda_file: Path to CUDA source file
            analysis: Analysis report from UnifiedAnalyzer
            
        Returns:
            SYCL code string
        """
        # Read CUDA code
        with open(cuda_file, 'r') as f:
            cuda_code = f.read()
        
        # Build prompt
        prompt = self._build_prompt(cuda_code, analysis)
        
        # Call AI model (opencode)
        sycl_code = await self._call_model(prompt)
        
        # Validate generated code
        if await self._validate_syntax(sycl_code):
            return sycl_code
        else:
            raise ConversionError("Model generated invalid code")
    
    def _build_prompt(self, cuda_code: str, analysis: dict) -> str:
        """Build prompt for AI model"""
        return self.user_prompt_template.format(
            cuda_code=cuda_code,
            kernel_name=analysis.get('kernel_name', 'unknown'),
            total_lines=analysis.get('total_lines', 0),
            device_functions=analysis.get('device_functions', 0),
            global_kernels=analysis.get('global_kernels', 0),
            complexity_level=analysis.get('complexity_level', 1),
            templates=analysis.get('templates', 0)
        )
    
    async def _call_model(self, prompt: str) -> str:
        """Call opencode AI model"""
        # Implementation depends on opencode API
        # For now, placeholder that reads from opencode response
        pass
    
    async def _validate_syntax(self, code: str) -> bool:
        """Quick syntax validation"""
        # Check for basic SYCL patterns
        required_patterns = [
            '#include <sycl/sycl.hpp>',
            'namespace sycldnn_backend'  # or appropriate namespace
        ]
        
        for pattern in required_patterns:
            if pattern not in code:
                return False
        
        return True
```

### 2. Create Enhanced UnifiedConverter

```python
class UnifiedConverter:
    """Enhanced converter with model-based and rule-based options"""
    
    def __init__(self, tracer: UnifiedTracer, use_model: bool = True):
        self.tracer = tracer
        self.use_model = use_model
        self.model_converter = ModelBasedConverter(tracer)
        self.rule_converter = RuleBasedConverter(tracer)
    
    async def convert(self, cuda_file: str, analysis: dict) -> str:
        """
        Convert CUDA to SYCL
        
        Strategy:
        1. Try model-based conversion (if enabled)
        2. If fails, fallback to rule-based
        3. Log which method was used
        """
        self.tracer.log("UnifiedConverter", "start_conversion", {
            "file": cuda_file,
            "use_model": self.use_model,
            "complexity": analysis.get('complexity_level', 1)
        })
        
        if self.use_model:
            try:
                print("   🤖 Attempting model-based conversion...")
                sycl_code = await self.model_converter.convert(cuda_file, analysis)
                
                self.tracer.log("UnifiedConverter", "model_conversion_success", {
                    "lines": len(sycl_code.split('\n'))
                })
                print("   ✅ Model-based conversion successful")
                return sycl_code
                
            except Exception as e:
                self.tracer.log("UnifiedConverter", "model_conversion_failed", {
                    "error": str(e)
                })
                print(f"   ⚠ Model conversion failed: {e}")
                print("   🔄 Falling back to rule-based conversion...")
        
        # Fallback to rule-based
        sycl_code = await self.rule_converter.convert(cuda_file, analysis)
        
        self.tracer.log("UnifiedConverter", "rule_conversion_complete", {
            "lines": len(sycl_code.split('\n'))
        })
        print("   ✅ Rule-based conversion complete")
        
        return sycl_code
```

### 3. Prompt Design

#### System Prompt (prompts/converter_system.txt)

```
You are an expert CUDA to SYCL converter. Your task is to convert CUDA kernel code to SYCL code following Intel's best practices.

Key Conversion Rules:
1. Headers:
   - Replace #include <cuda_runtime.h> with #include <sycl/sycl.hpp>
   - Remove #include <cuda_fp16.h>
   - Keep other standard headers

2. Function Qualifiers:
   - Remove __device__ and __global__ qualifiers
   - Convert __forceinline__ to inline
   - Keep template declarations

3. Namespace:
   - Change namespace cudnn_backend to sycldnn_backend
   - Or use appropriate LCZero namespace

4. Kernel Launch:
   - Replace <<< >>> syntax with sycl::queue::parallel_for
   - Use sycl::nd_range for 2D/3D kernels
   - Pass sycl::queue as parameter

5. Thread Indexing:
   - threadIdx.x -> item.get_local_id(0)
   - blockIdx.x -> item.get_group(0)
   - blockDim.x -> item.get_local_range(0)
   - gridDim.x -> item.get_group_range(0)

6. Memory Operations:
   - cudaMemcpy -> queue.memcpy
   - __shared__ -> sycl::local_accessor
   - __syncthreads() -> item.barrier()

7. Data Types:
   - half -> sycl::half
   - uint4 -> sycl::uint4
   - Keep float, double, int as-is

8. Macros to Functions:
   - Convert #define INDEX_NCHW(...) to inline int IndexNCHW(...)
   - Add runtime parameters for template-dependent macros

9. Error Handling:
   - Remove CUDA error checking
   - Add SYCL exception handling

10. Best Practices:
    - Use -fno-associative-math for deterministic results
    - Add appropriate SYCL kernel attributes
    - Maintain original algorithm logic

Output Format:
- Provide only the complete SYCL code
- No explanations or comments about the conversion
- Ensure the code compiles with Intel oneAPI compiler (icpx)
- Use proper indentation (2 spaces)

Quality Criteria:
- Code must compile without errors
- Maintain numerical accuracy
- Follow SYCL 2020 standard
- Support both float and sycl::half types
```

#### User Prompt Template (prompts/converter_user.txt)

```
Convert the following CUDA kernel to SYCL:

CUDA Source Code:
```cuda
{cuda_code}
```

Analysis:
- Kernel Name: {kernel_name}
- Total Lines: {total_lines}
- Device Functions: {device_functions}
- Global Kernels: {global_kernels}
- Templates: {templates}
- Complexity Level: {complexity_level}/3

Requirements:
1. Convert to SYCL 2020 standard
2. Use sycldnn_backend namespace
3. Maintain all template specializations
4. Handle __device__ function conversions
5. Convert all __global__ kernels
6. Use sycl::queue for execution
7. Support both float and sycl::half

Generate the complete SYCL code:
```

### 4. Integration with UnifiedOrchestrator

Update UnifiedOrchestrator to use enhanced converter:

```python
class UnifiedOrchestrator:
    def __init__(self, kernel_id: str, cuda_file: str, use_model: bool = True):
        # ... existing init ...
        self.converter = UnifiedConverter(self.tracer, use_model=use_model)
```

### 5. Testing Strategy

Test with 3 simple kernels first:

1. **vector_add** (Level 1)
   - Simple element-wise addition
   - No shared memory
   - 1D kernel

2. **matrix_multiply** (Level 1-2)
   - Basic matrix multiplication
   - May use shared memory
   - 2D kernel

3. **softmax** (Level 2)
   - Reduction operation
   - More complex indexing
   - Requires careful conversion

### 6. Error Handling & Fallback

```python
class ConversionError(Exception):
    """Custom exception for conversion errors"""
    pass

async def safe_convert(cuda_file: str, analysis: dict, tracer: UnifiedTracer) -> str:
    """
    Safely convert CUDA to SYCL with multiple fallback strategies
    
    Strategy:
    1. Try AI model conversion
    2. If fails, try rule-based with auto-fixes
    3. If still fails, raise error with details
    """
    converter = UnifiedConverter(tracer, use_model=True)
    
    try:
        return await converter.convert(cuda_file, analysis)
    except Exception as e:
        tracer.log("Conversion", "all_methods_failed", {"error": str(e)})
        raise ConversionError(f"Failed to convert {cuda_file}: {e}")
```

### 7. Performance Considerations

```python
# Cache conversion results
class ConversionCache:
    def __init__(self):
        self.cache = {}
    
    def get(self, cuda_file: str, file_hash: str) -> Optional[str]:
        key = f"{cuda_file}:{file_hash}"
        return self.cache.get(key)
    
    def set(self, cuda_file: str, file_hash: str, sycl_code: str):
        key = f"{cuda_file}:{file_hash}"
        self.cache[key] = sycl_code
```

## Success Criteria

- [ ] ModelBasedConverter class implemented
- [ ] System prompt created and tested
- [ ] User prompt template created
- [ ] Integration with UnifiedOrchestrator
- [ ] Fallback mechanism works
- [ ] 3 simple kernels converted successfully
- [ ] Compilation successful
- [ ] Accuracy test passes (>99%)

## Deliverables

1. ModelBasedConverter class in unified_converter.py
2. Prompts: prompts/converter_system.txt, prompts/converter_user.txt
3. Updated UnifiedConverter with fallback
4. test_phase3.py with unit tests
5. 3 converted kernel examples

## Testing Plan

1. Test prompt generation
2. Test model API integration (mock)
3. Test syntax validation
4. Test fallback mechanism
5. Test with real kernels (vector_add, matrix_multiply, softmax)
6. Verify compilation
7. Run accuracy tests
