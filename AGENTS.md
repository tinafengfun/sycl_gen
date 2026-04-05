# AGENTS.md - LCZero CUDA/SYCL Kernel Dataset

## Repository Overview

This is a **dataset repository** containing extracted CUDA and SYCL kernel implementations from the LCZero chess engine. It serves as a reference for GPU kernel development and CUDA-to-SYCL migration.

- **Total Kernels**: 30 unique kernel functions
- **Source**: https://github.com/LeelaChessZero/lc0
- **License**: GNU GPL v3 (with NVIDIA CUDA library exception)

The kernel dataset is located at `kernel_bench/kernel_dataset/`.

## Build/Test Commands

This is a **data repository** - no build system or tests. Files are:
- Raw kernel source code templates
- JSON index for programmatic access
- Reference implementations for study/migration

## Code Style Guidelines

### File Naming
- **CUDA**: `{kernel_name}_kernel.cu`
- **SYCL**: `{kernel_name}_kernel.dp.cpp`
- Use snake_case for kernel names
- Must end with `_kernel` suffix before extension

### File Structure
Each kernel file must include:
1. **GPL License Header** (exact format from existing files)
2. **Includes**: `cuda_runtime.h`, `cuda_fp16.h` for CUDA
3. **Namespace**: `lczero::cudnn_backend` (CUDA) or `lczero::sycldnn_backend` (SYCL)
4. **Standard Constants** (copy from existing files):
   - `kNumOutputPolicy = 1858`
   - `kMaxResBlockFusingChannels = 384`
   - `kMaxResBlockFusingSeKFp16Ampere = 512`
   - `kMaxResBlockFusingSeK = 128`
   - `kInputPlanes = 112`
5. **ActivationFunction enum** (9 values, copy exactly)
6. **DivUp helper**: `inline int DivUp(int a, int b) { return (a + b - 1) / b; }`
7. **Kernel documentation comments**:
   ```cpp
   // Kernel: {kernel_id}
   // Description: {description}
   // Category: {category}
   ```

### Code Formatting
- **Indentation**: 2 spaces (no tabs)
- **Braces**: Same line (K&R style)
- **Line length**: 80 characters max
- **Comments**: `//` for inline, `/* */` for license header only
- **Namespaces**: Indent content inside namespaces

### Naming Conventions
- **Constants**: `kCamelCase` (e.g., `kNumOutputPolicy`)
- **Enums**: `ALL_CAPS_WITH_UNDERSCORES`
- **Functions**: `snake_case`
- **Kernels**: `{name}_kernel` (e.g., `add_vectors_kernel`)
- **Templates**: `typename T` for data types
- **Variables**: `snake_case` or `camelCase` (follow existing patterns)

### CUDA-SYCL Correspondence
| CUDA | SYCL |
|------|------|
| `__global__` | Remove (lambda or functor) |
| `__shared__` | `sycl::local_accessor` |
| `__syncthreads()` | `item.barrier()` |
| `threadIdx.x` | `item.get_local_id(0)` |
| `blockIdx.x` | `item.get_group(0)` |
| `blockDim.x` | `item.get_local_range(0)` |
| `cudaStream_t` | `sycl::queue` |

### Adding New Kernels

1. Create file in appropriate directory (`cuda/` or `sycl/`)
2. Follow exact file structure above
3. Update `index.json`:
   - Add entry to `kernels` array
   - Assign unique `id` (snake_case)
   - Set `has_sycl_mapping` boolean
   - Specify `category` (use existing or add new)
   - Add to `categories` object if new category
4. Increment `statistics` in `index.json` if applicable

### Index.json Structure
```json
{
  "id": "kernel_name",
  "name": "Human Readable Name",
  "description": "What the kernel does",
  "category": "category_name",
  "has_sycl_mapping": true/false,
  "cuda": { "file": "cuda/kernel_name_kernel.cu" },
  "sycl": { "file": "sycl/kernel_name_kernel.dp.cpp" },
  "notes": "Optional notes"
}
```

### Special Cases
- **CUDA-only kernels** (e.g., CUTLASS): Set `has_sycl_mapping: false`, add `notes`
- **SYCL-only kernels**: Omit `cuda` field
- Keep placeholder comments for unimplemented kernels

### Git Workflow
- Commit message format: `[category] Add/Update kernel_name kernel`
- One kernel per commit preferred
- Update README.md if adding new categories

## Directory Structure
```
kernel_bench/kernel_dataset/
├── cuda/              # CUDA kernel files (.cu)
├── sycl/              # SYCL kernel files (.dp.cpp)
├── templates/         # Reserved for templates
├── index.json         # Kernel metadata and mappings
└── README.md          # Dataset documentation
```

## References
- Source CUDA backend: `src/neural/backends/cuda/`
- Source SYCL backend: `src/neural/backends/sycl/`
- LCZero: https://github.com/LeelaChessZero/lc0
