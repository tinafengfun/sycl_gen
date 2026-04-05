# SYCL Kernel Builder

Compile and test SYCL kernels in the B60 Docker container ($arg)

## Environment

- **Container**: `lsv-container_2026_3`
- **Compiler**: `icpx` (Intel DPC++/C++ 2025.3)
- **Path mapping**: `/home/intel/tianfeng/opencode_bench` → `/sandbox`, `/tmp/intel/` → `/workspace`
- **Source dir**: `/sandbox/` (代码)
- **Build dir**: `/workspace/build/` (编译产物)

## Execution

Based on the user's argument `$arg`, determine the action:

### 1. Compile a single SYCL kernel

If `$arg` is a .dp.cpp or .cpp file path → compile it:

```bash
docker exec lsv-container_2026_3 bash -c 'source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null && \
mkdir -p /workspace/build && \
icpx -fsycl -fsycl-targets=spir64_gen \
  -Xs -device\ bmg \
  -fsycl-unnamed-lambda -std=c++17 -O2 \
  /sandbox/<source> -o /workspace/build/<output> 2>&1'
```

### 2. Compile and run a test

If `$arg` is a test file with a run command → compile and execute:

```bash
# Compile
docker exec lsv-container_2026_3 bash -c 'source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null && \
mkdir -p /workspace/build && \
icpx -fsycl -fsycl-targets=spir64_gen -Xs -device\ bmg -fsycl-unnamed-lambda -std=c++17 -O2 \
  /sandbox/<test.cpp> -o /workspace/build/<test> 2>&1'

# Run
docker exec lsv-container_2026_3 bash -c '/workspace/build/<test> 2>&1'
```

### 3. Batch compile all kernels

If `$arg` is "all" → compile every .dp.cpp in `kernel_bench/kernel_dataset/sycl/`:

```bash
docker exec lsv-container_2026_3 bash -c 'source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null && \
mkdir -p /workspace/build && \
PASS=0 && FAIL=0 && \
for f in /sandbox/kernel_bench/kernel_dataset/sycl/*.dp.cpp; do
  name=$(basename "$f" .dp.cpp)
  if icpx -fsycl -fsycl-targets=spir64_gen -Xs -device\ bmg -fsycl-unnamed-lambda -std=c++17 -O2 \
    "$f" -c -o /workspace/build/${name}.o 2>/workspace/build/${name}.log; then
    echo "PASS $name"
    PASS=$((PASS+1))
  else
    echo "FAIL $name"
    FAIL=$((FAIL+1))
  fi
done && \
echo "=== PASS: $PASS  FAIL: $FAIL  TOTAL: $((PASS+FAIL)) ==="'
```

### 4. Check container status

If `$arg` is "check" → verify container, compiler, GPU:

```bash
docker exec lsv-container_2026_3 bash -c 'source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null && \
echo "=== Compiler ===" && icpx --version | head -1 && \
echo "=== GPU ===" && sycl-ls 2>&1'
```

### 5. Full optimization compile

If `$arg` contains "full" → use torch-xpu-ops optimization flags:

```bash
docker exec lsv-container_2026_3 bash -c 'source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null && \
icpx -fsycl -fsycl-targets=spir64_gen \
  -fsycl-unnamed-lambda -sycl-std=2020 -std=c++17 -O2 \
  -fno-fast-math -fma -no-ftz \
  -Xs -device\ bmg\ -options\ -cl-intel-enable-auto-large-GRF-mode\ -options\ -cl-fp32-correctly-rounded-divide-sqrt\ -options\ -cl-intel-greater-than-4GB-buffer-required\ -options\ -cl-poison-unsupported-fp64-kernels \
  --offload-compress \
  /sandbox/<source> -o /workspace/build/<output> 2>&1'
```

## Compiler Flags Quick Reference

| Flag | Purpose |
|------|---------|
| `-fsycl` | Enable SYCL |
| `-fsycl-targets=spir64_gen` | AOT for Intel GPU |
| `-Xs -device bmg` | Target Battlemage |
| `-fsycl-unnamed-lambda` | **必须** for lambda kernels |
| `-std=c++17` | C++ standard |
| `-O2` | Optimization |
| `-fno-fast-math -fma -no-ftz` | Numerical precision |
| `-options -cl-intel-enable-auto-large-GRF-mode` | Large register file (**推荐**) |
| `--offload-compress` | Compress AOT binary |

## Troubleshooting

- **Container not running**: `docker start lsv-container_2026_3`
- **`fp64 is not supported`**: BMG has no native FP64, use `-fsycl-fp64-conv-emu` or avoid `double`
- **`unnamed type is invalid`**: Add `-fsycl-unnamed-lambda`
- **AOT slow**: Use JIT for dev: `icpx -fsycl -std=c++17 -O2 <src> -o <out>`
- **`PI_ERROR_DEVICE_NOT_FOUND`**: Check `--device=/dev/dri` mapping, verify `sycl-ls`
- **`redefinition of X`**: Check if header already defines it, remove local duplicate
