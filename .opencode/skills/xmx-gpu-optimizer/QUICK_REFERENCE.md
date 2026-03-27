# XMX Optimizer - Quick Reference Card

## 30-Second Decision Tree

```
1. Matrix multiply involved?
   ├─ Matrix >= 256x256? → Type D-Large (XMX)
   └─ Matrix < 256x256?  → Type D-Small (single-thread)

2. Pooling/softmax/reduction?
   └─ Type C (single-thread-per-output)

3. Winograd/spatial transform?
   └─ Type B (tile optimization)

4. Element-wise operations?
   └─ Type A (vectorized memory, stop if <15% gain)
```

## Mandatory Compilation

```bash
icpx -fsycl -O3 -std=c++17 \
  -fsycl-targets=spir64_gen \
  -Xsycl-target-backend "-device bmg -options -ze-opt-large-register-file" \
  -o output input.cpp
```

## Key Insights (Don't Forget!)

1. **Single-thread-per-output is OPTIMAL for BMG**
   - Proven 60% gain (global_avg_pool)
   - Proven 18x gain (SE layer)
   - Use for: Type C (all), Type D-Small

2. **XMX requires matrices >= 256x256**
   - Small matrices: XMX overhead > benefit
   - Large matrices: 100+ TFLOPS achievable
   - SE layer (128x64): Single-thread 21 GFLOPS vs XMX 1.7 GFLOPS

3. **AOT compilation REQUIRED for XMX**
   - `-fsycl-targets=spir64_gen -device bmg`
   - JIT compilation WILL FAIL with XMX

4. **Large GRF mode essential**
   - `-ze-opt-large-register-file`
   - Complex kernels need more registers

## Expected Performance by Type

| Type | Speedup | Strategy |
|------|---------|----------|
| A | 1.05x | Vectorized, 1 round only |
| B | 1.4-1.6x | Tile optimization, 2-3 rounds |
| C | 1.5-1.7x | **Single-thread-per-output** |
| D-Small | 10-100x | Single-thread-per-output |
| D-Large | 12x+ | XMX joint_matrix |

## Common Mistakes

❌ **Using collaborative reduction** → ✅ Use single-thread-per-output
❌ **XMX for small matrices** → ✅ Check matrix size first
❌ **JIT compilation for XMX** → ✅ Use AOT `-device bmg`
❌ **No `-ze-opt-large-register-file`** → ✅ Always include
❌ **3 rounds for all kernels** → ✅ Type A stops at round 1

## Quick Checklist

### Before Starting
- [ ] Docker container running
- [ ] Source file in container: `/workspace/tests/`
- [ ] Kernel type identified
- [ ] Expected speedup realistic

### During Optimization
- [ ] Compilation with all mandatory flags
- [ ] Test 3 sizes: small, medium, large
- [ ] Compare with baseline
- [ ] Decision: continue or stop?

### After Optimization
- [ ] Results saved to CSV
- [ ] Logs archived
- [ ] Best version identified
- [ ] Next kernel ready

## Emergency Commands

```bash
# Check what's compiled
docker exec lsv-container ls -la /workspace/tests/

# Recompile with verbose output
docker exec -w /workspace/tests lsv-container \
  icpx -fsycl -O3 -std=c++17 -v \
  -fsycl-targets=spir64_gen \
  -Xsycl-target-backend "-device bmg -options -ze-opt-large-register-file" \
  -o test test.cpp 2>&1 | tee compile.log

# Run with timing
docker exec -w /workspace/tests lsv-container time ./test

# Copy results back
docker cp lsv-container:/workspace/tests/results.csv ./
```

## File Locations

```
.opencode/skills/xmx-gpu-optimizer/
├── SKILL.md                    # This skill
└── templates/                  # Copy-paste templates
    ├── type_a_elementwise.cpp
    ├── type_b_winograd.cpp
    ├── type_c_reduction.cpp
    ├── type_d_small_gemm.cpp
    └── type_d_large_xmx.cpp
```

## Version Info

Based on: LCZero kernel optimization project (2026-03-26)
Validated: 4/36 kernels (11%)
Key Finding: Single-thread-per-output pattern
Hardware: Intel BMG B60 (0xe211)
