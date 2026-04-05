# TurboDiffusion SYCL 增量优化指南

## 优化策略：测试驱动迭代

每步优化后必须验证：**准确度 + 性能**

---

## Phase 1: 基线测试 (当前状态)

### 测试命令
```bash
cd /home/intel/tianfeng/opencode_bench/turbodiffusion-sycl

# 1. 测试当前基线实现
python optimization_benchmark.py --phase baseline --device xpu

# 2. 查看结果
cat results/baseline_results.json
cat results/OPTIMIZATION_REPORT.md
```

### 期望基线指标
- **Flash Attention**: ~15-16 TFLOPS
- **最大误差**: < 0.01 (BF16标准)
- **无NaN/Inf**: 必须满足

---

## Phase 2: XMX优化 (预期 2-3x 加速)

### 优化内容
1. 启用Intel XMX矩阵引擎
2. 使用DPAS指令加速矩阵乘法
3. 优化tile尺寸为8x16x16 (Xe2原生)

### 实施步骤
```bash
# 1. 更新setup.py，添加flash_attention_xmx.cpp
# 2. 重新编译
export CC=icpx CXX=icpx
cd operators
python setup.py build_ext --inplace

# 3. 测试XMX版本
python optimization_benchmark.py --phase xmx --device xpu

# 4. 对比结果
python optimization_benchmark.py --compare
```

### 成功标准
- [ ] TFLOPS提升 > 50%
- [ ] 准确度保持在<0.01
- [ ] 无NaN/Inf

---

## Phase 3: 内存带宽优化 (预期 1.5-2x 加速)

### 优化内容
1. 共享内存访问模式优化
2. Bank conflict消除
3. 预取策略(Prefetching)

### 关键修改
```cpp
// 优化前
for (int d = 0; d < head_dim; d++) {
    dot += q_tile[d] * k_tile[d];
}

// 优化后 - 4路展开，步长避免bank conflict
#pragma unroll 4
for (int d = 0; d < head_dim; d += 4) {
    dot += q_tile[d] * k_tile[d];
    dot += q_tile[d+1] * k_tile[d+1];
    dot += q_tile[d+2] * k_tile[d+2];
    dot += q_tile[d+3] * k_tile[d+3];
}
```

### 测试步骤
```bash
python optimization_benchmark.py --phase memory --device xpu
python optimization_benchmark.py --compare
```

---

## Phase 4: Work-Group调优 (预期 1.2-1.5x 加速)

### Xe2架构参数
- EU数量: 8 EUs/subslice
- 线程: 256 threads/subslice  
- 推荐: 256 threads/work-group (2 subslices)

### 测试矩阵
测试不同work-group大小:
- 128 threads (1 subslice)
- 256 threads (2 subslices) ← 推荐
- 512 threads (4 subslices)

```bash
python optimization_benchmark.py --phase workgroup --device xpu
```

---

## Phase 5: 数值精度修复

### 当前问题
1. 最大误差 ~0.38-3.4 (目标 <0.01)
2. 部分测试出现NaN
3. Softmax数值稳定性

### 修复方案
1. **Online Softmax算法**
```cpp
// 当前：每次KV tile后重新计算softmax
// 优化：增量更新，避免重复exp计算
float new_max = max(old_max, tile_max);
float scale = exp(old_max - new_max);
accumulator = accumulator * scale + new_values;
```

2. **FP32累加**
```cpp
// BF16存储 + FP32计算
float dot = 0.0f;  // FP32累加器
for (...) {
    dot += (float)q_tile[d] * (float)k_tile[d];
}
```

3. **数值截断检查**
```cpp
if (!isfinite(result)) {
    result = 0.0f;  // 或者报告错误
}
```

### 精度测试
```bash
python optimization_benchmark.py --phase precision --device xpu
```

**成功标准**:
- [ ] 最大误差 < 0.01
- [ ] 无NaN/Inf
- [ ] 性能不下降超过5%

---

## 完整测试流程

### 单次完整测试
```bash
# 1. 基线测试
python optimization_benchmark.py --phase baseline --device xpu

# 2. XMX优化后测试
# [编辑代码...]
python optimization_benchmark.py --phase xmx --device xpu

# 3. 对比结果
python optimization_benchmark.py --compare
```

### 自动化回归测试
```bash
#!/bin/bash
# run_all_tests.sh

PHASES=("baseline" "xmx" "memory" "workgroup")

for phase in "${PHASES[@]}"; do
    echo "Testing phase: $phase"
    
    # 编译
    if [ "$phase" != "baseline" ]; then
        cd operators && python setup.py build_ext --inplace && cd ..
    fi
    
    # 测试
    python optimization_benchmark.py --phase $phase --device xpu
    
    # 检查是否通过
    if [ $? -ne 0 ]; then
        echo "FAILED: $phase"
        exit 1
    fi
done

# 最终对比
python optimization_benchmark.py --compare
```

---

## 性能目标检查表

| 阶段 | 目标TFLOPS | 相对于基线 | 状态 |
|------|-----------|-----------|------|
| 基线 | 15.7 | 1.0x | 当前 |
| XMX优化 | 35-45 | 2.5x | 待测试 |
| 内存优化 | 50-60 | 3.5x | 待测试 |
| Work-group优化 | 55-65 | 4.0x | 待测试 |
| **总体目标** | **≥47** | **≥3.0x** | **≥60% L20** |

---

## 调试技巧

### 1. 性能分析
```bash
# Intel VTune
vtune -collect gpu-hotspots -result-dir vtune_results -- python test.py

# 查看内存带宽
vtune -collect memory-access -result-dir mem_results -- python test.py
```

### 2. 精度调试
```python
# 打印中间结果
print(f"Q[0,0,:5] = {q[0,0,:5]}")
print(f"S[0,0,:5] = {scores[0,0,:5]}")
print(f"Max score = {scores.max()}")
print(f"Min score = {scores.min()}")

# 检查NaN位置
nan_mask = torch.isnan(output)
if nan_mask.any():
    print(f"NaN at positions: {nan_mask.nonzero()}")
```

### 3. 基准对比
```python
# 与PyTorch原生对比
import torch.nn.functional as F

# PyTorch reference
scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(d)
attn = F.softmax(scores, dim=-1)
ref_output = torch.matmul(attn, v)

# 对比
diff = (output - ref_output).abs()
print(f"Max diff: {diff.max()}")
print(f"Mean diff: {diff.mean()}")
```

---

## 下一步行动

1. **在目标环境运行基线测试**
   ```bash
   python optimization_benchmark.py --phase baseline --device xpu
   ```

2. **分析基线结果**
   - 记录当前TFLOPS
   - 确认准确度是否在可接受范围
   - 标记需要优化的热点

3. **开始XMX优化**
   - 编辑 `flash_attention_xmx.cpp`
   - 重新编译并测试
   - 验证性能提升和准确度保持

4. **迭代优化**
   - 每步优化后运行完整测试
   - 记录性能数据
   - 对比目标检查表

---

## 结果记录模板

```markdown
## Phase X: [优化名称]
**日期**: YYYY-MM-DD  
**修改**: [简述修改内容]

### 性能结果
| Config | 基线TFLOPS | 优化后TFLOPS | 加速比 |
|--------|-----------|-------------|--------|
| Wan2.1 | 15.7 | XX.X | X.Xx |

### 准确度结果
| Config | 最大误差 | 状态 |
|--------|---------|------|
| Test 1 | X.XXXX | ✓/✗ |

### 结论
- [ ] 达到性能目标
- [ ] 准确度可接受
- [ ] 无NaN/Inf

### 下一步
[描述下一步优化]
```

---

**开始测试吧！运行基线测试后，告诉我结果，我们继续优化。**
