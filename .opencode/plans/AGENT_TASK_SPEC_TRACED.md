---
name: cuda-to-sycl-converter-agent
description: 全自动CUDA-to-SYCL Kernel转换Agent Task，带完整Trace日志、循环检测、异常追踪和人工干预记录
license: MIT
compatibility: opencode
metadata:
  task_type: agentic_workflow
  version: "2.0"
  automation_level: fully_autonomous
  human_intervention: minimal
  trace_enabled: true
---

# CUDA-to-SYCL 全自动转换 Agent Task (with Trace)

## 任务概述

- **任务ID**: cuda-sycl-converter-v2-traced
- **执行模式**: 全自动Agentic Workflow + 完整Trace
- **人工干预**: 仅在连续失败5次后暂停
- **预期时间**: 2-4小时 (30个kernel)
- **试点Kernel**: winograd_input_transform (Level 3, 216行)

## Trace日志系统

### Trace架构

```
.traces/
├── sessions/
│   └── {session_id}/
│       ├── trace.json              # 主trace文件
│       ├── agent_steps.jsonl       # Agent步骤日志 (逐行JSON)
│       ├── tool_calls.json         # Tool调用统计
│       ├── patterns.json           # 循环pattern检测
│       ├── exceptions.json         # 异常记录
│       └── interventions.json      # 人工干预
└── aggregations/
    └── session_stats.json          # 会话统计
```

### Trace记录内容

1. **Agent Steps**: 每个Agent的每个动作
2. **Tool Calls**: 工具调用次数、成功率、耗时
3. **Invalid Calls**: 无效调用、重复调用
4. **Loop Patterns**: 循环检测、无限循环风险
5. **Exceptions**: 所有异常和错误
6. **Interventions**: 人工干预记录

---

## Agent架构

### 1. TracedOrchestrator (主控Agent)

**职责**:
- 初始化Trace系统
- 调度子Agent
- 监控执行状态
- 检测循环和异常
- 决策重试/跳过/人工干预

**Trace Points**:
- Task start/end
- Phase transitions
- Agent invocations
- Decision points

### 2. TracedAnalyzer (分析Agent)

**职责**:
- 深度解析CUDA代码
- 识别CUDA特性和复杂度
- 生成转换策略

**Trace Points**:
- File read operations
- Pattern detection results
- Complexity assessment
- Strategy generation

### 3. TracedConverter (转换Agent)

**职责**:
- 生成SYCL代码
- 应用转换规则
- 处理特殊模式

**Trace Points**:
- Rule applications
- Code generation steps
- Template substitutions

### 4. TracedValidator (编译验证Agent)

**职责**:
- 编译验证
- 错误分析
- 调用Fixer修复

### 5. TracedAccuracyTester (准确度测试Agent) ⭐ NEW

**职责**:
- 生成测试数据
- 执行CUDA和SYCL版本
- 对比输出结果
- 验证数值精度

**Trace Points**:
- Test data generation
- CUDA execution
- SYCL execution
- Result comparison
- Accuracy metrics

**准确度指标**:
- **绝对误差**: < 1e-5
- **相对误差**: < 1e-4
- **通过率**: >= 99.9%
- **测试覆盖率**: 边界值、随机值、特殊值

**Trace Points**:
- Build attempts
- Error classifications
- Fix applications
- Retry loops

### 5. TracedFixer (修复Agent)

**职责**:
- 自动修复常见错误
- 应用修复规则
- 验证修复效果

**Trace Points**:
- Error pattern matching
- Fix rule applications
- Success/failure tracking

---

## 自动执行流程

### Phase 0: 初始化 + Trace启动

**Duration**: 2 minutes

**Steps**:
1. 创建Trace会话
2. 检查环境连通性
3. 加载kernel配置
4. 初始化状态追踪

**Trace Output**:
```json
{
  "session_id": "winograd-conv-20260303-110000",
  "task": "winograd_input_transform_conversion",
  "start_time": "2026-03-03T11:00:00Z",
  "trace_version": "2.0"
}
```

### Phase 1: CUDA深度分析

**Duration**: 5-10 minutes

**Agent**: TracedAnalyzer

**Analysis**:
- 216行代码解析
- 识别: matrixMul_gpu_serial, InputTransform4x4, InputTransform_kernel
- 检测: __device__, __global__, 宏定义, 模板特化
- 评估: Level 3复杂度 (Winograd算法)

**Trace Output**:
- File reads: ~3-5次
- Pattern detections: ~10-15个
- Analysis duration: ~300s

### Phase 2: SYCL代码生成

**Duration**: 10-15 minutes

**Agent**: TracedConverter

**Conversion**:
- 应用13个基础映射规则
- 处理Winograd特殊逻辑
- 生成generated_v1.dp.cpp

**Trace Output**:
- Rules applied: ~20-30个
- Code generation steps: ~50步
- Output file: conversion/winograd/generated_v1.dp.cpp

### Phase 3: 编译验证循环

**Duration**: 15-30 minutes (含重试)

**Agent**: TracedValidator + TracedFixer

**Loop**:
```
Attempt 1: Compile -> Error detected -> Auto-fix
Attempt 2: Compile -> Error detected -> Auto-fix
Attempt 3: Compile -> Success
```

**Trace Output**:
- Build attempts: 1-5次
- Errors detected: N个
- Fixes applied: M个
- Success: boolean

### Phase 4: 编译验证循环

**Duration**: 15-30 minutes (含重试)

**Agent**: TracedValidator + TracedFixer

**Loop**:
```
Attempt 1: Compile -> Error detected -> Auto-fix
Attempt 2: Compile -> Error detected -> Auto-fix
Attempt 3: Compile -> Success
```

**Trace Output**:
- Build attempts: 1-5次
- Errors detected: N个
- Fixes applied: M个
- Success: boolean

### Phase 5: 准确度验证 ⭐ NEW

**Duration**: 10-20 minutes

**Agent**: TracedAccuracyTester

**步骤**:
1. **测试数据生成**
   - 边界值: 0, 1, -1, min, max
   - 随机值: 正态分布、均匀分布
   - 特殊值: inf, nan, subnormal
   - 规模: 小规模(1-64), 大规模(1024-65536)

2. **执行CUDA版本** (Baseline)
   ```cpp
   // 编译CUDA测试程序
   nvcc -o test_cuda test_framework.cu
   
   // 运行并保存输出
   ./test_cuda > cuda_output.bin
   ```

3. **执行SYCL版本** (Target)
   ```cpp
   // 编译SYCL测试程序
   icpx -fsycl -o test_sycl test_framework.dp.cpp
   
   // 运行并保存输出
   ./test_sycl > sycl_output.bin
   ```

4. **数值对比**
   ```cpp
   // 逐元素比较
   for (i = 0; i < n; i++) {
     float diff = fabs(cuda_out[i] - sycl_out[i]);
     float rel_diff = diff / (fabs(cuda_out[i]) + 1e-8);
     
     if (diff > ABS_TOL && rel_diff > REL_TOL) {
       record_mismatch(i, cuda_out[i], sycl_out[i]);
     }
   }
   ```

**准确度阈值**:
```yaml
accuracy_thresholds:
  absolute_tolerance: 1.0e-5
  relative_tolerance: 1.0e-4
  min_pass_rate: 99.9  # percentage
  
  special_cases:
    fp16:  # FP16 has lower precision
      absolute_tolerance: 1.0e-3
      relative_tolerance: 1.0e-2
    
    atomic_operations:  # Order of operations may vary
      absolute_tolerance: 1.0e-4
      relative_tolerance: 1.0e-3
```

**Trace Output**:
- Test cases generated: N
- CUDA execution time: X ms
- SYCL execution time: Y ms
- Mismatches found: M
- Pass rate: P%
- Accuracy verdict: PASS/FAIL

### Phase 6: 完成 + 报告

**Duration**: 2 minutes

**Actions**:
- 保存最终文件
- 更新index.json
- 生成Trace报告
- 生成准确度报告 ⭐ NEW
- 输出执行摘要

---

## 决策逻辑

### 自动决策树

```yaml
on_compile_error:
  - condition: "error in auto_fix_rules"
    action: "auto_fix_and_retry"
    max_attempts: 5
    trace: log_fix_attempt
  
  - condition: "attempts >= 5"
    action: "pause_and_request_intervention"
    trace: log_intervention_required
    notify: human
  
  - condition: "error_type == 'critical'"
    action: "skip_kernel_continue"
    trace: log_skip

on_loop_detected:
  - condition: "cycle_count >= 3"
    action: "escalate_to_human"
    trace: log_loop_pattern
    alert: true

on_accuracy_test_failure: ⭐ NEW
  - condition: "mismatch_rate <= 1%"
    action: "accept_with_warning"
    trace: log_minor_accuracy_deviation
    note: "Small numerical differences acceptable for this kernel"
  
  - condition: "mismatch_rate > 1% and mismatch_rate <= 5%"
    action: "flag_for_review"
    trace: log_significant_accuracy_deviation
    notify: human
    note: "Accuracy deviation requires investigation"
  
  - condition: "mismatch_rate > 5%"
    action: "reject_and_rollback"
    trace: log_critical_accuracy_failure
    alert: true
    note: "Conversion failed accuracy requirements"

on_exception:
  - condition: "recoverable == true"
    action: "attempt_recovery"
    trace: log_recovery_attempt
  
  - condition: "recoverable == false"
    action: "pause_and_escalate"
    trace: log_critical_exception
```

---

## 工具调用规范

### Trace-enabled Tool Wrapper

```python
# 所有工具调用都经过Trace包装
tool_call = {
  "tool": "read",
  "params": {"file": "..."},
  "trace_id": "step_001_call_001",
  "timestamp": "2026-03-03T11:00:00.123Z"
}

# Trace记录:
# - Tool name
# - Input params
# - Output/Error
# - Duration
# - Success/Failure
# - Duplicate detection
```

### 可调用的Tools

1. **read** - 读取文件
2. **write** - 写入文件
3. **edit** - 编辑文件
4. **bash** - 执行shell命令
5. **skill** - 加载skill

---

## 成功指标

### 任务级指标
- ✅ 编译成功率: >=90%
- ✅ **准确度通过率: >=99.9%** ⭐ NEW
- ✅ 自动化率: >=95%
- ✅ 完成时间: <=4小时
- ✅ 异常率: <=5%

### 准确度指标 ⭐ NEW
```yaml
accuracy_requirements:
  numerical_precision:
    float32:
      absolute_tolerance: 1.0e-5
      relative_tolerance: 1.0e-4
      max_mismatches: 0.1%  # of total elements
    
    float16:
      absolute_tolerance: 1.0e-3
      relative_tolerance: 1.0e-2
      max_mismatches: 1.0%
  
  test_coverage:
    boundary_values: 100%   # 0, 1, -1, min, max
    random_values: 100%     # multiple distributions
    special_values: 100%    # inf, nan, subnormal
    scale_variants: 100%    # small, medium, large
  
  execution_consistency:
    cuda_baseline: stable
    sycl_output: deterministic
    cross_platform: verified
```

### Trace指标
- ✅ Tool Call成功率: >=95%
- ✅ 重复调用率: <=10%
- ✅ 循环检测准确率: >=90%
- ✅ **准确度测试通过率: >=99.9%** ⭐ NEW
- ✅ 人工干预次数: <=3次

---

## 执行指令

### 启动任务

```
启动带Trace的CUDA-to-SYCL转换任务
目标: winograd_input_transform kernel
```

### 查看实时Trace

```bash
# 查看当前执行状态
tail -f .traces/sessions/{session_id}/agent_steps.jsonl

# 查看统计
cat .traces/sessions/{session_id}/tool_calls.json

# 查看异常
cat .traces/sessions/{session_id}/exceptions.json
```

---

版本: 2.0-with-trace | 创建: 2026-03-03
