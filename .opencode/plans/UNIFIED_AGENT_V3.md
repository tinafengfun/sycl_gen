---
name: cuda-to-sycl-converter-unified
description: 统一优化的CUDA-to-SYCL转换Agent系统。集成Analyzer、Converter、Validator、Fixer、AccuracyTester、Reporter六大Agent，全自动完成代码转换、编译验证和准确度测试。
license: MIT
compatibility: opencode
metadata:
  task_type: unified_agentic_workflow
  version: "3.0"
  agents: 6
  phases: 5
  trace_enabled: true
  accuracy_test_enabled: true
---

# CUDA-to-SYCL 统一转换Agent系统 v3.0

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│              UnifiedOrchestrator (主控Agent)                 │
│                    统一调度与状态管理                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
    ┌──────────────┼──────────────┬──────────────┬──────────────┐
    ▼              ▼              ▼              ▼              ▼
┌────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐   ┌──────────┐
│Analyzer│   │Converter │   │Validator │   │ Fixer  │   │Accuracy  │
│  (分析) │   │  (转换)   │   │ (编译验证)│   │(修复)  │   │Tester   │
└────┬───┘   └─────┬────┘   └─────┬────┘   └───┬────┘   │(准确度)  │
     │             │              │            │          └────┬─────┘
     └─────────────┴──────────────┴────────────┘               │
                   │                                          │
                   └──────────────────┬───────────────────────┘
                                      │
                               ┌──────┴──────┐
                               │  Reporter   │
                               │  (报告生成)  │
                               └─────────────┘
```

## Agent职责与协作流程

### 1. UnifiedOrchestrator (主控Agent)

**核心职责**:
- 初始化整个系统和Trace
- 管理全局状态机
- 协调各Agent执行顺序
- 监控循环和异常
- 决策重试/跳过/终止
- 汇总最终结果

**状态管理**:
```python
class UnifiedOrchestrator:
    def __init__(self, kernel_id):
        self.state = ConversionState(kernel_id)
        self.tracer = UnifiedTracer(kernel_id)
        
    async def execute_conversion(self):
        # Phase 1: 分析
        analysis = await self.run_analyzer()
        
        # Phase 2: 转换
        sycl_code = await self.run_converter(analysis)
        
        # Phase 3: 编译验证 (含自动修复)
        build_result = await self.run_validation_loop(sycl_code)
        
        # Phase 4: 准确度测试
        accuracy_result = await self.run_accuracy_test(build_result)
        
        # Phase 5: 报告
        return await self.generate_final_report()
```

### 2. UnifiedAnalyzer (统一分析Agent)

**输入**: CUDA源文件
**输出**: AnalysisReport (包含复杂度、策略、风险点)

**优化点**:
- 缓存分析结果，避免重复读取
- 并行分析多个组件
- 生成详细的转换策略树

### 3. UnifiedConverter (统一转换Agent)

**输入**: AnalysisReport + CUDA代码
**输出**: 初始SYCL代码

**优化点**:
- 基于策略树选择性应用规则
- 模板预编译加速
- 增量生成，支持断点续传

### 4. UnifiedValidator (统一验证Agent)

**输入**: SYCL代码
**输出**: 编译结果 + 错误列表

**集成Fixer**:
- 实时错误检测
- 自动修复应用
- 重试循环管理
- 最大5次自动修复

### 5. UnifiedAccuracyTester (统一准确度Agent) ⭐

**输入**: CUDA输出 (baseline) + SYCL输出 (target)
**输出**: 准确度报告

**核心功能**:
- 自动生成多样化测试数据
- 并行执行CUDA和SYCL
- 数值精度对比
- 自动判定通过/失败

### 6. UnifiedReporter (统一报告Agent)

**输入**: 所有阶段结果
**输出**: 综合报告

**报告内容**:
- 执行时间线
- Agent性能统计
- Trace日志汇总
- 准确度验证结果
- 可优化建议

## 优化执行流程 (5个Phase)

### Phase 1: 智能分析 (Optimized Analysis)

**Duration**: 3-5 minutes

```python
async def phase1_analysis(self):
    # 并行分析
    tasks = [
        self.analyze_structure(),      # 代码结构
        self.analyze_patterns(),       # CUDA模式
        self.analyze_complexity(),     # 复杂度
        self.generate_strategy()       # 策略
    ]
    
    results = await asyncio.gather(*tasks)
    
    # 智能缓存
    if self.cache_hit(kernel_id):
        return self.load_from_cache()
    
    analysis = self.merge_results(results)
    self.save_to_cache(analysis)
    
    return analysis
```

### Phase 2: 智能转换 (Optimized Conversion)

**Duration**: 5-10 minutes

```python
async def phase2_conversion(self, analysis):
    # 基于策略选择转换规则
    rules = self.select_rules(analysis.strategy)
    
    # 增量生成
    if self.partial_result_exists():
        code = self.load_partial()
        code = self.continue_from_checkpoint(code)
    else:
        code = self.generate_from_scratch(analysis)
    
    # 实时验证语法
    if not self.quick_syntax_check(code):
        raise ConversionError("Syntax validation failed")
    
    return code
```

### Phase 3: 编译验证 + 自动修复 (Validation + Auto-Fix Loop)

**Duration**: 10-20 minutes

```python
async def phase3_validation_loop(self, sycl_code):
    max_attempts = 5
    
    for attempt in range(1, max_attempts + 1):
        self.log_attempt(attempt)
        
        # 编译
        result = await self.compile(sycl_code)
        
        if result.success:
            self.log_success(attempt)
            return result
        
        # 分析错误
        errors = self.parse_errors(result)
        
        # 尝试自动修复
        fixes = await self.apply_auto_fixes(errors)
        
        if fixes:
            sycl_code = self.apply_fixes(sycl_code, fixes)
            self.log_fixes_applied(fixes)
        else:
            if attempt >= max_attempts:
                self.log_max_retries_reached()
                raise MaxRetriesExceeded(errors)
    
    return result
```

### Phase 4: 准确度验证 (Accuracy Validation) ⭐

**Duration**: 10-15 minutes

```python
async def phase4_accuracy_test(self, sycl_binary):
    # 生成测试数据
    test_data = self.generate_comprehensive_tests()
    
    # 并行执行
    cuda_task = self.run_cuda(test_data)
    sycl_task = self.run_sycl(test_data)
    
    cuda_result, sycl_result = await asyncio.gather(cuda_task, sycl_task)
    
    # 对比结果
    comparison = self.compare_results(cuda_result, sycl_result)
    
    # 判定
    verdict = self.determine_verdict(comparison)
    
    return {
        "status": verdict.status,  # PASS / WARN / FAIL
        "pass_rate": verdict.pass_rate,
        "max_error": verdict.max_error,
        "details": comparison
    }
```

### Phase 5: 统一报告 (Unified Reporting)

**Duration**: 1-2 minutes

```python
async def phase5_reporting(self, all_results):
    # 收集所有数据
    report_data = {
        "analysis": all_results.phase1,
        "conversion": all_results.phase2,
        "validation": all_results.phase3,
        "accuracy": all_results.phase4,
        "trace": self.tracer.get_full_log(),
        "performance": self.collect_performance_metrics()
    }
    
    # 生成多格式报告
    reports = await asyncio.gather(
        self.generate_json_report(report_data),
        self.generate_html_report(report_data),
        self.generate_markdown_summary(report_data)
    )
    
    return reports
```

## 关键优化特性

### 1. 智能缓存系统

```python
class SmartCache:
    def __init__(self):
        self.analysis_cache = {}
        self.conversion_cache = {}
        self.test_results_cache = {}
    
    def get_analysis(self, kernel_id, file_hash):
        key = f"{kernel_id}:{file_hash}"
        if key in self.analysis_cache:
            return self.analysis_cache[key]
        return None
    
    def should_reuse_conversion(self, kernel_id, partial_code):
        # 检查上次转换进度
        if self.has_checkpoint(kernel_id):
            return True
        return False
```

### 2. 并行执行优化

```python
async def parallel_execution(self):
    # Level 1/2 kernels 可以并行
    level1_kernels = self.get_kernels_by_level(1)
    level2_kernels = self.get_kernels_by_level(2)
    
    # 并行处理多个kernel
    tasks = []
    for kernel in level1_kernels[:8]:  # 最多8个并行
        tasks.append(self.convert_kernel(kernel))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results
```

### 3. 增量转换支持

```python
class IncrementalConverter:
    def save_checkpoint(self, kernel_id, code, progress):
        checkpoint = {
            "timestamp": datetime.now(),
            "code": code,
            "progress": progress,
            "completed_rules": self.applied_rules
        }
        self.checkpoint_db.save(kernel_id, checkpoint)
    
    def resume_from_checkpoint(self, kernel_id):
        checkpoint = self.checkpoint_db.load(kernel_id)
        return checkpoint.code, checkpoint.progress
```

### 4. 准确度测试集成

```python
class IntegratedAccuracyTester:
    async def run_integrated_test(self, kernel_id):
        # 复用编译后的二进制
        cuda_binary = self.get_compiled_cuda(kernel_id)
        sycl_binary = self.get_compiled_sycl(kernel_id)
        
        # 共享测试数据生成
        test_data = self.generate_test_suite(kernel_id)
        
        # 并行执行对比
        results = await self.execute_comparison(
            cuda_binary, sycl_binary, test_data
        )
        
        # 实时结果分析
        analysis = self.analyze_results_realtime(results)
        
        return analysis
```

## Trace集成 (统一追踪)

### Trace数据流

```
All Agents -> UnifiedTracer -> .traces/sessions/{session_id}/
                                    ├── unified_trace.json
                                    ├── agent_timeline.json
                                    ├── accuracy_results.json
                                    └── performance_stats.json
```

### 关键Trace点

1. **Phase Transitions**: 阶段切换
2. **Agent Handoffs**: Agent间传递
3. **Tool Calls**: 工具调用 (含重复检测)
4. **Error Events**: 错误和修复
5. **Accuracy Checks**: 准确度验证
6. **Decision Points**: 决策点

## 执行入口

### 启动完整流程

```bash
# 启动统一Agent系统
python3 tools/unified_converter.py \
  --kernel winograd_input_transform \
  --mode full \
  --accuracy-test \
  --trace

# 批量转换
python3 tools/unified_converter.py \
  --batch \
  --level 1,2 \
  --parallel 8 \
  --accuracy-test
```

### 在opencode中使用

```python
# 启动统一转换任务
/unified-convert kernel=winograd_input_transform accuracy=true trace=true

# 查看实时状态
/unified-status session={session_id}

# 查看准确度报告
/unified-accuracy-report kernel={kernel_id}
```

## 成功指标 v3.0

### 转换质量
- ✅ 编译成功率: >=95%
- ✅ **准确度通过率: >=99.9%** ⭐
- ✅ 零人工干预率: >=90%

### 性能指标
- ⚡ 平均转换时间: <30分钟/kernel
- ⚡ 准确度测试时间: <15分钟/kernel
- ⚡ 并行效率: >=80%

### 可靠性
- 🛡️ 自动修复成功率: >=85%
- 🛡️ 断点续传成功率: 100%
- 🛡️ Trace完整性: 100%

---

**版本**: 3.0 | **Agent数**: 6 | **Phase数**: 5 | **更新**: 2026-03-04
