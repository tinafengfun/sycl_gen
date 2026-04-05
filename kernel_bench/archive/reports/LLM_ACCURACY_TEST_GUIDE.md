# LLM Accuracy Test Agent - 完整使用指南

## 🎯 概述

LLM Accuracy Test Agent是一个**完全自动化、LLM驱动**的CUDA-to-SYCL转换准确度测试系统。

### 核心特性

- ✅ **全自动**: 从平台检测到报告生成全程自动化
- ✅ **LLM驱动**: 无硬编码，所有判断和生成由LLM完成
- ✅ **并行执行**: 支持4个LLM并发、1个CUDA、1个SYCL
- ✅ **自动修复**: 编译失败时自动用LLM修复（最多3次重试）
- ✅ **全面覆盖**: 13个测试配置，覆盖3种数据类型、4种维度、极端值
- ✅ **详细报告**: JSON格式报告，包含决策支持信息
- ✅ **进度监控**: 实时显示测试进度和阶段

## 📦 组件清单

```
tools/
├── platform_detector.py          # 平台能力检测
├── test_suite_generator.py       # 测试套件生成（13个配置）
├── llm_harness_generator.py      # LLM测试harness生成
├── async_test_executor.py        # 异步执行器+进度监控
├── json_report_generator.py      # JSON报告生成
└── llm_accuracy_test_agent.py    # 主Agent（集成所有组件）
```

## 🚀 快速开始

### 方法1: 使用便捷函数（推荐）

```python
import asyncio
from tools.llm_accuracy_test_agent import run_accuracy_test

async def main():
    result = await run_accuracy_test(
        kernel_id="copy_type_converted",
        cuda_file="kernel_dataset/cuda/copy_type_converted_kernel.cu",
        sycl_file="kernel_dataset/sycl/copy_type_converted_kernel.dp.cpp",
        output_dir="results/reports",
        max_llm_concurrency=4
    )
    
    # 查看结果
    print(f"Pass rate: {result['summary']['pass_rate']*100:.1f}%")
    print(f"Decision: {result['decision_support']['conversion_quality']['verdict']}")

asyncio.run(main())
```

### 方法2: 使用Agent类（更多控制）

```python
import asyncio
from tools.llm_accuracy_test_agent import LLMAccuracyTestAgent

async def main():
    # 创建Agent
    agent = LLMAccuracyTestAgent(
        kernel_id="my_kernel",
        max_llm_concurrency=4
    )
    
    # 运行测试
    result = await agent.run_full_accuracy_test(
        cuda_file="path/to/cuda.cu",
        sycl_file="path/to/sycl.dp.cpp",
        output_dir="results/reports"
    )
    
    if result.success:
        print(f"✅ Test completed in {result.duration_seconds:.1f}s")
        print(f"Report: {result.report['metadata']['test_timestamp']}")
    else:
        print(f"❌ Test failed: {result.error}")

asyncio.run(main())
```

### 方法3: 命令行使用

```bash
# 基本用法
python3 tools/llm_accuracy_test_agent.py \
    copy_type_converted \
    kernel_dataset/cuda/copy_type_converted_kernel.cu \
    kernel_dataset/sycl/copy_type_converted_kernel.dp.cpp

# 指定输出目录和并发数
python3 tools/llm_accuracy_test_agent.py \
    my_kernel \
    cuda/my_kernel.cu \
    sycl/my_kernel.dp.cpp \
    -o results/my_reports \
    -c 2
```

## 📊 测试覆盖

### 数据类型测试（根据平台能力动态调整）

| 数据类型 | 测试数量 | 说明 |
|---------|---------|------|
| float32 | 8 | 总是测试（标准、边界、特殊、大值、维度变化、极端值） |
| float16 | 3 | SYCL支持时测试 |
| bfloat16 | 2 | 双平台都支持时测试 |

**总计**: 13个测试配置（根据平台能力动态生成）

### 测试类别

1. **标准测试**: 随机数据、边界值、特殊值（inf/nan）
2. **维度测试**: 方形、矩形、单维度、非对齐
3. **极端值测试**: 零值、数学边界
4. **精度测试**: bf16精度损失测试

### 容差配置

```python
float32:  abs=1e-5, rel=1e-4
bfloat16: abs=1e-3, rel=1e-2
float16:  abs=1e-3, rel=1e-2
large:    abs=1e-4, rel=1e-3  (大数值放宽)
```

## 📈 输出报告

### 报告结构

```json
{
  "metadata": {
    "kernel_id": "copy_type_converted",
    "kernel_name": "copyTypeConverted",
    "test_timestamp": "2024-03-08T12:34:56Z",
    "total_duration_seconds": 485.3,
    "framework_version": "1.0.0"
  },
  
  "platform": {
    "sycl": {...},
    "cuda": {...}
  },
  
  "test_results": [...],
  
  "summary": {
    "total_tests": 13,
    "passed": 12,
    "failed": 0,
    "skipped": 1,
    "pass_rate": 0.923,
    "coverage": {...}
  },
  
  "decision_support": {
    "conversion_quality": {
      "score": "A",
      "verdict": "EXCELLENT"
    },
    "deployment_readiness": {
      "ready": true,
      "confidence": "HIGH"
    },
    "risks": [...],
    "next_steps": [...]
  }
}
```

### 决策支持

报告自动生成决策建议：

- **质量评分**: A/B/C/D
- **部署准备度**: READY/NON-READY + 置信度
- **风险评估**: 识别潜在问题
- **下一步建议**: 具体行动项

## 🔧 高级用法

### 自定义测试配置

```python
from tools.test_suite_generator import TestSuiteGenerator

# 自定义平台能力
platform_caps = {
    "sycl": {"float32": True, "float16": True, "bfloat16": False},
    "cuda": {"float32": True, "float16": True, "bfloat16": True}
}

# 生成测试套件
generator = TestSuiteGenerator(platform_caps)
configs = generator.generate_full_suite()

# 自定义测试数据
test_data = generator.get_test_data_generator(configs[0])
```

### 手动执行单个测试

```python
from tools.llm_harness_generator import LLMHarnessGenerator
from tools.async_test_executor import AsyncTestExecutor, ProgressMonitor

# 生成harness
generator = LLMHarnessGenerator()
harness = await generator.generate_full_harness(
    kernel_code=cuda_code,
    test_config=test_config
)

# 执行测试
monitor = ProgressMonitor()
executor = AsyncTestExecutor(monitor)
result = await executor.execute_test(
    test_id="my_test",
    kernel_code=cuda_code,
    test_config=test_config
)
```

### 生成报告

```python
from tools.json_report_generator import JSONReportGenerator

generator = JSONReportGenerator("kernel_id", "kernel_name")

# 添加trace
generator.add_trace("test_started")

# 添加问题
generator.add_issue(
    severity="warning",
    category="nan_behavior",
    message="NaN handling differs",
    recommendation="Consider explicit NaN checks"
)

# 生成报告
report = generator.generate_report(
    platform_info=platform_caps,
    test_configs=configs,
    test_results=results,
    llm_usage=llm_stats
)

# 保存报告
filepath = generator.save_report(report, "results/reports")
```

## 🐛 故障排除

### LLM调用失败

```python
# 减少并发数
result = await run_accuracy_test(
    ...,
    max_llm_concurrency=2  # 从4减少到2
)
```

### CUDA/SYCL执行失败

检查：
1. Docker容器是否运行: `docker ps | grep cuda`, `docker ps | grep lsv`
2. 远程主机是否可达: `ping 10.112.229.160`
3. SSH密钥是否配置

### 编译错误

系统会自动用LLM修复（最多3次），如果仍然失败：
1. 检查kernel代码是否完整
2. 查看详细错误信息在报告中
3. 手动检查CUDA/SYCL代码兼容性

## 📊 性能预期

- **单测试时间**: 30-60秒（取决于LLM生成速度）
- **完整套件**: 5-10分钟（13个测试串行执行）
- **LLM调用**: 每个测试约2-5次（生成+可能的修复）
- **总Token消耗**: 约50K-100K tokens per kernel

## 🎯 最佳实践

1. **先小规模测试**: 先用1-2个简单kernel验证流程
2. **监控资源**: 确保GPU和LLM API资源充足
3. **查看报告**: 仔细分析报告中的warnings和recommendations
4. **保存报告**: 所有报告自动保存，便于后续对比

## 📞 支持

如有问题，请检查：
1. 所有依赖模块是否正确导入（运行verify脚本）
2. 平台能力检测是否成功
3. LLM API是否可用

## 📝 更新日志

**v1.0.0** (2024-03-08)
- 初始版本
- 完整的LLM驱动测试框架
- 13个测试配置
- 自动编译错误修复
- JSON报告和决策支持
