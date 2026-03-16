# CUDA-to-SYCL Agent 系统完整执行计划
# Complete Execution Plan for Agent System Enhancement

**版本**: 1.0  
**创建日期**: 2026-03-04  
**执行模式**: Build Mode  
**总工期**: 6天  

---

## 📋 执行概览

### 核心目标
按照 **1→2→3** 顺序完善Agent系统，使用opencode完成模型生成，覆盖全部30个kernel测试。

### 执行原则
- ✅ 最小人工干预
- ✅ 使用opencode完成代码生成
- ✅ 全部30个kernel测试覆盖
- ✅ 提供完整部署手册

---

## Phase 1: UnifiedAccuracyTester (2天)

### Day 1 任务

#### 上午: 代码集成 (4小时)
- [ ] 分析现有 `accuracy_tester.py` (407行)
- [ ] 分析 `run_accuracy_test.py` 工作流程
- [ ] 重构 `UnifiedAccuracyTester` 类
- [ ] 迁移测试数据生成函数
- [ ] 迁移CUDA执行函数
- [ ] 迁移SYCL执行函数

#### 下午: 功能实现 (4小时)
- [ ] 实现并行执行逻辑
- [ ] 集成数值对比算法
- [ ] 添加错误处理和超时机制
- [ ] 集成Trace日志系统

### Day 2 任务

#### 上午: 测试验证 (4小时)
- [ ] 测试 winograd kernel
- [ ] 验证通过率 >= 99.9%
- [ ] 修复集成问题
- [ ] 添加边界情况处理

#### 下午: 文档和模板 (4小时)
- [ ] 创建 `prompts/phase1_accuracy_tester.md`
- [ ] 编写 Phase 1 完成报告
- [ ] 更新主计划状态

### 交付物
1. ✅ `unified_converter.py` (Phase 1更新版)
2. ✅ `prompts/phase1_accuracy_tester.md`
3. ✅ Phase 1 测试报告

---

## Phase 2: UnifiedReporter (1天)

### 单一工作日任务

#### 上午: 核心实现 (4小时)
- [ ] 创建 `UnifiedReporter` 类
- [ ] 实现 JSON 报告生成
- [ ] 实现 HTML 报告生成
- [ ] 实现 Markdown 报告生成

#### 下午: 集成和优化 (4小时)
- [ ] 集成到 Phase 5 (run_phase5_reporting)
- [ ] 收集所有 Phase 数据
- [ ] 性能统计实现
- [ ] 创建 `prompts/phase2_reporter.md`

### 交付物
1. ✅ `unified_converter.py` (添加 UnifiedReporter)
2. ✅ `prompts/phase2_reporter.md`
3. ✅ 示例报告文件 (JSON/HTML/Markdown)

---

## Phase 3: UnifiedConverter (2天)

### Day 1 任务

#### 上午: 框架搭建 (4小时)
- [ ] 分析手动生成的 winograd 代码
- [ ] 识别转换模式和规则
- [ ] 创建 `ModelBasedConverter` 框架
- [ ] 设计提示词模板结构

#### 下午: 提示词工程 (4小时)
- [ ] 编写 System Prompt
- [ ] 编写 User Prompt Template
- [ ] 添加错误处理示例
- [ ] 创建 `prompts/phase3_converter.md`

### Day 2 任务

#### 上午: 集成实现 (4小时)
- [ ] 集成模型生成到统一工作流
- [ ] 实现 fallback 机制 (规则替换)
- [ ] 添加快速验证逻辑
- [ ] 实现智能选择逻辑

#### 下午: 测试优化 (4小时)
- [ ] 测试 3 个简单 kernel
- [ ] 验证编译通过率
- [ ] 验证准确度 > 99%
- [ ] 优化提示词模板

### 交付物
1. ✅ `unified_converter.py` (完整增强版)
2. ✅ `prompts/phase3_converter.md`
3. ✅ 提示词模板文件 (system.txt, user.txt)

---

## 批量处理30个Kernel (1天)

### 任务清单

#### 上午: 批量脚本 (4小时)
- [ ] 创建 `tools/batch_convert.py`
- [ ] 实现并行处理逻辑
- [ ] 添加进度报告生成
- [ ] 测试批量处理流程

#### 下午: 执行和报告 (4小时)
- [ ] 执行全部30个kernel转换
- [ ] 收集所有结果数据
- [ ] 生成完成度报告
- [ ] 分析失败原因

### 交付物
1. ✅ `tools/batch_convert.py`
2. ✅ `results/CONVERSION_COMPLETION_REPORT.md`
3. ✅ 30个kernel的转换结果

---

## 文档编写 (并行进行)

### 文档清单

- [ ] `docs/OPENCODE_DEPLOYMENT_GUIDE.md`
  - 快速开始指南
  - 详细使用说明
  - API参考
  - 故障排除

- [ ] `docs/AGENT_IMPLEMENTATION_GUIDE.md`
  - 每个Agent的详细说明
  - 架构设计
  - 接口定义

- [ ] `.opencode/config.yaml`
  - 配置文件模板
  - 默认参数设置

---

## 文件结构规划

```
opencode_bench/
├── .opencode/
│   ├── plans/
│   │   ├── UNIFIED_AGENT_V3.md          (已有)
│   │   ├── AGENT_TASK_SPEC_TRACED.md    (已有)
│   │   └── EXECUTION_PLAN.md            (本文件)
│   └── config.yaml                      (新建)
│
├── prompts/
│   ├── phase1_accuracy_tester.md        (Phase 1)
│   ├── phase2_reporter.md               (Phase 2)
│   └── phase3_converter.md              (Phase 3)
│
├── tools/
│   ├── unified_converter.py             (最终版)
│   ├── batch_convert.py                 (批量处理)
│   ├── b60_sycl_builder.py              (已有-BMG版)
│   └── accuracy_tester.py               (已有)
│
├── docs/
│   ├── OPENCODE_DEPLOYMENT_GUIDE.md     (部署手册)
│   ├── AGENT_IMPLEMENTATION_GUIDE.md    (实现指南)
│   └── UPDATED_TOOLS_SUMMARY.md         (已有)
│
├── results/
│   ├── CONVERSION_COMPLETION_REPORT.md  (完成度报告)
│   └── {session_id}/                    (各次执行结果)
│
└── test/
    └── accuracy/
        └── run_accuracy_test.py         (已有)
```

---

## 执行检查清单

### Phase 1 检查点
- [ ] UnifiedAccuracyTester.test() 调用真实执行
- [ ] 5种测试配置都能正常工作
- [ ] 并行执行CUDA和SYCL
- [ ] 数值对比误差 < 1e-5
- [ ] winograd kernel 测试通过
- [ ] Trace日志完整记录

### Phase 2 检查点
- [ ] UnifiedReporter 独立Agent
- [ ] JSON报告包含完整数据
- [ ] HTML报告美观可读
- [ ] Markdown报告简洁明了
- [ ] 集成到 Phase 5
- [ ] 性能统计准确

### Phase 3 检查点
- [ ] ModelBasedConverter 框架
- [ ] 提示词模板可用
- [ ] 模型生成代码可编译
- [ ] fallback机制工作正常
- [ ] 3个简单kernel测试通过
- [ ] 准确度 > 99%

### 批量处理检查点
- [ ] batch_convert.py 可运行
- [ ] 30个kernel全部处理
- [ ] 成功率 > 90%
- [ ] 完成度报告生成
- [ ] 失败原因分析

### 文档检查点
- [ ] 部署手册完整
- [ ] API文档清晰
- [ ] 使用示例可运行
- [ ] 故障排除有效

---

## 风险管理

### 高风险项
1. **模型生成质量不稳定**
   - 缓解: 提供详细的提示词模板
   - 缓解: 保留规则替换作为fallback
   - 缓解: 充分测试后再批量处理

2. **30个kernel处理时间过长**
   - 缓解: 实现并行处理
   - 缓解: 分批次执行
   - 缓解: 设置超时机制

3. **准确度测试环境配置复杂**
   - 缓解: 复用已有验证的脚本
   - 缓解: 提供详细的配置说明
   - 缓解: 容器化环境

### 中风险项
1. **Trace日志过大**
   - 缓解: 实现日志轮转
   - 缓解: 压缩旧日志

2. **报告生成性能问题**
   - 缓解: 异步生成
   - 缓解: 缓存中间结果

---

## 每日更新要求

### 更新内容
1. 完成百分比
2. 遇到的问题
3. 解决方案
4. 明日计划
5. 风险评估

### 更新位置
- `.opencode/plans/EXECUTION_LOG.md`
- 或者作为本文件的附录

---

## 成功标准

### 功能标准
- ✅ 5个Phase全部可执行
- ✅ UnifiedAccuracyTester 真实测试
- ✅ UnifiedReporter 多格式报告
- ✅ UnifiedConverter 模型生成
- ✅ 30个kernel处理完成

### 性能标准
- ✅ 编译成功率 > 95%
- ✅ 准确度通过率 > 99%
- ✅ 平均处理时间 < 30分钟/kernel

### 文档标准
- ✅ 部署手册完整可用
- ✅ 提示词模板清晰有效
- ✅ 完成度报告详细准确

---

## 附录: 提示词模板预览

### Phase 1 提示词核心
```markdown
将 accuracy_tester.py 功能集成到 UnifiedAccuracyTester...
要求:
1. 真实执行CUDA/SYCL
2. 并行执行对比
3. 数值误差 < 1e-5
4. Trace日志集成
```

### Phase 2 提示词核心
```markdown
创建 UnifiedReporter Agent...
要求:
1. 3种报告格式
2. 性能统计
3. 集成到Phase 5
```

### Phase 3 提示词核心
```markdown
实现 ModelBasedConverter...
要求:
1. opencode模型生成
2. 详细提示词模板
3. fallback机制
```

---

**计划状态**: ✅ 已批准  
**执行开始**: 立即  
**下次更新**: Day 1 结束

---

**注意**: 本计划是动态文档，执行过程中可根据实际情况调整。
