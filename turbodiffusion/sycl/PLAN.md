# TurboDiffusion-SYCL 完整迁移项目计划

**项目目标**: 将TurboDiffusion所有Triton算子迁移到SYCL，实现端到端视频生成，并与NVIDIA L20进行对比  
**项目周期**: 约20天  
**团队规模**: 2 Agents并行  
**质量要求**: 100%功能正确，性能≥60% CUDA  

---

## 📋 任务分配

### Agent 1: Flash架构师
**职责**: 
- 复用sycle-tla_internal Flash Attention
- 整体架构设计 (model_patcher.py)
- 集成所有kernel到TurboDiffusion
- 性能调优与NVIDIA对比

**交付物**:
- sycl_flash_attention_wrapper.cpp (PyTorch binding)
- model_patcher.py (自动注入框架)
- kernel_registry.py (算子注册管理)
- 集成测试与性能报告

### Agent 2: Sparse专家
**职责**:
- Sparse Attention SYCL实现 (Triton转SYCL)
- RoPE SYCL实现
- 精度验证与误差分析
- 混合精度策略

**交付物**:
- sycl_sparse_attention.cpp (Forward + Backward)
- sycl_rope.cpp (3D RoPE)
- 单元测试套件
- 精度验证报告

---

## 🎯 里程碑时间表

### Milestone 0: 技术调研与架构设计 (Day 0-1)
**目标**: 深入理解现有实现，设计SYCL架构

**Agent 1 任务**:
- [ ] 研究sycle-tla_internal Flash Attention接口
  - 阅读applications/flash_attention_v2/文档
  - 分析API: prefill/decode/cachedKV
  - 记录头文件依赖和编译选项
- [ ] 分析TurboDiffusion集成点
  - 定位modify_model.py调用链
  - 理解attention调用位置
  - 分析tensor形状和数据流
- [ ] 设计集成架构
  - model_patcher.py接口设计
  - kernel registry设计
  - fallback机制设计

**Agent 2 任务**:
- [ ] 深度分析Sparse Attention算法
  - 精读SLA/kernel.py Triton代码
  - 理解LUT(look-up table)机制
  - 理解top-k稀疏模式
  - 画出数据流图
- [ ] 分析RoPE实现
  - 精读VideoRopePosition3DEmb类
  - 理解3D RoPE计算逻辑
  - 识别CUDA-specific代码
- [ ] 精度分析
  - BF16 vs FP32误差容忍度
  - 混合精度策略设计
  - 测试数据集准备

**验收标准**:
- [ ] 两份技术调研文档
- [ ] 架构设计文档
- [ ] 统一接口规范

**同步点**: Day 1结束，双方分享调研结果，统一接口

---

### Milestone 1: Flash Attention集成 (Day 2-4)
**负责人**: Agent 1
**目标**: 复用sycle-tla_internal，完成PyTorch集成

**任务清单**:
- [ ] Day 2: 创建sycl_flash_attention_wrapper.cpp
  - 封装sycle-tla_internal接口
  - 实现PyTorch C++ binding
  - 支持BF16/FP32
- [ ] Day 3: 编译测试
  - 验证与sycle-tla_internal链接
  - 单元测试: 随机输入验证正确性
  - 性能基准测试
- [ ] Day 4: 文档与review
  - API使用文档
  - 已知限制说明
  - 代码review准备

**验收标准**:
- [ ] 单元测试通过 (误差<1e-3)
- [ ] 性能达到sycle-tla_internal示例的90%
- [ ] 代码review通过

**阻塞风险**: sycle-tla_integration困难  
**缓解策略**: 早期POC验证，准备fallback方案

---

### Milestone 2: Sparse Attention实现 (Day 2-6)
**负责人**: Agent 2
**目标**: Triton转SYCL，实现Sparse Attention kernel

**任务清单**:
- [ ] Day 2-3: 算法设计与原型
  - 参考Triton代码设计SYCL版本
  - 设计tiling策略 (适配Intel SLM 128KB)
  - 设计LUT内存布局
  - 编写伪代码
- [ ] Day 4-5: 实现sycl_sparse_attention.cpp
  - Forward kernel (_attn_fwd)
  - Backward kernels (dQ, dK, dV)
  - 预处理kernel
- [ ] Day 6: 单元测试与优化
  - 正确性验证 (对比PyTorch)
  - 性能profiling
  - 内存使用分析

**关键技术点**:
```cpp
// Sparse Attention核心逻辑
class SparseAttentionKernel {
public:
    void operator()(sycl::nd_item<2> item) {
        // Q block in SLM
        sycl::local_accessor<float, 2> q_block({BLOCK_M, HEAD_DIM});
        
        // 从LUT读取活跃KV块索引
        int lut_start = lut_offsets[batch][head][q_block_idx];
        int num_kv_blocks = lut_data[lut_start];
        
        for (int i = 0; i < num_kv_blocks; ++i) {
            int kv_block_idx = lut_data[lut_start + 1 + i];
            // 加载KV块 (稀疏加载)
            // 计算部分attention
        }
    }
};
```

**验收标准**:
- [ ] Forward精度: 与PyTorch对比误差<1e-4
- [ ] Backward精度: 梯度误差<1e-3
- [ ] 性能: 比Dense Attention快30%+ (稀疏度50%)

**阻塞风险**: Sparse Attention算法复杂  
**缓解策略**: 每日代码review，遇到困难立即升级

---

### Milestone 3: 架构集成 (Day 5-7)
**负责人**: Agent 1 (主导) + Agent 2 (配合)
**目标**: 将所有kernel集成到TurboDiffusion

**任务清单**:
- [ ] Day 5: 实现model_patcher.py
  - 设备检测逻辑
  - 自动kernel替换
  - fallback管理
- [ ] Day 6: 实现kernel_registry.py
  - 注册所有SYCL kernel
  - 版本管理
  - 配置系统
- [ ] Day 7: 集成测试
  - 单模块测试
  - 多模块组合测试
  - 端到端冒烟测试

**集成点设计**:
```python
# model_patcher.py 核心逻辑
def patch_model_for_sycl(model, device):
    if device.type != "xpu":
        return model  # 非Intel GPU不处理
    
    # 替换norm层
    replace_norm_layers(model, use_sycl=True)
    
    # 替换attention
    replace_attention_layers(model, use_sycl=True)
    
    # 替换RoPE
    replace_rope_layers(model, use_sycl=True)
    
    return model
```

**验收标准**:
- [ ] 模型能正常加载
- [ ] 前向传播无错误
- [ ] 输出形状正确

---

### Milestone 4: 精度验证 (Day 8-10)
**负责人**: Agent 2 (主导) + Agent 1 (配合)
**目标**: 确保100%算法正确性

**任务清单**:
- [ ] Day 8: 端到端精度测试
  - 对比PyTorch native输出
  - 逐层误差分析
  - 关键帧SSIM/PSNR
- [ ] Day 9: 混合精度调优
  - Norm层: BF16 (已验证)
  - Attention: BF16测试稳定性
  - 识别需FP32回退的模块
- [ ] Day 10: 边缘案例测试
  - 长序列 (>4096)
  - 大batch
  - 极端稀疏模式

**精度标准**:
- Max Error < 1e-3
- Mean Error < 1e-4
- Cosine Similarity > 0.999

**验收标准**:
- [ ] 精度测试报告
- [ ] 混合精度策略文档
- [ ] 边缘案例通过

---

### Milestone 5: 性能调优 (Day 9-12)
**负责人**: Agent 1 (主导)
**目标**: 达到性能目标 (≥60% CUDA)

**任务清单**:
- [ ] Day 9-10: 性能profiling
  - Intel VTune分析
  - 识别热点kernel
  - 内存带宽分析
- [ ] Day 11: Work-group大小调优
  - 自动tuning脚本
  - 不同input size最优配置
- [ ] Day 12: 内存与调度优化
  - USM内存池调优
  - 减少host-device拷贝
  - 异步执行优化

**性能目标**:
| 指标 | 目标 |
|------|------|
| vs NVIDIA L20 | ≥60% |
| 比PyTorch XPU | ≥10x |
| 内存占用 | < 16GB |

---

### Milestone 6: L20环境准备与NVIDIA Baseline (Day 13)
**负责人**: Agent 1
**目标**: 准备NVIDIA对比环境

**任务清单**:
- [ ] 同步代码到L20 (10.112.229.160)
- [ ] 安装CUDA依赖
- [ ] 运行NVIDIA baseline测试
  - PyTorch native性能
  - 记录详细metrics
- [ ] 数据收集脚本

**环境信息**:
- Host: root@10.112.229.160
- Container: cuda12.9-test
- Path: /home/tianfeng -> /workspace

---

### Milestone 7: 完整对比测试与报告 (Day 14-15)
**双Agent并行**
**目标**: 生成功能+性能对比报告

**Agent 1 (L20环境)**:
- [ ] 运行完整benchmark
  - 不同分辨率 (480p, 720p)
  - 不同帧数 (16, 32, 81)
  - 多次运行取平均
- [ ] 收集metrics
  - 总时间分解
  - GPU利用率
  - 内存占用

**Agent 2 (Intel环境)**:
- [ ] 相同配置测试
  - 完全相同的prompt和seed
  - 相同输入尺寸
  - 相同模型配置
- [ ] 质量对比
  - 帧级SSIM/PSNR
  - 视频级质量评估

**交付物**:
- 性能对比表格
- 质量对比报告
- 技术总结文档

---

## ⚠️ 风险管理

### 高风险项

| 风险 | 概率 | 影响 | 应对策略 | 负责人 |
|------|------|------|----------|--------|
| Sparse Attention算法复杂 | 中 | **极高** | 1. 每日代码review<br>2. 遇到困难立即升级<br>3. 备选Dense方案 | Agent 2 |
| sycle-tla_integration困难 | 中 | 高 | 1. 早期POC验证<br>2. 准备fallback<br>3. 咨询Intel专家 | Agent 1 |
| 精度不达标 | 低 | **极高** | 1. 混合精度策略<br>2. 关键模块FP32<br>3. 误差逐层定位 | Agent 2 |
| 性能不达预期 | 中 | 高 | 1. VTune深度分析<br>2. Work-group tuning<br>3. Kernel fusion | Agent 1 |
| L20环境不可用 | 低 | **极高** | 1. 提前验证访问<br>2. 准备备用环境<br>3. 本地模拟测试 | Agent 1 |

### 技术风险详细分析

#### Risk 1: Sparse Attention LUT内存访问
**问题**: LUT随机访问可能导致cache miss严重  
**缓解**:
- 使用coalesced memory access
- 预加载LUT到shared memory
- 批量处理相同sparse pattern的query

#### Risk 2: Flash Attention版本兼容性
**问题**: sycle-tla与PyTorch版本冲突  
**缓解**:
- 早期做版本兼容性测试
- docker环境隔离
- 记录exact版本组合

#### Risk 3: BF16精度累积误差
**问题**: 多层BF16可能导致可观误差  
**缓解**:
- 关键层(residual前)用FP32
- 动态精度切换
- 误差监控机制

---

## 📊 进度追踪

### 每日同步模板

```markdown
## [Date] 进度汇报

### Agent 1 (Flash架构师)
**完成任务**:
- [x] 任务A (用时: X小时)
- [x] 任务B (用时: Y小时)

**进行中**:
- [ ] 任务C (预计完成: 明天)

**遇到问题**:
- 问题描述
- 已尝试的解决方案
- 需要的帮助

**明日计划**:
- 任务D
- 任务E

### Agent 2 (Sparse专家)
[同上格式]

### 同步会议纪要
- 决策1: xxx
- 决策2: xxx
- 待解决问题: xxx
```

### 里程碑验收模板

```markdown
## Milestone X: [名称]

### 完成状态: [✅/⚠️/❌]

### 实际用时: X天 (计划: Y天)

### 交付物:
- [ ] 代码: [link]
- [ ] 测试: [link]
- [ ] 文档: [link]

### 性能数据:
- 正确性: [误差值]
- 性能: [ms/iter]
- 目标达成: [X%]

### 问题与风险:
- [问题描述]

### 下一步调整:
- [调整计划]

### 负责人确认:
- [ ] Agent 1
- [ ] Agent 2
- [ ] 用户review
```

---

## 📝 关键文档

### 输入文档
- TurboDiffusion原代码 (已分析)
- sycle-tla_internal Flash Attention (已定位)
- Triton kernel源码 (SLA/kernel.py)

### 输出文档
- 技术调研报告 (M0)
- 架构设计文档 (M0)
- API文档 (M1)
- 精度验证报告 (M4)
- 性能调优报告 (M5)
- 最终对比报告 (M7)

---

## ✅ 启动前检查清单

- [x] Agent分工确认
- [x] 里程碑计划确认
- [x] 同步机制确认
- [x] L20环境验证
- [x] 质量要求确认
- [x] 风险应对确认

**项目状态**: ✅ 准备就绪，等待启动指令

**启动日期**: [填写实际启动日期]

**计划完成日期**: [启动日期 + 20天]
