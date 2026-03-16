# SYCL Kernel转换Agent - 待完成任务清单

**生成日期**: 2024-03-08
**当前状态**: 约40-50%完成

---

## 📊 当前统计

| 指标 | 数值 |
|------|------|
| **总CUDA内核** | 30 |
| **已完成SYCL转换** | 12 (40%) |
| **待转换** | 18 (60%) |
| **SYCL编译成功** | 8/13 (61.5%) |
| **SYCL编译失败** | 5 |

---

## 🔴 P0 - 关键任务（阻塞生产使用）

### 任务1: 修复CUDA GPU容器访问
**状态**: 🔴 待处理  
**优先级**: P0 - 关键  
**预估时间**: 1-2小时

**问题描述**:
```
CUDA error: no CUDA-capable device is detected
Docker容器创建时未使用 --gpus all 参数
```

**解决方案**:
```bash
# 在远程主机上执行
ssh root@10.112.229.160
docker stop cuda12.9-test
docker rm cuda12.9-test
docker run -d --gpus all --name cuda12.9-test-gpu \
  nvidia/cuda:12.9.0-devel-ubuntu22.04 sleep infinity

# 验证
docker exec cuda12.9-test-gpu nvidia-smi
```

**阻塞影响**: 
- ❌ 无法运行CUDA vs SYCL准确度对比
- ❌ 无法验证转换正确性
- ❌ 无法生成完整测试报告

---

### 任务2: 转换18个待处理的CUDA内核
**状态**: 🔴 待处理  
**优先级**: P0 - 关键  
**预估时间**: 2-3天

**待转换内核列表**:

#### 高优先级（基础操作）
1. `add_vectors` - 基础向量加法
2. `add_vectors_hnc_nhc` - 带转置的向量加法
3. `add_bias_batched` - 批处理偏置加法
4. `add_bias_nchw` - NCHW偏置加法
5. `nchw_to_nhwc` - 布局转换

#### 中优先级（归一化/池化）
6. `global_scale` - 全局缩放
7. `global_scale_fp16_nhwc` - FP16全局缩放
8. `global_avg_pool_nhwc_fp16` - FP16池化

#### 中优先级（输入/预处理）
9. `expand_planes_nhwc` - NHWC平面扩展
10. `input_gating` - 输入门控

#### 低优先级（Attention/Transformer）
11. `softmax` - 通用softmax
12. `gen_offset_pointers` - Attention偏移指针
13. `preprocess_attention_body` - Attention预处理
14. `promotion_logits` - 提升逻辑
15. `se_layer_nhwc` - Squeeze-and-excitation层

#### 低优先级（Winograd）
16. `winograd_output_relu_input` - 融合输出转换
17. `winograd_output_transform` - 输出转换
18. `winograd_output_se_relu_input` - 已有SYCL但编译失败

**注意**: `fused_mha_cutlass` 是CUDA-only（使用NVIDIA CUTLASS），不需要转换

**推荐方法**: 使用LLM直接生成（已证明比基于规则的转换更有效）

---

### 任务3: 修复5个损坏的SYCL内核
**状态**: 🔴 待处理  
**优先级**: P0 - 关键  
**预估时间**: 1天

**损坏的内核**:

| 内核 | 问题 | 状态 |
|------|------|------|
| `layer_norm` | 缺少头文件: `neural/tables/activation_function.h` | 🔴 |
| `winograd_filter_transform` | 编译错误 | 🔴 |
| `winograd_output_se_relu_input` | 编译错误 | 🔴 |
| `output_input_transform_fp16_shmem` | 编译错误 | 🔴 |
| `generated_v1.dp` | 源文件路径问题（测试产物） | 🔴 |

**修复策略**:
1. 内联缺少的头文件函数
2. 修复模板参数问题
3. 更新API调用以匹配SYCL标准

---

### 任务4: 解决头文件依赖问题
**状态**: 🔴 待处理  
**优先级**: P0 - 关键  
**预估时间**: 4-6小时

**缺少的头文件**:
- `neural/tables/activation_function.h`
- `neural/shared.h`
- 其他LCZero内部头文件

**影响的内核**:
- `layer_norm`
- `batch_norm`
- `winograd_filter_transform`
- `output_input_transform_fp16_shmem`
- `policy_map`

**解决方案选项**:
1. **选项A**: 内联所有需要的函数到kernel文件（推荐）
2. **选项B**: 创建独立的SYCL头文件
3. **选项C**: 修改include路径

**推荐**: 选项A - 使kernel文件自包含

---

## 🟡 P1 - 高优先级任务

### 任务5: 运行完整CUDA vs SYCL准确度对比
**状态**: 🟡 等待GPU配置  
**优先级**: P1  
**预估时间**: 4-6小时

**测试内容**:
- 13个测试配置
- 3种数据类型（float32, float16, bfloat16）
- 4种维度测试
- 极端值测试

**成功标准**:
- 绝对误差 < 1e-5 (float32)
- 绝对误差 < 1e-3 (float16/bfloat16)
- 通过率 > 95%

**依赖**: 任务1（GPU访问）

---

### 任务6: 改进转换器规则
**状态**: 🟡 部分完成  
**优先级**: P1  
**预估时间**: 1-2天

**当前问题**:
- 模板依赖的宏转换失败
- 简单的字符串替换不够智能
- 调用点更新不完整

**改进方向**:
1. 检测模板参数的隐式依赖
2. 将宏转为显式参数的函数
3. 自动更新所有调用点
4. 添加更多CUDA到SYCL的模式匹配

**示例**:
```cpp
// CUDA宏
#define INDEX_NCHW(n, c, h, w) ((n)*C * 8 * 8 + (c)*8 * 8 + (h)*8 + w)

// SYCL函数
inline int IndexNCHW(int n, int c, int h, int w, int C) {
  return (n) * C * 8 * 8 + (c) * 8 * 8 + (h) * 8 + w;
}
```

---

### 任务7: 批量转换剩余内核
**状态**: 🟡 准备中  
**优先级**: P1  
**预估时间**: 2-3天

**批量转换流程**:
1. 使用`batch_convert.py`工具
2. 并行转换多个内核
3. 自动错误恢复
4. 统一报告生成

**建议**: 一次转换3-5个内核，验证后再继续

---

## 🟢 P2 - 中等优先级

### 任务8: 性能基准测试
**状态**: 🟢 计划中  
**优先级**: P2  
**预估时间**: 1-2天

**测试内容**:
- CUDA vs SYCL执行时间对比
- 内存带宽利用率
- 计算吞吐量
- 不同batch size的性能

**指标**:
- 加速比（SYCL/CUDA）
- 延迟（毫秒）
- 吞吐量（GFLOPS）

---

### 任务9: 完成文档
**状态**: 🟢 进行中  
**优先级**: P2  
**预估时间**: 1天

**待完成文档**:
- [ ] 更新README.md（当前状态）
- [ ] 创建故障排除指南
- [ ] API文档
- [ ] 性能优化指南
- [ ] 部署手册

---

## 🔵 P3 - 低优先级

### 任务10: 准备生产环境部署
**状态**: 🔵 计划中  
**优先级**: P3  
**预估时间**: 2-3天

**内容**:
- 环境配置脚本
- 监控和日志
- 自动恢复机制
- 性能优化
- 安全加固

---

## 📈 项目路线图

### 第一阶段（1-2周）
- [x] 建立基础架构 ✅
- [x] 创建LLM Accuracy Test Agent ✅
- [x] 集成到Unified Converter ✅
- [ ] 修复CUDA GPU访问 🔴
- [ ] 转换前10个高优先级内核 🔴

### 第二阶段（2-3周）
- [ ] 转换剩余8个内核
- [ ] 修复5个损坏的SYCL内核
- [ ] 运行完整准确度测试
- [ ] 修复发现的问题

### 第三阶段（3-4周）
- [ ] 性能优化
- [ ] 文档完善
- [ ] 生产部署准备
- [ ] 验收测试

---

## 🎯 成功标准

### 转换完成标准
- [ ] 100%内核有SYCL版本（除了CUDA-only的）
- [ ] 100%SYCL内核编译成功
- [ ] 95%+准确度测试通过率
- [ ] 性能差距 < 20%（SYCL vs CUDA）

### 生产就绪标准
- [ ] 所有P0任务完成
- [ ] 文档完整
- [ ] 自动化测试通过
- [ ] 性能基准达标

---

## 💡 建议的工作模式

基于成功经验，推荐以下工作流程：

```
1. 选择3-5个相关内核
2. 使用LLM直接生成SYCL代码
3. 编译验证
4. 修复编译错误（最多3次迭代）
5. 运行准确度测试
6. 修复准确度问题
7. 提交并继续下一批
```

**关键成功因素**:
- ✅ LLM直接生成（避免模板转换复杂性）
- ✅ 迭代修复（快速解决问题）
- ✅ 完整测试（编译+运行+准确度）
- ✅ 小批量处理（降低风险）

---

## 📞 需要支持

### 需要管理员权限
- 修复远程主机的Docker GPU配置
- 安装/更新系统包

### 需要决策
- 是否内联所有头文件依赖？
- 是否保留CUTLASS内核为CUDA-only？
- 性能差距的可接受范围？

---

**最后更新**: 2024-03-08  
**下次审查**: 完成P0任务后
