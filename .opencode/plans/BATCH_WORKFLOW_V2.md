# 优化后的Agent执行规范和流程
# Optimized Agent Execution Standards & Workflow

**版本**: 2.0  
**更新**: 2026-03-04  
**目标**: 集成opencode API，批量处理30个kernel  

---

## 🎯 核心改进点

### 1. 代码生成模式升级
- **从**: Rule-based fallback  
- **到**: opencode API集成，真正AI生成  

### 2. 批处理优化
- 并行处理多个kernel
- 自动错误恢复
- 进度实时报告

### 3. 质量保证
- 每个kernel编译验证
- 准确度测试
- 失败自动重试

---

## 📋 opencode API集成规范

### API调用标准

```python
async def call_opencode_api(prompt: str, system_prompt: str = None) -> str:
    """
    调用opencode API生成代码
    
    Args:
        prompt: 用户提示词
        system_prompt: 系统提示词（可选）
        
    Returns:
        生成的SYCL代码
        
    Raises:
        APIError: API调用失败
        TimeoutError: 超时
    """
    # 实现方式1: 使用opencode CLI
    # 实现方式2: 使用opencode Python SDK
    # 实现方式3: 直接HTTP API调用
```

### 错误处理策略

```python
max_retries = 3
for attempt in range(max_retries):
    try:
        code = await call_opencode_api(prompt)
        if validate_code(code):
            return code
    except Exception as e:
        if attempt == max_retries - 1:
            raise
        await asyncio.sleep(2 ** attempt)  # 指数退避
```

### Prompt优化

```python
# 系统提示词 - 定义角色和规则
SYSTEM_PROMPT = """You are an expert CUDA to SYCL converter..."""

# 用户提示词 - 具体任务
USER_PROMPT_TEMPLATE = """
Convert this CUDA kernel to SYCL:

{cuda_code}

Requirements:
1. ...
2. ...
"""
```

---

## 🔄 批量处理流程

### Phase 1: 准备阶段
```
1. 扫描所有30个kernel
2. 按复杂度排序（简单优先）
3. 创建批处理队列
4. 初始化环境检查
```

### Phase 2: 并行转换
```
1. 启动N个worker（推荐: 4-8个）
2. 每个worker处理一个kernel:
   - 分析CUDA代码
   - 调用opencode API生成SYCL
   - 验证编译
   - 运行准确度测试
3. 收集结果到队列
```

### Phase 3: 结果汇总
```
1. 统计成功率
2. 分析失败原因
3. 生成完成度报告
4. 输出JSON/HTML/Markdown报告
```

---

## 🛡️ 错误恢复机制

### 失败分类

| 失败类型 | 处理策略 | 重试次数 |
|---------|---------|---------|
| API超时 | 指数退避重试 | 3次 |
| 生成代码编译失败 | 发送错误信息给模型修复 | 2次 |
| 准确度测试失败 | 调整参数重试 | 1次 |
| 严重错误 | 记录并跳过 | 0次 |

### 自动修复流程

```python
async def auto_fix_and_retry(kernel_id, error_info):
    """自动修复失败的任务"""
    # 1. 分析错误
    # 2. 构建修复提示词
    # 3. 重新调用API
    # 4. 验证修复
```

---

## 📊 进度监控

### 实时状态

```python
class BatchProgress:
    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.in_progress = 0
        
    def update(self, kernel_id, status):
        """更新进度并打印"""
        print(f"Progress: {self.completed}/{self.total} "
              f"({self.completed/self.total*100:.1f}%)")
```

### 日志分级

- **INFO**: 正常进度
- **WARNING**: 可恢复错误
- **ERROR**: 严重错误
- **SUCCESS**: 任务完成

---

## 🎯 质量标准

### 通过标准

每个kernel必须通过：
- ✅ 编译成功（无错误）
- ✅ 准确度测试通过率 ≥ 99%
- ✅ 最大绝对误差 < 1e-5
- ✅ 执行时间 < 300秒

### 失败容忍

- 编译失败: 允许2次重试
- 准确度失败: 允许1次重试
- 超时: 允许1次重试

---

## 🔧 执行命令

### 批量处理启动

```bash
# 处理所有30个kernel
python3 tools/batch_convert.py \
  --all \
  --workers 4 \
  --use-opencode \
  --output-dir results/batch_20260304

# 处理指定kernel
python3 tools/batch_convert.py \
  --kernels kernel1,kernel2,kernel3 \
  --workers 2
```

### 监控进度

```bash
# 实时查看日志
tail -f results/batch_20260304/progress.log

# 查看统计
cat results/batch_20260304/summary.json
```

---

## ✅ 检查清单

### 集成前检查

- [ ] opencode API密钥配置
- [ ] B60容器运行正常
- [ ] CUDA环境可访问
- [ ] 输出目录可写入
- [ ] 网络连接稳定

### 执行中检查

- [ ] API调用成功率 > 90%
- [ ] 编译成功率 > 95%
- [ ] 准确度测试 > 99%
- [ ] 无内存泄漏
- [ ] 进度正常推进

### 完成后检查

- [ ] 所有报告生成
- [ ] 完成度统计准确
- [ ] 失败原因记录完整
- [ ] 可交付物完整

---

## 📈 性能目标

### 时间估算

| 任务 | 估算时间 | 并行数 |
|------|---------|--------|
| 单个kernel转换 | 2-5分钟 | 4-8个 |
| 30个kernel总计 | 15-40分钟 | - |
| 报告生成 | 2分钟 | - |
| **总计** | **20-45分钟** | - |

### 资源使用

- CPU: < 50%（主要瓶颈是API调用）
- 内存: < 4GB
- 磁盘: < 1GB（临时文件）
- 网络: 稳定连接（API调用）

---

## 🚀 快速启动

### 1. 环境检查
```bash
./scripts/pre_flight_check.sh
```

### 2. 启动批量处理
```bash
python3 tools/batch_convert.py --all --workers 4
```

### 3. 监控进度
```bash
watch -n 5 'cat results/latest/summary.json'
```

### 4. 查看报告
```bash
firefox results/latest/final_report.html
```

---

**状态**: 准备执行批量转换  
**下一步**: 实现opencode API集成 → 批量处理30个kernel
