# Agent开发规范与流程优化
# Agent Development Standards & Process Optimization

**版本**: 1.0  
**更新日期**: 2026-03-04  

---

## 🎯 核心原则

### 1. 代码质量优先
- 每次编辑后立即语法检查
- 单元测试覆盖核心功能
- 避免复杂嵌套，保持代码清晰

### 2. 编码规范
- **统一使用UTF-8编码**
- 英文注释优先（避免编码问题）
- 复杂字符串使用format()而非f-string嵌套

### 3. 测试驱动
- 每个方法必须有测试
- 先测试，后集成
- 自动化测试脚本

---

## 📝 编码规范

### 字符串处理规范

#### ❌ 避免：复杂f-string嵌套
```python
# BAD: 容易出错
compile_cmd = f"""
docker exec ... {config["N"]} {config["C"]} ...
"""
```

#### ✅ 推荐：使用format()或提前提取变量
```python
# GOOD: 清晰安全
n_val = config['N']
c_val = config['C']
cmd_template = """
docker exec ... {} {} ...
"""
compile_cmd = cmd_template.format(n_val, c_val)

# 或提前格式化
params = {
    'N': config['N'],
    'C': config['C'],
    'dtype': config['dtype']
}
cmd = template.format(**params)
```

### 多行字符串规范

#### ❌ 避免：混合使用 """ 和 '''
```python
# BAD: 容易混淆
def method():
    return f'''content'''
    
def another():
    return """content"""
```

#### ✅ 推荐：统一使用 """
```python
# GOOD: 统一风格
def generate_cpp(self):
    return """
#include <header>
int main() {{
    return 0;
}}
"""
```

### 文件编码规范

#### 强制要求
```python
# 所有Python文件开头添加
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
```

#### 避免中文变量名和注释中的特殊字符
```python
# BAD: 可能编码问题
def 生成数据():  # 中文函数名
    """生成測试数据"""  # 繁体或特殊字符
    
# GOOD: 使用英文
def generate_test_data():
    """Generate test data for accuracy testing"""
```

---

## 🔧 开发流程优化

### Phase执行标准流程

```
Phase X 开始
    │
    ├── Step 1: 创建提示词模板
    │   └── prompts/phaseX_xxx.md
    │
    ├── Step 2: 实现核心代码
    │   ├── 编写代码
    │   ├── 语法检查 ⭐
    │   └── 修复错误
    │
    ├── Step 3: 单元测试
    │   ├── 创建test_phaseX.py
    │   ├── 测试核心方法
    │   └── 验证通过 ⭐
    │
    ├── Step 4: 集成测试（可选）
    │   └── 测试完整流程
    │
    └── Step 5: 更新日志
        └── .opencode/plans/PHASEX_EXECUTION_LOG.md
```

### 检查点清单

#### 每个编辑操作后必须：
- [ ] 运行 `python3 -m py_compile file.py`
- [ ] 确认无语法错误
- [ ] 如有错误，立即修复

#### 每个Phase完成后必须：
- [ ] 创建单元测试脚本
- [ ] 运行测试，通过率100%
- [ ] 更新执行日志
- [ ] 代码提交（commit）

---

## 🛡️ 错误预防措施

### 1. 自动化语法检查

```bash
# 添加到 .bashrc 或 Makefile
alias pycheck='python3 -m py_compile'

# 每次保存后自动检查
watch -n 1 'python3 -m py_compile tools/unified_converter.py 2>&1 | head -5'
```

### 2. 代码模板标准化

```python
# 方法模板
async def method_name(self, param1: type1, param2: type2) -> return_type:
    """
    Brief description
    
    Args:
        param1: description
        param2: description
        
    Returns:
        description
        
    Raises:
        ExceptionType: when/why
    """
    # Step 1: Validate inputs
    if not param1:
        raise ValueError("param1 cannot be empty")
    
    # Step 2: Main logic
    try:
        result = await self._do_something(param1)
    except SpecificError as e:
        self.tracer.log("AgentName", "error", {"error": str(e)})
        raise
    
    # Step 3: Return
    return result
```

### 3. 字符串常量提取

```python
# 复杂字符串统一放在类顶部
class UnifiedAccuracyTester:
    # Constants
    CUDA_TEMPLATE = """
#include <cuda_runtime.h>
int main(int argc, char* argv[]) {{
    // Implementation
    return 0;
}}
"""
    
    SYCL_TEMPLATE = """
#include <sycl/sycl.hpp>
int main(int argc, char* argv[]) {{
    // Implementation
    return 0;
}}
"""
    
    def _generate_cuda_test(self, ...):
        return self.CUDA_TEMPLATE.format(...)
```

---

## 📊 代码质量指标

### 每个Phase必须达到：

| 指标 | 要求 | 检查方式 |
|------|------|----------|
| 语法正确性 | 100% | py_compile |
| 单元测试覆盖率 | 核心方法100% | pytest |
| 代码注释 | 所有public方法 | 人工检查 |
| 类型注解 | 关键参数和返回值 | mypy |
| 复杂度 | 函数<50行 | 人工检查 |

---

## 🚀 更新后的Phase执行计划

### Phase X 执行步骤

#### Step 1: Preparation (5分钟)
1. 创建提示词模板
2. 定义接口和数据结构
3. 创建测试脚本框架

#### Step 2: Implementation (2-3小时)
1. 编写核心代码（小步快跑）
2. **每30分钟语法检查一次** ⭐
3. 实现错误处理
4. 添加日志记录

#### Step 3: Testing (1小时)
1. 运行语法检查
2. 执行单元测试
3. 修复发现的问题
4. 达到100%通过率

#### Step 4: Documentation (30分钟)
1. 更新执行日志
2. 记录关键决策
3. 更新API文档

---

## 🎯 Phase 2 特别注意事项

### UnifiedReporter 实现要点

1. **避免HTML模板中的引号冲突**
   ```python
   # BAD
   html = f"<div class='{class_name}'>...</div>"
   
   # GOOD
   html_template = "<div class='{}'>...</div>"
   html = html_template.format(class_name)
   ```

2. **JSON序列化错误处理**
   ```python
   import json
   try:
       with open(file, 'w') as f:
           json.dump(data, f, indent=2)
   except (TypeError, ValueError) as e:
       # Handle non-serializable data
       logger.error(f"JSON serialization failed: {e}")
   ```

3. **多格式报告的统一接口**
   ```python
   class UnifiedReporter:
       def generate_report(self, data, format_type):
           generators = {
               'json': self._generate_json,
               'html': self._generate_html,
               'markdown': self._generate_markdown
           }
           generator = generators.get(format_type)
           if not generator:
               raise ValueError(f"Unknown format: {format_type}")
           return generator(data)
   ```

---

## ✅ Pre-Phase 2 Checklist

在开始Phase 2之前确认：

- [x] Phase 1代码语法正确
- [x] Phase 1单元测试通过
- [x] 编码规范文档创建
- [x] 测试脚本模板准备
- [ ] Phase 2提示词模板创建

---

## 📝 执行命令速查

```bash
# 语法检查
python3 -m py_compile tools/unified_converter.py

# 运行测试
PYTHONPATH=tools:$PYTHONPATH python3 test_phase1.py

# 创建新测试
./scripts/create_test.py phase2

# 更新日志
echo "Phase 2 started" >> .opencode/plans/EXECUTION_LOG.md
```

---

**批准执行优化后的流程？**
