#!/bin/bash
# Accuracy Test Usage Example
# 准确度测试使用示例

cat <> 'EOF'
╔══════════════════════════════════════════════════════════════════╗
║           准确度测试使用指南                                      ║
╚══════════════════════════════════════════════════════════════════╝

## 快速开始

### 1. 测试单个Kernel

```bash
# 基本用法
python3 tools/accuracy_tester.py <kernel_id> <cuda_file> <sycl_file> <trace_session>

# 示例: 测试winograd kernel
python3 tools/accuracy_tester.py \\
  winograd_input_transform \\
  kernel_dataset/cuda/winograd_input_transform_kernel.cu \\
  kernel_dataset/sycl/winograd_input_transform_kernel.dp.cpp \\
  session_20260304_001
```

### 2. 测试套件

```bash
# 运行完整测试套件
./tools/run_accuracy_test_suite.sh

# 仅测试Level 1 kernel
./tools/run_accuracy_test_suite.sh --level 1

# 仅测试失败的kernel
./tools/run_accuracy_test_suite.sh --failed-only
```

### 3. 查看测试报告

```bash
# 查看JSON报告
cat .traces/sessions/{session_id}/accuracy_report_{kernel_id}.json

# 生成可视化报告
python3 tools/generate_accuracy_report.py --session {session_id}

# 对比多个kernel
python3 tools/compare_accuracy.py --kernels kernel1,kernel2,kernel3
```

## 测试类型

### 测试数据类型

1. **边界值测试** (boundary)
   - 值: 0, 1, -1, min, max, eps
   - 目的: 检测边界条件处理

2. **随机均匀分布** (random_uniform)
   - 范围: [-1, 1]
   - 目的: 检测一般情况精度

3. **随机正态分布** (random_normal)
   - 均值: 0, 标准差: 1
   - 目的: 检测统计特性保持

4. **特殊值测试** (special)
   - 值: inf, -inf, nan, tiny, -0.0
   - 目的: 检测异常值处理

### 准确度阈值

```yaml
Float32:
  绝对误差: < 1e-5
  相对误差: < 1e-4
  最大不匹配率: 0.1%

Float16:
  绝对误差: < 1e-3
  相对误差: < 1e-2
  最大不匹配率: 1%
```

## 测试结果解读

### 成功示例

```json
{
  "kernel_id": "add_vectors",
  "overall_status": "PASS",
  "pass_rate": 1.0,
  "test_details": [
    {
      "test_name": "boundary_small",
      "status": "PASS",
      "comparison": {
        "mismatch_rate": 0.0,
        "max_abs_diff": 1.2e-7
      }
    }
  ]
}
```

### 失败示例

```json
{
  "kernel_id": "winograd_transform",
  "overall_status": "FAIL",
  "pass_rate": 0.67,
  "test_details": [
    {
      "test_name": "random_uniform",
      "status": "FAIL",
      "comparison": {
        "mismatch_rate": 0.05,
        "max_abs_diff": 0.01,
        "note": "5% mismatch exceeds threshold of 0.1%"
      }
    }
  ]
}
```

## 故障排除

### 问题1: CUDA编译失败
**解决**: 检查CUDA环境
```bash
nvcc --version
```

### 问题2: SYCL编译失败
**解决**: 检查B60容器
```bash
docker exec lsv-container icpx --version
```

### 问题3: 测试结果不一致
**解决**: 
1. 检查随机种子
2. 增加测试次数
3. 检查数据类型

### 问题4: 准确度偏差大
**解决**:
1. 检查转换规则
2. 对比中间结果
3. 使用更高精度测试

## 集成到Agent Workflow

### 自动触发

```python
# 在Agent Task中自动触发
class TracedAccuracyTester:
    def run_after_compilation(self):
        if compilation_success:
            report = self.run_full_accuracy_test(
                cuda_file, sycl_file
            )
            
            if report["overall_status"] == "PASS":
                self.mark_success()
            elif report["pass_rate"] >= 0.99:
                self.accept_with_warning()
            else:
                self.flag_for_review()
```

### 决策流程

```
编译成功
  ↓
准确度测试
  ↓
判断结果:
  - PASS (100%): 直接接受
  - 99-100%: 接受但记录警告
  - 95-99%: 标记需人工审查
  - <95%: 拒绝并回滚
```

## 高级用法

### 自定义测试数据

```python
# 创建自定义测试
from accuracy_tester import AccuracyTester

tester = AccuracyTester("my_kernel", "session_001")

# 自定义测试数据
custom_data = np.array([1.0, 2.0, 3.0, ...])

# 运行测试
cuda_out = tester.run_cuda_test(custom_data, "cuda_kernel.cu")
sycl_out = tester.run_sycl_test(custom_data, "sycl_kernel.dp.cpp")
result = tester.compare_results(cuda_out, sycl_out, "float32")
```

### 批量测试

```bash
# 测试所有已转换的kernel
for kernel in kernel_dataset/sycl/*.dp.cpp; do
  python3 tools/accuracy_tester.py \\
    $(basename $kernel .dp.cpp) \\
    kernel_dataset/cuda/$(basename $kernel .dp.cpp).cu \\
    $kernel \\
    batch_session_001
done
```

### 性能对比

```bash
# 对比CUDA和SYCL性能
python3 tools/benchmark_performance.py \\
  --cuda kernel_dataset/cuda/*.cu \\
  --sycl kernel_dataset/sycl/*.dp.cpp \\
  --output performance_report.json
```

## 报告分析

### 关键指标

1. **Pass Rate**: 通过率
   - 目标: >= 99.9%
   - 警告: 99-99.9%
   - 失败: < 99%

2. **Max Absolute Difference**: 最大绝对误差
   - Float32: < 1e-5
   - Float16: < 1e-3

3. **Max Relative Difference**: 最大相对误差
   - Float32: < 1e-4
   - Float16: < 1e-2

4. **Mismatch Rate**: 不匹配率
   - 目标: < 0.1%
   - 警告: 0.1-1%
   - 失败: > 1%

## 最佳实践

1. **总是先编译再通过准确度测试**
2. **使用多种测试数据类型**
3. **记录所有测试结果便于对比**
4. **定期检查准确度趋势**
5. **对失败案例进行根因分析**

╔══════════════════════════════════════════════════════════════════╗
║  提示: 准确度测试是CUDA-to-SYCL转换的最后一道关卡！              ║
╚══════════════════════════════════════════════════════════════════╝
EOF
