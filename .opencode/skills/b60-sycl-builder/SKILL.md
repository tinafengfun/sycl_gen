---
name: b60-sycl-builder
description: 在本地B60 docker容器(lsv-container)中进行SYCL kernel编译和测试。自动处理目录创建、代码同步、编译执行和结果回传。
license: MIT
compatibility: opencode
metadata:
  environment: b60
  container: lsv-container
  compiler: icpx
  workspace: /workspace
  type: build-and-test
  version: "1.1"
---

## What I do

在B60环境的docker容器内进行SYCL kernel的编译和测试，自动处理所有目录创建和同步。

### 核心功能

1. **自动目录初始化**
   - 本地自动创建 `results/b60/` 和 `scripts/b60/`
   - 容器内自动创建 `/workspace/kernel_dataset/sycl/`
   - 如 `/workspace` 不存在，自动创建

2. **健壮的文件同步**
   - 先创建目标目录结构
   - 使用 `docker cp` 复制文件
   - 验证文件是否正确同步
   - 显示同步的文件数量

3. **编译脚本生成**
   - 每个kernel生成独立脚本
   - 包含源文件存在性检查
   - 脚本位置: `scripts/b60/build_<kernel>_YYYYMMDD_HHMMSS.sh`
   - 自动生成目录: `build_sycl/`, `results/b60/`

4. **错误处理和验证**
   - 检查Docker是否安装
   - 检查容器是否运行
   - 验证源文件存在
   - 验证编译输出
   - 详细的错误日志

5. **结果回传**
   - 编译日志: `results/b60/compile_<kernel>_YYYYMMDD_HHMMSS.log`
   - 编译产物: `results/b60/build_sycl/<kernel>.o`
   - 状态文件: `.build_status.json`

## 目录结构 (自动创建)

```
本地 workspace/
├── results/
│   └── b60/
│       ├── build_sycl/         # 编译产物
│       ├── compile_*.log       # 编译日志
│       ├── batch_status_*.jsonl # 批量编译状态
│       └── summary_*.json      # 汇总报告
├── scripts/
│   └── b60/
│       └── build_*_*.sh        # 生成的编译脚本
└── .build_status.json          # 构建状态

容器内 /workspace/
├── kernel_dataset/sycl/        # 同步的代码
├── build_sycl/                 # 编译输出
└── results/b60/                # 容器内日志
```

## Commands

### 编译单个文件

**流程**:
1. 检查Docker和容器状态
2. 在容器内创建工作目录（如不存在）
3. 生成编译脚本
4. 同步代码到容器
5. 在容器内执行编译
6. 回传编译产物和日志
7. 更新状态文件

**生成的编译脚本示例**:
```bash
#!/bin/bash
# SYCL Build Script
set -e
set -x

# 验证源文件存在
if [ ! -f "/workspace/kernel_dataset/sycl/<kernel>.dp.cpp" ]; then
    echo "[ERROR] Source file not found"
    exit 1
fi

# 创建输出目录
mkdir -p /workspace/build_sycl
mkdir -p /workspace/results/b60

# 编译
icpx -fsycl -O2 -std=c++17 \
  -c "/workspace/kernel_dataset/sycl/<kernel>.dp.cpp" \
  -o "/workspace/build_sycl/<kernel>.o"

# 验证输出
if [ -f "/workspace/build_sycl/<kernel>.o" ]; then
    echo "Compilation successful!"
else
    echo "[WARNING] Output file not found!"
    exit 1
fi
```

### 批量编译

**执行流程**:
```bash
# 查找所有 .dp.cpp 文件
for kernel in kernel_dataset/sycl/*.dp.cpp; do
    # 编译单个文件
    compile_single $kernel
    
    # 记录状态到 batch_status_YYYYMMDD_HHMMSS.jsonl
    echo '{"kernel":"...","status":"...","time":"..."}' >> batch_status.jsonl
done

# 生成汇总报告 summary_YYYYMMDD_HHMMSS.json
```

## Error Handling

### 自动修复的问题

| 问题 | 自动处理 | 说明 |
|------|---------|------|
| /workspace 不存在 | ✅ | 自动创建 |
| kernel_dataset/sycl/ 不存在 | ✅ | 自动创建 |
| results/b60/ 不存在 | ✅ | 自动创建 |
| scripts/b60/ 不存在 | ✅ | 自动创建 |
| build_sycl/ 不存在 | ✅ | 自动创建 |

### 错误分类

| 错误类型 | 检测方式 | 调试建议 |
|---------|---------|---------|
| Docker未安装 | `which docker` | 安装Docker |
| 容器未运行 | `docker ps` | `docker start lsv-container` |
| 源文件不存在 | 本地检查 | 检查文件路径 |
| 同步失败 | docker cp 退出码 | 检查容器状态 |
| 编译失败 | icpx 退出码 | 查看编译日志 |
| 输出文件缺失 | 文件存在性检查 | 检查编译错误 |

## Output Format

### 编译日志 (`results/b60/compile_<kernel>_YYYYMMDD_HHMMSS.log`)

```
Exit code: 0
=== STDOUT ===
=== SYCL Compilation Start ===
Timestamp: 20260303_092517
Kernel: <kernel_name>
Source: /workspace/kernel_dataset/sycl/<kernel>.dp.cpp
...
=== STDERR ===
+ echo '=== SYCL Compilation Start ==='
...
```

### 状态文件 (`.build_status.json`)

```json
{
  "metadata": {
    "last_updated": "2026-03-03T09:25:18.097312"
  },
  "environments": {
    "b60": {
      "type": "local",
      "container": "lsv-container",
      "compiler": "icpx",
      "kernels": {
        "<kernel_name>": {
          "status": "success",
          "last_build": "20260303_092517",
          "source_file": "kernel_dataset/sycl/<kernel>.dp.cpp",
          "log_file": "results/b60/compile_<kernel>_20260303_092517.log",
          "script_file": "scripts/b60/build_<kernel>_20260303_092517.sh",
          "duration_seconds": 0.1
        }
      },
      "statistics": {
        "total": 1,
        "success": 1,
        "failed": 0
      }
    }
  }
}
```

## When to use me

1. **编译验证**: 验证SYCL代码语法正确性
2. **批量构建**: 编译整个sycl目录
3. **错误调试**: 获取详细编译错误信息
4. **CI集成**: 生成结构化构建报告

## Prerequisites

- Docker守护进程运行中
- 容器 `lsv-container` 已启动
- 本地磁盘空间充足（保留历史日志）

**环境检查命令**:
```bash
# 检查Docker
docker ps | grep lsv-container

# 检查编译器
docker exec lsv-container which icpx

# 自动检查
./tools/b60_sycl_builder.sh check
```

## 已知问题和解决

### 问题1: 容器内 /workspace 不存在
**解决**: 工具自动检测并创建

### 问题2: docker cp 同步失败
**解决**: 先创建目标目录，再复制文件，最后验证同步

### 问题3: 编译成功但输出文件缺失
**解决**: 脚本内添加输出文件存在性检查

## 改进日志

- **v1.1** (2026-03-03): 改进版
  - 自动创建所有缺失目录
  - 改进文件同步逻辑（先创建目录再复制）
  - 添加文件存在性验证
  - 改进错误处理和日志记录
  - 添加路径安全检查
  - 改进状态文件更新（异常处理）
