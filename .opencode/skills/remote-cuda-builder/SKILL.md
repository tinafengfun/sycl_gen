---
name: remote-cuda-builder
description: 在远程节点(10.112.229.160)的CUDA docker(cuda12.9-test)中进行CUDA kernel编译和测试。自动处理多级目录创建、SSH/SCP同步、编译执行和结果回传。
license: MIT
compatibility: opencode
metadata:
  environment: remote-cuda
  ssh_host: root@10.112.229.160
  container: cuda12.9-test
  compiler: /usr/local/cuda/bin/nvcc
  workspace: /workspace
  type: build-and-test
  version: "1.1"
---

## What I do

通过SSH连接到远程节点，在CUDA docker内进行编译和测试，自动处理所有目录创建和多级同步。

### 核心功能

1. **健壮的多级目录创建**
   - 本地: `results/cuda/`, `scripts/cuda/`
   - 远程主机: `/workspace/`, `/workspace/scripts/cuda/`
   - 容器内: `/workspace/kernel_dataset/cuda/`, `/workspace/build_cuda/`, `/workspace/results/cuda/`

2. **改进的多级同步**
   - **L1**: Local → Remote Host (使用 scp)
     - 先创建远程目录: `mkdir -p /workspace`
     - 同步文件: `scp -r local/cuda remote:/workspace/`
     - 验证同步
   
   - **L2**: Remote Host → Container (使用 docker cp)
     - 先创建容器目录: `docker exec container mkdir -p /workspace/kernel_dataset/cuda`
     - 复制文件: `docker cp /workspace/cuda/. container:/workspace/kernel_dataset/cuda/`
     - 验证同步

3. **编译脚本管理**
   - 本地生成脚本: `scripts/cuda/build_<kernel>_YYYYMMDD_HHMMSS.sh`
   - 复制到远程: `scp script root@host:/workspace/scripts/cuda/`
   - 复制到容器: `docker cp remote_script container:/workspace/`

4. **错误处理和验证**
   - 检查SSH/SCP安装
   - 验证SSH连接
   - 检查容器运行状态
   - 验证NVCC可用性
   - 检查源文件存在
   - 验证编译输出

5. **结果回传**
   - 容器 → 远程主机: `docker cp container:/workspace/build_cuda remote:/workspace/results/`
   - 远程主机 → 本地: `scp -r remote:/workspace/results/cuda/* local/results/cuda/`

## 目录结构 (三级)

```
本地 Local/
├── results/cuda/
│   ├── build_cuda/             # 回传的编译产物
│   ├── compile_*.log           # 编译日志
│   ├── batch_status_*.jsonl    # 批量编译状态
│   └── summary_*.json          # 汇总报告
├── scripts/cuda/
│   └── build_*_*.sh            # 生成的编译脚本
└── .build_status.json

远程主机 Remote (10.112.229.160)/
└── /workspace/
    ├── cuda/                   # L1同步的代码
    ├── scripts/cuda/           # 编译脚本中转
    └── results/cuda/           # L2回传结果中转

容器 Container (cuda12.9-test)/
└── /workspace/
    ├── kernel_dataset/cuda/    # L2同步的代码
    ├── build_cuda/             # 编译输出
    └── results/cuda/           # 容器内日志
```

## Commands

### 编译单个文件

**完整流程**:
```
1. 检查SSH/SCP安装
2. 验证SSH连接 (ssh root@10.112.229.160)
3. 检查容器运行状态
4. 检查NVCC可用性
5. 生成编译脚本 (本地)
6. 同步代码 L1: Local → Remote Host
   - 创建远程目录: mkdir -p /workspace
   - scp -r kernel_dataset/cuda root@host:/workspace/
   - 验证: ls /workspace/cuda
7. 同步代码 L2: Remote Host → Container
   - 创建容器目录: docker exec container mkdir -p /workspace/kernel_dataset/cuda
   - docker cp /workspace/cuda/. container:/workspace/kernel_dataset/cuda/
   - 验证: docker exec container ls /workspace/kernel_dataset/cuda
8. 复制脚本到远程
   - 创建目录: ssh root@host "mkdir -p /workspace/scripts/cuda"
   - scp script root@host:/workspace/scripts/cuda/
9. 复制脚本到容器
   - docker cp /workspace/scripts/cuda/script container:/workspace/
10. 在容器内执行编译
    - docker exec container bash /workspace/build_script.sh
11. 回传编译产物
    - L3: Container → Remote Host
      docker cp container:/workspace/build_cuda/. /workspace/results/cuda/
    - L4: Remote Host → Local
      scp -r root@host:/workspace/results/cuda/* local/results/cuda/
12. 更新状态文件
```

**生成的编译脚本示例**:
```bash
#!/bin/bash
# CUDA Build Script
set -e
set -x

# 验证源文件存在
if [ ! -f "/workspace/kernel_dataset/cuda/<kernel>.cu" ]; then
    echo "[ERROR] Source file not found"
    ls -la $(dirname "/workspace/kernel_dataset/cuda/<kernel>.cu") || true
    exit 1
fi

# 创建输出目录
mkdir -p /workspace/build_cuda
mkdir -p /workspace/results/cuda

# 检查CUDA环境
which nvcc
nvcc --version

# 编译
/usr/local/cuda/bin/nvcc -O2 -arch=sm_70 \
  -c "/workspace/kernel_dataset/cuda/<kernel>.cu" \
  -o "/workspace/build_cuda/<kernel>.o"

# 验证输出
if [ -f "/workspace/build_cuda/<kernel>.o" ]; then
    echo "CUDA compilation successful!"
    ls -lh "/workspace/build_cuda/<kernel>.o"
else
    echo "[WARNING] Output file not found!"
    exit 1
fi
```

### 批量编译

**执行流程**:
```bash
# 查找所有 .cu 文件
for kernel in kernel_dataset/cuda/*.cu; do
    # 完整四级同步 + 编译 + 四级回传
    compile_single $kernel
    
    # 记录状态
    echo '{"kernel":"...","status":"..."}' >> batch_status.jsonl
done
```

## Error Handling

### 自动修复的问题

| 问题 | 自动处理 | 说明 |
|------|---------|------|
| 远程 /workspace 不存在 | ✅ | 自动创建: `mkdir -p /workspace` |
| 远程 /workspace/scripts/cuda 不存在 | ✅ | 自动创建 |
| 容器内 /workspace/kernel_dataset/cuda 不存在 | ✅ | 自动创建 |
| 本地 results/cuda 不存在 | ✅ | 自动创建 |

### 错误分类和解决

| 错误类型 | 检测方式 | 调试建议 |
|---------|---------|---------|
| SSH未安装 | `which ssh` | 安装SSH客户端 |
| SCP未安装 | `which scp` | 安装SCP |
| SSH连接失败 | `ssh host echo OK` | 配置SSH key: `ssh-copy-id root@10.112.229.160` |
| 容器未运行 | `docker ps` | `ssh root@10.112.229.160 'docker start cuda12.9-test'` |
| NVCC未找到 | `docker exec which nvcc` | 检查CUDA安装 |
| L1同步失败 | scp退出码 | 检查网络连接 |
| L2同步失败 | docker cp退出码 | 检查容器状态 |
| 编译失败 | nvcc退出码 | 查看编译日志 |
| 回传失败 | scp退出码 | 检查远程文件 |

## Output Format

### 编译日志 (`results/cuda/compile_<kernel>_YYYYMMDD_HHMMSS.log`)

```
Exit code: 0
=== STDOUT ===
=== CUDA Compilation in Remote Container ===
Timestamp: 20260303_092651
Container: cuda12.9-test
Kernel: <kernel_name>
Source: /workspace/kernel_dataset/cuda/<kernel>.cu
...
=== STDERR ===
+ echo '=== CUDA Compilation in Remote Container ==='
...
```

### 状态文件 (`.build_status.json`)

```json
{
  "metadata": {
    "last_updated": "2026-03-03T09:27:02.462346"
  },
  "environments": {
    "remote_cuda": {
      "type": "remote",
      "ssh_host": "root@10.112.229.160",
      "container": "cuda12.9-test",
      "compiler": "/usr/local/cuda/bin/nvcc",
      "kernels": {
        "<kernel_name>": {
          "status": "success",
          "last_build": "20260303_092651",
          "source_file": "kernel_dataset/cuda/<kernel>.cu",
          "local_log": "results/cuda/compile_<kernel>_20260303_092651.log",
          "remote_log": "/workspace/results/cuda/compile_<kernel>_20260303_092651.log",
          "script_file": "scripts/cuda/build_<kernel>_20260303_092651.sh",
          "duration_seconds": 8.25
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

1. **CUDA验证**: 验证CUDA代码在真实GPU上的编译
2. **跨环境对比**: 与SYCL实现对比
3. **批量构建**: 编译整个cuda目录
4. **远程CI**: 在远程GPU服务器上执行构建

## Prerequisites

- SSH免密登录配置: `ssh root@10.112.229.160` 无需密码
- 本地可执行 `ssh` 和 `scp`
- 远程docker已启动: `cuda12.9-test`
- 网络连通性良好

**环境检查命令**:
```bash
# 自动检查所有环境
./tools/remote_cuda_builder.sh check

# 手动检查SSH
ssh root@10.112.229.160 "echo 'OK'"

# 手动检查容器
ssh root@10.112.229.160 "docker ps | grep cuda12.9-test"

# 手动检查NVCC
ssh root@10.112.229.160 "docker exec cuda12.9-test nvcc --version"
```

## 已知问题和解决

### 问题1: L2同步失败 "Could not find the file"
**原因**: 容器内目标目录不存在
**解决**: 先创建目录再复制文件
```bash
# 修复前 (失败)
docker cp source container:/dest/path/

# 修复后 (成功)
docker exec container mkdir -p /dest/path
docker cp source container:/dest/path/
```

### 问题2: 远程脚本目录不存在
**原因**: scp前未创建目录结构
**解决**: 先ssh创建目录再scp
```bash
ssh root@host "mkdir -p /workspace/scripts/cuda"
scp script root@host:/workspace/scripts/cuda/
```

### 问题3: 编译成功但输出文件缺失
**原因**: 编译器异常退出或输出到错误路径
**解决**: 脚本内添加输出文件验证
```bash
if [ -f "$OUTPUT_FILE" ]; then
    echo "Success"
else
    echo "[WARNING] Output file not found!"
    exit 1
fi
```

## 性能优化建议

1. **保持SSH连接**: 使用SSH multiplexing减少连接开销
2. **增量同步**: 只同步变更的文件（使用rsync替代scp）
3. **并行编译**: 多个kernel可以并行编译（如果GPU资源允许）
4. **本地缓存**: 缓存未变更的编译产物

## 改进日志

- **v1.1** (2026-03-03): 改进版
  - 健壮的多级目录创建（远程主机+容器）
  - 改进的L1/L2同步逻辑
  - 添加文件存在性验证
  - 改进错误处理和日志记录
  - 添加路径安全检查
  - 改进状态文件更新（异常处理）
  - 添加编译输出验证
