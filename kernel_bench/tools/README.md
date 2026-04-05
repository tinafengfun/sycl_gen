# Kernel Builder Tools - 快速使用指南

## 概述

本工具集提供了统一的接口来在两种环境中编译和测试CUDA/SYCL kernel：

1. **B60环境**: 本地SYCL编译 (`lsv-container`)
2. **远程CUDA环境**: 远程NVIDIA GPU编译 (`cuda12.9-test@10.112.229.160`)

## 目录结构

```
.
├── tools/                          # 工具脚本
│   ├── build.sh                   # 统一入口脚本 ⭐
│   ├── b60_sycl_builder.sh        # B60 SYCL构建
│   ├── b60_sycl_builder.py        # B60 Python实现
│   ├── remote_cuda_builder.sh     # 远程CUDA构建
│   ├── remote_cuda_builder.py     # 远程CUDA Python实现
│   ├── test_connectivity.sh       # 连通性测试
│   ├── test_integration.sh        # 集成测试
│   └── test_builders.py           # 单元测试
├── results/                        # 构建结果
│   ├── b60/                       # B60构建日志
│   └── cuda/                      # CUDA构建日志
├── scripts/                        # 生成的编译脚本
│   ├── b60/
│   └── cuda/
├── .opencode/skills/              # opencode skills
│   ├── b60-sycl-builder/
│   └── remote-cuda-builder/
└── .build_status.json             # 构建状态追踪
```

## 快速开始

### 1. 检查环境

```bash
# 测试所有环境连通性
./tools/test_connectivity.sh
```

### 2. 编译Kernel

#### 统一入口 (推荐)

```bash
# 编译单个文件
./tools/build.sh b60 compile kernel_dataset/sycl/add_vectors_kernel.dp.cpp
./tools/build.sh cuda compile kernel_dataset/cuda/add_vectors_kernel.cu

# 编译所有kernel
./tools/build.sh b60 compile-all
./tools/build.sh cuda compile-all

# 编译所有环境的所有kernel
./tools/build.sh all compile-all
```

#### 单独使用

```bash
# B60 SYCL
./tools/b60_sycl_builder.sh compile kernel_dataset/sycl/softmax_kernel.dp.cpp
./tools/b60_sycl_builder.sh compile-all
./tools/b60_sycl_builder.sh status

# 远程CUDA
./tools/remote_cuda_builder.sh compile kernel_dataset/cuda/softmax_kernel.cu
./tools/remote_cuda_builder.sh compile-all
./tools/remote_cuda_builder.sh status
./tools/remote_cuda_builder.sh check    # 检查连通性
```

### 3. 查看状态

```bash
# 查看所有环境状态
./tools/build.sh all status

# 或单独查看
./tools/build.sh b60 status
./tools/build.sh cuda status
```

### 4. 清理构建

```bash
# 清理B60构建
./tools/build.sh b60 clean

# 清理CUDA构建
./tools/build.sh cuda clean

# 清理所有
./tools/build.sh all clean
```

## 输出文件

### 编译日志

格式: `results/{env}/compile_{kernel}_YYYYMMDD_HHMMSS.log`

```
results/b60/
├── compile_add_vectors_20260228_143022.log
├── compile_softmax_20260228_143105.log
└── summary_20260228_143500.json

results/cuda/
├── compile_add_vectors_20260228_143022.log
├── compile_softmax_20260228_143105.log
└── summary_20260228_143500.json
```

### 编译脚本

格式: `scripts/{env}/build_{kernel}_YYYYMMDD_HHMMSS.sh`

这些脚本包含完整的编译命令，可用于：
- 调试编译问题
- 手动重新执行
- 作为编译模板

### 状态文件

`.build_status.json` 包含所有kernel的构建状态：

```json
{
  "metadata": {
    "last_updated": "2026-02-28T14:30:25Z"
  },
  "environments": {
    "b60": {
      "type": "local",
      "container": "lsv-container",
      "compiler": "icpx",
      "kernels": {
        "add_vectors": {
          "status": "success",
          "last_build": "20260228_143022",
          "duration_seconds": 2.34
        }
      },
      "statistics": {
        "total": 30,
        "success": 15,
        "failed": 1
      }
    },
    "remote_cuda": {
      "type": "remote",
      "ssh_host": "root@10.112.229.160",
      "container": "cuda12.9-test",
      "compiler": "/usr/local/cuda/bin/nvcc",
      "kernels": {...}
    }
  }
}
```

## 测试

```bash
# 运行所有测试
./tools/test_integration.sh

# 运行单元测试
python3 tools/test_builders.py

# 测试连通性
./tools/test_connectivity.sh
```

## 故障排查

### B60环境

**问题**: 容器未运行
```bash
# 检查容器状态
docker ps | grep lsv-container

# 启动容器
docker start lsv-container
```

**问题**: icpx 不可用
```bash
# 检查编译器
docker exec lsv-container which icpx
docker exec lsv-container icpx --version
```

### 远程CUDA环境

**问题**: SSH连接失败
```bash
# 配置SSH免密登录
ssh-copy-id root@10.112.229.160

# 测试连接
ssh -o ConnectTimeout=5 root@10.112.229.160 "echo 'OK'"
```

**问题**: 容器未运行
```bash
# 在远程主机上启动
ssh root@10.112.229.160 "docker start cuda12.9-test"
```

**问题**: NVCC不可用
```bash
# 检查CUDA环境
ssh root@10.112.229.160 "docker exec cuda12.9-test nvcc --version"
```

## 工作流程示例

### 场景1: 验证新生成的SYCL kernel

```bash
# 1. 编译单个文件
./tools/build.sh b60 compile kernel_dataset/sycl/my_new_kernel.dp.cpp

# 2. 查看编译日志
cat results/b60/compile_my_new_kernel_*.log

# 3. 如成功，检查状态
./tools/build.sh b60 status
```

### 场景2: 批量编译并对比

```bash
# 1. 编译所有SYCL kernel
./tools/build.sh b60 compile-all

# 2. 编译所有CUDA kernel
./tools/build.sh cuda compile-all

# 3. 查看对比状态
./tools/build.sh all status
```

### 场景3: 调试编译错误

```bash
# 1. 尝试编译
./tools/build.sh b60 compile kernel_dataset/sycl/failing_kernel.dp.cpp

# 2. 查看详细错误日志
cat results/b60/compile_failing_kernel_*.log

# 3. 查看生成的编译脚本（用于手动调试）
cat scripts/b60/build_failing_kernel_*.sh

# 4. 修复代码后重新编译
./tools/build.sh b60 compile kernel_dataset/sycl/failing_kernel.dp.cpp
```

## 高级用法

### 使用Python API

```python
from tools.b60_sycl_builder import B60SyclBuilder
from tools.remote_cuda_builder import RemoteCudaBuilder

# B60构建
b60 = B60SyclBuilder()
b60.compile_single("kernel_dataset/sycl/add_vectors_kernel.dp.cpp")

# 远程CUDA构建
cuda = RemoteCudaBuilder()
cuda.compile_single("kernel_dataset/cuda/add_vectors_kernel.cu")

# 查看状态
print(b60.get_status())
print(cuda.get_status())
```

### opencode Skill使用

当使用opencode时，可以通过skill加载这些功能：

```python
# opencode会自动加载 .opencode/skills/ 中的skill
skill({ name: "b60-sycl-builder" })

# 然后请求
"请编译所有SYCL kernel"
"检查B60构建状态"
"编译失败的kernel有哪些"
```

## 配置文件

工具使用硬编码配置（在Python脚本顶部），如需修改：

**B60**: 编辑 `tools/b60_sycl_builder.py`
- `self.container = "lsv-container"`
- `self.workspace = "/workspace"`
- `self.compiler_flags = "-fsycl -O2 -std=c++17"`

**CUDA**: 编辑 `tools/remote_cuda_builder.py`
- `self.ssh_host = "root@10.112.229.160"`
- `self.container = "cuda12.9-test"`
- `self.compiler = "/usr/local/cuda/bin/nvcc"`

## 注意事项

1. **时间戳**: 所有文件使用 `YYYYMMDD_HHMMSS` 格式
2. **日志保留**: 历史日志不删除，便于追踪
3. **错误处理**: 详细错误信息保存在日志中
4. **并发**: 不同环境的编译可并行执行
5. **调试**: 使用 `set -x` 在脚本中显示所有执行的命令

## 更新日志

- **v1.0** (2026-02-28): 初始版本，支持B60和远程CUDA编译

## 获得帮助

```bash
# 显示帮助
./tools/build.sh help
./tools/b60_sycl_builder.sh help
./tools/remote_cuda_builder.sh help

# 运行测试
./tools/test_integration.sh
```
