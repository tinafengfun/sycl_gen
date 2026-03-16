# opencode Skill 集成指南

本文档说明如何将B60 SYCL Builder和Remote CUDA Builder skill集成到opencode中，供Agent使用。

## 目录结构

```
.opencode/
├── config.json          # opencode配置文件
└── skills/
    ├── b60-sycl-builder/
    │   └── SKILL.md     # SYCL编译skill
    └── remote-cuda-builder/
        └── SKILL.md     # CUDA编译skill
```

## 集成方法

### 方法1: 自动发现（推荐）

opencode会自动发现 `.opencode/skills/` 目录下的所有skill。

**步骤**:

1. **确保skill文件在正确位置**:
   ```bash
   .opencode/skills/<skill-name>/SKILL.md
   ```

2. **配置权限**（在 `.opencode/config.json` 中）:
   ```json
   {
     "permissions": {
       "skill": {
         "*": "allow",
         "b60-sycl-builder": "allow",
         "remote-cuda-builder": "allow"
       }
     }
   }
   ```

3. **重启opencode**或使用 `/reload` 命令刷新skill列表

### 方法2: 全局安装

将skill安装到用户主目录，所有项目都可用：

```bash
# 创建全局skill目录
mkdir -p ~/.config/opencode/skills

# 复制skill
cp -r .opencode/skills/b60-sycl-builder ~/.config/opencode/skills/
cp -r .opencode/skills/remote-cuda-builder ~/.config/opencode/skills/
```

### 方法3: 显式加载

在对话中显式加载skill：

```
/skill load b60-sycl-builder
/skill load remote-cuda-builder
```

## 使用方法

### 在对话中使用

一旦skill被加载，你可以在对话中直接请求：

#### 1. 编译单个Kernel

**SYCL**:
```
用户: 请编译 kernel_dataset/sycl/add_vectors_kernel.dp.cpp
Agent: [自动使用 b60-sycl-builder skill]
       1. 检查B60环境
       2. 同步代码到容器
       3. 执行编译
       4. 返回结果
```

**CUDA**:
```
用户: 请使用远程CUDA环境编译 kernel_dataset/cuda/add_vectors_kernel.cu
Agent: [自动使用 remote-cuda-builder skill]
       1. 检查SSH连接
       2. 检查远程容器
       3. 同步代码
       4. 执行远程编译
       5. 回传结果
```

#### 2. 批量编译

```
用户: 编译所有的SYCL kernel
Agent: [使用 b60-sycl-builder compile-all]
       正在批量编译 30 个kernel...
       [████████████████████] 100%
       结果: 成功 28 个, 失败 2 个
       查看详细报告: results/b60/summary_20260303_120000.json
```

#### 3. 检查状态

```
用户: 查看编译状态
Agent: [读取 .build_status.json]
       
       B60 SYCL Build Status
       ======================
       Container: lsv-container
       Compiler: icpx
       Total: 30
       Success: 28
       Failed: 2
       
       Failed kernels:
       - softmax_kernel: error: no matching function...
       - winograd_kernel: error: undefined reference...
```

#### 4. 环境检查

```
用户: 检查构建环境
Agent: [执行环境检查]
       
       B60 SYCL Environment: ✅ READY
       - Docker: ✅
       - Container lsv-container: ✅ Running
       - Compiler icpx: ✅ Available
       
       Remote CUDA Environment: ✅ READY
       - SSH connection: ✅
       - Container cuda12.9-test: ✅ Running
       - NVCC: ✅ Available
```

### 使用工具命令

Agent也可以直接调用工具脚本：

```python
# 使用bash工具调用构建脚本
bash({
    "command": "./tools/build.sh b60 compile kernel_dataset/sycl/softmax_kernel.dp.cpp",
    "description": "Compile SYCL kernel"
})

# 查看状态
bash({
    "command": "./tools/build.sh all status",
    "description": "Show build status"
})

# 读取编译日志
read({
    "filePath": "/home/intel/tianfeng/opencode_bench/results/b60/compile_softmax_kernel_20260303_120000.log"
})
```

## 高级用法

### 1. 对比编译

```
用户: 对比CUDA和SYCL的编译结果
Agent: 
1. 编译CUDA版本
   $ ./tools/build.sh cuda compile kernel_dataset/cuda/add_vectors_kernel.cu
   
2. 编译SYCL版本
   $ ./tools/build.sh b60 compile kernel_dataset/sycl/add_vectors_kernel.dp.cpp
   
3. 对比结果
   CUDA: ✅ Success (2.5s)
   SYCL: ✅ Success (0.8s)
   
   输出文件:
   - results/cuda/build_cuda/add_vectors_kernel.o (15KB)
   - results/b60/build_sycl/add_vectors_kernel.dp.o (31KB)
```

### 2. 批量转换和编译

```
用户: 帮我转换并编译所有的winograd kernel
Agent:
1. 转换CUDA winograd kernel到SYCL
   [转换过程...]
   
2. 编译SYCL版本
   $ ./tools/build.sh b60 compile kernel_dataset/sycl/winograd_input_transform_kernel.dp.cpp
   $ ./tools/build.sh b60 compile kernel_dataset/sycl/winograd_output_transform_kernel.dp.cpp
   ...
   
3. 验证编译结果
   检查 results/b60/build_sycl/ 下的输出文件
```

### 3. 错误诊断

```
用户: 为什么这个kernel编译失败了？
Agent:
1. 检查状态
   $ ./tools/build.sh b60 status
   
2. 读取错误日志
   [读取 results/b60/compile_failed_kernel_20260303_120000.log]
   
3. 分析错误
   错误类型: syntax error
   错误位置: line 45, column 12
   错误信息: expected ';' before '{'
   
4. 提供修复建议
   建议: 在line 44末尾添加分号
```

## 配置示例

### 完整的 `.opencode/config.json`

```json
{
  "name": "cuda-to-sycl-converter",
  "description": "CUDA to SYCL kernel conversion project",
  
  "permissions": {
    "skill": {
      "b60-sycl-builder": "allow",
      "remote-cuda-builder": "allow"
    },
    "bash": {
      "allowed_paths": [
        "/home/intel/tianfeng/opencode_bench/*",
        "/workspace/*"
      ]
    }
  },
  
  "skills": {
    "auto_discover": true,
    "directories": [
      ".opencode/skills"
    ]
  },
  
  "build_environments": {
    "b60": {
      "name": "B60 SYCL",
      "type": "sycl",
      "container": "lsv-container",
      "compiler": "icpx",
      "enabled": true
    },
    "cuda": {
      "name": "Remote CUDA",
      "type": "cuda",
      "host": "10.112.229.160",
      "container": "cuda12.9-test",
      "compiler": "/usr/local/cuda/bin/nvcc",
      "enabled": true
    }
  }
}
```

## 故障排除

### Skill未被发现

**问题**: opencode没有显示可用的skill

**解决**:
1. 检查skill目录结构:
   ```bash
   ls -la .opencode/skills/*/SKILL.md
   ```

2. 检查skill文件格式:
   - 必须以 `---` 开头（YAML frontmatter）
   - 必须包含 `name` 和 `description` 字段
   - 文件名必须大写: `SKILL.md`

3. 重启opencode或刷新skill:
   ```
   /reload
   /skill refresh
   ```

### Permission Denied

**问题**: Agent无法使用skill

**解决**:
在 `.opencode/config.json` 中添加:
```json
{
  "permissions": {
    "skill": {
      "b60-sycl-builder": "allow",
      "remote-cuda-builder": "allow"
    }
  }
}
```

### 环境检查失败

**问题**: Skill报告环境未就绪

**解决**:
1. 手动检查环境:
   ```bash
   ./tools/test_connectivity.sh
   ```

2. 启动缺失的服务:
   ```bash
   # B60容器
   docker start lsv-container
   
   # 远程CUDA容器
   ssh root@10.112.229.160 'docker start cuda12.9-test'
   ```

## 最佳实践

1. **始终在编译前检查环境**:
   ```
   用户: 先检查环境，然后编译所有kernel
   ```

2. **保存日志以便调试**:
   ```
   用户: 编译这个kernel并保存详细日志
   ```

3. **批量操作时先测试单个**:
   ```
   用户: 先编译一个测试，成功后再批量编译
   ```

4. **定期清理旧构建**:
   ```
   用户: 清理旧的构建产物
   Agent: $ ./tools/build.sh all clean
   ```

## 参考资源

- Skill文档: `.opencode/skills/*/SKILL.md`
- 工具脚本: `tools/*.py`, `tools/*.sh`
- 配置文件: `.opencode/config.json`
- 构建状态: `.build_status.json`
- 测试报告: `test/TEST_REPORT.md`

## 获取帮助

在opencode对话中:
```
/help skill           # 显示skill帮助
/skill list           # 列出所有skill
/skill info b60-sycl-builder    # 显示skill详情
```
