# opencode 使用示例对话

以下是在opencode中使用B60 SYCL Builder和Remote CUDA Builder skill的示例对话。

---

## 示例1: 基础编译

**用户**: 帮我编译一下这个SYCL kernel

**Agent**: 
我会帮你编译 `kernel_dataset/sycl/add_vectors_kernel.dp.cpp`。让我使用B60环境进行编译。

```bash
$ ./tools/build.sh b60 compile kernel_dataset/sycl/add_vectors_kernel.dp.cpp

[CHECK] Checking container: lsv-container
[OK] Container is ready
[SCRIPT] Generated: scripts/b60/build_add_vectors_kernel_20260303_100000.sh
[SYNC] Syncing code to container...
[SYNC] Creating directory: /workspace/kernel_dataset
[SYNC] Copying files...
[OK] Code synced successfully (4 items)
[BUILD] Starting compilation...
[SUCCESS] Compilation completed in 2.34s
[LOG] Saved: results/b60/compile_add_vectors_kernel_20260303_100000.log
[OUTPUT] results/b60/build_sycl/add_vectors_kernel.o
```

✅ **编译成功！**
- 编译时间: 2.34秒
- 输出文件: `results/b60/build_sycl/add_vectors_kernel.o` (45KB)
- 编译日志: `results/b60/compile_add_vectors_kernel_20260303_100000.log`

---

## 示例2: 批量编译

**用户**: 编译所有的SYCL kernel

**Agent**:
我会批量编译所有在 `kernel_dataset/sycl/` 目录下的kernel。

```bash
$ ./tools/build.sh b60 compile-all

[BATCH] Found 30 kernels to compile

[1/30] Processing: add_vectors_kernel.dp.cpp
[SUCCESS] Compilation completed in 2.34s

[2/30] Processing: add_bias_batched_kernel.dp.cpp
[SUCCESS] Compilation completed in 1.89s

...

[29/30] Processing: winograd_output_transform_kernel.dp.cpp
[FAILED] Compilation failed (exit code: 1)
[LOG] Saved: results/b60/compile_winograd_output_transform_kernel_20260303_100100.log

[30/30] Processing: se_layer_nhwc_kernel.dp.cpp
[SUCCESS] Compilation completed in 5.67s
```

📊 **批量编译报告**
```
============================================================
[SUMMARY] Batch Compilation Report
============================================================
Total:   30
Success: 28
Failed:  2
Rate:    93.3%
Report:  results/b60/summary_20260303_100200.json
============================================================
```

⚠️ **有2个kernel编译失败**:
1. `winograd_output_transform_kernel` - 语法错误: line 45
2. `se_layer_nhwc_kernel` - 类型不匹配

需要我帮你查看详细的错误日志吗？

---

## 示例3: 对比编译

**用户**: 对比CUDA和SYCL版本的编译结果

**Agent**:
我会分别编译CUDA和SYCL版本，然后对比结果。

**步骤1: 编译CUDA版本**
```bash
$ ./tools/build.sh cuda compile kernel_dataset/cuda/add_vectors_kernel.cu

[CHECK] SSH connection successful
[CHECK] Container cuda12.9-test is running
[CHECK] NVCC available
[SYNC L1] Local -> Remote host...
[OK] Synced to remote host (32 items)
[SYNC L2] Remote host -> Container...
[OK] Synced to container (32 items)
[BUILD] Starting remote compilation...
[SUCCESS] Compilation completed in 8.45s
```

**步骤2: 编译SYCL版本**
```bash
$ ./tools/build.sh b60 compile kernel_dataset/sycl/add_vectors_kernel.dp.cpp

[CHECK] Container lsv-container is ready
[BUILD] Starting compilation...
[SUCCESS] Compilation completed in 2.34s
```

**步骤3: 对比结果**
```
╔═══════════════════════════════════════════════════════╗
║           CUDA vs SYCL 编译对比                        ║
╚═══════════════════════════════════════════════════════╝

Kernel: add_vectors
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CUDA (Remote GPU):
  ✓ 编译状态: 成功
  ⏱ 编译时间: 8.45s (含网络传输)
  📦 输出大小: 15.2KB
  🔧 编译器: NVCC 12.9
  🎯 目标架构: sm_70
  📄 日志文件: results/cuda/compile_add_vectors_*.log

SYCL (B60 Local):
  ✓ 编译状态: 成功
  ⏱ 编译时间: 2.34s
  📦 输出大小: 31.4KB
  🔧 编译器: Intel oneAPI 2025.1
  🎯 目标架构: x86_64 + Intel GPU
  📄 日志文件: results/b60/compile_add_vectors_*.log

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
结论:
  • SYCL编译更快 (2.34s vs 8.45s)
  • SYCL输出文件更大 (31.4KB vs 15.2KB)
  • 两者都编译成功
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 示例4: 错误诊断

**用户**: 为什么winograd kernel编译失败了？

**Agent**:
让我帮你检查winograd kernel的编译错误。

**步骤1: 检查状态**
```bash
$ ./tools/build.sh b60 status

B60 SYCL Build Status
============================================================
Container: lsv-container
Compiler:  icpx
Total:     30
Success:   28
Failed:    2
============================================================

Failed kernels:
  - winograd_output_transform_kernel: error: no matching function for call to 'atomicAdd'
  - se_layer_nhwc_kernel: error: expected ';' before '{' token
```

**步骤2: 查看详细日志**
```bash
$ cat results/b60/compile_winograd_output_transform_kernel_20260303_100100.log
```

**错误分析**:
```
Exit code: 1
=== STDOUT ===
=== SYCL Compilation Start ===
...

=== STDERR ===
/workspaces/kernel_dataset/sycl/winograd_output_transform_kernel.dp.cpp:127:15:
error: no matching function for call to 'atomicAdd'
    atomicAdd(&sum, val);
    ^~~~~~~~~

note: candidate function not viable: no known conversion from 'sycl::atomic_ref<float>' 
      to 'float *' for 1st argument
```

**问题诊断**:
错误类型: SYCL原子操作API不兼容
位置: winograd_output_transform_kernel.dp.cpp, line 127
原因: CUDA的 `atomicAdd(float*, float)` 与SYCL的 `atomic_ref` 接口不同

**修复建议**:
```cpp
// CUDA版本
atomicAdd(&sum, val);

// 需要改为SYCL版本
sycl::atomic_ref<float, sycl::memory_order::relaxed,
                 sycl::memory_scope::device,
                 sycl::access::address_space::global_space> atomic_sum(sum);
atomic_sum.fetch_add(val);
```

需要我帮你自动修复这个问题吗？

---

## 示例5: 环境检查

**用户**: 检查环境是否都准备好了

**Agent**:
让我检查所有的构建环境。

```bash
$ ./tools/test_connectivity.sh

╔════════════════════════════════════════════════════════╗
║       Environment Connectivity Quick Test             ║
╚════════════════════════════════════════════════════════╝

[B60] Testing B60 SYCL Environment...
✓ Docker is available
✓ Container 'lsv-container' is running
✓ SYCL compiler (icpx) is available

[CUDA] Testing Remote CUDA Environment...
✓ SSH is available
✓ SCP is available
✓ SSH connection successful
✓ Container 'cuda12.9-test' is running
✓ CUDA compiler (nvcc) is available

═════════════════════════════════════════════════════════
Summary:

  ✓ B60 SYCL Environment: READY
  ✓ Remote CUDA Environment: READY
═════════════════════════════════════════════════════════

All environments are ready for building!
```

✅ **所有环境已就绪！**

你现在可以开始编译kernel了。有什么需要编译的吗？

---

## 示例6: 清理和重置

**用户**: 清理所有旧的构建产物

**Agent**:
我会清理所有的构建产物和日志文件。

```bash
$ ./tools/build.sh all clean

[WARN] Cleaning B60 build artifacts...
[SUCCESS] Cleaned B60 build artifacts

[WARN] Cleaning CUDA build artifacts...
[SUCCESS] Cleaned CUDA build artifacts

[SUCCESS] Cleaned all build artifacts
```

✅ **清理完成！**

已删除:
- results/b60/* (编译产物和日志)
- results/cuda/* (编译产物和日志)
- scripts/b60/* (生成的脚本)
- scripts/cuda/* (生成的脚本)
- .build_status.json (状态文件)

环境已重置，可以开始新的构建了。

---

## 示例7: 状态查询

**用户**: 查看所有kernel的编译状态

**Agent**:
让我查看完整的编译状态。

```bash
$ ./tools/build.sh all status

=== B60 SYCL Environment ===
B60 SYCL Build Status
============================================================
Container: lsv-container
Compiler:  icpx
Total:     30
Success:   28
Failed:    2
============================================================

Failed kernels:
  - winograd_output_transform_kernel: error: no matching function...
  - se_layer_nhwc_kernel: error: expected ';' before...

=== Remote CUDA Environment ===
Remote CUDA Build Status
============================================================
SSH Host:  root@10.112.229.160
Container: cuda12.9-test
Compiler:  /usr/local/cuda/bin/nvcc
Total:     30
Success:   30
Failed:    0
============================================================

✓ All CUDA kernels compiled successfully!
```

📊 **状态总结**:
- SYCL: 28/30 成功 (93.3%)
- CUDA: 30/30 成功 (100%)

需要我帮你修复失败的SYCL kernel吗？

---

## 使用技巧

### 技巧1: 快速编译最近修改的文件

**用户**: 编译最近修改的kernel

**Agent**:
```bash
# 查找最近修改的.dp.cpp文件
$ find kernel_dataset/sycl -name "*.dp.cpp" -mtime -1 -exec ./tools/build.sh b60 compile {} \;
```

### 技巧2: 只编译失败的kernel

**用户**: 重新编译上次失败的kernel

**Agent**:
```bash
# 从状态文件中提取失败的kernel
$ cat .build_status.json | jq '.environments.b60.kernels | to_entries[] | select(.value.status == "failed") | .key'

# 然后逐个编译
for kernel in failed_kernels; do
    ./tools/build.sh b60 compile "kernel_dataset/sycl/${kernel}.dp.cpp"
done
```

### 技巧3: 监控编译进度

**用户**: 批量编译时显示进度

**Agent**:
```bash
$ ./tools/build.sh b60 compile-all 2>&1 | tee /tmp/build.log | grep -E "(Processing|SUCCESS|FAILED)"

[1/30] Processing: add_vectors_kernel.dp.cpp
[SUCCESS] Compilation completed in 2.34s
[2/30] Processing: add_bias_batched_kernel.dp.cpp
[SUCCESS] Compilation completed in 1.89s
...
```

---

## 故障排除对话

**用户**: 编译失败了，帮我看看什么问题

**Agent**: 
让我帮你诊断编译失败的问题。

**步骤1: 确认是哪个环境失败**
是SYCL编译失败还是CUDA编译失败？

**用户**: SYCL失败了

**Agent**:
让我检查SYCL环境并查看错误日志。

```bash
# 检查环境
$ ./tools/build.sh b60 check

# 查看最近失败的日志
$ ls -lt results/b60/compile_*.log | head -5
$ cat results/b60/compile_$(ls -t results/b60/compile_*.log | head -1)
```

[分析错误日志...]

**诊断结果**:
错误类型: 头文件未找到
文件: kernel_dataset/sycl/new_kernel.dp.cpp
错误: `#include <sycl/sycl.hpp>` file not found

**原因**: SYCL运行时环境未正确配置

**解决方案**:
1. 检查容器中是否正确安装了Intel oneAPI
2. 确认环境变量是否设置正确
3. 或者修改代码使用标准C++库

需要我帮你执行这些修复步骤吗？

---

## 总结

在opencode对话中，你可以：

1. **直接请求** - "编译xxx文件"
2. **批量操作** - "编译所有kernel"
3. **对比结果** - "对比CUDA和SYCL编译"
4. **诊断问题** - "为什么编译失败了？"
5. **检查环境** - "环境准备好了吗？"
6. **查看状态** - "查看编译状态"
7. **清理重置** - "清理旧的构建"

Agent会自动使用对应的skill来帮助你完成任务！
