---
name: cuda-docker-test
description: 在远程CUDA Docker环境中编译和测试lc0 SYCL内核。用于在CUDA机器上收集参考数据。
disable-model-invocation: true
---

# CUDA Docker测试流程

在远程CUDA机器（10.112.229.160）的Docker容器中编译和测试lc0测试代码。

## 前置条件

- 远程机器：root@10.112.229.160
- Docker容器：cuda12.9-test
- 代码映射：/home/tianfeng (host) -> /workspace (docker)

## 测试步骤

### 1. 拷贝代码到远程机器

```bash
# 从本地拷贝test_cuda到远程机器
scp -r /home/intel/tianfeng/lc0/test_cuda root@10.112.229.160:/home/tianfeng/
```

### 2. 登录远程机器

```bash
ssh root@10.112.229.160
```

### 3. 进入Docker容器

```bash
docker exec -it cuda12.9-test bash
cd /workspace/test_cuda
```

### 4. 编译测试

```bash
# 清理并编译
make clean
make all
```

### 5. 运行测试

```bash
# 运行所有测试
make run

# 或单独运行特定测试
make policy      # PolicyMap测试
make se          # SE Layer测试
make winograd    # Winograd变换测试
make fc          # FC Layer测试
make attention   # Attention Body测试
```

### 6. 收集结果

```bash
# 在docker内检查输出
ls -lh reference_data/

# 退出docker，在host上打包
tar -czvf cuda_reference_data.tar.gz reference_data/
```

### 7. 拷贝结果回本地（可选）

```bash
# 在本地机器执行
scp root@10.112.229.160:/home/tianfeng/test_cuda/cuda_reference_data.tar.gz /home/intel/tianfeng/lc0/test_cuda/
```

## 一键执行脚本

如果需要自动化，可以使用以下脚本：

```bash
#!/bin/bash
set -e

# 1. 拷贝代码
echo "=== 拷贝代码到远程机器 ==="
scp -r /home/intel/tianfeng/lc0/test_cuda root@10.112.229.160:/home/tianfeng/

# 2. 在docker中编译和测试
echo "=== 在Docker中编译和测试 ==="
ssh root@10.112.229.160 << 'EOF'
cd /home/tianfeng/test_cuda
docker exec cuda12.9-test bash -c "cd /workspace/test_cuda && make clean && make all && make run"
ls -lh reference_data/
tar -czvf cuda_reference_data.tar.gz reference_data/
EOF

# 3. 拷贝结果回本地
echo "=== 拷贝结果回本地 ==="
scp root@10.112.229.160:/home/tianfeng/test_cuda/cuda_reference_data.tar.gz /home/intel/tianfeng/lc0/test_cuda/

echo "=== 完成 ==="
```

## 故障排除

### Docker容器未运行
```bash
docker start cuda12.9-test
```

### 编译错误
- 检查CUDA版本：nvcc --version
- 检查GPU架构：nvidia-smi
- 确保在/workspace目录下编译

### 权限问题
```bash
chown -R root:root /home/tianfeng/test_cuda
```

## 输出文件

测试完成后，reference_data/目录应包含：

```
reference_data/
├── PolicyMap_input.f32.bin
├── PolicyMap_output.f32.bin
├── SELayer_stage*.f32.bin
├── Winograd_*.f32.bin
├── FCLayer_*.f32.bin
└── AttentionBody_*.f32.bin
```
