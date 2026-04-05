# Skill: SYCL BMG Kernel 开发编译环境

## 环境概述
在持久运行的 Docker 容器中编译和测试 SYCL kernel，目标设备为 Intel BMG (Battlemage) 离散 GPU。
编译选项参考自 [intel/torch-xpu-ops](https://github.com/intel/torch-xpu-ops) 的 `cmake/BuildFlags.cmake`。

## 容器信息
- **容器名**: `lsv-container_2026_3`
- **基础镜像**: `intel/llm-scaler-vllm:0.14.0-b8.1`
- **编译器**: `icpx` (Intel DPC++/C++ Compiler 2025.3, 已安装)
- **代理**: `https_proxy=http://child-prc.intel.com:912`

## 目录映射
```
宿主机                                    → 容器内        用途
─────────────────────────────────────────────────────────────
/home/intel/tianfeng/opencode_bench       → /sandbox     代码目录 (持久)
/tmp/intel/                               → /workspace   编译输出/临时目录 (WORKDIR)
/home/intel/                              → /intel       宿主机 home (只读参考)
```

### 使用规则
- **源码**放在 `/sandbox/`，编辑在宿主机 `/home/intel/tianfeng/opencode_bench/` 同步可见
- **编译产物**输出到 `/workspace/build/`
- **不要**在 `/sandbox/` 内做 build，避免污染代码目录

## 日常操作

### 进入容器
```bash
docker exec -it lsv-container_2026_3 bash
```

### 初始化环境（每次 exec 进入后执行）
```bash
source /opt/intel/oneapi/compiler/latest/env/vars.sh
# 如果 pti/umf/ccl 则一并 source:
# source /opt/intel/oneapi/pti/latest/env/vars.sh
# source /opt/intel/oneapi/umf/latest/env/vars.sh
icpx --version
sycl-ls
```

> 建议写入 `~/.bashrc` 实现自动化，避免每次手动 source。

## 编译命令

### 最简编译（日常开发推荐）
```bash
icpx -fsycl \
  -fsycl-targets=spir64_gen \
  -Xs "-device bmg" \
  -fsycl-unnamed-lambda \
  -std=c++17 -O2 \
  /sandbox/kernel.cpp -o /workspace/build/kernel
```

### 完整编译（参考 torch-xpu-ops 全部优化选项）
```bash
icpx -fsycl \
  -fsycl-targets=spir64_gen \
  -fsycl-unnamed-lambda \
  -sycl-std=2020 \
  -std=c++17 -O2 \
  -fno-fast-math -fma -no-ftz \
  -Xs "-device bmg \
       -options -cl-intel-enable-auto-large-GRF-mode \
       -options -cl-fp32-correctly-rounded-divide-sqrt \
       -options -cl-intel-greater-than-4GB-buffer-required \
       -options -cl-poison-unsupported-fp64-kernels" \
  --offload-compress \
  /sandbox/kernel.cpp -o /workspace/build/kernel
```

### Debug 编译
```bash
icpx -fsycl \
  -fsycl-targets=spir64_gen \
  -Xs "-device bmg" \
  -fsycl-unnamed-lambda \
  -std=c++17 -g -O0 \
  -Rno-debug-disables-optimization \
  /sandbox/kernel.cpp -o /workspace/build/kernel_dbg
```

### AOT + JIT fallback（生成可在非 BMG 设备上 JIT 回退的二进制）
```bash
icpx -fsycl \
  -fsycl-targets=spir64_gen,spir64 \
  -Xs "-device bmg" \
  -fsycl-unnamed-lambda \
  -std=c++17 -O2 \
  /sandbox/kernel.cpp -o /workspace/build/kernel
```

### 多文件编译
```bash
# 分步编译
icpx -fsycl -fsycl-targets=spir64_gen -fsycl-unnamed-lambda -std=c++17 -O2 -c /sandbox/a.cpp -o /workspace/build/a.o
icpx -fsycl -fsycl-targets=spir64_gen -fsycl-unnamed-lambda -std=c++17 -O2 -c /sandbox/b.cpp -o /workspace/build/b.o

# 链接
icpx -fsycl -fsycl-targets=spir64_gen \
  -Xs "-device bmg -options -cl-intel-enable-auto-large-GRF-mode" \
  --offload-compress \
  /workspace/build/a.o /workspace/build/b.o -o /workspace/build/app
```

### CMake 项目
```bash
mkdir -p /workspace/build && cd /workspace/build
cmake /sandbox \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=spir64_gen -fsycl-unnamed-lambda" \
  -DCMAKE_EXE_LINKER_FLAGS='-Xs "-device bmg"' \
  -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

对应的最小 CMakeLists.txt:
```cmake
cmake_minimum_required(VERSION 3.20)
project(my_sycl_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
add_executable(kernel kernel.cpp)
target_compile_options(kernel PRIVATE -fsycl -fsycl-targets=spir64_gen -fsycl-unnamed-lambda)
target_link_options(kernel PRIVATE -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg")
```

## 运行与测试
```bash
# 运行
cd /workspace/build
./kernel

# 查看 GPU 信息
sycl-ls
xpu-smi discovery   # 如果安装了 xpu-smi

# 设置默认设备 (多 GPU 时)
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu:0"
./kernel
```

## 编译标志速查

### icpx 编译标志
| 标志 | 作用 | 何时使用 |
|------|------|---------|
| `-fsycl` | 启用 SYCL 模式 | **必须** |
| `-fsycl-targets=spir64_gen` | AOT 编译 | **BMG 必须** |
| `-fsycl-targets=spir64_gen,spir64` | AOT + JIT 回退 | 需跨设备分发时 |
| `-std=c++17` | C++ 标准 | 默认用 17 |
| `-std=c++20` | C++ 标准 | 用 SYCL-TLA 时 |
| `-sycl-std=2020` | SYCL 标准 | 可选 |
| `-fsycl-unnamed-lambda` | 匿名 lambda | **必须**（我们的 kernel 用 lambda 写法） |
| `-fno-fast-math` | 禁用 fast-math | 需要精度时 |
| `-fma` | 启用 FMA 指令 | 性能+精度 |
| `-no-ftz` | 禁用 flush-to-zero | 需要 denorm 精度时 |
| `-fsycl-fp64-conv-emu` | FP64 模拟 | 目标含 dg2 时 |
| `--offload-compress` | 压缩设备代码 | 链接时减小体积 |

### 离线编译器选项（通过 -Xs 传递）
| 选项 | 作用 |
|------|------|
| `-device bmg` | **目标设备** |
| `-device pvc,bmg` | 多设备 |
| `-options -cl-intel-enable-auto-large-GRF-mode` | 自动大寄存器（**推荐**） |
| `-options -cl-fp32-correctly-rounded-divide-sqrt` | FP32 精确除法开方 |
| `-options -cl-intel-greater-than-4GB-buffer-required` | 支持 >4GB buffer |
| `-options -cl-poison-unsupported-fp64-kernels` | FP64 不支持时报错 |

### 设备代号
| 代号 | GPU | FP64 |
|------|-----|------|
| `bmg` | Arc B-Series (Battlemage) | 无原生 |
| `pvc` | Data Center GPU Max (Ponte Vecchio) | 有 |
| `dg2` | Arc A-Series (Alchemist) | 无原生 |
| `mtl-h` | Core Ultra (Meteor Lake) | 无原生 |
| `lnl-m` | Lunar Lake | 无原生 |

## 常见问题

**Q: 编译报 `fp64 is not supported`**
BMG 无原生 FP64。避免在 kernel 中使用 `double`，或加 `-fsycl-fp64-conv-emu`（性能差）。

**Q: 运行时 `PI_ERROR_DEVICE_NOT_FOUND`**
容器未正确映射 GPU。确认 `--device=/dev/dri` 以及 `sycl-ls` 能看到设备。

**Q: AOT 编译慢**
只指定 `-device bmg`，不要编译多设备。日常开发可去掉 AOT 用纯 JIT:
```bash
icpx -fsycl -std=c++17 -O2 kernel.cpp -o kernel
```

**Q: 编译报 `unnamed type is invalid`**
缺少 `-fsycl-unnamed-lambda`。**不要**用 `-fno-sycl-unnamed-lambda`，它和匿名 lambda kernel 冲突。

**Q: 每次 exec 都要 source oneapi**
写入 bashrc:
```bash
echo 'source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null' >> ~/.bashrc
```

## 快捷别名（建议加入 ~/.bashrc）
```bash
# BMG 快速编译
alias bmg='f(){ source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null; icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg -options -cl-intel-enable-auto-large-GRF-mode" -fsycl-unnamed-lambda -std=c++17 -O2 "$@"; }; f'
# 用法: bmg /sandbox/test.cpp -o /workspace/build/test

# BMG debug 编译
alias bmg-dbg='f(){ source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null; icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg" -fsycl-unnamed-lambda -std=c++17 -g -O0 "$@"; }; f'

# 纯 JIT 快速编译（开发迭代用，跳过 AOT）
alias sycl-jit='f(){ source /opt/intel/oneapi/compiler/latest/env/vars.sh 2>/dev/null; icpx -fsycl -std=c++17 -O2 "$@"; }; f'
```
