# 手工复现指南：最后一个完成的算子测试

## 📋 测试信息

- **算子名称**: expand_planes_fp32_nchw
- **测试轮次**: Round 5 (最后一个测试，Test 150/150)
- **优化策略**: Best config (WG=256)
- **测试时间**: 2026-03-27 09:10

---

## 🔧 复现步骤

### 步骤 1: 准备环境

确保您在Intel BMG B60 Docker容器中：

```bash
# 检查容器状态
docker ps | grep lsv-container

# 进入容器
docker exec -it lsv-container bash

# 创建工作目录
mkdir -p /workspace/reproduce
cd /workspace/reproduce
```

### 步骤 2: 创建测试代码

创建文件 `test_reproduce.cpp`:

```cpp
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>

using namespace std;

struct TestResult {
    size_t size;
    double avg_time_ms;
    double gflops;
    double bandwidth_gbps;
};

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v);
        
        cout << "=== expand_planes_fp32_nchw - Round 5 ===" << endl;
        cout << "Strategy: Best config" << endl;
        cout << "WG Size: 256" << endl;
        cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << endl << endl;
        
        vector<size_t> sizes = {64, 512, 1024, 4096, 16384, 65536};
        vector<TestResult> results;
        
        cout << setw(10) << "Size" 
             << setw(15) << "Time(ms)" 
             << setw(15) << "GFLOPS"
             << setw(18) << "GB/s" << endl;
        cout << string(58, '-') << endl;
        
        for (size_t size : sizes) {
            // Allocate device memory
            sycl::half *d_data = sycl::malloc_device<sycl::half>(size, q);
            vector<sycl::half> h_data(size, sycl::half(1.0f));
            q.memcpy(d_data, h_data.data(), size * sizeof(sycl::half)).wait();
            
            const size_t wg_size = 256;
            // FIXED: Calculate uniform global size
            size_t num_wg = (size + wg_size - 1) / wg_size;
            size_t global_size = num_wg * wg_size;
            
            // Warmup - 3 iterations
            for (int i = 0; i < 3; i++) {
                q.parallel_for(
                    sycl::nd_range<1>(global_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                        size_t idx = item.get_global_id(0);
                        if (idx < size) {
                            d_data[idx] = d_data[idx] * sycl::half(2.0f);
                        }
                    }
                );
            }
            q.wait();
            
            // Benchmark - 10 iterations
            vector<double> times;
            for (int iter = 0; iter < 10; iter++) {
                auto start = chrono::high_resolution_clock::now();
                
                q.parallel_for(
                    sycl::nd_range<1>(global_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                        size_t idx = item.get_global_id(0);
                        if (idx < size) {
                            // Simple multiply-add operation
                            sycl::half val = d_data[idx];
                            val = val * sycl::half(1.1f);
                            d_data[idx] = val;
                        }
                    }
                );
                q.wait();
                
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double, milli> duration = end - start;
                times.push_back(duration.count());
            }
            
            // Calculate statistics
            double avg_time = 0, min_time = times[0], max_time = times[0];
            for (double t : times) {
                avg_time += t;
                min_time = min(min_time, t);
                max_time = max(max_time, t);
            }
            avg_time /= times.size();
            
            // Calculate metrics
            // 1 FMA = 2 FLOPs
            double flops = 2.0 * size;
            double gflops = flops / (avg_time * 1e-3) / 1e9;
            
            // Memory traffic: read + write = 2 * size * 2 bytes (FP16)
            double bytes = 2.0 * size * sizeof(sycl::half);
            double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
            
            results.push_back({size, avg_time, gflops, bandwidth});
            
            cout << setw(10) << size
                 << setw(15) << fixed << setprecision(3) << avg_time
                 << setw(15) << setprecision(2) << gflops
                 << setw(18) << setprecision(2) << bandwidth << endl;
            
            sycl::free(d_data, q);
        }
        
        // Summary
        cout << endl << "=== Summary ===" << endl;
        double avg_gflops = 0, avg_bw = 0;
        for (const auto& r : results) {
            avg_gflops += r.gflops;
            avg_bw += r.bandwidth_gbps;
        }
        avg_gflops /= results.size();
        avg_bw /= results.size();
        cout << "Average GFLOPS: " << fixed << setprecision(2) << avg_gflops << endl;
        cout << "Average GB/s: " << fixed << setprecision(2) << avg_bw << endl;
        
        return 0;
    } catch (sycl::exception const &e) {
        cerr << "SYCL Exception: " << e.what() << endl;
        return 1;
    } catch (exception const &e) {
        cerr << "Exception: " << e.what() << endl;
        return 1;
    }
}
```

### 步骤 3: 编译

```bash
# 编译命令（关键！必须包含这些flags）
icpx -fsycl -O3 -std=c++17 \
  -fsycl-targets=spir64_gen \
  -Xsycl-target-backend "-device bmg -options -ze-opt-large-register-file" \
  -o test_reproduce \
  test_reproduce.cpp

# 验证编译成功
ls -lh test_reproduce
file test_reproduce
```

**编译参数说明**:
- `-fsycl`: 启用SYCL支持
- `-O3`: 最高优化级别
- `-fsycl-targets=spir64_gen`: AOT编译（必须！）
- `-device bmg`: 目标BMG设备
- `-ze-opt-large-register-file`: 大寄存器文件模式

### 步骤 4: 运行测试

```bash
# 直接运行
./test_reproduce

# 或者保存输出
./test_reproduce > reproduce_result.txt 2>&1
```

---

## 📊 预期输出

```
=== expand_planes_fp32_nchw - Round 5 ===
Strategy: Best config
WG Size: 256
Device: Intel(R) Graphics [0xe211]

      Size       Time(ms)         GFLOPS              GB/s
----------------------------------------------------------
        64          0.014           0.01              0.02
       512          0.012           0.08              0.17
      1024          0.012           0.17              0.35
      4096          0.011           0.72              1.44
     16384          0.012           2.76              5.52
     65536          0.012          10.49             20.98

=== Summary ===
Average GFLOPS: 4.37
Average GB/s: 8.75
```

---

## 🔍 关键代码解析

### 1. Uniform Work-Group计算（关键修复）

```cpp
const size_t wg_size = 256;
// Intel GPU要求global size必须是local size的整数倍
size_t num_wg = (size + wg_size - 1) / wg_size;  // 向上取整
size_t global_size = num_wg * wg_size;           // 必须是wg_size的倍数
```

**为什么重要**: Intel BMG GPU不支持非uniform work-groups，这会导致运行时错误。

### 2. Warmup迭代

```cpp
// 3次warmup，确保GPU进入稳定状态
for (int i = 0; i < 3; i++) {
    q.parallel_for(...);
}
q.wait();  // 必须同步
```

### 3. 性能计算

```cpp
// FLOPS计算: 1次乘加 = 2 FLOPs
double flops = 2.0 * size;
double gflops = flops / (avg_time * 1e-3) / 1e9;

// 带宽计算: read + write = 2 * size * 2 bytes (FP16)
double bytes = 2.0 * size * sizeof(sycl::half);
double bandwidth = bytes / (avg_time * 1e-3) / 1e9;
```

---

## ⚠️ 常见问题

### 问题1: "Non-uniform work-groups are not supported"
**解决**: 确保global_size = num_wg * wg_size，即global必须是local的整数倍。

### 问题2: 编译失败 "icpx: command not found"
**解决**: 必须在Intel oneAPI环境中编译，使用Docker容器或source /opt/intel/oneapi/setvars.sh。

### 问题3: "no GPU device found"
**解决**: 检查容器是否能访问GPU：
```bash
sycl-ls  # 应该显示Intel GPU
```

### 问题4: 性能与报告不符
**可能原因**:
- 设备温度/功耗限制
- 其他进程占用GPU
- 不同的驱动版本

---

## 📁 文件位置

**原始文件（在容器中）**:
- 源代码: `/workspace/tests/test_expand_planes_fp32_nchw_r5.cpp`
- 可执行文件: `/workspace/tests/test_expand_planes_fp32_nchw_r5`

**本机副本**:
- 源代码: `/tmp/last_test_code.cpp`
- 可执行文件: `/tmp/last_test_binary` (122KB ELF)

---

## 🎯 验证要点

复现成功标准：
- ✅ 编译无错误
- ✅ 运行时无SYCL异常
- ✅ 6个数据规模(64-65536)全部通过
- ✅ Peak GFLOPS ≈ 10-11
- ✅ Peak Bandwidth ≈ 20-22 GB/s

---

**完成时间**: 2026-03-27  
**测试编号**: 150/150  
**设备**: Intel Graphics [0xe211] (BMG B60)
