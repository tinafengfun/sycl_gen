#!/bin/bash
# 批量修复SYCL内核的头文件依赖
# Batch fix SYCL kernel header dependencies

echo "========================================"
echo "批量修复头文件依赖"
echo "========================================"
echo ""

# 创建标准头文件内容
STANDARD_DEFS='
// ActivationFunction enum
enum ActivationFunction {
  ACTIVATION_NONE,
  ACTIVATION_RELU,
  ACTIVATION_RELU_2,
  ACTIVATION_TANH,
  ACTIVATION_SIGMOID,
  ACTIVATION_SELU,
  ACTIVATION_MISH,
  ACTIVATION_SWISH,
  ACTIVATION_DEFAULT,
  ACTIVATION_SOFTMAX
};

// Common constants
static constexpr int kNumOutputPolicy = 1858;
static constexpr int kMaxResBlockFusingChannels = 384;
static constexpr int kMaxResBlockFusingSeKFp16Ampere = 512;
static constexpr int kMaxResBlockFusingSeK = 128;
static constexpr int kInputPlanes = 112;
static constexpr int kOpInpTransformBlockSize = 64;

// Helper functions
inline int DivUp(int a, int b) { return (a + b - 1) / b; }
'

# 处理每个文件
for f in kernel_dataset/sycl/*.dp.cpp; do
    echo -n "处理 $(basename $f)... "
    
    # 检查是否有缺失的头文件
    if grep -q '#include "cuda_common.h"' "$f" 2>/dev/null || \
       grep -q '#include "neural/' "$f" 2>/dev/null || \
       grep -q '#include "utils/' "$f" 2>/dev/null; then
        
        # 备份原文件
        cp "$f" "$f.bak"
        
        # 创建临时文件
        TEMP=$(mktemp)
        
        # 写入标准定义
        echo '#include <algorithm>' > "$TEMP"
        echo '#include <cmath>' >> "$TEMP"
        echo '#include <sycl/sycl.hpp>' >> "$TEMP"
        echo '' >> "$TEMP"
        echo "$STANDARD_DEFS" >> "$TEMP"
        echo '' >> "$TEMP"
        
        # 追加原文件内容，跳过特定的include
        grep -v '#include "cuda_common.h"' "$f.bak" | \
        grep -v '#include "neural/' | \
        grep -v '#include "utils/' | \
        grep -v '#include <cuda_runtime.h>' | \
        grep -v '#include <cuda_fp16.h>' | \
        grep -v '#include <assert>' >> "$TEMP"
        
        # 替换文件
        mv "$TEMP" "$f"
        
        echo "✅ 已修复头文件"
    else
        echo "⏭️  无需修复"
    fi
done

echo ""
echo "========================================"
echo "开始编译测试..."
echo "========================================"
echo ""

PASS=0
FAIL=0

for f in kernel_dataset/sycl/*.dp.cpp; do
    kernel=$(basename "$f" .dp.cpp)
    echo -n "编译 $kernel... "
    
    docker cp "$f" lsv-container:/workspace/test.cpp 2>/dev/null
    docker exec lsv-container bash -c "cd /workspace && icpx -fsycl -c test.cpp -o test.o 2>/dev/null" >/dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅"
        PASS=$((PASS + 1))
    else
        echo "❌"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "========================================"
echo "编译统计:"
echo "  通过: $PASS"
echo "  失败: $FAIL"
echo "========================================"
