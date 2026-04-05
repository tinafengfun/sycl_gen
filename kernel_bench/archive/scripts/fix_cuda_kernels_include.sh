#!/bin/bash
# 修复所有CUDA内核文件 - 添加winograd_helper.inc包含

echo "========================================"
echo "🔧 修复所有CUDA内核文件"
echo "========================================"
echo ""
echo "📋 修复内容: 添加 #include \"winograd_helper.inc\""
echo "📍 原因: 提供activate函数定义"
echo ""

FIXED_COUNT=0
SKIPPED_COUNT=0
ERROR_COUNT=0

# 遍历所有CUDA内核文件
for cu_file in kernel_dataset/cuda/*_kernel.cu; do
    kernel_name=$(basename "$cu_file" .cu)
    
    # 检查是否已包含winograd_helper.inc
    if grep -q "winograd_helper.inc" "$cu_file" 2>/dev/null; then
        echo "⏭️  $kernel_name - 已包含winograd_helper.inc"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
    fi
    
    # 检查是否使用activate函数
    if ! grep -q "activate(" "$cu_file" 2>/dev/null; then
        echo "⏭️  $kernel_name - 不使用activate函数"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
    fi
    
    echo "🔧 修复: $kernel_name"
    
    # 创建备份
    cp "$cu_file" "${cu_file}.bak" 2>/dev/null
    
    # 查找包含utils/exception.h或类似头文件的行，在其后添加winograd_helper.inc
    # 使用sed在包含"utils/exception.h"的行后添加新包含
    if grep -q "utils/exception.h" "$cu_file" 2>/dev/null; then
        sed -i '/utils\/exception.h/a #include "winograd_helper.inc"  // For activate function' "$cu_file"
        echo "  ✅ 已添加包含 (在utils/exception.h后)"
        FIXED_COUNT=$((FIXED_COUNT + 1))
    elif grep -q "#include.*cuda_common.h" "$cu_file" 2>/dev/null; then
        # 如果没找到utils/exception.h，在cuda_common.h后添加
        sed -i '/cuda_common.h/a #include "winograd_helper.inc"  // For activate function' "$cu_file"
        echo "  ✅ 已添加包含 (在cuda_common.h后)"
        FIXED_COUNT=$((FIXED_COUNT + 1))
    else
        # 在第一个#include后添加
        sed -i '0,/#include/s/#include/#include\n#include "winograd_helper.inc"  \/\/ For activate function\n#include/' "$cu_file"
        echo "  ✅ 已添加包含 (在第一个include后)"
        FIXED_COUNT=$((FIXED_COUNT + 1))
    fi
done

echo ""
echo "========================================"
echo "📊 修复统计"
echo "========================================"
echo "  已修复: $FIXED_COUNT 个文件"
echo "  已跳过: $SKIPPED_COUNT 个文件"
echo "  错误:   $ERROR_COUNT 个文件"
echo ""

# 显示修复的文件
if [ $FIXED_COUNT -gt 0 ]; then
    echo "✅ 已修复的文件列表:"
    for cu_file in kernel_dataset/cuda/*_kernel.cu; do
        if [ -f "${cu_file}.bak" ]; then
            kernel_name=$(basename "$cu_file" .cu)
            echo "  • $kernel_name"
        fi
    done
    echo ""
    echo "💾 备份文件: *.cu.bak"
fi

echo "========================================"
