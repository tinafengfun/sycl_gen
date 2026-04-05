#!/bin/bash
# GitHub 提交指南 | GitHub Submission Guide

echo "=========================================="
echo "🚀 CUDA-SYCL Harnesses - GitHub 提交指南"
echo "=========================================="
echo ""

# 检查 git 是否安装
if ! command -v git &> /dev/null; then
    echo "❌ Git 未安装，请先安装 Git"
    exit 1
fi

echo "📋 提交步骤："
echo ""

# 1. 初始化仓库（如果需要）
echo "1. 初始化 Git 仓库（如果是新项目）："
echo "   git init"
echo ""

# 2. 添加文件
echo "2. 添加所有文件到暂存区："
echo "   git add ."
echo ""

# 3. 提交
echo "3. 提交更改："
echo "   git commit -m 'Initial commit: CUDA-SYCL kernel accuracy test suite'"
echo ""

# 4. 添加远程仓库
echo "4. 添加远程仓库："
echo "   git remote add origin https://github.com/YOUR_USERNAME/cuda-sycl-harnesses.git"
echo ""

# 5. 推送到 GitHub
echo "5. 推送到 GitHub："
echo "   git push -u origin main"
echo ""

echo "=========================================="
echo "📁 提交文件清单："
echo "=========================================="
echo ""

# 列出所有要提交的文件
find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "LICENSE" -o -name ".gitignore" \) ! -path "./.git/*" | sort | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "  ✅ ${file:2} (${size})"
done

echo ""
echo "=========================================="
echo "🎯 GitHub 仓库建议设置："
echo "=========================================="
echo ""
echo "1. 仓库名称：cuda-sycl-harnesses"
echo "2. 描述：CUDA to SYCL kernel accuracy test suite for LCZero chess engine"
echo "3. 可见性：Public"
echo "4. 添加 topics：cuda, sycl, oneapi, lczero, gpu, machine-learning"
echo ""
echo "=========================================="
echo "📝 提交前检查清单："
echo "=========================================="
echo ""
echo "□ 所有代码文件已添加"
echo "□ README.md 完整且准确"
echo "□ LICENSE 文件包含"
echo "□ .gitignore 配置正确"
echo "□ 敏感信息（密码、IP）已移除或替换为占位符"
echo "□ 代码可以正常运行"
echo ""
echo "=========================================="
echo "✅ 准备就绪！执行以下命令提交："
echo "=========================================="
echo ""
echo "git init"
echo "git add ."
echo "git commit -m 'Initial commit: CUDA-SYCL kernel accuracy test suite'"
echo "git remote add origin https://github.com/YOUR_USERNAME/cuda-sycl-harnesses.git"
echo "git push -u origin main"
echo ""
