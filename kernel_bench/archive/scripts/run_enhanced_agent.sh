#!/bin/bash
# 启动增强版Agent v2.0
# 使用 DeepSeek-R1-G2-static-671B + Qwen3-Coder-30B

echo "========================================"
echo "🚀 启动增强版Agent v2.0"
echo "========================================"
echo ""
echo "📊 模型配置:"
echo "  • 预处理: Qwen3-Coder-30B-A3B-Instruct"
echo "  • 转换: DeepSeek-R1-G2-static-671B"
echo "  • 修复: DeepSeek-R1-G2-static-671B"
echo "  • 测试生成: Qwen3-Coder-30B-A3B-Instruct"
echo ""
echo "✨ 新特性:"
echo "  ✓ LLM高比重使用 (5个Stage全部使用LLM)"
echo "  ✓ 智能预处理 (依赖分析+头文件内联)"
echo "  ✓ 5轮渐进式修复"
echo "  ✓ 自动转换计划生成"
echo ""
echo "📈 预期改进:"
echo "  • 编译通过率: 27.6% → 80%+"
echo "  • 自动解决头文件依赖"
echo "  • 智能错误修复"
echo ""
echo "========================================"
echo ""

# 创建结果目录
mkdir -p results/enhanced

# 运行增强版Agent
echo "🔄 开始转换..."
python3 enhanced_agent_v2.py

echo ""
echo "========================================"
echo "✅ 增强版Agent运行完成!"
echo "========================================"
echo ""
echo "📁 结果文件:"
echo "  • results/enhanced_conversion_results.json"
echo ""
echo "💡 提示:"
echo "  查看详细结果了解转换成功率"
echo "  失败的内核可以再次运行以尝试修复"
echo "========================================"
