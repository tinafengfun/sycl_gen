#!/bin/bash
# 批量转换21个失败的内核

echo "========================================"
echo "🚀 增强版Agent v2.0 - 批量转换"
echo "========================================"
echo ""
echo "📊 转换计划:"
echo "  目标: 21个编译失败的CUDA内核"
echo "  模型: minimax-m2.5"
echo "  策略: LLM全流程驱动 (5个Stage)"
echo "  预计: 1-2小时"
echo ""
echo "📝 内核列表:"
python3 -c "
failed_kernels = [
    'add_bias_batched',
    'add_bias_nchw',
    'add_vectors_hnc_nhc',
    'add_vectors',
    'expand_planes_nhwc',
    'gen_offset_pointers',
    'global_avg_pool_nhwc_fp16',
    'global_scale_fp16_nhwc',
    'global_scale',
    'input_gating',
    'layer_norm',
    'nchw_to_nhwc',
    'output_input_transform_fp16_shmem',
    'preprocess_attention_body',
    'promotion_logits',
    'se_layer_nhwc',
    'softmax',
    'winograd_filter_transform',
    'winograd_output_relu_input',
    'winograd_output_se_relu_input',
    'winograd_output_transform'
]
for i, k in enumerate(failed_kernels, 1):
    print(f'  {i:2}. {k}')
"
echo ""
echo "⚠️  注意:"
echo "  完整转换21个内核需要1-2小时"
echo "  建议先测试3-5个内核验证流程"
echo ""
echo "💡 选项:"
echo "  1. 测试3个内核 (约10-15分钟)"
echo "  2. 测试5个内核 (约20-30分钟)"
echo "  3. 转换全部21个内核 (约1-2小时)"
echo ""
echo "========================================"
echo ""

# 询问用户选择
echo "请选择转换模式 [1/2/3]:"
read -r choice

case $choice in
  1)
    echo "🔄 测试3个内核..."
    TEST_KERNELS="add_vectors,add_bias_batched,layer_norm"
    ;;
  2)
    echo "🔄 测试5个内核..."
    TEST_KERNELS="add_vectors,add_bias_batched,layer_norm,softmax,global_scale"
    ;;
  3)
    echo "🔄 转换全部21个内核..."
    TEST_KERNELS="all"
    ;;
  *)
    echo "默认: 测试3个内核"
    TEST_KERNELS="add_vectors,add_bias_batched,layer_norm"
    ;;
esac

echo ""
echo "========================================"
echo "🚀 开始转换..."
echo "========================================"

# 创建结果目录
mkdir -p results/enhanced_batch_conversion

# 运行增强版Agent
if [ "$TEST_KERNELS" == "all" ]; then
    # 转换全部21个
    timeout 7200 python3 enhanced_agent_v2.py 2>&1 | tee results/enhanced_batch_conversion/conversion_log.txt
else
    # 测试指定数量
    echo "测试内核: $TEST_KERNELS"
    timeout 1800 python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')
from enhanced_agent_v2 import EnhancedConversionAgent

kernels = '$TEST_KERNELS'.split(',')
print(f'测试 {len(kernels)} 个内核: {kernels}')

agent = EnhancedConversionAgent()
results = asyncio.run(agent.run_batch_conversion(kernels))

# 保存结果
import json
with open('results/enhanced_batch_conversion/test_results.json', 'w') as f:
    json.dump(results, f, indent=2)
" 2>&1 | tee results/enhanced_batch_conversion/test_log.txt
fi

echo ""
echo "========================================"
echo "✅ 转换完成!"
echo "========================================"
echo ""
echo "📁 结果位置:"
echo "  results/enhanced_batch_conversion/"
echo ""
echo "📊 查看结果:"
echo "  cat results/enhanced_batch_conversion/*_results.json"
echo ""
echo "🔍 验证编译:"
echo "  ./check_compilation.sh"
echo "========================================"
