#!/bin/bash
# 执行所有 TODO 任务
# Execute all TODO tasks

cd /home/intel/tianfeng/opencode_bench

echo "========================================"
echo "开始执行所有 TODO 任务"
echo "开始时间: $(date)"
echo "========================================"
echo ""

# 创建结果目录
mkdir -p results/batch_conversion
mkdir -p results/accuracy_tests
mkdir -p results/todo_completion

LOG_FILE="results/todo_completion/all_tasks_$(date +%Y%m%d_%H%M%S).log"

echo "日志文件: $LOG_FILE"
echo ""

# 任务1: 修复CUDA GPU访问 - 已完成 ✅
echo "✅ 任务1: CUDA GPU访问 - 已完成"
echo ""

# 任务2: 转换18个待处理的CUDA内核
echo "🚀 任务2: 启动内核转换（4批并行）..."
nohup ./convert_batch1.sh > results/batch_conversion/batch1.log 2>&1 &
BATCH1_PID=$!
nohup ./convert_batch2.sh > results/batch_conversion/batch2.log 2>&1 &
BATCH2_PID=$!
nohup ./convert_batch3.sh > results/batch_conversion/batch3.log 2>&1 &
BATCH3_PID=$!
nohup ./convert_batch4.sh > results/batch_conversion/batch4.log 2>&1 &
BATCH4_PID=$!

echo "  批次1 (基础操作): PID $BATCH1_PID"
echo "  批次2 (全局/预处理): PID $BATCH2_PID"
echo "  批次3 (Attention/Softmax): PID $BATCH3_PID"
echo "  批次4 (Winograd): PID $BATCH4_PID"
echo ""

# 任务3: 修复5个损坏的SYCL内核
echo "🚀 任务3: 启动损坏内核修复..."
nohup ./fix_broken_kernels.sh > results/batch_conversion/fix_broken.log 2>&1 &
FIX_PID=$!
echo "  修复损坏内核: PID $FIX_PID"
echo ""

echo "========================================"
echo "所有任务已启动"
echo "========================================"
echo ""
echo "监控命令:"
echo "  查看批次1: tail -f results/batch_conversion/batch1.log"
echo "  查看批次2: tail -f results/batch_conversion/batch2.log"
echo "  查看批次3: tail -f results/batch_conversion/batch3.log"
echo "  查看批次4: tail -f results/batch_conversion/batch4.log"
echo "  查看修复:  tail -f results/batch_conversion/fix_broken.log"
echo "  查看所有:  ps aux | grep batch_convert"
echo ""
echo "预计完成时间: 2-4 小时"
