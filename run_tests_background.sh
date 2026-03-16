#!/bin/bash
# 后台运行完整准确度测试
# Run full accuracy tests in background

LOG_FILE="results/accuracy_tests/full_test_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "启动完整准确度测试套件"
echo "启动时间: $(date)"
echo "日志文件: $LOG_FILE"
echo "=========================================="
echo ""
echo "预计耗时: 5-10 小时"
echo "测试 28 个内核 × 13 个配置 = 364 个测试用例"
echo ""

# 运行测试并记录日志
python3 run_full_accuracy_tests.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "测试完成时间: $(date)"
echo "日志文件: $LOG_FILE"
echo "=========================================="
