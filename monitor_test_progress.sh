#!/bin/bash
# 监控测试进度
# Monitor test progress

LOG_FILE="$1"

if [ -z "$LOG_FILE" ]; then
    # 找到最新的日志文件
    LOG_FILE=$(ls -t results/accuracy_tests/*.log 2>/dev/null | head -1)
    if [ -z "$LOG_FILE" ]; then
        echo "未找到日志文件"
        echo "用法: $0 <log_file>"
        exit 1
    fi
fi

echo "监控日志文件: $LOG_FILE"
echo "按 Ctrl+C 停止监控"
echo ""

# 显示最后50行并持续监控
tail -f -n 50 "$LOG_FILE"
