#!/bin/bash
# Improved Agent v3.0 运行脚本
# 基于过程反思的CUDA→SYCL转换系统

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 帮助信息
show_help() {
    echo -e "${BLUE}改进版 CUDA→SYCL 转换 Agent v3.0${NC}"
    echo ""
    echo "用法: $0 [选项] [命令]"
    echo ""
    echo "命令:"
    echo "  full              运行完整的批量转换流程"
    echo "  test              测试特定内核"
    echo "  verify            验证已转换的内核"
    echo "  fix               修复失败的转换"
    echo "  analyze           分析转换结果"
    echo "  report            生成详细报告"
    echo ""
    echo "选项:"
    echo "  -k, --kernels     指定要处理的内核ID（逗号分隔）"
    echo "  -s, --strategy    转换策略: auto/direct/template_expansion"
    echo "  -p, --priority    是否优先处理简单内核 (true/false)"
    echo "  -f, --fix-loop    启用自动修复循环 (true/false)"
    echo "  -v, --verbose     详细输出"
    echo "  -h, --help        显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 full                           # 转换所有内核"
    echo "  $0 test -k add_vectors            # 测试单个内核"
    echo "  $0 full -s direct -p true         # 使用直接策略，优先简单内核"
    echo "  $0 fix -k batch_norm,layer_norm   # 修复特定内核"
    echo ""
}

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查环境
check_environment() {
    log_info "检查运行环境..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "未找到 Python3"
        exit 1
    fi
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "未找到 Docker"
        exit 1
    fi
    
    # 检查Docker容器
    if ! docker ps | grep -q "lsv-container"; then
        log_warn "SYCL容器未运行，尝试启动..."
        docker start lsv-container 2>/dev/null || true
    fi
    
    # 检查CUDA环境
    if ! ssh root@10.112.229.160 "docker ps | grep -q cuda12.9-test" 2>/dev/null; then
        log_warn "CUDA容器未运行，尝试启动..."
        ssh root@10.112.229.160 "docker start cuda12.9-test" 2>/dev/null || true
    fi
    
    log_success "环境检查通过"
}

# 检查依赖
check_dependencies() {
    log_info "检查Python依赖..."
    
    python3 -c "import aiohttp" 2>/dev/null || {
        log_warn "安装 aiohttp..."
        pip install aiohttp -q
    }
    
    python3 -c "import numpy" 2>/dev/null || {
        log_warn "安装 numpy..."
        pip install numpy -q
    }
    
    log_success "依赖检查完成"
}

# 运行完整转换
run_full_conversion() {
    log_info "启动完整批量转换流程..."
    
    local strategy="${1:-auto}"
    local priority="${2:-true}"
    
    log_info "策略: $strategy"
    log_info "优先简单内核: $priority"
    
    python3 improved_agent_v3.py 2>&1 | tee "results/improved_agent_v3/run_$(date +%Y%m%d_%H%M%S).log"
    
    log_success "转换流程完成"
}

# 测试特定内核
run_test() {
    local kernel_id="$1"
    
    if [ -z "$kernel_id" ]; then
        log_error "请指定内核ID"
        exit 1
    fi
    
    log_info "测试内核: $kernel_id"
    
    python3 << EOF
import asyncio
import sys
sys.path.insert(0, '.')
from improved_agent_v3 import ImprovedConversionAgent

async def test():
    agent = ImprovedConversionAgent()
    if "$kernel_id" in agent.kernels:
        kernel_info = agent.kernels["$kernel_id"]
        success = await agent.process_kernel(kernel_info)
        print(f"\n结果: {'通过' if success else '失败'}")
    else:
        print(f"错误: 内核 $kernel_id 不存在")

asyncio.run(test())
EOF
}

# 验证已转换的内核
run_verify() {
    log_info "验证已转换的内核..."
    
    python3 << EOF
import asyncio
import sys
sys.path.insert(0, '.')
from improved_agent_v3 import ImprovedConversionAgent, ConversionStatus

async def verify():
    agent = ImprovedConversionAgent()
    
    # 找出所有已转换的内核
    converted_kernels = [
        k for k in agent.kernels.values()
        if k.sycl_file.exists()
    ]
    
    print(f"找到 {len(converted_kernels)} 个已转换的内核")
    
    passed = 0
    failed = 0
    
    for kernel_info in converted_kernels:
        print(f"\n验证: {kernel_info.kernel_id}")
        success = await agent.verify_kernel(kernel_info)
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n验证完成: {passed} 通过, {failed} 失败")

asyncio.run(verify())
EOF
}

# 修复失败的转换
run_fix() {
    local kernel_ids="$1"
    
    log_info "修复失败的转换..."
    
    if [ -z "$kernel_ids" ]; then
        # 自动找出失败的内核
        python3 << EOF
import asyncio
import sys
sys.path.insert(0, '.')
from improved_agent_v3 import ImprovedConversionAgent

async def fix_failed():
    agent = ImprovedConversionAgent()
    
    # 检查之前的运行结果
    import json
    result_file = Path("results/improved_agent_v3/session_results.json")
    if result_file.exists():
        with open(result_file) as f:
            results = json.load(f)
        
        failed_kernels = [
            d['kernel_id'] for d in results.get('details', [])
            if d['status'] != 'passed'
        ]
        
        print(f"发现 {len(failed_kernels)} 个失败的内核需要修复")
        
        for kernel_id in failed_kernels:
            if kernel_id in agent.kernels:
                print(f"\n修复: {kernel_id}")
                kernel_info = agent.kernels[kernel_id]
                # 重置状态
                kernel_info.status = agent.ConversionStatus.PENDING
                kernel_info.fix_attempts = 0
                success = await agent.process_kernel(kernel_info)
                print(f"结果: {'成功' if success else '失败'}")

asyncio.run(fix_failed())
EOF
    else
        # 修复指定的内核
        IFS=',' read -ra kernels <<< "$kernel_ids"
        for kernel_id in "${kernels[@]}"; do
            run_test "$kernel_id"
        done
    fi
}

# 分析转换结果
run_analyze() {
    log_info "分析转换结果..."
    
    python3 << 'EOF'
import json
from pathlib import Path

result_file = Path("results/improved_agent_v3/session_results.json")
if not result_file.exists():
    print("错误: 找不到结果文件")
    exit(1)

with open(result_file) as f:
    results = json.load(f)

print("\n" + "="*70)
print("转换结果分析")
print("="*70)

# 统计
total = results.get('kernels_total', 0)
verified = results.get('kernels_verified', 0)
print(f"\n总体统计:")
print(f"  总内核: {total}")
print(f"  验证通过: {verified}")
print(f"  成功率: {verified/total*100:.1f}%" if total > 0 else "  N/A")

# 按类别统计
categories = {}
for detail in results.get('details', []):
    cat = detail.get('category', 'unknown')
    if cat not in categories:
        categories[cat] = {'total': 0, 'passed': 0}
    categories[cat]['total'] += 1
    if detail['status'] == 'passed':
        categories[cat]['passed'] += 1

if categories:
    print(f"\n按类别统计:")
    for cat, stats in categories.items():
        rate = stats['passed']/stats['total']*100 if stats['total'] > 0 else 0
        print(f"  {cat}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")

# 错误分析
error_types = {}
for detail in results.get('details', []):
    if detail['status'] != 'passed':
        err_type = detail.get('error_type', 'unknown')
        error_types[err_type] = error_types.get(err_type, 0) + 1

if error_types:
    print(f"\n常见错误类型:")
    for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
        print(f"  {err_type}: {count}次")

# LLM统计
llm_stats = results.get('llm_stats', {})
if llm_stats:
    print(f"\nLLM使用统计:")
    print(f"  调用次数: {llm_stats.get('call_count', 0)}")
    print(f"  Token数: {llm_stats.get('total_tokens', 0)}")
    print(f"  成功率: {llm_stats.get('success_rate', 0):.1f}%")

print("\n" + "="*70)
EOF
}

# 生成报告
run_report() {
    log_info "生成详细报告..."
    
    local report_file="results/improved_agent_v3/final_report_$(date +%Y%m%d_%H%M%S).md"
    
    python3 << 'EOF'
import json
from pathlib import Path
from datetime import datetime

result_file = Path("results/improved_agent_v3/session_results.json")
if not result_file.exists():
    print("错误: 找不到结果文件")
    exit(1)

with open(result_file) as f:
    results = json.load(f)

report = f"""# CUDA→SYCL 转换报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 执行摘要

- **总内核数**: {results.get('kernels_total', 0)}
- **验证通过**: {results.get('kernels_verified', 0)}
- **成功率**: {results.get('kernels_verified', 0) / max(results.get('kernels_total', 1), 1) * 100:.1f}%

## LLM使用统计

- **总调用次数**: {results.get('llm_stats', {}).get('call_count', 0)}
- **总Token数**: {results.get('llm_stats', {}).get('total_tokens', 0)}
- **调用成功率**: {results.get('llm_stats', {}).get('success_rate', 0):.1f}%

## 编译统计

### CUDA平台
- **尝试次数**: {results.get('compilation_stats', {}).get('cuda_attempts', 0)}
- **成功次数**: {results.get('compilation_stats', {}).get('cuda_success', 0)}
- **成功率**: {results.get('compilation_stats', {}).get('cuda_success_rate', 0):.1f}%

### SYCL平台
- **尝试次数**: {results.get('compilation_stats', {}).get('sycl_attempts', 0)}
- **成功次数**: {results.get('compilation_stats', {}).get('sycl_success', 0)}
- **成功率**: {results.get('compilation_stats', {}).get('sycl_success_rate', 0):.1f}%

## 详细结果

| 内核ID | 状态 | 复杂度 | 修复次数 | 错误类型 |
|--------|------|--------|----------|----------|
"""

for detail in results.get('details', []):
    status_icon = "✅" if detail['status'] == 'passed' else '❌'
    report += f"| {detail['kernel_id']} | {status_icon} {detail['status']} | {detail.get('complexity_score', 0):.1f} | {detail.get('fix_attempts', 0)} | {detail.get('error_type', '-')} |\n"

report += """
## 失败分析

"""

failed = [d for d in results.get('details', []) if d['status'] != 'passed']
if failed:
    for detail in failed[:10]:  # 只显示前10个
        report += f"""### {detail['kernel_id']}
- **状态**: {detail['status']}
- **复杂度**: {detail.get('complexity_score', 0):.1f}
- **错误类型**: {detail.get('error_type', 'unknown')}
- **尝试次数**: {detail.get('conversion_attempts', 0)}

"""
else:
    report += "所有内核均已成功转换和验证！\n"

report += """
## 改进建议

基于本次转换过程，建议：

1. **优先处理简单内核**: 简单内核（复杂度<3）成功率更高
2. **增强错误诊断**: 对于复杂模板，使用模板展开策略
3. **增加修复轮数**: 对于关键内核，可以增加修复尝试次数
4. **持续学习**: 记录成功/失败模式，优化后续转换

---
*由 Improved Agent v3.0 自动生成*
"""

print(report)
EOF
    > "$report_file"
    
    log_success "报告已生成: $report_file"
}

# 主函数
main() {
    # 默认参数
    COMMAND=""
    KERNELS=""
    STRATEGY="auto"
    PRIORITY="true"
    FIX_LOOP="true"
    VERBOSE="false"
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            full|test|verify|fix|analyze|report)
                COMMAND="$1"
                shift
                ;;
            -k|--kernels)
                KERNELS="$2"
                shift 2
                ;;
            -s|--strategy)
                STRATEGY="$2"
                shift 2
                ;;
            -p|--priority)
                PRIORITY="$2"
                shift 2
                ;;
            -f|--fix-loop)
                FIX_LOOP="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查必需参数
    if [ -z "$COMMAND" ]; then
        log_error "请指定命令"
        show_help
        exit 1
    fi
    
    # 检查环境
    check_environment
    check_dependencies
    
    # 创建输出目录
    mkdir -p results/improved_agent_v3
    
    # 执行命令
    case $COMMAND in
        full)
            run_full_conversion "$STRATEGY" "$PRIORITY"
            ;;
        test)
            run_test "$KERNELS"
            ;;
        verify)
            run_verify
            ;;
        fix)
            run_fix "$KERNELS"
            ;;
        analyze)
            run_analyze
            ;;
        report)
            run_report
            ;;
        *)
            log_error "未知命令: $COMMAND"
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
