#!/bin/bash
# B60 SYCL Builder - Shell Wrapper (Improved)
# Usage: ./b60_sycl_builder.sh [compile|compile-all|status|check|clean] [file]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="${SCRIPT_DIR}"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# 显示帮助
show_help() {
    cat << EOF
B60 SYCL Builder
================
Compile SYCL kernels in B60 docker container (lsv-container)

Usage:
    $0 compile <file>     Compile a single kernel file
    $0 compile-all         Compile all sycl kernels
    $0 status              Show build status
    $0 check               Check environment connectivity
    $0 clean               Clean build artifacts
    $0 help                Show this help message

Examples:
    $0 compile kernel_dataset/sycl/add_vectors_kernel.dp.cpp
    $0 compile-all
    $0 status
    $0 check

Output:
    - Logs: results/b60/
    - Scripts: scripts/b60/
    - Status: .build_status.json
EOF
}

# 检查环境
check_environment() {
    log_info "Checking B60 environment..."
    
    # 检查docker
    if ! command -v docker > /dev/null 2>&1; then
        log_error "Docker not found. Please install Docker."
        return 1
    fi
    log_success "Docker is available"
    
    # 检查容器
    if ! docker ps | grep -q "lsv-container"; then
        log_error "Container 'lsv-container' is not running!"
        echo "  Start with: docker start lsv-container"
        return 1
    fi
    log_success "Container 'lsv-container' is running"
    
    # 检查workspace
    if ! docker exec lsv-container ls /workspace > /dev/null 2>&1; then
        log_warn "Workspace not found, creating..."
        docker exec lsv-container mkdir -p /workspace
    fi
    log_success "Workspace is accessible"
    
    # 检查编译器
    if ! docker exec lsv-container which icpx > /dev/null 2>&1; then
        log_error "SYCL compiler (icpx) not found in container!"
        return 1
    fi
    log_success "SYCL compiler (icpx) is available"
    
    echo ""
    log_success "All environment checks passed!"
    return 0
}

# 清理构建
clean_build() {
    log_warn "Cleaning B60 build artifacts..."
    
    rm -rf "${WORKSPACE_DIR}/results/b60"
    rm -rf "${WORKSPACE_DIR}/scripts/b60"
    
    log_success "Cleaned B60 build artifacts"
}

# 主函数
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 1
    fi

    COMMAND=$1
    shift

    case $COMMAND in
        compile)
            if [ $# -eq 0 ]; then
                log_error "Please specify a file to compile"
                echo "Usage: $0 compile <file>"
                exit 1
            fi
            
            # 检查文件存在
            if [ ! -f "$1" ]; then
                log_error "File not found: $1"
                exit 1
            fi
            
            python3 "${TOOLS_DIR}/b60_sycl_builder.py" compile "$@"
            ;;
        compile-all)
            python3 "${TOOLS_DIR}/b60_sycl_builder.py" compile-all
            ;;
        status)
            python3 "${TOOLS_DIR}/b60_sycl_builder.py" status
            ;;
        check)
            check_environment
            ;;
        clean)
            clean_build
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
