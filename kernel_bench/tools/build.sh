#!/bin/bash
# Unified Kernel Builder
# 统一的kernel构建入口，支持SYCL和CUDA两种环境

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助
show_help() {
    cat << EOF
Unified Kernel Builder
======================
Build and test CUDA/SYCL kernels across different environments

Usage:
    $0 <environment> <command> [options]

Environments:
    b60         Local B60 SYCL environment (docker: lsv-container)
    cuda        Remote CUDA environment (docker: cuda12.9-test@10.112.229.160)
    all         Both environments (for compile-all and status)

Commands:
    compile <file>    Compile a single kernel file
    compile-all       Compile all kernels in the environment
    status            Show build status
    check             Check environment connectivity
    clean             Clean build artifacts and logs

Examples:
    # Compile single kernel
    $0 b60 compile kernel_dataset/sycl/add_vectors_kernel.dp.cpp
    $0 cuda compile kernel_dataset/cuda/softmax_kernel.cu

    # Compile all kernels
    $0 b60 compile-all
    $0 cuda compile-all
    $0 all compile-all          # Compile in both environments

    # Check status
    $0 b60 status
    $0 cuda status
    $0 all status               # Show both environments

    # Check connectivity
    $0 b60 check
    $0 cuda check

    # Clean build artifacts
    $0 all clean

Output Structure:
    results/
    ├── b60/                    # B60 SYCL build results
    │   ├── compile_*.log
    │   └── summary_*.json
    ├── cuda/                   # Remote CUDA build results
    │   ├── compile_*.log
    │   └── summary_*.json
    └── .build_status.json      # Unified status file

    scripts/
    ├── b60/                    # Generated build scripts
    └── cuda/
EOF
}

# 打印带颜色的消息
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

# 编译单个文件
compile_single() {
    local env=$1
    local file=$2
    
    if [ -z "$file" ]; then
        log_error "No file specified"
        return 1
    fi
    
    if [ ! -f "$file" ]; then
        log_error "File not found: $file"
        return 1
    fi
    
    case $env in
        b60)
            log_info "Compiling with B60 SYCL builder..."
            "${SCRIPT_DIR}/b60_sycl_builder.sh" compile "$file"
            ;;
        cuda)
            log_info "Compiling with Remote CUDA builder..."
            "${SCRIPT_DIR}/remote_cuda_builder.sh" compile "$file"
            ;;
        *)
            log_error "Unknown environment: $env"
            return 1
            ;;
    esac
}

# 编译所有
compile_all() {
    local env=$1
    
    case $env in
        b60)
            log_info "Compiling all SYCL kernels..."
            "${SCRIPT_DIR}/b60_sycl_builder.sh" compile-all
            ;;
        cuda)
            log_info "Compiling all CUDA kernels..."
            "${SCRIPT_DIR}/remote_cuda_builder.sh" compile-all
            ;;
        all)
            log_info "Compiling all kernels in both environments..."
            echo ""
            log_info "=== B60 SYCL Environment ==="
            "${SCRIPT_DIR}/b60_sycl_builder.sh" compile-all || true
            echo ""
            log_info "=== Remote CUDA Environment ==="
            "${SCRIPT_DIR}/remote_cuda_builder.sh" compile-all || true
            ;;
        *)
            log_error "Unknown environment: $env"
            return 1
            ;;
    esac
}

# 显示状态
show_status() {
    local env=$1
    
    case $env in
        b60)
            "${SCRIPT_DIR}/b60_sycl_builder.sh" status
            ;;
        cuda)
            "${SCRIPT_DIR}/remote_cuda_builder.sh" status
            ;;
        all)
            echo ""
            log_info "=== B60 SYCL Environment ==="
            "${SCRIPT_DIR}/b60_sycl_builder.sh" status
            echo ""
            log_info "=== Remote CUDA Environment ==="
            "${SCRIPT_DIR}/remote_cuda_builder.sh" status
            ;;
        *)
            log_error "Unknown environment: $env"
            return 1
            ;;
    esac
}

# 检查环境
check_env() {
    local env=$1
    
    case $env in
        b60)
            log_info "Checking B60 environment..."
            if docker ps | grep -q "lsv-container"; then
                log_success "Container 'lsv-container' is running"
            else
                log_error "Container 'lsv-container' is not running"
                return 1
            fi
            ;;
        cuda)
            log_info "Checking Remote CUDA environment..."
            "${SCRIPT_DIR}/remote_cuda_builder.sh" check
            ;;
        all)
            log_info "Checking all environments..."
            echo ""
            log_info "=== B60 SYCL Environment ==="
            check_env b60 || true
            echo ""
            log_info "=== Remote CUDA Environment ==="
            check_env cuda || true
            ;;
        *)
            log_error "Unknown environment: $env"
            return 1
            ;;
    esac
}

# 清理构建产物
clean_build() {
    local env=$1
    
    log_warn "Cleaning build artifacts for: $env"
    
    case $env in
        b60)
            rm -rf "${WORKSPACE_DIR}/results/b60"
            rm -rf "${WORKSPACE_DIR}/scripts/b60"
            log_success "Cleaned B60 build artifacts"
            ;;
        cuda)
            rm -rf "${WORKSPACE_DIR}/results/cuda"
            rm -rf "${WORKSPACE_DIR}/scripts/cuda"
            log_success "Cleaned CUDA build artifacts"
            ;;
        all)
            rm -rf "${WORKSPACE_DIR}/results"
            rm -rf "${WORKSPACE_DIR}/scripts"
            rm -f "${WORKSPACE_DIR}/.build_status.json"
            log_success "Cleaned all build artifacts"
            ;;
        *)
            log_error "Unknown environment: $env"
            return 1
            ;;
    esac
}

# 主函数
main() {
    if [ $# -lt 2 ]; then
        show_help
        exit 1
    fi

    ENV=$1
    COMMAND=$2
    shift 2

    case $COMMAND in
        compile)
            compile_single "$ENV" "$1"
            ;;
        compile-all)
            compile_all "$ENV"
            ;;
        status)
            show_status "$ENV"
            ;;
        check)
            check_env "$ENV"
            ;;
        clean)
            clean_build "$ENV"
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
