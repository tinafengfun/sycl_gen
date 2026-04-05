#!/bin/bash
# Remote CUDA Builder - Shell Wrapper (Improved)
# Usage: ./remote_cuda_builder.sh [compile|compile-all|status|check|clean] [file]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="${SCRIPT_DIR}"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SSH_HOST="root@10.112.229.160"
CONTAINER="cuda12.9-test"

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
Remote CUDA Builder
===================
Compile CUDA kernels in remote docker container (cuda12.9-test@10.112.229.160)

Usage:
    $0 compile <file>     Compile a single kernel file
    $0 compile-all         Compile all cuda kernels
    $0 status              Show build status
    $0 check               Check environment connectivity
    $0 clean               Clean build artifacts
    $0 help                Show this help message

Examples:
    $0 compile kernel_dataset/cuda/add_vectors_kernel.cu
    $0 compile-all
    $0 status
    $0 check

Configuration:
    - SSH Host: ${SSH_HOST}
    - Container: ${CONTAINER}
    - Compiler: /usr/local/cuda/bin/nvcc

Output:
    - Logs: results/cuda/
    - Scripts: scripts/cuda/
    - Status: .build_status.json
EOF
}

# 检查环境
check_environment() {
    log_info "Checking Remote CUDA environment..."
    
    # 检查SSH
    if ! command -v ssh > /dev/null 2>&1; then
        log_error "SSH not found. Please install SSH client."
        return 1
    fi
    log_success "SSH is available"
    
    # 检查SCP
    if ! command -v scp > /dev/null 2>&1; then
        log_error "SCP not found. Please install SCP."
        return 1
    fi
    log_success "SCP is available"
    
    # 检查SSH连接
    log_info "Testing SSH connection to ${SSH_HOST}..."
    if ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "${SSH_HOST}" "echo 'SSH_OK'" > /dev/null 2>&1 | grep -q "SSH_OK"; then
        log_error "SSH connection failed!"
        echo "  Make sure SSH key is configured:"
        echo "    ssh-copy-id ${SSH_HOST}"
        return 1
    fi
    log_success "SSH connection successful"
    
    # 检查容器
    log_info "Checking container '${CONTAINER}'..."
    if ! ssh "${SSH_HOST}" "docker ps --filter 'name=${CONTAINER}' --format '{{.Names}}'" 2>/dev/null | grep -q "${CONTAINER}"; then
        log_error "Container '${CONTAINER}' is not running!"
        echo "  Start with: ssh ${SSH_HOST} 'docker start ${CONTAINER}'"
        return 1
    fi
    log_success "Container '${CONTAINER}' is running"
    
    # 检查NVCC
    if ! ssh "${SSH_HOST}" "docker exec ${CONTAINER} which nvcc" > /dev/null 2>&1; then
        log_error "CUDA compiler (nvcc) not found in container!"
        return 1
    fi
    log_success "CUDA compiler (nvcc) is available"
    
    echo ""
    log_success "All environment checks passed!"
    return 0
}

# 清理构建
clean_build() {
    log_warn "Cleaning CUDA build artifacts..."
    
    rm -rf "${WORKSPACE_DIR}/results/cuda"
    rm -rf "${WORKSPACE_DIR}/scripts/cuda"
    
    log_success "Cleaned CUDA build artifacts"
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
            
            python3 "${TOOLS_DIR}/remote_cuda_builder.py" compile "$@"
            ;;
        compile-all)
            python3 "${TOOLS_DIR}/remote_cuda_builder.py" compile-all
            ;;
        status)
            python3 "${TOOLS_DIR}/remote_cuda_builder.py" status
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
