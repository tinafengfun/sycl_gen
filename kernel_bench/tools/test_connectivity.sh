#!/bin/bash
# Quick test for environment connectivity
# 快速测试环境连通性

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║       Environment Connectivity Quick Test             ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo ""
}

test_b60() {
    echo -e "${BLUE}[B60] Testing B60 SYCL Environment...${NC}"
    echo ""
    
    # 检查docker
    if ! command -v docker > /dev/null 2>&1; then
        echo -e "${RED}✗ Docker not installed${NC}"
        return 1
    fi
    echo "✓ Docker is available"
    
    # 检查容器
    if docker ps | grep -q "lsv-container"; then
        echo -e "${GREEN}✓ Container 'lsv-container' is running${NC}"
        
        # 检查编译器
        if docker exec lsv-container which icpx > /dev/null 2>&1; then
            echo -e "${GREEN}✓ SYCL compiler (icpx) is available${NC}"
        else
            echo -e "${YELLOW}⚠ SYCL compiler (icpx) not found in container${NC}"
        fi
    else
        echo -e "${RED}✗ Container 'lsv-container' is not running${NC}"
        echo "  Start with: docker start lsv-container"
        return 1
    fi
    
    echo ""
    return 0
}

test_remote_cuda() {
    echo -e "${BLUE}[CUDA] Testing Remote CUDA Environment...${NC}"
    echo ""
    
    # 检查ssh
    if ! command -v ssh > /dev/null 2>&1; then
        echo -e "${RED}✗ SSH not installed${NC}"
        return 1
    fi
    echo "✓ SSH is available"
    
    # 检查scp
    if ! command -v scp > /dev/null 2>&1; then
        echo -e "${RED}✗ SCP not installed${NC}"
        return 1
    fi
    echo "✓ SCP is available"
    
    # 测试SSH连接
    echo "Testing SSH connection to root@10.112.229.160..."
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@10.112.229.160 "echo 'SSH_OK'" 2>/dev/null | grep -q "SSH_OK"; then
        echo -e "${GREEN}✓ SSH connection successful${NC}"
    else
        echo -e "${RED}✗ SSH connection failed${NC}"
        echo "  Make sure SSH key is configured:"
        echo "    ssh-copy-id root@10.112.229.160"
        return 1
    fi
    
    # 检查远程容器
    echo "Checking remote container 'cuda12.9-test'..."
    if ssh root@10.112.229.160 "docker ps --filter 'name=cuda12.9-test' --format '{{.Names}}'" 2>/dev/null | grep -q "cuda12.9-test"; then
        echo -e "${GREEN}✓ Container 'cuda12.9-test' is running${NC}"
        
        # 检查NVCC
        if ssh root@10.112.229.160 "docker exec cuda12.9-test which nvcc" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ CUDA compiler (nvcc) is available${NC}"
            
            # 显示版本
            local nvcc_version
            nvcc_version=$(ssh root@10.112.229.160 "docker exec cuda12.9-test nvcc --version" 2>/dev/null | head -1)
            echo "  Version: ${nvcc_version}"
        else
            echo -e "${YELLOW}⚠ CUDA compiler (nvcc) not found in container${NC}"
        fi
    else
        echo -e "${RED}✗ Container 'cuda12.9-test' is not running${NC}"
        return 1
    fi
    
    echo ""
    return 0
}

main() {
    print_header
    
    local b60_ok=false
    local cuda_ok=false
    
    # 测试B60
    if test_b60; then
        b60_ok=true
    fi
    
    # 测试远程CUDA
    if test_remote_cuda; then
        cuda_ok=true
    fi
    
    # 总结
    echo "═════════════════════════════════════════════════════════"
    echo "Summary:"
    echo ""
    
    if $b60_ok; then
        echo -e "  ${GREEN}✓ B60 SYCL Environment: READY${NC}"
    else
        echo -e "  ${RED}✗ B60 SYCL Environment: NOT READY${NC}"
    fi
    
    if $cuda_ok; then
        echo -e "  ${GREEN}✓ Remote CUDA Environment: READY${NC}"
    else
        echo -e "  ${RED}✗ Remote CUDA Environment: NOT READY${NC}"
    fi
    
    echo "═════════════════════════════════════════════════════════"
    echo ""
    
    if $b60_ok && $cuda_ok; then
        echo -e "${GREEN}All environments are ready for building!${NC}"
        echo ""
        echo "Quick start:"
        echo "  ./tools/build.sh b60 compile-all      # Build all SYCL kernels"
        echo "  ./tools/build.sh cuda compile-all     # Build all CUDA kernels"
        exit 0
    else
        echo -e "${YELLOW}Some environments are not ready. Please fix the issues above.${NC}"
        exit 1
    fi
}

main "$@"
