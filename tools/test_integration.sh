#!/bin/bash
# Integration test script for builder tools
# 集成测试脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# 测试1: 检查文件结构
test_file_structure() {
    log_info "Testing file structure..."
    
    local files=(
        "tools/b60_sycl_builder.py"
        "tools/remote_cuda_builder.py"
        "tools/b60_sycl_builder.sh"
        "tools/remote_cuda_builder.sh"
        "tools/build.sh"
        "tools/test_builders.py"
        ".opencode/skills/b60-sycl-builder/SKILL.md"
        ".opencode/skills/remote-cuda-builder/SKILL.md"
    )
    
    local all_exist=true
    for file in "${files[@]}"; do
        if [ -f "${WORKSPACE_DIR}/${file}" ]; then
            echo "  ✓ ${file}"
        else
            echo "  ✗ ${file} (missing)"
            all_exist=false
        fi
    done
    
    if $all_exist; then
        log_pass "All required files exist"
        return 0
    else
        log_fail "Some files are missing"
        return 1
    fi
}

# 测试2: Python语法检查
test_python_syntax() {
    log_info "Testing Python syntax..."
    
    local python_files=(
        "tools/b60_sycl_builder.py"
        "tools/remote_cuda_builder.py"
        "tools/test_builders.py"
    )
    
    local all_valid=true
    for file in "${python_files[@]}"; do
        if python3 -m py_compile "${WORKSPACE_DIR}/${file}" 2>/dev/null; then
            echo "  ✓ ${file}"
        else
            echo "  ✗ ${file} (syntax error)"
            all_valid=false
        fi
    done
    
    if $all_valid; then
        log_pass "All Python files have valid syntax"
        return 0
    else
        log_fail "Some Python files have syntax errors"
        return 1
    fi
}

# 测试3: Shell脚本语法检查
test_shell_syntax() {
    log_info "Testing shell script syntax..."
    
    local shell_files=(
        "tools/b60_sycl_builder.sh"
        "tools/remote_cuda_builder.sh"
        "tools/build.sh"
    )
    
    local all_valid=true
    for file in "${shell_files[@]}"; do
        if bash -n "${WORKSPACE_DIR}/${file}" 2>/dev/null; then
            echo "  ✓ ${file}"
        else
            echo "  ✗ ${file} (syntax error)"
            all_valid=false
        fi
    done
    
    if $all_valid; then
        log_pass "All shell scripts have valid syntax"
        return 0
    else
        log_fail "Some shell scripts have syntax errors"
        return 1
    fi
}

# 测试4: 目录权限
test_directories() {
    log_info "Testing directory structure..."
    
    mkdir -p "${WORKSPACE_DIR}/results/b60"
    mkdir -p "${WORKSPACE_DIR}/results/cuda"
    mkdir -p "${WORKSPACE_DIR}/scripts/b60"
    mkdir -p "${WORKSPACE_DIR}/scripts/cuda"
    
    if [ -d "${WORKSPACE_DIR}/results" ] && [ -d "${WORKSPACE_DIR}/scripts" ]; then
        log_pass "Directory structure is valid"
        return 0
    else
        log_fail "Failed to create directories"
        return 1
    fi
}

# 测试5: Python单元测试
test_python_unit() {
    log_info "Running Python unit tests..."
    
    cd "${WORKSPACE_DIR}"
    if python3 tools/test_builders.py; then
        log_pass "Python unit tests passed"
        return 0
    else
        log_fail "Some Python unit tests failed"
        return 1
    fi
}

# 测试6: 检查help输出
test_help_output() {
    log_info "Testing help output..."
    
    # Test unified build.sh help
    if "${WORKSPACE_DIR}/tools/build.sh" help > /dev/null 2>&1; then
        echo "  ✓ build.sh help"
    else
        echo "  ✗ build.sh help failed"
        return 1
    fi
    
    log_pass "Help output works correctly"
    return 0
}

# 主测试流程
main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║     Kernel Builder Tools - Integration Test Suite      ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo ""
    
    local failed=0
    local passed=0
    
    # 运行所有测试
    tests=(
        test_file_structure
        test_python_syntax
        test_shell_syntax
        test_directories
        test_python_unit
        test_help_output
    )
    
    for test_func in "${tests[@]}"; do
        if $test_func; then
            ((passed++))
        else
            ((failed++))
        fi
        echo ""
    done
    
    # 汇总
    echo "═════════════════════════════════════════════════════════"
    echo "Test Summary:"
    echo "  Passed: ${passed}"
    echo "  Failed: ${failed}"
    echo "═════════════════════════════════════════════════════════"
    echo ""
    
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}Some tests failed!${NC}"
        exit 1
    fi
}

main "$@"
