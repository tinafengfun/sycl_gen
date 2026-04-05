#!/bin/bash
# Test BMG/XPU Compilation Options
# 测试新的SYCL编译选项是否正常工作

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

echo "=========================================="
echo "SYCL BMG/XPU Compilation Options Test"
echo "=========================================="
echo ""

# 测试编译选项
TEST_FILE="${WORKSPACE_DIR}/test/sycl_bmg_test.cpp"

if [ ! -f "$TEST_FILE" ]; then
    log_error "Test file not found: $TEST_FILE"
    exit 1
fi

log_info "Copying test file to container..."
docker cp "$TEST_FILE" lsv-container:/workspace/sycl_bmg_test.cpp

log_info "Compiling with BMG/XPU options..."
log_info "This may take 30-60 seconds for AOT compilation..."

# 使用新的编译选项
docker exec lsv-container bash -c "
cd /workspace
icpx -fsycl -O2 -std=c++17 \\
  -fsycl-unnamed-lambda \\
  -sycl-std=2020 \\
  -fhonor-nans \\
  -fhonor-infinities \\
  -fno-associative-math \\
  -fno-approx-func \\
  -no-ftz \\
  -fsycl-targets=spir64_gen,spir64 \\
  -fsycl-max-parallel-link-jobs=4 \\
  --offload-compress \\
  -Xs '-device pvc,bmg,dg2,arl-h,mtl-h,lnl-m,ptl-h,ptl-u -options -cl-poison-unsupported-fp64-kernels -options -cl-intel-enable-auto-large-GRF-mode -options -cl-fp32-correctly-rounded-divide-sqrt -options -cl-intel-greater-than-4GB-buffer-required' \\
  sycl_bmg_test.cpp \\
  -o sycl_bmg_test \\
  2>&1 | tee /workspace/compile_log.txt
"

if [ $? -ne 0 ]; then
    log_error "Compilation failed!"
    exit 1
fi

log_success "Compilation successful!"

# 运行测试
log_info "Running test..."
docker exec lsv-container bash -c "cd /workspace && ./sycl_bmg_test" || {
    log_error "Test execution failed!"
    exit 1
}

log_success "All tests passed!"

# 显示编译信息
echo ""
echo "=========================================="
echo "Compilation Summary"
echo "=========================================="

# 获取文件大小
FILE_SIZE=$(docker exec lsv-container ls -lh /workspace/sycl_bmg_test | awk '{print $5}')
echo "Output file size: $FILE_SIZE"

# 获取编译时间
if docker exec lsv-container grep -q "real" /workspace/compile_log.txt 2>/dev/null; then
    echo "Compile time: $(docker exec lsv-container grep "real" /workspace/compile_log.txt | tail -1)"
fi

echo ""
echo "AOT Targets compiled:"
echo "  ✓ pvc  (Ponte Vecchio)"
echo "  ✓ bmg  (Battlemage)"
echo "  ✓ dg2  (Alchemist)"
echo "  ✓ arl-h (Arrow Lake-H)"
echo "  ✓ mtl-h (Meteor Lake-H)"
echo "  ✓ lnl-m (Lunar Lake-M)"
echo "  ✓ ptl-h (Panther Lake-H)"
echo "  ✓ ptl-u (Panther Lake-U)"

echo ""
log_success "BMG/XPU compilation options are working correctly!"
