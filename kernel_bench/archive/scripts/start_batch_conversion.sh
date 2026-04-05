#!/bin/bash
# Start batch conversion of 30 kernels
# 启动30个kernel的批量转换

echo "=========================================="
echo "🚀 CUDA-to-SYCL Batch Conversion"
echo "=========================================="
echo ""

# Check environment
echo "📋 Checking environment..."

# Check B60 container
if docker ps | grep -q "lsv-container"; then
    echo "✅ B60 container (SYCL) is running"
else
    echo "❌ B60 container not found. Starting..."
    docker start lsv-container
fi

# Check CUDA environment
if ssh root@10.112.229.160 "docker ps | grep -q cuda12.9-test" 2>/dev/null; then
    echo "✅ CUDA container is running"
else
    echo "⚠️  CUDA container not found on remote host"
fi

echo ""

# Create output directory
OUTPUT_DIR="results/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# Count kernels
KERNEL_COUNT=$(ls kernel_dataset/cuda/*_kernel.cu 2>/dev/null | wc -l)
echo "📊 Found $KERNEL_COUNT kernels to process"
echo ""

# Start batch conversion
echo "🚀 Starting batch conversion..."
echo "=========================================="
echo ""

python3 tools/batch_convert.py \
  --all \
  --output "$OUTPUT_DIR" \
  --workers 4

EXIT_CODE=$?

echo ""
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Batch conversion completed successfully!"
    echo ""
    echo "📊 Results:"
    cat "$OUTPUT_DIR/completion_report.md" | head -30
    echo ""
    echo "📁 Full report: $OUTPUT_DIR/completion_report.md"
    echo "📁 JSON report: $OUTPUT_DIR/completion_report.json"
else
    echo "❌ Batch conversion failed or incomplete"
    echo "📁 Check logs: $OUTPUT_DIR/batch_conversion.log"
fi

echo "=========================================="

exit $EXIT_CODE
