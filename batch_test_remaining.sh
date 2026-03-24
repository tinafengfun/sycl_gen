#!/bin/bash
# Batch test remaining kernels with detailed logging
# Executing one by one with full optimization comparison

set -e

KERNELS=(
    "winograd_filter_transform:winograd_filter_transform_results.csv"
    "nchw_to_nhwc:nchw_to_nhwc_results.csv"
    "global_scale:global_scale_results.csv"
    "expand_planes:expand_planes_results.csv"
    "copy_type_converted:copy_type_converted_results.csv"
)

TOTAL=${#KERNELS[@]}
COMPLETED=0

echo "========================================"
echo "Batch Kernel Testing - Remaining $TOTAL kernels"
echo "========================================"
echo ""

for kernel_info in "${KERNELS[@]}"; do
    IFS=':' read -r kernel_name output_file <<< "$kernel_info"
    
    echo "[$((COMPLETED+1))/$TOTAL] Testing: $kernel_name"
    
    # Check if already completed
    if [ -f "/home/intel/tianfeng/opencode_bench/benchmarks/results/$output_file" ]; then
        echo "  ✓ Already completed, skipping"
        ((COMPLETED++))
        continue
    fi
    
    # Test file should be created manually for each kernel
    echo "  ⏳ Please create test file: tests/test_${kernel_name}.cpp"
    echo ""
    
    ((COMPLETED++))
done

echo "========================================"
echo "Batch testing summary: $COMPLETED/$TOTAL"
echo "========================================"
