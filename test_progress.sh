#!/bin/bash
# Batch test script for remaining kernels
# Run one by one with full optimization comparison

KERNELS=(
    "se_layer_nhwc:se_layer"
    "winograd_input_transform:winograd_input"
    "winograd_filter_transform:winograd_filter"
    "nchw_to_nhwc:nchw_conversion"
    "global_scale:global_scale"
    "expand_planes:expand_planes"
    "copy_type_converted:copy_convert"
    "add_vectors_hnc_nhc:add_vectors_hnc"
    "global_avg_pool_nhwc_fp16:global_avg_pool_fp16"
    "global_scale_fp16_nhwc:global_scale_fp16"
    "expand_planes_nchw:expand_planes_nchw"
    "expand_planes_fp32_nchw:expand_planes_fp32"
    "policy_map:policy_map"
    "promotion_logits:promotion_logits"
    "preprocess_attention_body:preprocess_attn"
    "input_gating:input_gating"
    "gen_offset_pointers:gen_offsets"
    "output_input_transform_fp16_shmem:output_input_fp16"
)

echo "================================================"
echo "GPU Kernel Testing Progress"
echo "================================================"
echo ""

COMPLETED=0
TOTAL=${#KERNELS[@]}

for kernel_info in "${KERNELS[@]}"; do
    IFS=':' read -r kernel_name file_name <<< "$kernel_info"
    
    echo "Testing: $kernel_name"
    
    # Check if results already exist
    if [ -f "benchmarks/results/${file_name}_results.csv" ]; then
        echo "  ✓ Already completed"
        ((COMPLETED++))
        continue
    fi
    
    echo "  ⏳ Waiting for test file creation..."
    # The test files need to be created individually
    
done

echo ""
echo "Progress: $COMPLETED/$TOTAL completed"
