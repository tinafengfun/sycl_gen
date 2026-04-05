#!/bin/bash
# Quick test of the benchmark functionality

set -e

CONTAINER_DIR="/workspace/turbodiffusion-sycl"
CONTAINER="lsv-container"

echo "=== Running Quick Test ==="
docker exec -w $CONTAINER_DIR $CONTAINER bash -c '
    source /opt/intel/oneapi/setvars.sh --force
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH
    
    echo "Testing module imports:"
    python3 -c "
import torch
import turbodiffusion_sycl_ops
print(\"✓ turbodiffusion_sycl_ops imported\")

from turbodiffusion_sycl import FlashAttentionSYCL
print(\"✓ FlashAttentionSYCL imported\")

# Quick test with small tensors
print(\"\\nTesting with small tensors...\")
q = torch.randn(1, 2, 64, 32, dtype=torch.bfloat16)
k = torch.randn(1, 2, 64, 32, dtype=torch.bfloat16)
v = torch.randn(1, 2, 64, 32, dtype=torch.bfloat16)

print(f\"Q shape: {q.shape}\")
print(f\"K shape: {k.shape}\")
print(f\"V shape: {v.shape}\")

# Test forward pass
print(\"\\nTesting FlashAttention forward pass...\")
fa = FlashAttentionSYCL(head_dim=32, num_heads=2)
output = fa(q, k, v)
print(f\"Output shape: {output.shape}\")
print(f\"Output dtype: {output.dtype}\")
print(\"✓ Forward pass completed\")
    "
' 2>&1

echo ""
echo "=== Quick test complete ==="
