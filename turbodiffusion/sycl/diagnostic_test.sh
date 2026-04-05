#!/bin/bash
# Quick diagnostic test with smaller tensors

set -e

CONTAINER_DIR="/workspace/turbodiffusion-sycl"
CONTAINER="lsv-container"

echo "=== Quick Diagnostic Test ==="
docker exec -w $CONTAINER_DIR $CONTAINER bash -c '
    source /opt/intel/oneapi/setvars.sh --force
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH
    
    python3 -c "
import torch
import time
from turbodiffusion_sycl import FlashAttentionSYCL, SparseAttentionSYCL

print(\"Testing Flash Attention with small tensors...\")

# Test case 1: Very small
print(\"\\nTest 1: B=1, H=2, S=64, D=32\")
q = torch.randn(1, 2, 64, 32, dtype=torch.bfloat16)
k = torch.randn(1, 2, 64, 32, dtype=torch.bfloat16)
v = torch.randn(1, 2, 64, 32, dtype=torch.bfloat16)

fa = FlashAttentionSYCL(head_dim=32, num_heads=2)
start = time.time()
out = fa(q, k, v)
elapsed = time.time() - start
print(f\"  ✓ Completed in {elapsed:.3f}s\")
print(f\"  Output shape: {out.shape}\")

# Test case 2: Medium
print(\"\\nTest 2: B=1, H=8, S=256, D=64\")
q = torch.randn(1, 8, 256, 64, dtype=torch.bfloat16)
k = torch.randn(1, 8, 256, 64, dtype=torch.bfloat16)
v = torch.randn(1, 8, 256, 64, dtype=torch.bfloat16)

fa = FlashAttentionSYCL(head_dim=64, num_heads=8)
start = time.time()
out = fa(q, k, v)
elapsed = time.time() - start
print(f\"  ✓ Completed in {elapsed:.3f}s\")
print(f\"  Output shape: {out.shape}\")

# Test case 3: Accuracy check
print(\"\\nTest 3: Accuracy check\")
torch.manual_seed(42)
q = torch.randn(1, 2, 64, 32, dtype=torch.bfloat16)
k = torch.randn(1, 2, 64, 32, dtype=torch.bfloat16)
v = torch.randn(1, 2, 64, 32, dtype=torch.bfloat16)

# Reference
scores = torch.matmul(q, k.transpose(-2, -1)) / (32 ** 0.5)
attn = torch.softmax(scores, dim=-1)
ref = torch.matmul(attn, v)

# SYCL
fa = FlashAttentionSYCL(head_dim=32, num_heads=2)
sycl_out = fa(q, k, v)

max_err = (ref.float() - sycl_out.float()).abs().max().item()
print(f\"  Max error: {max_err:.6f}\")
if max_err < 0.01:
    print(f\"  ✓ PASS\")
else:
    print(f\"  ✗ FAIL\")

print(\"\\n=== All tests completed ===\")
    "
' 2>&1

echo ""
echo "=== Test Complete ==="
