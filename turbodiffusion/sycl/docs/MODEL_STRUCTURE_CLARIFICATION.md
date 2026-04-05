# TurboDiffusion-SYCL: Model Structure Clarification Report

## Date: 2026-04-01
## Model: Wan2.1 1.3B T2V-480P

---

## 1. Model Architecture Overview

### Basic Information
- **Model Type**: Wan2.1 1.3B Text-to-Video (T2V)
- **Total Parameters**: 1.42B (1,419,492,160)
- **Number of Blocks**: 30 (indexed 0-29)
- **Hidden Dimension**: 1536
- **FFN Intermediate Dimension**: 8960
- **Number of Attention Heads**: 12 (inferred: 1536/128)
- **Head Dimension**: 128

### Data Type Distribution
- **BFloat16**: 705 tensors (79.7%) - Model weights
- **Float32**: 180 tensors (20.3%) - Norm layers and biases

### Top-Level Structure
```
├── patch_embedding (2 tensors)
├── text_embedding (4 tensors)  
├── time_embedding (4 tensors)
├── time_projection (2 tensors)
├── blocks (870 tensors, 30 blocks)
│   ├── blocks.0
│   ├── blocks.1
│   └── ... 
│   └── blocks.29
└── head (3 tensors)
    ├── head.modulation (BF16)
    ├── head.head.weight (BF16)
    └── head.head.bias (BF16)
```

---

## 2. Critical Discovery: Normalization Layer Structure

### ❌ Previous Assumption (Incorrect)
Based on typical transformer models, we assumed:
- `head.norm` (LayerNorm)
- `blocks.*.norm1` (LayerNorm - Self-attention input)
- `blocks.*.norm2` (LayerNorm - FFN input)
- `blocks.*.norm_q` / `norm_k` (RMSNorm)

### ✅ Actual Model Structure
The Wan2.1 model uses a **different architecture**:

#### Self-Attention Path:
```
Input → RMSNorm(norm_q) → Q projection
      → RMSNorm(norm_k) → K projection
      → V projection (no norm)
      → Attention → O projection
```

#### Cross-Attention Path:
```
Input → RMSNorm(cross_attn.norm_q) → Q projection
Text  → RMSNorm(cross_attn.norm_k) → K projection
      → V projection
      → Attention → O projection
```

#### FFN Path:
```
Input → LayerNorm(norm3) → FFN(0) → Activation → FFN(2) → Output
```

### Complete Norm Layer List (180 total):

| Layer Path | Type | Count | Weight Shape | Has Bias |
|-----------|------|-------|--------------|----------|
| `blocks.*.self_attn.norm_q` | RMSNorm | 30 | [1536] | No |
| `blocks.*.self_attn.norm_k` | RMSNorm | 30 | [1536] | No |
| `blocks.*.cross_attn.norm_q` | RMSNorm | 30 | [1536] | No |
| `blocks.*.cross_attn.norm_k` | RMSNorm | 30 | [1536] | No |
| `blocks.*.norm3` | LayerNorm | 30 | [1536] | Yes (weight+bias) |

**Total**: 180 tensors (120 RMSNorm + 60 LayerNorm)

---

## 3. Per-Block Structure Detail

### Block Components (29 tensors per block):

#### Self-Attention (12 tensors):
- `self_attn.q.weight` [1536, 1536]
- `self_attn.q.bias` [1536]
- `self_attn.k.weight` [1536, 1536]
- `self_attn.k.bias` [1536]
- `self_attn.v.weight` [1536, 1536]
- `self_attn.v.bias` [1536]
- `self_attn.o.weight` [1536, 1536]
- `self_attn.o.bias` [1536]
- `self_attn.norm_q.weight` [1536] ← **SYCL Target**
- `self_attn.norm_k.weight` [1536] ← **SYCL Target**
- `self_attn.attn_op.local_attn.proj_l.weight` [128, 128]
- `self_attn.attn_op.local_attn.proj_l.bias` [128]

#### Cross-Attention (10 tensors):
- `cross_attn.q.weight` [1536, 1536]
- `cross_attn.q.bias` [1536]
- `cross_attn.k.weight` [1536, 1536]
- `cross_attn.k.bias` [1536]
- `cross_attn.v.weight` [1536, 1536]
- `cross_attn.v.bias` [1536]
- `cross_attn.o.weight` [1536, 1536]
- `cross_attn.o.bias` [1536]
- `cross_attn.norm_q.weight` [1536] ← **SYCL Target**
- `cross_attn.norm_k.weight` [1536] ← **SYCL Target**

#### FFN (6 tensors):
- `ffn.0.weight` [8960, 1536]
- `ffn.0.bias` [8960]
- `ffn.2.weight` [1536, 8960]
- `ffn.2.bias` [1536]

#### Other (2 tensors):
- `norm3.weight` [1536] ← **SYCL Target**
- `norm3.bias` [1536] ← **SYCL Target**
- `modulation` [1, 2, 1536] (AdaLN modulation parameters)

---

## 4. Head Structure

**Note**: The head does NOT have normalization layers!

```python
head.modulation: [1, 2, 1536]  # BF16
head.head.weight: [64, 1536]    # BF16 (patch prediction)
head.head.bias: [64]            # BF16
```

The head is a simple linear projection from hidden dim (1536) to patch dim (64).

---

## 5. Data Type Considerations

### BF16 vs FP32 Distribution:
- **BF16**: Main model weights (attention Q/K/V/O, FFN, patch embedding)
- **FP32**: All normalization layers (RMSNorm and LayerNorm weights/biases)

### Implications for SYCL:
1. Our SYCL kernels currently use FP32
2. Model's norm weights are FP32 - ✅ Good match
3. Input/output tensors will be BF16 - ⚠️ Need conversion
4. **Performance impact**: BF16→FP32 conversion on every forward pass

---

## 6. Action Items for Implementation

### Immediate Actions:

1. **Update Phase 2 Tests** ✅
   - Remove tests for `head.norm` (doesn't exist)
   - Remove tests for `blocks.*.norm1` (doesn't exist)
   - Remove tests for `blocks.*.norm2` (doesn't exist)
   - Add tests for `blocks.*.norm3` (LayerNorm)
   - Add tests for `blocks.*.cross_attn.norm_q/k` (RMSNorm)
   - Keep tests for `blocks.*.self_attn.norm_q/k` (RMSNorm)

2. **Update LayerRegistry** ✅
   - Remove `get_head_norm()`
   - Remove `get_block_norm1()`
   - Remove `get_block_norm2()`
   - Add `get_block_norm3()`
   - Add `get_block_cross_attn_norm_q/k()`

3. **Update Test Architecture**
   - Tests should target actual model layers
   - Test data should match BF16 format where appropriate

### Phase 3 Integration Plan Updates:

**Phase 3.1**: ✅ Complete - Model loaded, structure understood

**Phase 3.2**: Single layer replacement
- Target: `blocks.0.self_attn.norm_q` (RMSNorm)
- Simpler than full model test

**Phase 3.3**: Multiple layer replacement
- Target: All `blocks.*.self_attn.norm_q` (30 layers)
- Test attention mechanism still works

**Phase 3.4**: All norm layer types
- Target: All 180 norm layers
- Includes self_attn, cross_attn, and norm3

**Phase 3.5**: Full inference
- Complete forward pass with SYCL kernels
- Compare output quality

**Phase 3.6**: Video quality validation
- Generate video frames
- SSIM/PSNR metrics

---

## 7. Performance Considerations

### Memory Layout:
- Model is BF16 but our kernels are FP32
- Need conversion: BF16 input → FP32 compute → BF16 output
- **Optimization opportunity**: Add BF16 support to SYCL kernels

### Kernel Launch Frequency:
- 30 blocks × (2 self_attn RMSNorm + 2 cross_attn RMSNorm + 1 LayerNorm) = 150 norm ops per forward pass
- Each norm op = 1 SYCL kernel launch
- **Total**: ~150 kernel launches per inference step
- **Critical**: Must implement P0 optimizations (custom operator, USM) before Phase 3

### Attention Complexity:
- Self-attention: Full sequence
- Cross-attention: Sequence attends to text embeddings
- Local attention: Uses `proj_l` for local window attention

---

## 8. Open Questions

1. **BF16 Support**: Should we add BF16 kernel variants for better performance?
2. **Cross-Attention**: Do we replace cross-attention norms or focus on self-attention only?
3. **Local Attention**: Is `proj_l` part of the attention computation we need to handle?
4. **Modulation**: How does `modulation` tensor interact with norm layers?

---

## 9. Summary

### What We Got Wrong:
- ❌ Assumed standard transformer layout (norm1, norm2)
- ❌ Assumed head has norm layer
- ❌ Assumed all data is FP32

### What We Learned:
- ✅ Wan2.1 uses unique architecture (norm_q/k instead of norm1/2)
- ✅ 5 types of norm layers: self_attn.norm_q/k, cross_attn.norm_q/k, norm3
- ✅ 180 total norm tensors to potentially replace
- ✅ Model is BF16, norms are FP32
- ✅ 30 blocks, 1536 hidden dim, 8960 FFN dim

### Impact on Project:
- **Scope**: Larger than expected (180 norm tensors vs ~60 assumed)
- **Complexity**: Higher (cross-attention adds complexity)
- **Performance**: More kernel launches (150 vs ~30)
- **Priority**: P0 optimizations are CRITICAL before integration

---

## 10. Updated Test Priorities

### Must Test (High Impact):
1. `blocks.*.self_attn.norm_q` - Most frequent operation
2. `blocks.*.self_attn.norm_k` - Paired with norm_q
3. `blocks.*.norm3` - FFN path normalization

### Should Test (Medium Impact):
4. `blocks.*.cross_attn.norm_q` - Cross-attention query
5. `blocks.*.cross_attn.norm_k` - Cross-attention key

### Won't Test:
- ❌ `head.norm` - Doesn't exist
- ❌ `blocks.*.norm1` - Doesn't exist  
- ❌ `blocks.*.norm2` - Doesn't exist

---

**Next Step**: Proceed with Phase 3 integration tests using correct model structure
