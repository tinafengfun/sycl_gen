"""
Tests for TurboDiffusion SYCL Integration

Verifies that the SYCL attention modules can be used as drop-in replacements
for the original CUDA-based modules.
"""

import torch
import torch.nn as nn
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import SYCL wrappers
try:
    from turbodiffusion_sycl import (
        FlashAttentionSYCL,
        SparseAttentionSYCL,
        replace_attention_sycl,
        is_sycl_available
    )
    SYCL_AVAILABLE = is_sycl_available()
except ImportError as e:
    print(f"Warning: Could not import turbodiffusion_sycl: {e}")
    SYCL_AVAILABLE = False
    FlashAttentionSYCL = None
    SparseAttentionSYCL = None
    replace_attention_sycl = None


class TestFlashAttentionSYCL(unittest.TestCase):
    """Test Flash Attention SYCL wrapper."""
    
    @unittest.skipUnless(SYCL_AVAILABLE, "SYCL not available")
    def test_flash_attention_forward(self):
        """Test basic Flash Attention forward pass."""
        batch_size = 2
        num_heads = 8
        seq_len = 128
        head_dim = 64
        
        # Create Flash Attention module
        flash_attn = FlashAttentionSYCL(
            head_dim=head_dim,
            num_heads=num_heads,
            causal=True
        )
        
        # Create input tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
        
        # Move to XPU if available
        if torch.xpu.is_available():
            q = q.to('xpu')
            k = k.to('xpu')
            v = v.to('xpu')
            flash_attn = flash_attn.to('xpu')
        
        # Forward pass
        output = flash_attn(q, k, v)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, num_heads, seq_len, head_dim))
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())
    
    @unittest.skipUnless(SYCL_AVAILABLE, "SYCL not available")
    def test_flash_attention_gqa(self):
        """Test Flash Attention with GQA (Grouped Query Attention)."""
        batch_size = 2
        num_heads_q = 16
        num_heads_kv = 4
        seq_len = 128
        head_dim = 64
        
        flash_attn = FlashAttentionSYCL(
            head_dim=head_dim,
            num_heads=num_heads_q,
            num_heads_kv=num_heads_kv,
            causal=True
        )
        
        q = torch.randn(batch_size, num_heads_q, seq_len, head_dim, dtype=torch.bfloat16)
        k = torch.randn(batch_size, num_heads_kv, seq_len, head_dim, dtype=torch.bfloat16)
        v = torch.randn(batch_size, num_heads_kv, seq_len, head_dim, dtype=torch.bfloat16)
        
        if torch.xpu.is_available():
            q = q.to('xpu')
            k = k.to('xpu')
            v = v.to('xpu')
            flash_attn = flash_attn.to('xpu')
        
        output = flash_attn(q, k, v)
        
        # Output should have same number of heads as query
        self.assertEqual(output.shape, (batch_size, num_heads_q, seq_len, head_dim))


class TestSparseAttentionSYCL(unittest.TestCase):
    """Test Sparse Attention SYCL wrapper."""
    
    @unittest.skipUnless(SYCL_AVAILABLE, "SYCL not available")
    def test_sparse_attention_forward(self):
        """Test basic Sparse Attention forward pass."""
        batch_size = 2
        num_heads = 8
        seq_len = 256
        head_dim = 64
        
        sparse_attn = SparseAttentionSYCL(
            head_dim=head_dim,
            topk=0.2,
            BLKQ=64,
            BLKK=64,
            use_bf16=True
        )
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        if torch.xpu.is_available():
            q = q.to('xpu')
            k = k.to('xpu')
            v = v.to('xpu')
            sparse_attn = sparse_attn.to('xpu')
        
        output = sparse_attn(q, k, v)
        
        self.assertEqual(output.shape, (batch_size, num_heads, seq_len, head_dim))
        self.assertFalse(torch.isnan(output).any())
    
    @unittest.skipUnless(SYCL_AVAILABLE, "SYCL not available")
    def test_sparse_attention_with_sparsity_return(self):
        """Test Sparse Attention with sparsity ratio return."""
        batch_size = 2
        num_heads = 8
        seq_len = 256
        head_dim = 64
        
        sparse_attn = SparseAttentionSYCL(
            head_dim=head_dim,
            topk=0.2,
            BLKQ=64,
            BLKK=64
        )
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        if torch.xpu.is_available():
            q = q.to('xpu')
            k = k.to('xpu')
            v = v.to('xpu')
            sparse_attn = sparse_attn.to('xpu')
        
        output, sparsity = sparse_attn(q, k, v, return_sparsity=True)
        
        self.assertEqual(output.shape, (batch_size, num_heads, seq_len, head_dim))
        self.assertIsInstance(sparsity, float)
        self.assertGreater(sparsity, 0.0)
        self.assertLessEqual(sparsity, 1.0)
    
    def test_sparse_attention_pytorch_fallback(self):
        """Test Sparse Attention PyTorch fallback (without SYCL)."""
        # This test doesn't require SYCL to be available
        if not SYCL_AVAILABLE:
            self.skipTest("Testing fallback requires SYCL to be unavailable")
        
        batch_size = 2
        num_heads = 4
        seq_len = 64
        head_dim = 32
        
        sparse_attn = SparseAttentionSYCL(
            head_dim=head_dim,
            topk=0.5,
            BLKQ=32,
            BLKK=32
        )
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # Force use of PyTorch fallback by temporarily removing SYCL ops
        import turbodiffusion_sycl.attention as attn_module
        original_sycl_ops = attn_module.sycl_ops
        attn_module.sycl_ops = None
        
        try:
            output = sparse_attn._sparse_attention_pytorch(
                q.reshape(batch_size * num_heads, seq_len, head_dim),
                k.reshape(batch_size * num_heads, seq_len, head_dim),
                v.reshape(batch_size * num_heads, seq_len, head_dim),
                torch.zeros(batch_size * num_heads, 2, 2, dtype=torch.int64),  # Dummy LUT
                2
            )
            
            self.assertEqual(output.shape, (batch_size * num_heads, seq_len, head_dim))
        finally:
            # Restore original
            attn_module.sycl_ops = original_sycl_ops


class TestModelUtilities(unittest.TestCase):
    """Test model modification utilities."""
    
    def test_get_model_info(self):
        """Test model info extraction."""
        from turbodiffusion_sycl.model_utils import get_model_info
        
        # Create a simple model with attention-like module
        class MockAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_op = type('obj', (object,), {
                    'local_attn': nn.Identity()
                })()
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = MockAttention()
                self.norm = nn.LayerNorm(64)
        
        model = MockModel()
        info = get_model_info(model)
        
        self.assertEqual(info['total_modules'], 4)  # MockModel + MockAttention + obj + norm
        self.assertEqual(len(info['attention_modules']), 1)
        self.assertEqual(len(info['norm_modules']), 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for SYCL modules."""
    
    @unittest.skipUnless(SYCL_AVAILABLE, "SYCL not available")
    def test_end_to_end_sparse_attention(self):
        """Test end-to-end sparse attention replacement and execution."""
        # Create a mock model that mimics Wan model structure
        class MockWanAttention(nn.Module):
            def __init__(self, dim, num_heads):
                super().__init__()
                self.dim = dim
                self.num_heads = num_heads
                self.attn_op = type('obj', (object,), {
                    'local_attn': nn.Identity()
                })()
        
        class MockWanModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([
                    MockWanAttention(512, 8),
                    MockWanAttention(512, 8)
                ])
        
        model = MockWanModel()
        
        # Replace attention with SYCL version
        model = replace_attention_sycl(
            model,
            attention_type='sparse',
            topk=0.2,
            verbose=False
        )
        
        # Check that attention was replaced
        for block in model.blocks:
            self.assertIsInstance(block.attn_op.local_attn, SparseAttentionSYCL)
    
    @unittest.skipUnless(SYCL_AVAILABLE, "SYCL not available")
    def test_end_to_end_flash_attention(self):
        """Test end-to-end flash attention replacement."""
        class MockWanAttention(nn.Module):
            def __init__(self, dim, num_heads):
                super().__init__()
                self.dim = dim
                self.num_heads = num_heads
                self.attn_op = type('obj', (object,), {
                    'local_attn': nn.Identity()
                })()
        
        class MockWanModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([MockWanAttention(512, 8)])
        
        model = MockWanModel()
        
        model = replace_attention_sycl(
            model,
            attention_type='flash',
            verbose=False
        )
        
        for block in model.blocks:
            self.assertIsInstance(block.attn_op.local_attn, FlashAttentionSYCL)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
