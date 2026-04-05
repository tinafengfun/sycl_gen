"""
Test script for Flash Attention SYCL implementation

Tests:
1. Basic functionality with small tensors
2. GQA (Grouped Query Attention) support
3. BF16 precision
4. Variable length sequences
"""

import torch
import numpy as np

def test_flash_attention_basic():
    """Test basic flash attention with small tensors"""
    print("=" * 60)
    print("Test 1: Basic Flash Attention")
    print("=" * 60)
    
    try:
        import turbodiffusion_sycl_ops as ops
    except ImportError as e:
        print(f"❌ Failed to import module: {e}")
        print("Note: Module needs to be built with: CC=icpx CXX=icpx python setup.py build_ext --inplace")
        # Try importing with path
        import sys
        sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/turbodiffusion-sycl')
        sys.path.insert(0, '/home/intel/tianfeng/opencode_bench/turbodiffusion-sycl/operators')
        try:
            import turbodiffusion_sycl_ops as ops
            print("✅ Module imported with path adjustment")
        except ImportError as e2:
            print(f"❌ Still failed: {e2}")
            return False
    
    # Create test tensors
    B, H_q, H_kv, S, D = 2, 8, 2, 64, 64  # GQA: 8 query heads, 2 KV heads
    
    # Create BF16 tensors on XPU
    try:
        q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16, device='xpu')
        k = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device='xpu')
        v = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device='xpu')
    except RuntimeError as e:
        print(f"❌ XPU not available: {e}")
        print("Note: This test requires Intel XPU (GPU) support")
        return False
    
    print(f"Input shapes:")
    print(f"  Q: {q.shape} (dtype={q.dtype}, device={q.device})")
    print(f"  K: {k.shape} (dtype={k.dtype}, device={k.device})")
    print(f"  V: {v.shape} (dtype={v.dtype}, device={v.device})")
    
    # Run Flash Attention
    try:
        output = ops.flash_attention_forward(q, k, v, None, 0.0)
        print(f"\n✅ Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        print(f"   Output device: {output.device}")
        
        # Verify output shape
        expected_shape = (B, H_q, S, D)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
        print("✅ Shape check passed")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Flash Attention failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flash_attention_gqa():
    """Test GQA (Grouped Query Attention) support"""
    print("\n" + "=" * 60)
    print("Test 2: Grouped Query Attention (GQA)")
    print("=" * 60)
    
    try:
        import turbodiffusion_sycl_ops as ops
    except ImportError:
        print("❌ Module not available")
        return False
    
    # Test different GQA configurations
    test_cases = [
        (1, 8, 1, 32, 64),   # MQA: 8 query heads share 1 KV head
        (2, 12, 4, 64, 64),  # GQA: 3 groups
        (1, 16, 8, 128, 64), # GQA: 2 groups
    ]
    
    for B, H_q, H_kv, S, D in test_cases:
        print(f"\nTesting B={B}, H_q={H_q}, H_kv={H_kv}, S={S}, D={D}")
        
        try:
            q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16, device='xpu')
            k = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device='xpu')
            v = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device='xpu')
            
            output = ops.flash_attention_forward(q, k, v, None, 0.0)
            
            expected_shape = (B, H_q, S, D)
            assert output.shape == expected_shape
            print(f"  ✅ GQA config passed: {H_q}/{H_kv} = {H_q//H_kv} groups")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            return False
    
    print("\n✅ All GQA configurations passed!")
    return True


def test_flash_attention_varlen():
    """Test variable length sequence support"""
    print("\n" + "=" * 60)
    print("Test 3: Variable Length Sequences")
    print("=" * 60)
    
    try:
        import turbodiffusion_sycl_ops as ops
    except ImportError:
        print("❌ Module not available")
        return False
    
    # Create packed sequences
    # Batch with 3 sequences of different lengths
    seq_lengths = [32, 64, 48]
    total_tokens = sum(seq_lengths)
    H_q, H_kv, D = 8, 2, 64
    
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Total tokens: {total_tokens}")
    
    # Create cumulative lengths
    cu_seqlens = torch.tensor([0, 32, 96, 144], dtype=torch.int32, device='xpu')
    
    try:
        q = torch.randn(total_tokens, H_q, D, dtype=torch.bfloat16, device='xpu')
        k = torch.randn(total_tokens, H_kv, D, dtype=torch.bfloat16, device='xpu')
        v = torch.randn(total_tokens, H_kv, D, dtype=torch.bfloat16, device='xpu')
        
        output = ops.flash_attention_forward_varlen(
            q, k, v, cu_seqlens, max(seq_lengths), 0.0
        )
        
        expected_shape = (total_tokens, H_q, D)
        assert output.shape == expected_shape
        print(f"✅ Output shape: {output.shape}")
        print("✅ Variable length test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """Quick performance check"""
    print("\n" + "=" * 60)
    print("Test 4: Performance Check")
    print("=" * 60)
    
    try:
        import turbodiffusion_sycl_ops as ops
    except ImportError:
        print("❌ Module not available")
        return False
    
    # Larger test case
    B, H_q, H_kv, S, D = 4, 16, 4, 256, 64
    
    print(f"Config: B={B}, H_q={H_q}, H_kv={H_kv}, S={S}, D={D}")
    
    try:
        q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16, device='xpu')
        k = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device='xpu')
        v = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device='xpu')
        
        # Warmup
        for _ in range(3):
            _ = ops.flash_attention_forward(q, k, v, None, 0.0)
        torch.xpu.synchronize()
        
        # Benchmark
        import time
        start = time.time()
        iterations = 10
        for _ in range(iterations):
            _ = ops.flash_attention_forward(q, k, v, None, 0.0)
        torch.xpu.synchronize()
        elapsed = time.time() - start
        
        avg_time = elapsed / iterations * 1000  # ms
        print(f"✅ Average time: {avg_time:.2f} ms")
        print(f"   Throughput: {iterations / elapsed:.1f} iterations/sec")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("TurboDiffusion Flash Attention SYCL Tests")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Basic", test_flash_attention_basic()))
    results.append(("GQA", test_flash_attention_gqa()))
    results.append(("Variable Length", test_flash_attention_varlen()))
    results.append(("Performance", test_performance()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed")


if __name__ == "__main__":
    main()
