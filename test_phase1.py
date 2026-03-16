#!/usr/bin/env python3
"""
Phase 1 Test - UnifiedAccuracyTester
测试准确度测试Agent的核心功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from unified_converter import (
    UnifiedTracer, UnifiedAccuracyTester, Phase, Status, ConversionResult
)
import numpy as np

async def test_generate_test_data():
    """测试数据生成功能"""
    print("\n🧪 测试1: generate_test_data()")
    
    tracer = UnifiedTracer("test_session", "test_kernel")
    tester = UnifiedAccuracyTester(tracer)
    
    # Test random config
    config = {"N": 1, "C": 64, "dtype": "float", "layout": "nchw", "test_type": "random", "name": "test_random"}
    data = tester.generate_test_data(config)
    
    expected_size = 1 * 64 * 8 * 8  # N * C * 8 * 8
    assert len(data) == expected_size, f"Size mismatch: {len(data)} != {expected_size}"
    assert data.dtype == np.float32, f"Dtype mismatch: {data.dtype}"
    print(f"  ✅ Random data: {len(data)} elements, dtype={data.dtype}")
    
    # Test ones config
    config["test_type"] = "ones"
    data = tester.generate_test_data(config)
    assert np.all(data == 1.0), "Ones test failed"
    print(f"  ✅ Ones data: all values are 1.0")
    
    # Test sequential config
    config["test_type"] = "sequential"
    data = tester.generate_test_data(config)
    expected = (np.arange(len(data)) % 100) / 100.0
    assert np.allclose(data, expected), "Sequential test failed"
    print(f"  ✅ Sequential data: correct pattern")
    
    print("  ✅ All generate_test_data tests passed!")
    return True

async def test_compare_results():
    """测试结果对比功能"""
    print("\n🧪 测试2: compare_results()")
    
    tracer = UnifiedTracer("test_session", "test_kernel")
    tester = UnifiedAccuracyTester(tracer)
    
    # Test identical arrays (should PASS)
    cuda_out = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    sycl_out = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    config = {"dtype": "float", "name": "test_identical"}
    
    result = tester.compare_results(cuda_out, sycl_out, config)
    
    assert result['status'] == 'PASS', f"Expected PASS, got {result['status']}"
    assert result['max_abs_diff'] == 0.0, f"Expected 0.0, got {result['max_abs_diff']}"
    print(f"  ✅ Identical arrays: PASS (max_err=0.0)")
    
    # Test with small differences (should PASS)
    sycl_out = np.array([1.0, 2.0, 3.000001, 4.0, 5.0], dtype=np.float32)
    result = tester.compare_results(cuda_out, sycl_out, config)
    
    assert result['status'] == 'PASS', f"Expected PASS, got {result['status']}"
    assert result['max_abs_diff'] < 1e-5, f"Error too large: {result['max_abs_diff']}"
    print(f"  ✅ Small differences: PASS (max_err={result['max_abs_diff']:.2e})")
    
    # Test with large differences (should FAIL)
    sycl_out = np.array([1.0, 2.0, 300.0, 4.0, 5.0], dtype=np.float32)
    result = tester.compare_results(cuda_out, sycl_out, config)
    
    assert result['status'] == 'FAIL', f"Expected FAIL, got {result['status']}"
    print(f"  ✅ Large differences: FAIL (as expected)")
    
    # Test size mismatch
    sycl_out = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = tester.compare_results(cuda_out, sycl_out, config)
    
    assert result['status'] == 'SIZE_MISMATCH', f"Expected SIZE_MISMATCH, got {result['status']}"
    print(f"  ✅ Size mismatch: detected correctly")
    
    print("  ✅ All compare_results tests passed!")
    return True

async def test_class_structure():
    """测试类结构是否正确"""
    print("\n🧪 测试3: Class Structure")
    
    tracer = UnifiedTracer("test_session", "test_kernel")
    tester = UnifiedAccuracyTester(tracer)
    
    # Check attributes
    assert hasattr(tester, 'tracer'), "Missing tracer attribute"
    assert hasattr(tester, 'test_configs'), "Missing test_configs attribute"
    assert hasattr(tester, 'tolerance'), "Missing tolerance attribute"
    assert len(tester.test_configs) == 5, f"Expected 5 configs, got {len(tester.test_configs)}"
    
    print(f"  ✅ Has tracer attribute")
    print(f"  ✅ Has {len(tester.test_configs)} test configs")
    print(f"  ✅ Has tolerance configs for {list(tester.tolerance.keys())}")
    
    # Check methods
    methods = ['test', 'generate_test_data', 'compile_and_run_cuda', 
               'compile_and_run_sycl', 'compare_results', '_generate_summary',
               '_generate_cuda_test_cpp', '_generate_sycl_test_cpp']
    
    for method in methods:
        assert hasattr(tester, method), f"Missing method: {method}"
        assert callable(getattr(tester, method)), f"Method not callable: {method}"
        print(f"  ✅ Has method: {method}()")
    
    print("  ✅ All class structure tests passed!")
    return True

async def main():
    """主测试函数"""
    print("="*60)
    print("🚀 Phase 1 Test - UnifiedAccuracyTester")
    print("="*60)
    
    tests = [
        ("Generate Test Data", test_generate_test_data),
        ("Compare Results", test_compare_results),
        ("Class Structure", test_class_structure),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ Test '{name}' failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✅ All Phase 1 unit tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
