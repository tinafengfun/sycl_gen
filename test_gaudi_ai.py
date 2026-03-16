#!/usr/bin/env python3
"""
Test Gaudi AI Integration
测试Gaudi AI集成
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from gaudi_ai_client import GaudiAIClient
from model_config import MODEL_CONFIG, SYSTEM_PROMPT


async def test_connection():
    """测试API连接"""
    print("\n🧪 Test 1: API Connection")
    
    api_key = MODEL_CONFIG["options"]["apiKey"]
    base_url = MODEL_CONFIG["options"]["baseURL"]
    
    print(f"   API URL: {base_url}")
    print(f"   API Key: {api_key[:20]}...")
    
    client = GaudiAIClient(api_key=api_key, base_url=base_url)
    
    try:
        # 测试简单请求
        result = await client.generate_code(
            prompt="Say 'OK'",
            max_tokens=10,
            timeout=30
        )
        print(f"   Response: {result}")
        print("   ✅ API connection successful")
        return True
    except Exception as e:
        print(f"   ❌ API connection failed: {e}")
        return False


async def test_simple_conversion():
    """测试简单代码转换"""
    print("\n🧪 Test 2: Simple CUDA to SYCL Conversion")
    
    cuda_code = """
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
"""
    
    prompt = f"""Convert this CUDA kernel to SYCL:

{cuda_code}

Requirements:
1. Convert to SYCL 2020
2. Use sycldnn_backend namespace
3. Replace threadIdx/blockIdx with sycl equivalents

Generate only the SYCL code:"""
    
    api_key = MODEL_CONFIG["options"]["apiKey"]
    base_url = MODEL_CONFIG["options"]["baseURL"]
    model = MODEL_CONFIG["default_model"]
    
    client = GaudiAIClient(api_key=api_key, base_url=base_url)
    
    try:
        print(f"   Model: {model}")
        print("   Generating SYCL code...")
        
        result = await client.generate_code(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            model=model,
            temperature=0.1,
            max_tokens=2048,
            timeout=60
        )
        
        print("   ✅ Generation successful")
        print("\n   Generated code preview:")
        print("   " + "="*50)
        # 只显示前500字符
        preview = result[:500] + "..." if len(result) > 500 else result
        for line in preview.split('\n')[:20]:
            print(f"   {line}")
        print("   " + "="*50)
        
        # 检查是否包含关键元素
        checks = [
            ('#include <sycl/sycl.hpp>', 'SYCL header'),
            ('namespace sycldnn_backend', 'Namespace'),
            ('parallel_for', 'SYCL kernel launch')
        ]
        
        print("\n   Validation:")
        all_passed = True
        for pattern, name in checks:
            if pattern in result:
                print(f"   ✅ {name}")
            else:
                print(f"   ⚠️  {name} not found")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_model_variants():
    """测试不同模型"""
    print("\n🧪 Test 3: Testing Different Models")
    
    api_key = MODEL_CONFIG["options"]["apiKey"]
    base_url = MODEL_CONFIG["options"]["baseURL"]
    
    models = [
        "Qwen3-Coder-30B-A3B-Instruct",
        "DeepSeek-R1-G2-static-671B"
    ]
    
    for model in models:
        print(f"\n   Testing model: {model}")
        client = GaudiAIClient(api_key=api_key, base_url=base_url)
        
        try:
            result = await client.generate_code(
                prompt="Generate a simple 'Hello World' function in C++",
                model=model,
                max_tokens=100,
                timeout=30
            )
            print(f"   ✅ {model} works")
        except Exception as e:
            print(f"   ❌ {model} failed: {e}")
    
    return True


async def main():
    """主测试函数"""
    print("="*60)
    print("🚀 Gaudi AI Integration Test")
    print("="*60)
    
    tests = [
        ("API Connection", test_connection),
        ("Simple Conversion", test_simple_conversion),
        ("Model Variants", test_model_variants),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if await test_func():
                passed += 1
            else:
                failed += 1
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
        print("\n✅ All Gaudi AI integration tests passed!")
        print("Ready to use Gaudi AI for batch conversion.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        print("Please check API configuration and network connectivity.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
