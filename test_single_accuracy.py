#!/usr/bin/env python3
"""
测试单个内核准确度 - 用于验证流程
Test single kernel accuracy - for workflow verification
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'tools'))

from llm_accuracy_test_agent import LLMAccuracyTestAgent

async def test_single_kernel():
    """测试一个简单内核"""
    
    kernel_id = "copy_type_converted"
    cuda_file = "kernel_dataset/cuda/copy_type_converted_kernel.cu"
    sycl_file = "kernel_dataset/sycl/copy_type_converted_kernel.dp.cpp"
    
    print(f"🧪 测试内核: {kernel_id}")
    print(f"   CUDA: {cuda_file}")
    print(f"   SYCL: {sycl_file}")
    print()
    
    agent = LLMAccuracyTestAgent(kernel_id=kernel_id)
    
    result = await agent.run_full_accuracy_test(
        cuda_file=cuda_file,
        sycl_file=sycl_file,
        output_dir="results/test_single"
    )
    
    if result.success:
        print("\n✅ 测试成功!")
        if result.report:
            print(f"报告: {result.report}")
    else:
        print(f"\n❌ 测试失败: {result.error}")
    
    return result

if __name__ == '__main__':
    asyncio.run(test_single_kernel())
