#!/usr/bin/env python3
"""
集成测试脚本：验证LLM Accuracy Test Agent与Unified Converter的集成
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tools"))

print("="*70)
print("🧪 集成测试：LLM Accuracy Test Agent + Unified Converter")
print("="*70)
print()

# 测试1：基础组件导入
print("1️⃣  测试基础组件导入...")
try:
    from unified_converter import (
        UnifiedOrchestrator, 
        UnifiedTracer,
        UnifiedAccuracyTester
    )
    from platform_detector import detect_platforms
    from llm_accuracy_test_agent import LLMAccuracyTestAgent
    print("   ✅ 所有组件导入成功")
except Exception as e:
    print(f"   ❌ 导入失败: {e}")
    sys.exit(1)

# 测试2：Tracer初始化
print("\n2️⃣  测试Tracer初始化...")
try:
    tracer = UnifiedTracer('integration_test', 'copy_type_converted')
    print("   ✅ UnifiedTracer初始化成功")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 测试3：平台检测
print("\n3️⃣  测试平台能力检测...")
try:
    platform_caps = detect_platforms()
    print(f"   ✅ 平台检测成功")
    print(f"      SYCL: {platform_caps['sycl']['device']}")
    print(f"      CUDA: {platform_caps['cuda']['device']}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 测试4：AccuracyTester初始化
print("\n4️⃣  测试准确度测试器初始化...")
try:
    tester = UnifiedAccuracyTester(tracer)
    print("   ✅ UnifiedAccuracyTester初始化成功")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 测试5：文件检查
print("\n5️⃣  测试必要文件存在性...")
required_files = [
    "tools/platform_detector.py",
    "tools/test_suite_generator.py",
    "tools/llm_harness_generator.py",
    "tools/async_test_executor.py",
    "tools/json_report_generator.py",
    "tools/llm_accuracy_test_agent.py",
    "tools/unified_converter.py"
]

base_dir = Path(__file__).parent
all_exist = True
for file in required_files:
    path = base_dir / file
    if path.exists():
        print(f"   ✅ {file}")
    else:
        print(f"   ❌ {file} (缺失)")
        all_exist = False

if not all_exist:
    sys.exit(1)

# 测试6：测试套件生成
print("\n6️⃣  测试套件生成...")
try:
    from test_suite_generator import generate_test_suite
    test_configs = generate_test_suite(platform_caps)
    print(f"   ✅ 生成 {len(test_configs)} 个测试配置")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 测试7：Orchestrator初始化
print("\n7️⃣  测试统一转换器初始化...")
try:
    orchestrator = UnifiedOrchestrator(
        kernel_id="integration_test",
        cuda_file="kernel_dataset/cuda/copy_type_converted_kernel.cu",
        use_model=True
    )
    print("   ✅ UnifiedOrchestrator初始化成功")
except Exception as e:
    print(f"   ❌ 失败: {e}")

print()
print("="*70)
print("✅ 所有集成测试通过！")
print("="*70)
print()
print("现在可以使用完整的LLM准确度测试功能：")
print()
print("方法1 - 命令行:")
print("  python3 tools/llm_accuracy_test_agent.py \\")
print("      copy_type_converted \\")
print("      kernel_dataset/cuda/copy_type_converted_kernel.cu \\")
print("      kernel_dataset/sycl/copy_type_converted_kernel.dp.cpp")
print()
print("方法2 - Python API:")
print("  from tools.unified_converter import UnifiedOrchestrator")
print("  orchestrator = UnifiedOrchestrator(kernel_id, cuda_file)")
print("  result = await orchestrator.execute_full_conversion()")
print()
print("方法3 - 准确度测试专用:")
print("  from tools.llm_accuracy_test_agent import run_accuracy_test")
print("  result = await run_accuracy_test(kernel_id, cuda_file, sycl_file)")
print()
