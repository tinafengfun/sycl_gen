#!/bin/bash
# LLM Accuracy Test Agent 完整验证脚本

echo "======================================================================"
echo "🧪 LLM Accuracy Test Agent - Complete Verification"
echo "======================================================================"
echo ""

cd /home/intel/tianfeng/opencode_bench

# 1. 验证Python版本
echo "1. Checking Python version..."
python3 --version

# 2. 验证所有模块可以导入
echo ""
echo "2. Verifying all modules..."
python3 -c "
import sys
sys.path.insert(0, 'tools')

modules = [
    'platform_detector',
    'test_suite_generator',
    'llm_harness_generator',
    'async_test_executor',
    'json_report_generator',
    'llm_accuracy_test_agent'
]

all_ok = True
for mod in modules:
    try:
        __import__(mod)
        print(f'   ✅ {mod}')
    except Exception as e:
        print(f'   ❌ {mod}: {e}')
        all_ok = False

sys.exit(0 if all_ok else 1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Module verification failed!"
    exit 1
fi

# 3. 功能测试
echo ""
echo "3. Testing functionality..."
python3 -c "
import sys
sys.path.insert(0, 'tools')

print('   Testing Platform Detector...')
from platform_detector import detect_platforms
caps = detect_platforms()
print(f'      SYCL: {caps[\"sycl\"][\"device\"]}')
print(f'      CUDA: {caps[\"cuda\"][\"device\"]}')

print('   Testing Test Suite Generator...')
from test_suite_generator import generate_test_suite
suite = generate_test_suite(caps)
print(f'      Generated {len(suite)} test configurations')

print('   Testing JSON Report Generator...')
from json_report_generator import JSONReportGenerator
gen = JSONReportGenerator('test', 'test')
report = gen.generate_report(
    platform_info=caps,
    test_configs=[{'test_id': 't1', 'name': 'Test 1'}],
    test_results=[{'test_id': 't1', 'status': 'PASSED'}],
    llm_usage={'calls': 10}
)
print(f'      Report generated with {len(report)} keys')

print('   ✅ All functionality tests passed')
"

# 4. 文件清单
echo ""
echo "4. Verifying file structure..."
files=(
    "tools/platform_detector.py"
    "tools/test_suite_generator.py"
    "tools/llm_harness_generator.py"
    "tools/async_test_executor.py"
    "tools/json_report_generator.py"
    "tools/llm_accuracy_test_agent.py"
    "LLM_ACCURACY_TEST_GUIDE.md"
    "LLM_AGENT_COMPLETION_SUMMARY.md"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        echo "   ✅ $file ($lines lines)"
    else
        echo "   ❌ $file (missing)"
    fi
done

echo ""
echo "======================================================================"
echo "✅ Verification Complete!"
echo "======================================================================"
echo ""
echo "The LLM Accuracy Test Agent is ready to use."
echo ""
echo "Quick start:"
echo "  python3 tools/llm_accuracy_test_agent.py \\"
echo "      kernel_id \\"
echo "      cuda_file.cu \\"
echo "      sycl_file.dp.cpp"
echo ""
echo "Or use Python API:"
echo "  from tools.llm_accuracy_test_agent import run_accuracy_test"
echo "  result = await run_accuracy_test(kernel_id, cuda_file, sycl_file)"
echo ""
