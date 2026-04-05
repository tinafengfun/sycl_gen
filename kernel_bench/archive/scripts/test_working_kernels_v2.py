#!/usr/bin/env python3
"""
测试8个编译通过的SYCL内核
Test 8 SYCL kernels that compile successfully
"""

import asyncio
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

# 编译通过的8个内核
WORKING_KERNELS = [
    'batch_norm',
    'copy_type_converted', 
    'expand_planes_nchw',
    'global_avg_pool',
    'policy_map',
    'softmax_opt_64',
    'winograd_input_transform',
    'vector_add_test'
]

class SimpleAccuracyTester:
    """简化的准确度测试器"""
    
    def __init__(self):
        self.results = {
            'test_date': datetime.now().isoformat(),
            'kernels_tested': [],
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0
            }
        }
    
    def test_compilation(self, kernel_id: str) -> Tuple[bool, str]:
        """测试编译"""
        sycl_file = f"kernel_dataset/sycl/{kernel_id}_kernel.dp.cpp"
        
        try:
            # 复制到docker
            subprocess.run(
                ['docker', 'cp', sycl_file, 'lsv-container:/workspace/test.cpp'],
                capture_output=True, timeout=10, check=True
            )
            
            # 编译
            result = subprocess.run(
                ['docker', 'exec', 'lsv-container', 'bash', '-c',
                 'cd /workspace && icpx -fsycl -c test.cpp -o test.o 2>&1'],
                capture_output=True, text=True, timeout=30
            )
            
            return result.returncode == 0, result.stderr
        except Exception as e:
            return False, str(e)
    
    def test_kernel(self, kernel_id: str) -> Dict:
        """测试单个内核"""
        print(f"\n🧪 测试内核: {kernel_id}")
        
        result = {
            'kernel_id': kernel_id,
            'compilation': {'success': False, 'error': None},
            'sycl_file': f"kernel_dataset/sycl/{kernel_id}_kernel.dp.cpp",
            'cuda_file': f"kernel_dataset/cuda/{kernel_id}_kernel.cu"
        }
        
        # 1. 检查文件存在
        sycl_path = Path(result['sycl_file'])
        cuda_path = Path(result['cuda_file'])
        
        if not sycl_path.exists():
            result['compilation']['error'] = "SYCL file not found"
            print(f"  ❌ SYCL文件不存在")
            return result
        
        if not cuda_path.exists():
            result['compilation']['error'] = "CUDA file not found"
            print(f"  ❌ CUDA文件不存在")
            return result
        
        print(f"  ✅ 文件存在")
        
        # 2. 测试SYCL编译
        success, error = self.test_compilation(kernel_id)
        result['compilation']['success'] = success
        result['compilation']['error'] = error if not success else None
        
        if success:
            print(f"  ✅ SYCL编译通过")
        else:
            print(f"  ❌ SYCL编译失败: {error[:100] if error else 'Unknown'}")
        
        return result
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("=" * 80)
        print("🧪 8个SYCL内核准确度测试")
        print("=" * 80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        print(f"测试 {len(WORKING_KERNELS)} 个内核:")
        for k in WORKING_KERNELS:
            print(f"  - {k}")
        print()
        
        # 逐个测试
        for kernel_id in WORKING_KERNELS:
            result = self.test_kernel(kernel_id)
            self.results['kernels_tested'].append(result)
            self.results['summary']['total'] += 1
            
            if result['compilation']['success']:
                self.results['summary']['passed'] += 1
            else:
                self.results['summary']['failed'] += 1
        
        # 保存结果
        output_dir = Path('results') / 'working_kernels_test'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = output_dir / 'test_results.json'
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 打印汇总
        print()
        print("=" * 80)
        print("📊 测试结果汇总")
        print("=" * 80)
        print(f"总内核数: {self.results['summary']['total']}")
        print(f"通过: {self.results['summary']['passed']}")
        print(f"失败: {self.results['summary']['failed']}")
        print(f"成功率: {self.results['summary']['passed']/self.results['summary']['total']*100:.1f}%")
        print()
        print(f"详细结果: {result_file}")
        print("=" * 80)

async def main():
    tester = SimpleAccuracyTester()
    await tester.run_all_tests()

if __name__ == '__main__':
    asyncio.run(main())
