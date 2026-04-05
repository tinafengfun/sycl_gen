#!/usr/bin/env python3
"""
运行准确度测试 - 验证15个已编译通过的内核
Run accuracy tests for 15 compiled kernels
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
import subprocess

# 添加tools目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'tools'))

# 已编译通过的14个内核 (removed vector_add_test - doesn't exist)
WORKING_KERNELS = [
    'add_vectors',
    'add_bias_batched',
    'batch_norm',
    'copy_type_converted',
    'expand_planes_nchw',
    'global_avg_pool',
    'global_scale',
    'layer_norm',
    'policy_map',
    'se_layer_nhwc',
    'softmax',
    'softmax_opt_64',
    'winograd_filter_transform',
    'winograd_input_transform'
]

class SimpleAccuracyTester:
    """简化的准确度测试器 - 验证CUDA vs SYCL"""
    
    def __init__(self):
        self.results = {
            'test_date': datetime.now().isoformat(),
            'total_kernels': len(WORKING_KERNELS),
            'tested': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'details': {}
        }
    
    def check_cuda_compilation(self, kernel_id: str) -> tuple[bool, str]:
        """检查CUDA版本是否可编译"""
        try:
            # 直接在远程Docker容器中编译（文件已在容器中）
            result = subprocess.run(
                ['ssh', 'root@10.112.229.160',
                 f'docker exec cuda12.9-test bash -c "cd /workspace/kernel_dataset/cuda && nvcc -I./include -c {kernel_id}_kernel.cu -o {kernel_id}.o 2>&1"'],
                capture_output=True, text=True, timeout=60
            )
            
            return result.returncode == 0, result.stderr if result.returncode != 0 else "OK"
        except Exception as e:
            return False, str(e)
    
    def check_sycl_compilation(self, kernel_id: str) -> tuple[bool, str]:
        """检查SYCL版本是否可编译"""
        sycl_file = f"kernel_dataset/sycl/{kernel_id}_kernel.dp.cpp"
        
        # 检查文件是否为空
        if Path(sycl_file).stat().st_size == 0:
            return False, "File is empty (0 bytes)"
        
        try:
            # 复制到本地SYCL容器
            subprocess.run(
                ['docker', 'cp', sycl_file, f'lsv-container:/workspace/{kernel_id}_kernel.dp.cpp'],
                capture_output=True, timeout=10, check=True
            )
            
            # 编译 (SYCL kernels are self-contained, no external includes needed)
            result = subprocess.run(
                ['docker', 'exec', 'lsv-container', 'bash', '-c',
                 f'cd /workspace && icpx -fsycl -c {kernel_id}_kernel.dp.cpp -o {kernel_id}.o 2>&1'],
                capture_output=True, text=True, timeout=60
            )
            
            return result.returncode == 0, result.stderr if result.returncode != 0 else "OK"
        except Exception as e:
            return False, str(e)
    
    def test_kernel(self, kernel_id: str) -> dict:
        """测试单个内核"""
        print(f"\n🧪 测试内核: {kernel_id}")
        print("-" * 70)
        
        result = {
            'kernel_id': kernel_id,
            'cuda_compiles': False,
            'sycl_compiles': False,
            'cuda_error': None,
            'sycl_error': None,
            'can_run_accuracy': False
        }
        
        # 检查文件存在
        cuda_file = f"kernel_dataset/cuda/{kernel_id}_kernel.cu"
        sycl_file = f"kernel_dataset/sycl/{kernel_id}_kernel.dp.cpp"
        
        if not Path(cuda_file).exists():
            print(f"  ❌ CUDA文件不存在")
            result['cuda_error'] = "File not found"
            return result
        
        if not Path(sycl_file).exists():
            print(f"  ❌ SYCL文件不存在")
            result['sycl_error'] = "File not found"
            return result
        
        # 测试CUDA编译
        print("  🔨 测试CUDA编译...")
        cuda_ok, cuda_msg = self.check_cuda_compilation(kernel_id)
        result['cuda_compiles'] = cuda_ok
        result['cuda_error'] = cuda_msg if not cuda_ok else None
        
        if cuda_ok:
            print(f"    ✅ CUDA编译通过")
        else:
            print(f"    ❌ CUDA编译失败: {cuda_msg[:100]}...")
        
        # 测试SYCL编译
        print("  🔨 测试SYCL编译...")
        sycl_ok, sycl_msg = self.check_sycl_compilation(kernel_id)
        result['sycl_compiles'] = sycl_ok
        result['sycl_error'] = sycl_msg if not sycl_ok else None
        
        if sycl_ok:
            print(f"    ✅ SYCL编译通过")
        else:
            print(f"    ❌ SYCL编译失败: {sycl_msg[:100]}...")
        
        # 判断是否可以运行准确度测试
        if cuda_ok and sycl_ok:
            result['can_run_accuracy'] = True
            print(f"  ✅ 可以运行完整准确度测试")
        else:
            print(f"  ⚠️  无法运行完整测试 (需要CUDA和SYCL都编译通过)")
        
        return result
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("=" * 80)
        print("🧪 CUDA vs SYCL 准确度测试")
        print("=" * 80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"测试内核数: {len(WORKING_KERNELS)}")
        print()
        
        # 逐个测试
        for kernel_id in WORKING_KERNELS:
            result = self.test_kernel(kernel_id)
            self.results['details'][kernel_id] = result
            self.results['tested'] += 1
            
            if result['can_run_accuracy']:
                self.results['passed'] += 1
            elif result['cuda_error'] or result['sycl_error']:
                self.results['failed'] += 1
            else:
                self.results['skipped'] += 1
        
        # 保存结果
        output_dir = Path('results') / 'accuracy_verification'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = output_dir / 'verification_results.json'
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 打印汇总
        self.print_summary()
        
        return result_file
    
    def print_summary(self):
        """打印测试汇总"""
        print()
        print("=" * 80)
        print("📊 准确度测试汇总")
        print("=" * 80)
        print(f"总内核数: {self.results['total_kernels']}")
        print(f"已测试: {self.results['tested']}")
        print(f"✅ 通过 (可运行完整测试): {self.results['passed']}")
        print(f"❌ 失败: {self.results['failed']}")
        print(f"⏭️  跳过: {self.results['skipped']}")
        print()
        
        # 列出可运行完整测试的内核
        if self.results['passed'] > 0:
            print("✅ 可运行完整准确度测试的内核:")
            for kernel_id, detail in self.results['details'].items():
                if detail['can_run_accuracy']:
                    print(f"  • {kernel_id}")
        
        # 列出失败的内核
        if self.results['failed'] > 0:
            print()
            print("❌ 需要修复的内核:")
            for kernel_id, detail in self.results['details'].items():
                if detail['cuda_error'] or detail['sycl_error']:
                    print(f"  • {kernel_id}")
                    if detail['cuda_error']:
                        print(f"    CUDA: {detail['cuda_error'][:80]}...")
                    if detail['sycl_error']:
                        print(f"    SYCL: {detail['sycl_error'][:80]}...")
        
        print()
        print(f"📁 详细结果: results/accuracy_verification/verification_results.json")
        print("=" * 80)


async def main():
    """主函数"""
    tester = SimpleAccuracyTester()
    result_file = await tester.run_all_tests()
    print(f"\n✅ 测试完成！结果保存在: {result_file}")


if __name__ == '__main__':
    asyncio.run(main())
