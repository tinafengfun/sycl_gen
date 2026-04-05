#!/usr/bin/env python3
"""
Batch Converter - Process all 30 kernels
批量转换器 - 处理所有30个kernel

Usage:
  python3 tools/batch_convert.py --all
  python3 tools/batch_convert.py --kernels kernel1,kernel2,kernel3
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import tempfile
import shutil

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from unified_converter import (
    UnifiedOrchestrator, ConversionError
)


class BatchConverter:
    """批量转换器"""
    
    def __init__(self, output_dir: str, workers: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        
        # 统计信息
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "in_progress": 0,
            "start_time": datetime.now().isoformat(),
            "results": []
        }
        
        # 日志文件
        self.log_file = self.output_dir / "batch_conversion.log"
        
    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}\n"
        print(log_line.strip())
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line)
    
    def get_all_kernels(self) -> List[str]:
        """获取所有kernel列表"""
        cuda_dir = Path("kernel_dataset/cuda")
        if not cuda_dir.exists():
            self.log("ERROR: kernel_dataset/cuda directory not found", "ERROR")
            return []
        
        kernels = []
        for cu_file in sorted(cuda_dir.glob("*_kernel.cu")):
            kernel_name = cu_file.stem.replace("_kernel", "")
            kernels.append(kernel_name)
        
        self.log(f"Found {len(kernels)} kernels to process")
        return kernels
    
    async def convert_single_kernel(self, kernel_id: str) -> Dict:
        """转换单个kernel"""
        self.log(f"Starting conversion: {kernel_id}")
        
        cuda_file = f"kernel_dataset/cuda/{kernel_id}_kernel.cu"
        
        if not os.path.exists(cuda_file):
            self.log(f"ERROR: CUDA file not found: {cuda_file}", "ERROR")
            return {
                "kernel_id": kernel_id,
                "status": "FAILED",
                "error": "CUDA file not found"
            }
        
        try:
            # 创建转换器实例
            orchestrator = UnifiedOrchestrator(
                kernel_id=kernel_id,
                cuda_file=cuda_file,
                use_model=False  # 使用rule-based，更稳定
            )
            
            # 执行转换
            result = await orchestrator.execute_full_conversion()
            
            # 记录结果
            self.log(f"Completed {kernel_id}: {result.status}")
            
            return {
                "kernel_id": kernel_id,
                "status": result.status.upper(),
                "duration": result.duration_seconds,
                "compilation": "SUCCESS" if result.compilation_success else "FAILED",
                "accuracy": f"{result.accuracy_pass_rate*100:.1f}%",
                "output_file": result.output_file
            }
            
        except Exception as e:
            self.log(f"ERROR converting {kernel_id}: {str(e)}", "ERROR")
            return {
                "kernel_id": kernel_id,
                "status": "FAILED",
                "error": str(e)
            }
    
    async def process_batch(self, kernels: List[str]):
        """批量处理kernel"""
        self.stats["total"] = len(kernels)
        self.log(f"Starting batch conversion of {len(kernels)} kernels")
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"Workers: {self.workers}")
        
        # 使用信号量限制并发数
        semaphore = asyncio.Semaphore(self.workers)
        
        async def process_with_limit(kernel_id: str):
            async with semaphore:
                self.stats["in_progress"] += 1
                result = await self.convert_single_kernel(kernel_id)
                self.stats["in_progress"] -= 1
                
                if result["status"] == "SUCCESS":
                    self.stats["success"] += 1
                else:
                    self.stats["failed"] += 1
                
                self.stats["results"].append(result)
                return result
        
        # 创建所有任务
        tasks = [process_with_limit(kernel) for kernel in kernels]
        
        # 等待所有任务完成
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.log(f"Batch conversion completed")
    
    def generate_report(self):
        """生成完成度报告"""
        self.stats["end_time"] = datetime.now().isoformat()
        
        # 计算成功率
        success_rate = self.stats["success"] / max(self.stats["total"], 1) * 100
        
        # 保存JSON报告
        report_file = self.output_dir / "completion_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        
        # 生成Markdown报告
        md_report = self._generate_markdown_report(success_rate)
        md_file = self.output_dir / "completion_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        self.log(f"Report saved to: {report_file}")
        self.log(f"Markdown report: {md_file}")
        
        return success_rate
    
    def _generate_markdown_report(self, success_rate: float) -> str:
        """生成Markdown报告"""
        report = f"""# CUDA-to-SYCL Batch Conversion Report

## Summary

- **Total Kernels**: {self.stats['total']}
- **Successful**: {self.stats['success']}
- **Failed**: {self.stats['failed']}
- **Success Rate**: {success_rate:.1f}%
- **Start Time**: {self.stats['start_time']}
- **End Time**: {self.stats.get('end_time', 'N/A')}

## Results

| Kernel | Status | Duration | Compilation | Accuracy |
|--------|--------|----------|-------------|----------|
"""
        
        for result in self.stats["results"]:
            kernel_id = result["kernel_id"]
            status = result["status"]
            duration = result.get("duration", "N/A")
            compilation = result.get("compilation", "N/A")
            accuracy = result.get("accuracy", "N/A")
            
            emoji = "✅" if status == "SUCCESS" else "❌"
            report += f"| {kernel_id} | {emoji} {status} | {duration}s | {compilation} | {accuracy} |\n"
        
        report += """
## Failed Kernels

"""
        
        failed_kernels = [r for r in self.stats["results"] if r["status"] != "SUCCESS"]
        if failed_kernels:
            for result in failed_kernels:
                report += f"- **{result['kernel_id']}**: {result.get('error', 'Unknown error')}\n"
        else:
            report += "All kernels converted successfully!\n"
        
        report += f"""
---

*Generated by BatchConverter on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        return report


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Batch convert CUDA kernels to SYCL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all kernels
  python3 tools/batch_convert.py --all
  
  # Convert specific kernels
  python3 tools/batch_convert.py --kernels kernel1,kernel2
  
  # Specify output directory
  python3 tools/batch_convert.py --all --output results/batch_001
  
  # Use more workers
  python3 tools/batch_convert.py --all --workers 8
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all kernels in kernel_dataset/cuda/"
    )
    
    parser.add_argument(
        "--kernels",
        type=str,
        help="Comma-separated list of kernel names to convert"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=f"results/batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if not args.all and not args.kernels:
        parser.error("Please specify --all or --kernels")
    
    # 创建批量转换器
    converter = BatchConverter(
        output_dir=args.output,
        workers=args.workers
    )
    
    # 获取要处理的kernel列表
    if args.all:
        kernels = converter.get_all_kernels()
    else:
        kernels = [k.strip() for k in args.kernels.split(",")]
    
    if not kernels:
        print("ERROR: No kernels to process")
        sys.exit(1)
    
    # 开始批量处理
    print("="*60)
    print("🚀 Batch CUDA-to-SYCL Conversion")
    print("="*60)
    print(f"Kernels to process: {len(kernels)}")
    print(f"Output directory: {args.output}")
    print(f"Workers: {args.workers}")
    print("="*60)
    
    await converter.process_batch(kernels)
    
    # 生成报告
    success_rate = converter.generate_report()
    
    print("\n" + "="*60)
    print("📊 Batch Conversion Complete")
    print("="*60)
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Successful: {converter.stats['success']}/{converter.stats['total']}")
    print(f"Failed: {converter.stats['failed']}/{converter.stats['total']}")
    print(f"Output: {args.output}/")
    print("="*60)
    
    # 返回退出码
    sys.exit(0 if success_rate >= 90 else 1)


if __name__ == "__main__":
    asyncio.run(main())
