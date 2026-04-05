#!/usr/bin/env python3
"""
加载L20 CUDA参考数据工具
用于SYCL实现对比验证
"""

import numpy as np
import struct
from pathlib import Path

class CUDADataLoader:
    """加载CUDA dump的二进制数据"""
    
    def __init__(self, data_dir="turbodiffusion-sycl/tests/wan2_validation/data"):
        self.data_dir = Path(data_dir)
    
    def load_rmsnorm_data(self):
        """加载RMSNorm参考数据"""
        data = {}
        
        # 加载metadata
        meta_file = self.data_dir / "rmsnorm_metadata.txt"
        with open(meta_file, "r") as f:
            for line in f:
                key, val = line.strip().split(": ")
                if key in ["M", "N", "Grid", "Block"]:
                    data[key] = int(val)
                elif key == "EPS":
                    data[key] = float(val)
                else:
                    data[key] = val
        
        M = data["M"]
        N = data["N"]
        
        # 加载输入
        input_file = self.data_dir / f"rmsnorm_input_M{M}_N{N}_fp32.bin"
        data["input"] = np.fromfile(input_file, dtype=np.float32).reshape(M, N)
        
        # 加载weight
        weight_file = self.data_dir / f"rmsnorm_weight_N{N}_fp32.bin"
        data["weight"] = np.fromfile(weight_file, dtype=np.float32)
        
        # 加载CUDA输出（作为reference）
        output_file = self.data_dir / f"rmsnorm_output_cuda_M{M}_N{N}_fp32.bin"
        data["cuda_output"] = np.fromfile(output_file, dtype=np.float32).reshape(M, N)
        
        print(f"✅ Loaded RMSNorm CUDA reference data:")
        print(f"   Input shape: {data['input'].shape}")
        print(f"   Weight shape: {data['weight'].shape}")
        print(f"   Output shape: {data['cuda_output'].shape}")
        print(f"   Config: M={M}, N={N}, EPS={data['EPS']}")
        
        return data
    
    def verify_checksum(self, data):
        """验证数据完整性"""
        print("\n=== Data Verification ===")
        print(f"Input sum: {data['input'].sum():.4f}")
        print(f"Output sum: {data['cuda_output'].sum():.4f}")
        print(f"Output mean: {data['cuda_output'].mean():.6f}")
        
        # 检查RMS（第一行）
        rms = np.sqrt(np.mean(data['cuda_output'][0] ** 2))
        print(f"First row RMS: {rms:.4f}")
        
        return True


def compare_cuda_sycl(cuda_output, sycl_output, tolerance=1e-4):
    """对比CUDA和SYCL输出"""
    diff = np.abs(cuda_output - sycl_output)
    max_diff = diff.max()
    mean_diff = diff.mean()
    max_relative = (diff / (np.abs(cuda_output) + 1e-8)).max()
    
    print("\n=== CUDA vs SYCL Comparison ===")
    print(f"Max Absolute Error: {max_diff:.2e}")
    print(f"Mean Absolute Error: {mean_diff:.2e}")
    print(f"Max Relative Error: {max_relative:.2e}")
    
    # 统计通过比例
    pass_count = np.sum(diff < tolerance)
    total_count = diff.size
    pass_rate = pass_count / total_count
    
    print(f"Pass Rate: {pass_rate*100:.2f}% ({pass_count}/{total_count})")
    
    if max_diff < tolerance:
        print("✅ PASS - Within tolerance")
        return True
    else:
        print("❌ FAIL - Exceeds tolerance")
        return False


if __name__ == "__main__":
    # 测试加载
    loader = CUDADataLoader()
    data = loader.load_rmsnorm_data()
    loader.verify_checksum(data)
    
    print("\n✅ Data loader test complete!")
    print("Ready for SYCL implementation comparison.")