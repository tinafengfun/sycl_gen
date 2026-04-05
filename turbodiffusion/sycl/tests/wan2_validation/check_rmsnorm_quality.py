#!/usr/bin/env python3
"""
RMSNorm数据质量详细检查
"""

import numpy as np
import sys
from pathlib import Path

# 直接加载数据，不通过模块
sys.path.insert(0, str(Path(__file__).parent))
from load_cuda_reference import CUDADataLoader

def detailed_quality_check():
    """详细数据质量检查"""
    print("=" * 60)
    print("RMSNorm CUDA Dump 数据质量详细检查")
    print("=" * 60)
    
    loader = CUDADataLoader()
    data = loader.load_rmsnorm_data()
    
    # 1. 基本统计信息
    print("\n📊 基本统计信息")
    print("-" * 40)
    print(f"输入数据:")
    print(f"  形状: {data['input'].shape}")
    print(f"  范围: [{data['input'].min():.6f}, {data['input'].max():.6f}]")
    print(f"  均值: {data['input'].mean():.6f}")
    print(f"  标准差: {data['input'].std():.6f}")
    
    print(f"\n输出数据:")
    print(f"  形状: {data['cuda_output'].shape}")
    print(f"  范围: [{data['cuda_output'].min():.6f}, {data['cuda_output'].max():.6f}]")
    print(f"  均值: {data['cuda_output'].mean():.6f}")
    print(f"  标准差: {data['cuda_output'].std():.6f}")
    
    # 2. RMSNorm特性验证
    print("\n✅ RMSNorm特性验证")
    print("-" * 40)
    
    # 验证RMS ≈ 1.0 (RMSNorm的特性)
    for i in range(min(5, data['M'])):
        row = data['cuda_output'][i]
        rms = np.sqrt(np.mean(row ** 2))
        print(f"  Row {i}: RMS = {rms:.6f}", end="")
        if abs(rms - 1.0) < 0.01:
            print(" ✓")
        else:
            print(f" ✗ (偏离 {abs(rms-1.0):.4f})")
    
    # 3. 数值范围检查
    print("\n🔍 数值范围检查")
    print("-" * 40)
    
    # 检查是否有异常值
    input_finite = np.isfinite(data['input']).all()
    output_finite = np.isfinite(data['cuda_output']).all()
    
    print(f"输入数据全有限: {'✓' if input_finite else '✗'}")
    print(f"输出数据全有限: {'✓' if output_finite else '✗'}")
    
    # 4. 分布检查
    print("\n📈 分布检查")
    print("-" * 40)
    
    # 输入分布
    input_percentiles = np.percentile(data['input'], [1, 5, 25, 50, 75, 95, 99])
    print("输入数据分位数:")
    print(f"  1%: {input_percentiles[0]:.6f}")
    print(f"  5%: {input_percentiles[1]:.6f}")
    print(f"  25%: {input_percentiles[2]:.6f}")
    print(f"  50%: {input_percentiles[3]:.6f}")
    print(f"  75%: {input_percentiles[4]:.6f}")
    print(f"  95%: {input_percentiles[5]:.6f}")
    print(f"  99%: {input_percentiles[6]:.6f}")
    
    # 5. 对称性检查
    print("\n🔄 对称性检查")
    print("-" * 40)
    input_skew = np.mean((data['input'] - data['input'].mean()) ** 3) / (data['input'].std() ** 3)
    print(f"输入偏度: {input_skew:.6f}", end="")
    if abs(input_skew) < 0.1:
        print(" ✓ (近似对称)")
    else:
        print(f" (偏斜)")
    
    # 6. 最终质量评估
    print("\n📝 最终质量评估")
    print("-" * 40)
    
    checks = {
        "数据完整性": input_finite and output_finite,
        "RMS ≈ 1.0": all(abs(np.sqrt(np.mean(data['cuda_output'][i] ** 2)) - 1.0) < 0.1 
                      for i in range(min(5, data['M']))),
        "数值范围合理": data['input'].max() <= 1.0 and data['input'].min() >= -1.0,
        "无NaN/Inf": not (np.isnan(data['cuda_output']).any() or np.isinf(data['cuda_output']).any()),
    }
    
    all_pass = True
    for check, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check}: {status}")
        if not passed:
            all_pass = False
    
    print("\n" + "=" * 60)
    if all_pass:
        print("🎉 数据质量检查全部通过！")
        print("可以继续生成其他kernel的dump数据。")
    else:
        print("⚠️  数据质量检查发现问题，请检查。")
    print("=" * 60)
    
    return all_pass

if __name__ == "__main__":
    passed = detailed_quality_check()
    sys.exit(0 if passed else 1)