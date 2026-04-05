#!/usr/bin/env python3
"""
验证测试：确认 SYCL 是否真的被调用
"""

import sys
sys.path.insert(0, '/workspace/turbodiffusion-sycl')
sys.path.insert(0, '/workspace/turbodiffusion-sycl/bindings')
sys.path.insert(0, '/workspace/turbodiffusion-sycl/hooks')

import numpy as np
import torch
import torch.nn as nn

print("="*60)
print("验证测试：SYCL 是否真正被调用")
print("="*60)

# 导入 SYCL
import turbodiffusion_sycl as tds

# 创建简单的 LayerNorm 模块
class SimpleLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        return torch.nn.functional.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

# 创建模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = SimpleLayerNorm(1536)

model = SimpleModel()

# 测试输入
x = torch.randn(2, 64, 1536)

print(f"\n输入: {x.shape}")
print(f"输入均值: {x.mean().item():.6f}")

# 参考输出 (PyTorch)
with torch.no_grad():
    ref_output = model.norm(x)

print(f"\nPyTorch 输出: {ref_output.shape}")
print(f"PyTorch 输出均值: {ref_output.mean().item():.6f}")

# 手动调用 SYCL
print("\n--- 直接调用 SYCL ---")
x_np = x.numpy()
x_2d = x_np.reshape(-1, 1536)
m, n = x_2d.shape

output_2d = np.empty_like(x_2d)
gamma = np.ones(n, dtype=np.float32)
beta = np.zeros(n, dtype=np.float32)

print(f"调用 tds.layernorm({m}, {n})...")
tds.layernorm(x_2d, gamma, beta, output_2d, eps=1e-5, m=m, n=n)

sycl_output_np = output_2d.reshape(2, 64, 1536)
sycl_output = torch.from_numpy(sycl_output_np)

print(f"SYCL 输出: {sycl_output.shape}")
print(f"SYCL 输出均值: {sycl_output.mean().item():.6f}")

# 对比
max_error = (ref_output - sycl_output).abs().max().item()
print(f"\n对比结果:")
print(f"最大误差: {max_error:.2e}")

if max_error > 1e-6:
    print("✅ SYCL 确实被调用了，且有微小误差（正常）")
else:
    print("⚠️  误差太小，检查是否真的调用了 SYCL")

# 验证：故意传入不同的 gamma 看结果是否变化
print("\n--- 验证测试：使用不同的 gamma ---")
gamma2 = np.ones(n, dtype=np.float32) * 2.0  # gamma = 2
output2_2d = np.empty_like(x_2d)
tds.layernorm(x_2d, gamma2, beta, output2_2d, eps=1e-5, m=m, n=n)

output2 = torch.from_numpy(output2_2d.reshape(2, 64, 1536))
max_error2 = (ref_output - output2).abs().max().item()

print(f"gamma=2 时的误差: {max_error2:.2e}")
if max_error2 > 0.1:  # 应该有很大差异
    print("✅ 验证通过：gamma 变化导致输出变化，SYCL 确实在执行计算")
else:
    print("❌ 验证失败：输出没有变化")

print("\n" + "="*60)
print("验证完成")
print("="*60)
