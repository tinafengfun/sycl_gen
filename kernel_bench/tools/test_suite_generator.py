#!/usr/bin/env python3
"""
Complete Test Suite Generator
完整测试套件生成器

生成18个全面的测试配置，覆盖所有场景
"""

from typing import List, Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class TestConfiguration:
    """测试配置"""
    test_id: str
    name: str
    dtype: str
    N: int
    C: int
    H: int
    W: int
    data_gen: str
    data_strategy: str
    min_val: float
    max_val: float
    seed: int
    template_types: List[str]
    tolerance_abs: float
    tolerance_rel: float
    description: str


class TestSuiteGenerator:
    """测试套件生成器"""
    
    def __init__(self, platform_caps: Dict):
        """
        Args:
            platform_caps: 平台能力信息，来自PlatformDetector
        """
        self.platform_caps = platform_caps
        self.base_seed = 42
        
    def generate_full_suite(self) -> List[TestConfiguration]:
        """生成完整测试套件（18个测试）"""
        configs = []
        
        # 根据平台能力决定测试哪些数据类型
        dtypes_to_test = ["float32"]  # 总是测试
        
        if self.platform_caps.get("sycl", {}).get("bfloat16", False) and \
           self.platform_caps.get("cuda", {}).get("bfloat16", False):
            dtypes_to_test.append("bfloat16")
        
        if self.platform_caps.get("sycl", {}).get("float16", False):
            dtypes_to_test.append("float16")
        
        # 为每种数据类型生成测试
        for dtype in dtypes_to_test:
            configs.extend(self._generate_dtype_tests(dtype))
        
        # 维度变化测试（只用float32，避免重复）
        configs.extend(self._generate_dimension_tests())
        
        # 极端值测试（只用float32）
        configs.extend(self._generate_extreme_value_tests())
        
        return configs
    
    def _generate_dtype_tests(self, dtype: str) -> List[TestConfiguration]:
        """为特定数据类型生成测试"""
        configs = []
        
        # 容差配置
        tolerances = {
            "float32": (1e-5, 1e-4),
            "bfloat16": (1e-3, 1e-2),
            "float16": (1e-3, 1e-2)
        }
        
        abs_tol, rel_tol = tolerances.get(dtype, tolerances["float32"])
        
        # 1. 小尺寸随机测试
        configs.append(TestConfiguration(
            test_id=f"{dtype[:3]}_small_random",
            name=f"{dtype.upper()} Small Random",
            dtype=dtype,
            N=1, C=64, H=8, W=8,
            data_gen="random_uniform",
            data_strategy="random",
            min_val=-1.0,
            max_val=1.0,
            seed=self.base_seed,
            template_types=["float" if dtype == "float32" else dtype],
            tolerance_abs=abs_tol,
            tolerance_rel=rel_tol,
            description=f"Small tensor with random values in [-1, 1]"
        ))
        
        # 2. 边界值测试
        configs.append(TestConfiguration(
            test_id=f"{dtype[:3]}_boundary",
            name=f"{dtype.upper()} Boundary Values",
            dtype=dtype,
            N=1, C=64, H=8, W=8,
            data_gen="boundary_values",
            data_strategy="boundary",
            min_val=0.0,
            max_val=1.0,
            seed=self.base_seed,
            template_types=["float" if dtype == "float32" else dtype],
            tolerance_abs=abs_tol,
            tolerance_rel=rel_tol,
            description=f"Boundary values: 0, 1, -1, min, max, epsilon"
        ))
        
        # 3. 大尺寸随机测试（放大容差）
        large_abs_tol = abs_tol * 10
        large_rel_tol = rel_tol * 10
        
        configs.append(TestConfiguration(
            test_id=f"{dtype[:3]}_large",
            name=f"{dtype.upper()} Large Random",
            dtype=dtype,
            N=16, C=256, H=32, W=32,
            data_gen="random_uniform",
            data_strategy="random",
            min_val=-100.0,
            max_val=100.0,
            seed=self.base_seed,
            template_types=["float" if dtype == "float32" else dtype],
            tolerance_abs=large_abs_tol,
            tolerance_rel=large_rel_tol,
            description=f"Large tensor with values in [-100, 100]"
        ))
        
        # 4. 特殊值测试（bf16/fp16特殊处理）
        if dtype == "float32":
            configs.append(TestConfiguration(
                test_id=f"{dtype[:3]}_special",
                name=f"{dtype.upper()} Special Values",
                dtype=dtype,
                N=1, C=64, H=8, W=8,
                data_gen="special_values",
                data_strategy="special",
                min_val=0.0,
                max_val=1.0,
                seed=self.base_seed,
                template_types=["float" if dtype == "float32" else dtype],
                tolerance_abs=abs_tol,
                tolerance_rel=rel_tol,
                description=f"Special values: inf, -inf, nan"
            ))
        
        # 5. bf16精度损失测试
        if dtype == "bfloat16":
            configs.append(TestConfiguration(
                test_id="bf16_precision_loss",
                name="BFloat16 Precision Loss",
                dtype=dtype,
                N=1, C=64, H=8, W=8,
                data_gen="precision_test",
                data_strategy="small_variations",
                min_val=-0.001,
                max_val=0.001,
                seed=self.base_seed,
                template_types=["float"],
                tolerance_abs=1e-2,
                tolerance_rel=5e-2,
                description="Test bf16 precision loss with small variations"
            ))
        
        return configs
    
    def _generate_dimension_tests(self) -> List[TestConfiguration]:
        """生成维度变化测试（仅float32）"""
        configs = []
        
        # 方形tensor
        configs.append(TestConfiguration(
            test_id="dim_square",
            name="Dimension Square",
            dtype="float32",
            N=4, C=128, H=16, W=16,
            data_gen="random_uniform",
            data_strategy="random",
            min_val=-1.0,
            max_val=1.0,
            seed=self.base_seed,
            template_types=["float"],
            tolerance_abs=1e-5,
            tolerance_rel=1e-4,
            description="Square tensor (H=W)"
        ))
        
        # 矩形tensor (H≠W)
        configs.append(TestConfiguration(
            test_id="dim_rectangular",
            name="Dimension Rectangular",
            dtype="float32",
            N=2, C=128, H=8, W=16,
            data_gen="random_uniform",
            data_strategy="random",
            min_val=-1.0,
            max_val=1.0,
            seed=self.base_seed,
            template_types=["float"],
            tolerance_abs=1e-5,
            tolerance_rel=1e-4,
            description="Rectangular tensor (H≠W)"
        ))
        
        # 单维度 (H=1)
        configs.append(TestConfiguration(
            test_id="dim_line",
            name="Dimension Line",
            dtype="float32",
            N=1, C=64, H=1, W=64,
            data_gen="random_uniform",
            data_strategy="random",
            min_val=-1.0,
            max_val=1.0,
            seed=self.base_seed,
            template_types=["float"],
            tolerance_abs=1e-5,
            tolerance_rel=1e-4,
            description="Line tensor (H=1)"
        ))
        
        # 非对齐尺寸
        configs.append(TestConfiguration(
            test_id="dim_unaligned",
            name="Dimension Unaligned",
            dtype="float32",
            N=3, C=100, H=17, W=31,
            data_gen="random_uniform",
            data_strategy="random",
            min_val=-1.0,
            max_val=1.0,
            seed=self.base_seed,
            template_types=["float"],
            tolerance_abs=1e-5,
            tolerance_rel=1e-4,
            description="Non-aligned dimensions (not power of 2)"
        ))
        
        return configs
    
    def _generate_extreme_value_tests(self) -> List[TestConfiguration]:
        """生成极端值测试（仅float32）"""
        configs = []
        
        # 零值测试
        configs.append(TestConfiguration(
            test_id="extreme_zeros",
            name="Extreme Zeros",
            dtype="float32",
            N=1, C=64, H=8, W=8,
            data_gen="zeros",
            data_strategy="zeros",
            min_val=0.0,
            max_val=0.0,
            seed=self.base_seed,
            template_types=["float"],
            tolerance_abs=1e-5,
            tolerance_rel=1e-4,
            description="All zeros input"
        ))
        
        # 数学边界测试
        configs.append(TestConfiguration(
            test_id="extreme_math",
            name="Extreme Math Boundaries",
            dtype="float32",
            N=1, C=64, H=8, W=8,
            data_gen="math_boundary",
            data_strategy="math",
            min_val=0.0,
            max_val=1.0,
            seed=self.base_seed,
            template_types=["float"],
            tolerance_abs=1e-3,
            tolerance_rel=1e-2,
            description="Math boundaries: exp(88), large numbers, etc."
        ))
        
        return configs
    
    def get_test_data_generator(self, config: TestConfiguration):
        """获取测试数据生成器"""
        np.random.seed(config.seed)
        
        size = config.N * config.C * config.H * config.W
        
        if config.data_strategy == "random":
            return np.random.uniform(config.min_val, config.max_val, size).astype(np.float32)
        
        elif config.data_strategy == "boundary":
            # 边界值：0, 1, -1, min, max, epsilon
            values = [
                0.0, -0.0, 1.0, -1.0,
                np.finfo(np.float32).max,
                np.finfo(np.float32).min,
                np.finfo(np.float32).tiny,
                np.finfo(np.float32).eps
            ]
            data = np.tile(values, (size // len(values)) + 1)[:size]
            return data.astype(np.float32)
        
        elif config.data_strategy == "special":
            # 特殊值：inf, -inf, nan
            values = [np.inf, -np.inf, np.nan, 1.0, -1.0, 0.0]
            data = np.tile(values, (size // len(values)) + 1)[:size]
            return data.astype(np.float32)
        
        elif config.data_strategy == "zeros":
            return np.zeros(size, dtype=np.float32)
        
        elif config.data_strategy == "math":
            # 数学边界值
            values = [88.0, -88.0, 1e10, 1e-10, 1e38, 1e-38]
            data = np.tile(values, (size // len(values)) + 1)[:size]
            return data.astype(np.float32)
        
        elif config.data_strategy == "small_variations":
            # 小变化（用于测试精度损失）
            base = np.random.uniform(-0.001, 0.001, size)
            noise = np.random.uniform(-1e-7, 1e-7, size)
            return (base + noise).astype(np.float32)
        
        else:
            # 默认随机
            return np.random.uniform(-1.0, 1.0, size).astype(np.float32)


def generate_test_suite(platform_caps: Dict) -> List[Dict]:
    """便捷函数：生成测试套件"""
    generator = TestSuiteGenerator(platform_caps)
    configs = generator.generate_full_suite()
    
    # 转换为字典列表
    return [
        {
            "test_id": c.test_id,
            "name": c.name,
            "dtype": c.dtype,
            "N": c.N,
            "C": c.C,
            "H": c.H,
            "W": c.W,
            "data_gen": c.data_gen,
            "data_strategy": c.data_strategy,
            "min_val": c.min_val,
            "max_val": c.max_val,
            "seed": c.seed,
            "template_types": c.template_types,
            "tolerance": {"abs": c.tolerance_abs, "rel": c.tolerance_rel},
            "description": c.description
        }
        for c in configs
    ]


# 测试
if __name__ == "__main__":
    # 模拟平台能力
    platform_caps = {
        "sycl": {"float32": True, "float16": True, "bfloat16": False},
        "cuda": {"float32": True, "float16": True, "bfloat16": True}
    }
    
    suite = generate_test_suite(platform_caps)
    
    print("="*70)
    print("📋 Generated Test Suite")
    print("="*70)
    print(f"\nTotal tests: {len(suite)}\n")
    
    for i, test in enumerate(suite, 1):
        print(f"{i:2d}. {test['test_id']:25} | {test['dtype']:10} | {test['name']}")
        print(f"    Size: {test['N']}x{test['C']}x{test['H']}x{test['W']}")
        print(f"    Strategy: {test['data_strategy']}")
        print(f"    Tolerance: abs={test['tolerance']['abs']}, rel={test['tolerance']['rel']}")
        print()
