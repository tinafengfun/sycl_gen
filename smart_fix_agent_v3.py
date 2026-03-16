#!/usr/bin/env python3
"""
智能修复Agent v3.0 - 专门针对疑难问题
使用LLM处理失败内核的特定错误类型
"""

import asyncio
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import aiohttp

class SmartFixAgent:
    """智能修复Agent - 针对特定错误类型使用LLM修复"""
    
    def __init__(self):
        self.api_key = "sk-43Y6B7acOvFuOPelLuLw8Q"
        self.base_url = "http://10.112.110.111/v1"
        self.model = "minimax-m2.5"
        self.fix_history = []
        
    async def call_llm(self, prompt: str, system_prompt: str = "") -> str:
        """调用LLM"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        if not system_prompt:
            system_prompt = "你是CUDA到SYCL转换专家，专门修复编译错误。"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 4096
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                    else:
                        print(f"⚠️  LLM调用失败: HTTP {response.status}")
                        return ""
        except Exception as e:
            print(f"⚠️  LLM调用异常: {e}")
            return ""
    
    def detect_error_type(self, error_msg: str) -> str:
        """智能识别错误类型"""
        error_patterns = {
            'item_undefined': r"use of undeclared identifier 'item'",
            'uint2_undefined': r"unknown type name 'uint2'|uint2",
            'uint3_undefined': r"unknown type name 'uint3'|uint3",
            'uint4_undefined': r"unknown type name 'uint4'|uint4",
            'missing_inc': r"file not found.*\.inc",
            'missing_header': r"file not found",
            'thread_idx': r"threadIdx|blockIdx",
            'cuda_syntax': r"__global__|__device__|__shared__",
        }
        
        for error_type, pattern in error_patterns.items():
            if re.search(pattern, error_msg, re.IGNORECASE):
                return error_type
        
        return 'unknown'
    
    async def fix_item_undefined(self, code: str, error_msg: str) -> str:
        """修复item变量未定义错误 - 使用LLM"""
        print("  🔧 使用LLM修复: item变量未定义...")
        
        prompt = f"""请修复以下SYCL代码中的"item变量未定义"错误。

错误信息：
{error_msg}

当前代码：
```cpp
{code}
```

修复要求：
1. 在SYCL lambda中添加正确的item参数定义
2. 对于多维索引，使用sycl::item<2>或sycl::item<3>
3. 确保所有item.get_group(), item.get_local_id()等调用都正确
4. 参考以下模式：
   
   // 一维
   parallel_for(range<1>(N), [=](item<1> item) {{
       int idx = item.get_id(0);
   }});
   
   // 二维
   parallel_for(range<2>(M, N), [=](item<2> item) {{
       int row = item.get_id(0);
       int col = item.get_id(1);
   }});

请输出完整的修复后代码："""

        response = await self.call_llm(prompt)
        
        # 提取代码
        code_match = re.search(r'```(?:cpp)?\s*\n([\s\S]*?)\n```', response)
        if code_match:
            return code_match.group(1)
        
        return response
    
    async def fix_vector_types(self, code: str, error_msg: str) -> str:
        """
        处理CUDA向量类型未定义错误
        策略: 直接报错，不进行类型转换
        原因: 数据类型不对称可能导致精度丢失或逻辑错误
        """
        print("  ❌ 检测到CUDA向量类型错误，放弃修复")
        print("     原因: 数据类型不对称，转换可能导致精度丢失")
        print("     建议: 手动检查并修复类型映射")
        
        # 记录错误但不进行转换
        self.fix_history.append({
            'error_type': 'vector_type_mismatch',
            'error_msg': error_msg[:200],
            'action': 'skipped',
            'reason': 'Data type asymmetry - conversion may cause precision loss'
        })
        
        # 返回原代码，标记为失败
        return code
    
    async def fix_missing_inc(self, code: str, inc_file: str) -> str:
        """修复.inc文件缺失错误 - 使用LLM生成替代代码"""
        print(f"  🔧 使用LLM修复: 缺失的{inc_file}...")
        
        # 查找.inc文件
        inc_path = None
        for root in ['kernel_dataset/cuda', 'kernel_dataset/sycl', '.']:
            potential_path = Path(root) / inc_file
            if potential_path.exists():
                inc_path = potential_path
                break
        
        if inc_path:
            # 读取.inc文件内容
            with open(inc_path, 'r') as f:
                inc_content = f.read()
            
            prompt = f"""请将以下.inc文件内容内联到主代码中。

.inc文件内容 ({inc_file}):
```cpp
{inc_content}
```

当前主代码：
```cpp
{code}
```

修复要求：
1. 删除 #include "{inc_file}" 行
2. 将.inc文件的内容插入到合适的位置
3. 确保所有宏定义和辅助函数都被包含
4. 处理可能的命名空间问题

请输出完整的内联后代码："""
        else:
            # 如果找不到.inc文件，尝试让LLM推断内容
            prompt = f"""请修复以下代码中的缺失include问题。

当前代码：
```cpp
{code}
```

问题：找不到 "{inc_file}"

修复要求：
1. 如果这是不必要的include，请删除它
2. 如果是必需的，请根据上下文推断需要的内容并实现
3. 确保代码仍然可以编译

请输出完整的修复后代码："""
        
        response = await self.call_llm(prompt)
        
        code_match = re.search(r'```(?:cpp)?\s*\n([\s\S]*?)\n```', response)
        if code_match:
            return code_match.group(1)
        
        return response
    
    async def intelligent_fix(self, kernel_id: str, sycl_file: str) -> Tuple[bool, str]:
        """智能修复入口 - 自动识别并修复错误"""
        print(f"\n🔄 智能修复内核: {kernel_id}")
        print("="*70)
        
        # 读取当前代码
        with open(sycl_file, 'r') as f:
            code = f.read()
        
        # 测试编译获取错误信息
        success, error_msg = self.test_compile(code, kernel_id)
        
        if success:
            print("  ✅ 代码已经编译通过，无需修复")
            return True, code
        
        print(f"  ❌ 发现编译错误，开始智能修复...")
        print(f"  📋 错误信息前100字符: {error_msg[:100]}...")
        
        # 识别错误类型
        error_type = self.detect_error_type(error_msg)
        print(f"  🔍 识别错误类型: {error_type}")
        
        # 根据错误类型选择修复策略
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            print(f"\n  🔧 修复尝试 {attempt}/{max_attempts}...")
            
            if error_type == 'item_undefined':
                fixed_code = await self.fix_item_undefined(code, error_msg)
            elif error_type in ['uint2_undefined', 'uint3_undefined', 'uint4_undefined']:
                # 数据类型不对称，直接报错不转换
                print(f"  ❌ 错误类型 '{error_type}': 数据类型不对称")
                print("     策略: 放弃转换，需要手动修复")
                return False, code
            elif error_type == 'missing_inc':
                # 提取.inc文件名
                inc_match = re.search(r'"([^"]+\.inc)"', error_msg)
                if inc_match:
                    inc_file = inc_match.group(1)
                    fixed_code = await self.fix_missing_inc(code, inc_file)
                else:
                    fixed_code = code
            else:
                # 通用修复
                fixed_code = await self.generic_fix(code, error_msg)
            
            # 测试修复后的代码
            success, new_error = self.test_compile(fixed_code, kernel_id)
            
            if success:
                print(f"  ✅ 修复成功 (尝试 {attempt})!")
                # 保存修复后的代码
                with open(sycl_file, 'w') as f:
                    f.write(fixed_code)
                return True, fixed_code
            else:
                print(f"  ⚠️  修复未成功，新错误: {new_error[:80]}...")
                error_msg = new_error
                code = fixed_code
        
        print(f"  ❌ 达到最大修复次数，修复失败")
        return False, code
    
    async def generic_fix(self, code: str, error_msg: str) -> str:
        """通用修复 - 使用LLM分析并修复"""
        print("  🔧 使用LLM进行通用修复...")
        
        prompt = f"""请修复以下SYCL代码中的编译错误。

编译错误信息：
{error_msg}

当前代码：
```cpp
{code}
```

修复要求：
1. 分析编译错误原因
2. 修复所有编译错误
3. 保持代码逻辑不变
4. 输出完整的修复后代码
5. 添加注释说明主要修复内容

请输出修复后的完整代码："""

        response = await self.call_llm(prompt)
        
        code_match = re.search(r'```(?:cpp)?\s*\n([\s\S]*?)\n```', response)
        if code_match:
            return code_match.group(1)
        
        return response
    
    def test_compile(self, code: str, kernel_id: str) -> Tuple[bool, str]:
        """测试编译"""
        try:
            # 保存到临时文件
            temp_file = f"/tmp/{kernel_id}_fix_test.cpp"
            with open(temp_file, 'w') as f:
                f.write(code)
            
            # 复制到docker并编译
            subprocess.run(
                ['docker', 'cp', temp_file, 'lsv-container:/workspace/test.cpp'],
                capture_output=True, timeout=10, check=True
            )
            
            result = subprocess.run(
                ['docker', 'exec', 'lsv-container', 'bash', '-c',
                 'cd /workspace && icpx -fsycl -c test.cpp -o test.o 2>&1'],
                capture_output=True, text=True, timeout=60
            )
            
            return result.returncode == 0, result.stderr
            
        except Exception as e:
            return False, str(e)
    
    async def batch_fix(self, kernel_ids: List[str]):
        """批量修复多个内核"""
        print("="*70)
        print("🚀 智能修复Agent v3.0 - 批量修复")
        print("="*70)
        print(f"📊 修复内核数: {len(kernel_ids)}")
        print(f"🤖 使用模型: {self.model}")
        print("="*70)
        print()
        
        results = []
        for kernel_id in kernel_ids:
            sycl_file = f"kernel_dataset/sycl/{kernel_id}_kernel.dp.cpp"
            
            if not Path(sycl_file).exists():
                print(f"⚠️  跳过 {kernel_id}: 文件不存在")
                continue
            
            success, code = await self.intelligent_fix(kernel_id, sycl_file)
            results.append({
                'kernel_id': kernel_id,
                'success': success,
                'file': sycl_file
            })
        
        # 打印汇总
        print("\n" + "="*70)
        print("📊 修复完成汇总")
        print("="*70)
        
        total = len(results)
        success_count = sum(1 for r in results if r['success'])
        
        print(f"  总内核数: {total}")
        print(f"  ✅ 修复成功: {success_count} ({success_count/total*100:.1f}%)")
        print(f"  ❌ 修复失败: {total - success_count} ({(total-success_count)/total*100:.1f}%)")
        
        print("\n  详细结果:")
        for r in results:
            status = "✅" if r['success'] else "❌"
            print(f"    {status} {r['kernel_id']}")
        
        print("="*70)
        
        return results


async def main():
    """主函数 - 修复失败的3个内核"""
    
    # 需要修复的内核
    failed_kernels = [
        'add_bias_batched',
        'global_scale',
        # 'add_vectors'  # 已有可用版本，跳过
    ]
    
    agent = SmartFixAgent()
    results = await agent.batch_fix(failed_kernels)
    
    # 保存结果
    output_file = 'results/smart_fix_results.json'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 详细结果已保存: {output_file}")
    print("\n✅ 智能修复完成!")


if __name__ == '__main__':
    asyncio.run(main())
