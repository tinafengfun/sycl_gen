#!/usr/bin/env python3
"""
增强版 CUDA→SYCL 转换 Agent (Enhanced Agent v2.0)
使用 DeepSeek-R1-G2-static-671B + Qwen3-Coder-30B
大幅提高LLM使用比重，实现智能预处理和多轮修复
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp

# 加载增强配置
CONFIG_FILE = Path(__file__).parent / 'model_config_enhanced.json'

class EnhancedLLMClient:
    """增强版LLM客户端 - 使用更强的模型"""
    
    def __init__(self):
        with open(CONFIG_FILE, 'r') as f:
            self.config = json.load(f)
        
        self.provider = self.config['provider']['gaudi_ai']
        self.base_url = self.provider['options']['baseURL']
        self.api_key = self.provider['options']['apiKey']
        self.settings = self.provider['settings']
        
        # 根据任务选择模型
        self.models = self.config['models']
    
    async def call_llm(self, prompt: str, task_type: str = "conversion") -> str:
        """调用LLM，根据任务类型选择模型"""
        model = self.models.get(task_type, self.models['conversion'])
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.get_system_prompt(task_type)},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.settings['temperature'],
            "max_tokens": self.settings['max_tokens'],
            "top_p": self.settings['top_p']
        }
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.settings['retry_attempts']):
                try:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.settings['timeout'])
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data['choices'][0]['message']['content']
                        else:
                            print(f"⚠️  LLM调用失败 (尝试 {attempt+1}): HTTP {response.status}")
                except Exception as e:
                    print(f"⚠️  LLM调用异常 (尝试 {attempt+1}): {e}")
                    if attempt < self.settings['retry_attempts'] - 1:
                        await asyncio.sleep(2 ** attempt)  # 指数退避
        
        return ""
    
    def get_system_prompt(self, task_type: str) -> str:
        """根据任务类型获取系统提示"""
        prompts = {
            "preprocess": """你是一个专业的CUDA代码分析专家。请分析CUDA内核代码，提取所有依赖关系、头文件、宏定义和模板参数。
提供详细的分析报告，包括：
1. 外部头文件依赖
2. 宏定义和常量
3. 模板参数和类型
4. CUDA特定API调用
5. 代码复杂度评估
请用结构化格式输出。""",
            
            "conversion": """你是一个专家级的CUDA到SYCL转换工程师。请将CUDA内核代码转换为等效的SYCL代码。
转换规则：
1. __global__ → 移除，使用SYCL lambda
2. __device__ → 移除
3. threadIdx.x → item.get_local_id(0)
4. blockIdx.x → item.get_group(0)
5. __shared__ → sycl::local_accessor
6. __expf → sycl::exp
7. __shfl_xor_sync → sycl::group_broadcast
8. 保持原有算法逻辑不变
9. 添加必要的SYCL头文件
10. 确保代码完整可编译
请输出完整、可直接编译的SYCL代码。""",
            
            "fix": """你是一个专业的编译错误修复专家。请分析编译错误信息，修复SYCL代码中的问题。
常见修复策略：
1. 缺失的头文件 → 添加正确的include
2. 未定义的标识符 → 检查拼写或添加定义
3. CUDA残留语法 → 转换为SYCL等效语法
4. 类型错误 → 添加正确的类型转换
5. 模板错误 → 修复模板参数
请输出修复后的完整代码。""",
            
            "test_generation": """你是一个测试代码生成专家。请为给定的内核生成完整的测试harness代码。
包括：
1. 输入数据生成
2. 内核调用包装
3. 输出结果验证
4. 性能计时
5. 错误处理
生成可直接编译运行的完整测试程序。"""
        }
        return prompts.get(task_type, prompts['conversion'])


class LLMPreProcessor:
    """LLM驱动的预处理器 - 高比重使用LLM"""
    
    def __init__(self, llm_client: EnhancedLLMClient):
        self.llm = llm_client
    
    async def analyze_dependencies(self, cuda_code: str) -> Dict:
        """使用LLM分析依赖关系"""
        prompt = f"""请详细分析以下CUDA内核代码的依赖关系：

```cuda
{cuda_code}
```

请提供以下信息（JSON格式）：
{{
    "headers": ["依赖的头文件列表"],
    "macros": ["使用的宏定义"],
    "cuda_apis": ["CUDA特定API调用"],
    "templates": ["模板定义和使用"],
    "constants": ["常量定义"],
    "complexity_score": 1-10,
    "risk_factors": ["潜在转换难点"]
}}"""
        
        response = await self.llm.call_llm(prompt, "preprocess")
        
        # 尝试解析JSON响应
        try:
            # 提取JSON部分
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # 如果解析失败，返回结构化分析
        return self.parse_analysis_text(response)
    
    def parse_analysis_text(self, text: str) -> Dict:
        """从文本解析分析结果"""
        analysis = {
            'headers': [],
            'macros': [],
            'cuda_apis': [],
            'templates': [],
            'constants': [],
            'complexity_score': 5,
            'risk_factors': []
        }
        
        # 提取头文件
        header_pattern = r'#include\s+[<"]([^>"]+)[>"]'
        analysis['headers'] = re.findall(header_pattern, text)
        
        # 提取宏定义
        macro_pattern = r'#define\s+(\w+)'
        analysis['macros'] = re.findall(macro_pattern, text)
        
        # 提取CUDA API
        cuda_apis = ['__expf', '__logf', '__shfl', 'threadIdx', 'blockIdx', '__shared__']
        for api in cuda_apis:
            if api in text:
                analysis['cuda_apis'].append(api)
        
        return analysis
    
    async def inline_headers(self, cuda_code: str, headers: List[str]) -> str:
        """使用LLM内联头文件"""
        prompt = f"""请将以下CUDA代码中的外部头文件依赖内联到代码中。
需要内联的头文件：{headers}

原始代码：
```cuda
{cuda_code}
```

要求：
1. 提取头文件中的关键定义（宏、常量、辅助函数）
2. 将这些定义直接嵌入到代码中
3. 移除原始的#include语句
4. 确保代码仍然完整可编译
5. 添加注释说明内联的内容

请输出完整的内联后代码："""
        
        response = await self.llm.call_llm(prompt, "preprocess")
        
        # 提取代码块
        code_match = re.search(r'```(?:cuda|cpp)?\s*\n([\s\S]*?)\n```', response)
        if code_match:
            return code_match.group(1)
        
        return response
    
    async def generate_conversion_plan(self, analysis: Dict) -> Dict:
        """
        智能生成转换计划
        基于数据分析选择最优策略
        """
        # 根据分析结果智能选择策略
        complexity = analysis.get('complexity_score', 5)
        templates = analysis.get('templates', [])
        cuda_apis = analysis.get('cuda_apis', [])
        
        # 策略选择逻辑 (基于数据分析):
        # - 'direct'策略成功率75%，优于'template_expansion'的67%
        # - 默认使用'direct'，除非检测到复杂模板
        if complexity > 7 or len(templates) > 2:
            # 复杂内核使用 template_expansion
            strategy = 'template_expansion'
            risk_level = 'high'
            estimated_time = '10-15min'
        elif complexity > 5 or len(templates) > 0:
            # 中等复杂度使用 template_expansion
            strategy = 'template_expansion'
            risk_level = 'medium'
            estimated_time = '5-10min'
        else:
            # 简单内核使用 direct (成功率更高)
            strategy = 'direct'
            risk_level = 'low'
            estimated_time = '3-5min'
        
        # 使用LLM细化计划
        prompt = f"""基于以下CUDA代码分析结果，生成详细的SYCL转换计划：

分析结果：
{json.dumps(analysis, indent=2)}

已选择的策略: {strategy}
复杂度评分: {complexity}/10
检测到的模板: {len(templates)}个

请提供转换计划（JSON格式）：
{{
    "strategy": "{strategy}",
    "estimated_time": "{estimated_time}",
    "risk_level": "{risk_level}",
    "steps": [
        "步骤1: ...",
        "步骤2: ..."
    ],
    "special_handling": ["需要特殊处理的部分"]
}}"""
        
        response = await self.llm.call_llm(prompt, "preprocess")
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                plan = json.loads(json_match.group())
                # 确保使用我们选择的策略
                plan['strategy'] = strategy
                return plan
        except:
            pass
        
        # 默认返回
        return {
            'strategy': strategy,
            'estimated_time': estimated_time,
            'risk_level': risk_level,
            'steps': ['直接转换' if strategy == 'direct' else '模板展开转换'],
            'special_handling': []
        }


class LLMConverter:
    """LLM驱动的转换器 - 主要使用LLM而非规则"""
    
    def __init__(self, llm_client: EnhancedLLMClient):
        self.llm = llm_client
    
    async def convert(self, cuda_code: str, context: Dict = None) -> str:
        """使用LLM进行转换"""
        context_str = json.dumps(context, indent=2) if context else "无"
        
        prompt = f"""请将以下CUDA内核代码转换为SYCL代码。

上下文信息：
{context_str}

CUDA代码：
```cuda
{cuda_code}
```

转换要求：
1. 将CUDA语法完全转换为SYCL语法
2. 保留所有算法逻辑
3. 添加必要的SYCL头文件
4. 确保代码可直接编译
5. 添加注释说明关键转换点
6. 处理所有CUDA特定API

请输出完整的SYCL代码："""
        
        response = await self.llm.call_llm(prompt, "conversion")
        
        # 提取代码
        code_match = re.search(r'```(?:cpp|sycl)?\s*\n([\s\S]*?)\n```', response)
        if code_match:
            return code_match.group(1)
        
        return response


class LLMFixer:
    """LLM驱动的修复器 - 5轮渐进式修复"""
    
    def __init__(self, llm_client: EnhancedLLMClient):
        self.llm = llm_client
        self.fix_history = []
    
    async def fix(self, sycl_code: str, error_msg: str, attempt: int = 1) -> Tuple[str, bool]:
        """使用LLM修复编译错误"""
        if attempt > 5:
            print("❌ 达到最大修复次数")
            return sycl_code, False
        
        print(f"  🔧 LLM修复尝试 {attempt}/5...")
        
        prompt = f"""请修复以下SYCL代码中的编译错误。

错误信息：
{error_msg}

当前代码：
```cpp
{sycl_code}
```

修复要求：
1. 分析错误原因
2. 修复所有编译错误
3. 保持代码逻辑不变
4. 输出完整修复后的代码
5. 添加注释说明修复内容

请输出修复后的完整代码："""
        
        response = await self.llm.call_llm(prompt, "fix")
        
        # 提取修复后的代码
        code_match = re.search(r'```(?:cpp|sycl)?\s*\n([\s\S]*?)\n```', response)
        if code_match:
            fixed_code = code_match.group(1)
        else:
            fixed_code = response
        
        # 记录修复历史
        self.fix_history.append({
            'attempt': attempt,
            'error': error_msg[:200],
            'changes': 'LLM自动修复'
        })
        
        return fixed_code, True


class EnhancedConversionAgent:
    """增强版转换Agent - LLM高比重使用"""
    
    def __init__(self):
        self.llm = EnhancedLLMClient()
        self.preprocessor = LLMPreProcessor(self.llm)
        self.converter = LLMConverter(self.llm)
        self.fixer = LLMFixer(self.llm)
        
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'fix_attempts': 0
        }
    
    async def convert_kernel(self, kernel_id: str, cuda_file: str) -> Dict:
        """转换单个内核 - LLM全流程驱动"""
        print(f"\n🔄 转换内核: {kernel_id}")
        print("=" * 70)
        
        result = {
            'kernel_id': kernel_id,
            'success': False,
            'stages': {}
        }
        
        # Stage 1: 读取CUDA代码
        with open(cuda_file, 'r') as f:
            cuda_code = f.read()
        
        # Stage 2: LLM预处理分析
        print("🧠 Stage 1: LLM预处理分析...")
        analysis = await self.preprocessor.analyze_dependencies(cuda_code)
        result['stages']['analysis'] = analysis
        print(f"  ✓ 复杂度评分: {analysis.get('complexity_score', 'N/A')}/10")
        print(f"  ✓ 发现依赖: {len(analysis.get('headers', []))}个头文件")
        
        # Stage 3: LLM内联头文件
        if analysis.get('headers'):
            print("📦 Stage 2: LLM内联头文件...")
            cuda_code = await self.preprocessor.inline_headers(
                cuda_code, analysis['headers']
            )
            result['stages']['inlined'] = True
            print("  ✓ 头文件内联完成")
        
        # Stage 4: LLM生成转换计划
        print("📋 Stage 3: LLM生成转换计划...")
        plan = await self.preprocessor.generate_conversion_plan(analysis)
        result['stages']['plan'] = plan
        print(f"  ✓ 策略: {plan['strategy']}")
        print(f"  ✓ 预计时间: {plan['estimated_time']}")
        
        # Stage 5: LLM执行转换
        print("🔄 Stage 4: LLM执行转换...")
        context = {
            'analysis': analysis,
            'plan': plan,
            'kernel_id': kernel_id
        }
        sycl_code = await self.converter.convert(cuda_code, context)
        result['stages']['converted'] = True
        print("  ✓ 转换完成")
        
        # Stage 6: LLM多轮修复
        print("🔧 Stage 5: LLM多轮修复...")
        max_fix_attempts = 5
        for attempt in range(1, max_fix_attempts + 1):
            # 测试编译
            success, error = self.test_compile(sycl_code, kernel_id)
            
            if success:
                print(f"  ✓ 编译通过 (尝试 {attempt})")
                result['success'] = True
                result['stages']['fix_attempts'] = attempt - 1
                break
            else:
                print(f"  ⚠️ 编译失败 (尝试 {attempt}): {error[:100]}...")
                if attempt < max_fix_attempts:
                    sycl_code, can_continue = await self.fixer.fix(
                        sycl_code, error, attempt
                    )
                    if not can_continue:
                        break
                else:
                    print("  ❌ 达到最大修复次数")
                    result['stages']['final_error'] = error
        
        # 保存结果
        if result['success']:
            output_file = f"kernel_dataset/sycl/{kernel_id}_kernel.dp.cpp"
            with open(output_file, 'w') as f:
                f.write(sycl_code)
            print(f"  ✓ 已保存: {output_file}")
        
        return result
    
    def test_compile(self, sycl_code: str, kernel_id: str) -> Tuple[bool, str]:
        """测试编译"""
        try:
            # 保存到临时文件
            temp_file = f"/tmp/{kernel_id}_test.cpp"
            with open(temp_file, 'w') as f:
                f.write(sycl_code)
            
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
    
    async def run_batch_conversion(self, kernel_ids: List[str]):
        """批量转换"""
        print("=" * 80)
        print("🚀 增强版Agent批量转换")
        print(f"模型: DeepSeek-R1-G2-static-671B + Qwen3-Coder-30B")
        print("=" * 80)
        
        results = []
        for kernel_id in kernel_ids:
            cuda_file = f"kernel_dataset/cuda/{kernel_id}_kernel.cu"
            if not Path(cuda_file).exists():
                print(f"⚠️  跳过 {kernel_id}: CUDA文件不存在")
                continue
            
            result = await self.convert_kernel(kernel_id, cuda_file)
            results.append(result)
            
            self.stats['total'] += 1
            if result['success']:
                self.stats['success'] += 1
            else:
                self.stats['failed'] += 1
            self.stats['fix_attempts'] += result['stages'].get('fix_attempts', 0)
        
        # 打印汇总
        print("\n" + "=" * 80)
        print("📊 转换完成汇总")
        print("=" * 80)
        print(f"总内核数: {self.stats['total']}")
        print(f"成功: {self.stats['success']} ({self.stats['success']/self.stats['total']*100:.1f}%)")
        print(f"失败: {self.stats['failed']}")
        print(f"平均修复次数: {self.stats['fix_attempts']/self.stats['total']:.1f}")
        print("=" * 80)
        
        return results


# 主函数
async def main():
    """主函数"""
    # 21个编译失败的CUDA内核列表
    failed_kernels = [
        'add_bias_batched',
        'add_bias_nchw',
        'add_vectors_hnc_nhc',
        'add_vectors',
        'expand_planes_nhwc',
        'gen_offset_pointers',
        'global_avg_pool_nhwc_fp16',
        'global_scale_fp16_nhwc',
        'global_scale',
        'input_gating',
        'layer_norm',
        'nchw_to_nhwc',
        'output_input_transform_fp16_shmem',
        'preprocess_attention_body',
        'promotion_logits',
        'se_layer_nhwc',
        'softmax',
        'winograd_filter_transform',
        'winograd_output_relu_input',
        'winograd_output_se_relu_input',
        'winograd_output_transform'
    ]
    
    print(f"🎯 准备转换 {len(failed_kernels)} 个失败的内核")
    print(f"🤖 使用模型: minimax-m2.5")
    print(f"⏱️  预计耗时: 1-2小时")
    print("="*70)
    print()
    
    agent = EnhancedConversionAgent()
    results = await agent.run_batch_conversion(failed_kernels)
    
    # 保存详细结果
    with open('results/enhanced_conversion_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ 增强版Agent转换完成!")
    print("📁 详细结果: results/enhanced_conversion_results.json")


if __name__ == '__main__':
    asyncio.run(main())
