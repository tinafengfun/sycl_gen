#!/usr/bin/env python3
"""
验证LLM API连接
"""

import json
import asyncio
import aiohttp

async def test_llm_connection():
    """测试LLM连接"""
    print("🧪 测试LLM API连接...")
    print()
    
    # 配置
    api_key = "sk-43Y6B7acOvFuOPelLuLw8Q"
    base_url = "http://10.112.110.111/v1"
    model = "minimax-m2.5"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个CUDA到SYCL转换专家。"},
            {"role": "user", "content": "请将 __global__ void add(float* a, float* b) { int i = threadIdx.x; a[i] = a[i] + b[i]; } 转换为SYCL代码"}
        ],
        "temperature": 0.1,
        "max_tokens": 500
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            print(f"📤 发送请求到: {base_url}/chat/completions")
            print(f"🤖 使用模型: {model}")
            print()
            
            async with session.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                print(f"📥 响应状态: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    print()
                    print("✅ API调用成功!")
                    print(f"📝 模型: {data.get('model', 'unknown')}")
                    print()
                    print("🔄 转换结果:")
                    content = data['choices'][0]['message']['content']
                    print(content[:500] + "..." if len(content) > 500 else content)
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ 错误: {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False

if __name__ == '__main__':
    print("="*60)
    print("🔧 LLM API连接验证")
    print("="*60)
    print()
    
    success = asyncio.run(test_llm_connection())
    
    print()
    print("="*60)
    if success:
        print("✅ LLM服务连接正常!")
        print("   可以开始运行增强版Agent v2.0")
    else:
        print("❌ LLM服务连接失败")
        print("   请检查网络配置")
    print("="*60)
