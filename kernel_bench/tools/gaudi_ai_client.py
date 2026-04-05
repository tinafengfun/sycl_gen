#!/usr/bin/env python3
"""
Gaudi AI Model Client
Gaudi AI模型客户端 - 用于调用DeepSeek等模型生成代码
"""

import asyncio
import aiohttp
import json
from typing import Optional, Dict, Any


class GaudiAIClient:
    """Gaudi AI API Client"""
    
    def __init__(self, api_key: str, base_url: str = "http://10.112.110.111/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_code(
        self,
        prompt: str,
        model: str = "Qwen3-Coder-30B-A3B-Instruct",
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 120
    ) -> str:
        """
        Generate code using Gaudi AI model
        
        Args:
            prompt: User prompt
            model: Model name (default: Qwen3-Coder-30B-A3B-Instruct)
            system_prompt: System prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            
        Returns:
            Generated code string
            
        Raises:
            Exception: If API call fails
        """
        url = f"{self.base_url}/chat/completions"
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Build request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
                    
                    data = await response.json()
                    
                    # Extract generated code
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0].get("message", {}).get("content", "")
                        return content
                    else:
                        raise Exception("No content generated")
                        
            except asyncio.TimeoutError:
                raise Exception(f"Request timeout after {timeout}s")
            except Exception as e:
                raise Exception(f"API call failed: {str(e)}")
    
    async def test_connection(self) -> bool:
        """Test API connection"""
        try:
            # Simple test request
            result = await self.generate_code(
                prompt="Say 'OK'",
                max_tokens=10,
                timeout=30
            )
            return "OK" in result or len(result) > 0
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
