"""Provider plumbing for the one thing the clipper asks a model to do.

This used to also generate Xiaohongshu copy, hashtags and personalised notes.
That half of the product is gone; what remains is a thin multi-provider client
that `core.semantic_scoring` uses to score candidate windows.
"""
import os
from typing import Dict, Optional

import aiohttp
from loguru import logger


class LLMService:
    """大语言模型服务类"""
    
    def __init__(self):
        def configured(name: str) -> bool:
            value = os.getenv(name, "").strip()
            return bool(value) and not value.lower().startswith(("your_", "replace-", "example"))

        # 支持多种API配置
        self.apis = {
            "openai": {
                "base_url": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                "enabled": configured("OPENAI_API_KEY")
            },
            "claude": {
                "base_url": os.getenv("CLAUDE_API_BASE", "https://api.anthropic.com/v1"),
                "api_key": os.getenv("CLAUDE_API_KEY", ""),
                "model": os.getenv("CLAUDE_MODEL", "claude-sonnet-5"),
                "enabled": configured("CLAUDE_API_KEY")
            },
            "gemini": {
                "base_url": os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"),
                "api_key": os.getenv("GEMINI_API_KEY", ""),
                "model": os.getenv("GEMINI_MODEL", "gemini-pro"),
                "enabled": configured("GEMINI_API_KEY")
            },
            "zhipu": {
                "base_url": os.getenv("ZHIPU_API_BASE", "https://open.bigmodel.cn/api/paas/v4"),
                "api_key": os.getenv("ZHIPU_API_KEY", ""),
                "model": os.getenv("ZHIPU_MODEL", "glm-4"),
                "enabled": configured("ZHIPU_API_KEY")
            },
            "qwen": {
                "base_url": os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/api/v1"),
                "api_key": os.getenv("QWEN_API_KEY", ""),
                "model": os.getenv("QWEN_MODEL", "qwen-turbo"),
                "enabled": configured("QWEN_API_KEY")
            }
        }
        
        # 选择可用的API
        self.current_api = self._select_available_api()
        
    def _select_available_api(self) -> Optional[str]:
        """选择可用的API"""
        for api_name, config in self.apis.items():
            if config["enabled"]:
                logger.info(f"选择LLM API: {api_name}")
                return api_name
        
        logger.warning("未配置 LLM API，语义评分将使用词典规则")
        return None
    
    
    
    
    
    def is_configured(self) -> bool:
        """True when at least one provider has a usable API key."""
        return self.current_api is not None

    @property
    def active_model(self) -> Optional[str]:
        if not self.current_api:
            return None
        return self.apis[self.current_api]["model"]

    async def complete(self, prompt: str) -> str:
        """Send one prompt to the configured provider and return the raw text."""
        if not self.current_api:
            raise RuntimeError("未配置任何 LLM API")
        return await self._call_llm_api(prompt)

    async def _call_llm_api(self, prompt: str) -> str:
        """调用LLM API"""
        api_config = self.apis[self.current_api]
        
        if self.current_api == "openai":
            return await self._call_openai_api(prompt, api_config)
        elif self.current_api == "claude":
            return await self._call_claude_api(prompt, api_config)
        elif self.current_api == "gemini":
            return await self._call_gemini_api(prompt, api_config)
        elif self.current_api == "zhipu":
            return await self._call_zhipu_api(prompt, api_config)
        elif self.current_api == "qwen":
            return await self._call_qwen_api(prompt, api_config)
        else:
            raise Exception(f"不支持的API类型: {self.current_api}")
    
    async def _call_openai_api(self, prompt: str, config: Dict) -> str:
        """调用OpenAI API"""
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": config["model"],
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.7
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API错误: {response.status} - {error_text}")
    
    async def _call_claude_api(self, prompt: str, config: Dict) -> str:
        """调用Claude API"""
        headers = {
            "x-api-key": config['api_key'],
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": config["model"],
            "max_tokens": 1500,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config['base_url']}/messages",
                headers=headers,
                json=data,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["content"][0]["text"]
                else:
                    error_text = await response.text()
                    raise Exception(f"Claude API错误: {response.status} - {error_text}")
    
    async def _call_gemini_api(self, prompt: str, config: Dict) -> str:
        """调用Gemini API"""
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1500,
                "topP": 0.8,
                "topK": 10
            }
        }
        
        url = f"{config['base_url']}/models/{config['model']}:generateContent?key={config['api_key']}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=data,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if "candidates" in result and len(result["candidates"]) > 0:
                        content = result["candidates"][0]["content"]["parts"][0]["text"]
                        return content
                    else:
                        raise Exception("Gemini API返回格式异常")
                else:
                    error_text = await response.text()
                    raise Exception(f"Gemini API错误: {response.status} - {error_text}")
    
    async def _call_qwen_api(self, prompt: str, config: Dict) -> str:
        """调用通义千问API"""
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": config["model"],
            "input": {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {
                "max_tokens": 1500,
                "temperature": 0.7
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config['base_url']}/services/aigc/text-generation/generation",
                headers=headers,
                json=data,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["output"]["text"]
                else:
                    error_text = await response.text()
                    raise Exception(f"通义千问API错误: {response.status} - {error_text}")

    async def _call_zhipu_api(self, prompt: str, config: Dict) -> str:
        """调用智谱AI API"""
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": config["model"],
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.7
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"智谱AI API错误: {response.status} - {error_text}")
    
    
    
    
    
    
    
    

# 全局LLM服务实例
_llm_service = None

def get_llm_service() -> LLMService:
    """获取LLM服务实例"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
