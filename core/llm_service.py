"""
LLM服务模块 - 支持多种大模型API调用
"""
import json
import os
import aiohttp
import random
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
from datetime import datetime

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
        
        logger.warning("未配置任何LLM API，将使用本地模拟")
        return None
    
    async def generate_xiaohongshu_content(
        self, 
        theme: str, 
        photo_descriptions: List[str],
        highlights: str = "",
        feeling: str = "",
        style: str = "活泼",
        length: str = "medium",
        custom_requirements: str = ""
    ) -> Dict[str, Any]:
        """生成小红书文案"""
        
        if not self.current_api:
            return self._generate_fallback_content(theme, photo_descriptions, style, length)
        
        # 构建提示词
        prompt = self._build_xiaohongshu_prompt(
            theme, photo_descriptions, highlights, feeling, style, length, custom_requirements
        )
        
        try:
            response = await self._call_llm_api(prompt)
            return self._parse_xiaohongshu_response(response, theme)
            
        except Exception as e:
            logger.error(f"LLM API调用失败: {str(e)}")
            return self._generate_fallback_content(theme, photo_descriptions, style, length)
    
    async def generate_pro_content(
        self,
        topic: str,
        content_type: str = "complete",
        style: str = "professional",
        target_audience: str = "general"
    ) -> Dict[str, Any]:
        """生成Pro功能个性化文案"""
        
        if not self.current_api:
            return self._generate_fallback_pro_content(topic, content_type)
        
        prompt = self._build_pro_content_prompt(topic, content_type, style, target_audience)
        
        try:
            response = await self._call_llm_api(prompt)
            return self._parse_pro_content_response(response, content_type)
            
        except Exception as e:
            logger.error(f"Pro内容生成失败: {str(e)}")
            return self._generate_fallback_pro_content(topic, content_type)
    
    def _build_xiaohongshu_prompt(
        self, theme: str, photo_descriptions: List[str], 
        highlights: str, feeling: str, style: str, length: str, custom_requirements: str
    ) -> str:
        """构建小红书文案生成提示词"""
        
        length_map = {
            "short": "100字以内，简洁明了",
            "medium": "150-250字，详略得当", 
            "long": "300字以上，内容丰富"
        }
        
        style_map = {
            "活泼": "语言活泼可爱，多使用emoji表情，语气轻松愉快",
            "专业": "语言专业严谨，逻辑清晰，信息准确",
            "简约": "语言简洁清新，不拖泥带水，重点突出",
            "情感": "语言富有感染力，能够引起共鸣，情感丰富"
        }
        
        photo_context = ""
        if photo_descriptions:
            photo_context = f"根据以下照片内容：{', '.join(photo_descriptions[:5])}"
        
        prompt = f"""请为小红书平台生成一篇关于"{theme}"的优质内容。

要求：
1. 风格：{style_map.get(style, style)}
2. 长度：{length_map.get(length, "150-250字")}
3. {photo_context}

用户特别想强调：{highlights if highlights else "无特殊要求"}
用户的感受：{feeling if feeling else "无特殊感受"}
额外要求：{custom_requirements if custom_requirements else "无"}

请生成包含以下部分的内容：
1. 吸引人的标题（15-25字）
2. 正文内容（符合长度要求）
3. 相关话题标签（8-12个）
4. 互动引导语

请以JSON格式返回：
{{
    "title": "标题",
    "content": "正文内容",
    "hashtags": ["标签1", "标签2", "..."],
    "engagement": "互动引导语",
    "word_count": 实际字数
}}"""
        
        return prompt
    
    def _build_pro_content_prompt(
        self, topic: str, content_type: str, style: str, target_audience: str
    ) -> str:
        """构建Pro功能内容生成提示词"""
        
        type_instructions = {
            "title": "生成3-5个吸引人的标题选项",
            "description": "生成详细的描述性内容",
            "hashtags": "生成15-20个相关的话题标签",
            "complete": "生成完整的内容包括标题、正文、标签"
        }
        
        prompt = f"""请为主题"{topic}"生成{type_instructions.get(content_type, '内容')}。

目标受众：{target_audience}
内容风格：{style}
生成类型：{content_type}

请确保内容：
1. 原创性高，避免套模板
2. 符合目标受众的阅读习惯
3. 具有实用价值或娱乐性
4. 语言流畅自然

请以JSON格式返回结果。"""
        
        return prompt
    
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
    
    def _parse_xiaohongshu_response(self, response: str, theme: str) -> Dict[str, Any]:
        """解析小红书内容生成响应"""
        try:
            # 尝试解析JSON
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            result = json.loads(json_str)
            
            # 验证必要字段
            required_fields = ["title", "content", "hashtags"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"缺少必要字段: {field}")
            
            return {
                "status": "success",
                "data": {
                    "title": result.get("title", theme),
                    "content": result.get("content", ""),
                    "hashtags": result.get("hashtags", []),
                    "engagement": result.get("engagement", "你们觉得怎么样？评论区聊聊！"),
                    "word_count": result.get("word_count", len(result.get("content", ""))),
                    "generated_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"解析LLM响应失败: {str(e)}")
            # 返回原始文本作为内容
            return {
                "status": "partial_success",
                "data": {
                    "title": theme,
                    "content": response[:500] + "..." if len(response) > 500 else response,
                    "hashtags": [f"#{theme}", "#AI生成"],
                    "engagement": "AI生成内容，欢迎反馈！",
                    "word_count": len(response),
                    "generated_at": datetime.now().isoformat()
                }
            }
    
    def _parse_pro_content_response(self, response: str, content_type: str) -> Dict[str, Any]:
        """解析Pro内容生成响应"""
        try:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response.strip()
            
            result = json.loads(json_str)
            
            return {
                "status": "success",
                "type": content_type,
                "data": result,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"解析Pro内容响应失败: {str(e)}")
            return {
                "status": "partial_success",
                "type": content_type,
                "data": {"content": response},
                "generated_at": datetime.now().isoformat()
            }
    
    def _generate_fallback_content(
        self, theme: str, photo_descriptions: List[str], style: str, length: str
    ) -> Dict[str, Any]:
        """生成备用内容（当API不可用时）- 智能化本地生成"""
        
        # 分析主题关键词
        theme_keywords = self._extract_theme_keywords(theme)
        
        # 根据照片描述生成内容要点
        content_points = self._generate_content_points(photo_descriptions, theme_keywords)
        
        # 生成个性化内容
        title, content, engagement = self._generate_personalized_content(
            theme, content_points, style, length
        )
        
        # 生成智能标签
        hashtags = self._generate_smart_hashtags(theme, theme_keywords, content_points)
        
        return {
            "status": "fallback",
            "data": {
                "title": title,
                "content": content,
                "hashtags": hashtags,
                "engagement": engagement,
                "word_count": len(content.replace('\n', '').replace(' ', '')),
                "generated_at": datetime.now().isoformat(),
                "note": "使用智能本地生成器，质量已优化"
            }
        }
    
    def _extract_theme_keywords(self, theme: str) -> List[str]:
        """提取主题关键词"""
        keyword_map = {
            "亲子": ["family", "孩子", "宝宝", "亲情", "成长", "陪伴"],
            "旅游": ["travel", "风景", "体验", "探索", "发现", "回忆"],
            "美食": ["food", "味道", "享受", "品尝", "满足", "幸福"],
            "购物": ["shopping", "好物", "推荐", "种草", "心动", "满意"],
            "运动": ["fitness", "健康", "活力", "坚持", "挑战", "成就"],
            "学习": ["study", "知识", "成长", "进步", "收获", "感悟"],
            "工作": ["work", "职场", "效率", "专业", "团队", "成果"],
            "生活": ["life", "日常", "感受", "分享", "记录", "美好"]
        }
        
        keywords = []
        for key, values in keyword_map.items():
            if key in theme:
                keywords.extend(values[:3])  # 取前3个相关词
        
        # 如果没有匹配，使用通用关键词
        if not keywords:
            keywords = ["体验", "分享", "推荐"]
        
        return keywords
    
    def _generate_content_points(self, photo_descriptions: List[str], theme_keywords: List[str]) -> List[str]:
        """根据照片描述生成内容要点"""
        points = []
        
        for desc in photo_descriptions[:5]:  # 最多处理5张照片
            if "海边" in desc or "沙滩" in desc:
                points.append("海边的浪花声和孩子的笑声交织在一起")
            elif "乐园" in desc or "游乐" in desc:
                points.append("孩子在游乐设施上的开心模样")
            elif "餐厅" in desc or "晚餐" in desc or "美食" in desc:
                points.append("温馨的用餐时光，满满的幸福感")
            elif "风景" in desc or "景色" in desc:
                points.append("令人心旷神怡的美丽景色")
            elif "建筑" in desc or "景点" in desc:
                points.append("独特的建筑风格让人印象深刻")
            else:
                points.append(f"记录下{desc}的美好瞬间")
        
        return points
    
    def _generate_personalized_content(
        self, theme: str, content_points: List[str], style: str, length: str
    ) -> Tuple[str, str, str]:
        """生成个性化标题、内容和互动语"""
        
        # 标题生成
        title_templates = {
            "活泼": [
                f"✨{theme}✨ 超治愈的美好时光！",
                f"🎉{theme}记录 | 满满都是爱",
                f"💕{theme}日记 | 幸福感爆棚"
            ],
            "专业": [
                f"{theme} | 深度体验全记录",
                f"关于{theme}的完整攻略分享",
                f"{theme}体验报告：值得收藏"
            ],
            "简约": [
                f"{theme} | 简单记录",
                f"{theme}分享",
                f"关于{theme}"
            ],
            "情感": [
                f"❤️{theme} | 温暖的回忆",
                f"{theme} | 感动满满的时光",
                f"💝{theme} | 珍贵的美好"
            ]
        }
        
        title = random.choice(title_templates.get(style, title_templates["活泼"]))
        
        # 内容生成
        if style == "活泼":
            content_start = f"今天的{theme}真的太棒了！🌟\n\n"
            content_body = "\n".join([f"✨ {point}" for point in content_points[:3]])
            content_end = "\n\n这样的时光真的很珍贵，希望能一直记录下去~ 💕"
            
        elif style == "专业":
            content_start = f"关于{theme}的详细体验分享：\n\n"
            content_body = "\n".join([f"• {point}" for point in content_points[:4]])
            content_end = "\n\n总体来说是一次很不错的体验，推荐给有需要的朋友。"
            
        elif style == "简约":
            content_start = f"{theme}\n\n"
            content_body = "\n".join(content_points[:2])
            content_end = "\n\n很棒 ⭐⭐⭐⭐⭐"
            
        else:  # 情感
            content_start = f"这次{theme}让我感触很深...\n\n"
            content_body = "\n".join([f"那种{point.replace('孩子', '小家伙').replace('的', '时的')}感觉" for point in content_points[:3]])
            content_end = "\n\n希望这份美好能传递给更多人 ❤️"
        
        content = content_start + content_body + content_end
        
        # 长度调整
        if length == "short" and len(content) > 100:
            content = content_start + content_points[0] if content_points else content_start.replace('\n\n', '')
        elif length == "long":
            if len(content_points) > 3:
                extra_points = "\n".join([f"• {point}" for point in content_points[3:]])
                content = content.replace(content_end, f"\n\n另外：\n{extra_points}{content_end}")
        
        # 互动语生成
        engagement_templates = {
            "活泼": ["你们有类似的体验吗？快来评论区聊聊！🤗", "大家觉得怎么样？欢迎分享你们的故事~", "有同款体验的小伙伴吗？一起交流呀！"],
            "专业": ["欢迎大家分享自己的经验和看法", "如果有问题可以在评论区讨论", "期待听到更多朋友的分享"],
            "简约": ["你们觉得呢？", "欢迎交流", "有什么想法吗？"],
            "情感": ["希望你们也能感受到这份美好 💝", "愿每个人都能拥有这样温暖的时光", "分享快乐，传递温暖 ❤️"]
        }
        
        engagement = random.choice(engagement_templates.get(style, engagement_templates["活泼"]))
        
        return title, content, engagement
    
    def _generate_smart_hashtags(self, theme: str, keywords: List[str], content_points: List[str]) -> List[str]:
        """生成智能标签"""
        hashtags = [f"#{theme}"]
        
        # 基于关键词的标签
        keyword_tags = [f"#{kw}" for kw in keywords[:3]]
        hashtags.extend(keyword_tags)
        
        # 基于内容要点的标签
        content_tags = []
        for point in content_points:
            if "海边" in point or "沙滩" in point:
                content_tags.extend(["#海边", "#沙滩"])
            if "孩子" in point or "宝宝" in point:
                content_tags.extend(["#亲子时光", "#宝宝成长"])
            if "美食" in point or "餐厅" in point:
                content_tags.extend(["#美食", "#餐厅推荐"])
            if "风景" in point or "景色" in point:
                content_tags.extend(["#风景", "#打卡"])
        
        hashtags.extend(list(set(content_tags))[:3])  # 去重并限制数量
        
        # 通用热门标签
        popular_tags = ["#生活记录", "#美好时光", "#值得推荐", "#幸福感", "#记录生活"]
        hashtags.extend(random.sample(popular_tags, 2))
        
        return list(set(hashtags))[:10]  # 去重并限制为10个标签
    
    def _generate_fallback_pro_content(self, topic: str, content_type: str) -> Dict[str, Any]:
        """生成Pro功能备用内容"""
        
        if content_type == "title":
            titles = [
                f"关于{topic}的深度思考",
                f"{topic} - 全面解析",
                f"你不知道的{topic}秘密",
                f"{topic}完全指南"
            ]
            data = {"titles": titles}
        elif content_type == "hashtags":
            data = {
                "hashtags": [f"#{topic}", "#专业", "#分享", "#知识", "#经验", "#推荐", "#学习", "#成长"]
            }
        else:
            data = {
                "content": f"关于{topic}的内容正在生成中，请稍后再试或配置LLM API以获得更好的生成效果。"
            }
        
        return {
            "status": "fallback",
            "type": content_type,
            "data": data,
            "generated_at": datetime.now().isoformat()
        }

# 全局LLM服务实例
_llm_service = None

def get_llm_service() -> LLMService:
    """获取LLM服务实例"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
