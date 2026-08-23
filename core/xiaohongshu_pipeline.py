"""
小红书一键出稿流水线
面向旅行博主的智能内容生成系统
"""

import re
import logging
from typing import List, Dict
from pathlib import Path
from datetime import datetime
import random

logger = logging.getLogger(__name__)


class PhotoRankingService:
    """照片选优服务"""
    
    def __init__(self):
        self.aesthetic_keywords = [
            '风景', '建筑', '美食', '人物', '天空', '夕阳', '夜景', '街道',
            'landscape', 'architecture', 'food', 'portrait', 'sky', 'sunset'
        ]
        
    def rank_photos(self, photos: List[str], top_k: int = 15) -> List[Dict]:
        """
        照片选优排序
        
        Args:
            photos: 照片路径或URL列表
            top_k: 返回前k张照片
            
        Returns:
            排序后的照片信息
        """
        try:
            ranked_photos = []
            
            for i, photo_path in enumerate(photos):
                # 基础评分（简化版，实际可接入CLIP/美学模型）
                score = self._calculate_photo_score(photo_path, i)
                
                ranked_photos.append({
                    'photo_id': f'photo_{i:03d}',
                    'path': photo_path,
                    'score': score,
                    'rank': 0,  # 将在排序后设置
                    'tags': self._extract_photo_tags(photo_path),
                    'aesthetic_score': score * 0.7,
                    'composition_score': score * 0.8,
                    'uniqueness_score': score * 0.9
                })
            
            # 按分数排序
            ranked_photos.sort(key=lambda x: x['score'], reverse=True)
            
            # 设置排名并返回前k个
            for i, photo in enumerate(ranked_photos[:top_k]):
                photo['rank'] = i + 1
            
            return ranked_photos[:top_k]
            
        except Exception as e:
            logger.error(f"照片排序失败: {str(e)}")
            return []
    
    def _calculate_photo_score(self, photo_path: str, index: int) -> float:
        """计算照片评分（简化版）"""
        try:
            # 基于文件名和位置的启发式评分
            filename = Path(photo_path).name.lower()
            
            base_score = 0.5
            
            # 文件名包含美学关键词
            for keyword in self.aesthetic_keywords:
                if keyword in filename:
                    base_score += 0.1
            
            # 文件大小启发式（假设更大的文件质量更好）
            if Path(photo_path).exists():
                file_size = Path(photo_path).stat().st_size
                size_score = min(file_size / (1024 * 1024), 5.0) / 10.0  # MB转换为0-0.5分
                base_score += size_score
            
            # 位置多样性（避免连续相似照片）
            position_bonus = 0.1 * (1 - (index % 5) / 10)
            base_score += position_bonus
            
            # 随机因子增加多样性
            random_factor = random.uniform(0.8, 1.2)
            
            return min(base_score * random_factor, 1.0)
            
        except Exception:
            return 0.5
    
    def _extract_photo_tags(self, photo_path: str) -> List[str]:
        """提取照片标签（简化版）"""
        filename = Path(photo_path).name.lower()
        tags = []
        
        tag_mapping = {
            'food': ['美食', '餐厅', '小吃'],
            'landscape': ['风景', '自然', '景色'],
            'architecture': ['建筑', '古建', '现代'],
            'portrait': ['人物', '自拍', '合影'],
            'night': ['夜景', '灯光', '夜晚'],
            'street': ['街景', '街道', '城市']
        }
        
        for eng_tag, cn_tags in tag_mapping.items():
            if eng_tag in filename:
                tags.extend(cn_tags)
        
        return tags[:3]  # 最多返回3个标签


class StorylineGenerator:
    """故事线生成器"""
    
    def __init__(self):
        self.section_templates = {
            'opening': ['初印象', '到达', '第一眼', '开始探索'],
            'experience': ['核心体验', '必打卡', '亮点时刻', '特色体验'],
            'food': ['美食发现', '觅食时光', '味蕾惊喜', '当地特色'],
            'hidden': ['小众发现', '意外收获', '隐藏宝藏', '本地秘密'],
            'ending': ['总结感受', '离别时刻', '回味无穷', '下次再来']
        }
    
    def generate_storyline(self, transcript_mmss: List[Dict], notes: str, 
                          city: str = "", date: str = "", style: str = "治愈") -> Dict:
        """
        生成旅行故事线大纲
        
        Args:
            transcript_mmss: 带时间戳的转录文本
            notes: 作者要点
            city: 城市名称
            date: 日期
            style: 风格（治愈/专业/踩雷等）
            
        Returns:
            故事线结构
        """
        try:
            # 分析转录内容
            content_analysis = self._analyze_transcript(transcript_mmss)
            
            # 解析作者要点
            parsed_notes = self._parse_notes(notes)
            
            # 生成故事段落
            sections = self._generate_sections(content_analysis, parsed_notes, style)
            
            # 提取实用信息
            tips = self._extract_tips(content_analysis, parsed_notes)
            costs = self._extract_costs(content_analysis, parsed_notes)
            pois = self._extract_pois(content_analysis, parsed_notes, city)
            keywords = self._extract_keywords(content_analysis, parsed_notes, city)
            
            return {
                'city': city,
                'date': date,
                'style': style,
                'sections': sections,
                'tips': tips,
                'costs': costs,
                'pois': pois,
                'keywords': keywords,
                'duration_estimate': self._estimate_duration(sections),
                'highlight_moments': self._identify_highlights(content_analysis)
            }
            
        except Exception as e:
            logger.error(f"故事线生成失败: {str(e)}")
            return self._get_fallback_storyline(city, notes)
    
    def _analyze_transcript(self, transcript_mmss: List[Dict]) -> Dict:
        """分析转录内容"""
        analysis = {
            'total_duration': 0,
            'segments': [],
            'topics': set(),
            'emotions': [],
            'locations': set(),
            'activities': set()
        }
        
        # 关键词映射
        topic_keywords = {
            'food': ['吃', '美食', '餐厅', '小吃', '味道', '好吃', '饭', '菜'],
            'scenery': ['风景', '景色', '美丽', '漂亮', '山', '水', '天空', '夕阳'],
            'culture': ['文化', '历史', '古', '传统', '博物馆', '寺庙', '建筑'],
            'shopping': ['买', '购物', '商店', '市场', '便宜', '贵', '价格'],
            'transport': ['坐', '走', '开车', '地铁', '公交', '打车', '路']
        }
        
        emotion_keywords = {
            'positive': ['好', '棒', '喜欢', '开心', '兴奋', '惊喜', '满意'],
            'negative': ['不好', '失望', '累', '贵', '难吃', '坑', '后悔'],
            'neutral': ['还行', '一般', '普通', '可以', '凑合']
        }
        
        for segment in transcript_mmss:
            text = segment.get('text', '').lower()
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            
            # 分析主题
            for topic, keywords in topic_keywords.items():
                if any(keyword in text for keyword in keywords):
                    analysis['topics'].add(topic)
            
            # 分析情感
            for emotion, keywords in emotion_keywords.items():
                if any(keyword in text for keyword in keywords):
                    analysis['emotions'].append({
                        'emotion': emotion,
                        'timestamp': f"{int(start_time//60):02d}:{int(start_time%60):02d}",
                        'text': text[:50]
                    })
            
            analysis['segments'].append({
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'text': text,
                'timestamp': f"{int(start_time//60):02d}:{int(start_time%60):02d}"
            })
            
            analysis['total_duration'] = max(analysis['total_duration'], end_time)
        
        return analysis
    
    def _parse_notes(self, notes: str) -> Dict:
        """解析作者要点"""
        parsed = {
            'key_points': [],
            'locations': [],
            'costs': {},
            'tips': [],
            'avoid': []
        }
        
        lines = notes.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 解析费用信息
            cost_match = re.search(r'(\d+)元|(\d+)块|花费(\d+)|人均(\d+)', line)
            if cost_match:
                cost = next(g for g in cost_match.groups() if g)
                parsed['costs']['estimated'] = int(cost)
            
            # 解析地点
            if any(word in line for word in ['在', '去', '到', '位于']):
                parsed['locations'].append(line)
            
            # 解析避雷点
            if any(word in line for word in ['避免', '不要', '注意', '坑', '踩雷']):
                parsed['avoid'].append(line)
            
            # 解析建议
            if any(word in line for word in ['建议', '推荐', '最好', 'tip', 'tips']):
                parsed['tips'].append(line)
            else:
                parsed['key_points'].append(line)
        
        return parsed
    
    def _generate_sections(self, content_analysis: Dict, parsed_notes: Dict, style: str) -> List[Dict]:
        """生成故事段落"""
        sections = []
        total_duration = content_analysis['total_duration']
        segments = content_analysis['segments']
        
        if not segments:
            return self._get_default_sections()
        
        # 按时间分配段落
        section_count = min(5, len(segments) // 2 + 2)
        time_per_section = total_duration / section_count
        
        section_types = ['opening', 'experience', 'food', 'hidden', 'ending']
        
        for i in range(section_count):
            start_time = i * time_per_section
            end_time = min((i + 1) * time_per_section, total_duration)
            
            # 找到时间范围内的片段
            relevant_segments = [
                seg for seg in segments 
                if seg['start'] >= start_time and seg['end'] <= end_time
            ]
            
            section_type = section_types[min(i, len(section_types) - 1)]
            title_options = self.section_templates[section_type]
            
            sections.append({
                'title': random.choice(title_options),
                'type': section_type,
                'start_time': f"{int(start_time//60):02d}:{int(start_time%60):02d}",
                'end_time': f"{int(end_time//60):02d}:{int(end_time%60):02d}",
                'used_timestamps': [seg['timestamp'] for seg in relevant_segments],
                'content_summary': self._summarize_segments(relevant_segments),
                'shot_suggestions': self._get_shot_suggestions(section_type),
                'style_notes': self._get_style_notes(section_type, style)
            })
        
        return sections
    
    def _get_shot_suggestions(self, section_type: str) -> List[str]:
        """获取镜头建议"""
        shot_mapping = {
            'opening': ['广角全景', '建立镜头', '环境展示'],
            'experience': ['特写重点', '动态跟拍', '多角度切换'],
            'food': ['近景特写', '制作过程', '品尝反应'],
            'hidden': ['发现镜头', '对比展示', '细节捕捉'],
            'ending': ['回顾蒙太奇', '情感特写', '告别镜头']
        }
        return shot_mapping.get(section_type, ['常规拍摄'])
    
    def _get_style_notes(self, section_type: str, style: str) -> str:
        """获取风格注释"""
        style_templates = {
            '治愈': '温暖治愈的氛围，注重情感共鸣',
            '专业': '信息密度高，实用性强',
            '踩雷': '重点突出避坑指南，对比明显'
        }
        return style_templates.get(style, '自然真实的表达')
    
    def _extract_tips(self, content_analysis: Dict, parsed_notes: Dict) -> List[Dict]:
        """提取实用建议"""
        tips = []
        
        # 从笔记中提取
        for tip in parsed_notes['tips']:
            tips.append({
                'type': 'general',
                'content': tip,
                'priority': 'high'
            })
        
        # 从避雷点提取
        for avoid in parsed_notes['avoid']:
            tips.append({
                'type': 'warning',
                'content': avoid,
                'priority': 'critical'
            })
        
        # 从转录内容推断
        if 'transport' in content_analysis['topics']:
            tips.append({
                'type': 'transport',
                'content': '建议提前查看交通路线',
                'priority': 'medium'
            })
        
        return tips[:5]  # 最多5个建议
    
    def _extract_costs(self, content_analysis: Dict, parsed_notes: Dict) -> Dict:
        """提取费用信息"""
        costs = parsed_notes['costs'].copy()
        
        # 默认预估
        if not costs:
            costs = {
                'estimated': 200,
                'range': '150-300',
                'category': '中等消费'
            }
        
        return costs
    
    def _extract_pois(self, content_analysis: Dict, parsed_notes: Dict, city: str) -> List[Dict]:
        """提取兴趣点"""
        pois = []
        
        for location in parsed_notes['locations']:
            pois.append({
                'name': location,
                'city': city,
                'category': 'attraction',
                'mentioned_at': '00:00'  # 简化版
            })
        
        return pois
    
    def _extract_keywords(self, content_analysis: Dict, parsed_notes: Dict, city: str) -> List[str]:
        """提取关键词"""
        keywords = set()
        
        # 城市相关
        if city:
            keywords.add(city)
            keywords.add(f'{city}旅行')
        
        # 主题相关
        topic_mapping = {
            'food': ['美食', '觅食', '当地特色'],
            'scenery': ['风景', '打卡', '拍照'],
            'culture': ['文化', '历史', '人文'],
            'shopping': ['购物', '逛街', '买买买']
        }
        
        for topic in content_analysis['topics']:
            keywords.update(topic_mapping.get(topic, []))
        
        # 风格标签
        keywords.update(['一日游', '攻略', '旅行日记', '城市探索'])
        
        return list(keywords)[:15]
    
    def _get_fallback_storyline(self, city: str, notes: str) -> Dict:
        """获取默认故事线"""
        return {
            'city': city,
            'sections': self._get_default_sections(),
            'tips': [{'type': 'general', 'content': '提前做好行程规划', 'priority': 'medium'}],
            'costs': {'estimated': 200},
            'pois': [],
            'keywords': [city, '旅行', '攻略'] if city else ['旅行']
        }
    
    def _get_default_sections(self) -> List[Dict]:
        """获取默认段落"""
        return [
            {
                'title': '初印象',
                'type': 'opening',
                'start_time': '00:00',
                'end_time': '02:00',
                'shot_suggestions': ['广角全景'],
                'style_notes': '建立氛围'
            }
        ]
    
    def _summarize_segments(self, segments: List[Dict]) -> str:
        """总结片段内容"""
        if not segments:
            return "暂无内容"
        
        texts = [seg['text'][:30] for seg in segments]
        return ' | '.join(texts)
    
    def _estimate_duration(self, sections: List[Dict]) -> str:
        """估算游览时长"""
        section_count = len(sections)
        if section_count <= 2:
            return "2-3小时"
        elif section_count <= 4:
            return "半天"
        else:
            return "一天"
    
    def _identify_highlights(self, content_analysis: Dict) -> List[Dict]:
        """识别高光时刻"""
        highlights = []
        
        for emotion in content_analysis['emotions']:
            if emotion['emotion'] == 'positive':
                highlights.append({
                    'timestamp': emotion['timestamp'],
                    'reason': '情感高光',
                    'content': emotion['text']
                })
        
        return highlights[:3]


class XiaohongshuDraftGenerator:
    """小红书文案生成器"""
    
    def __init__(self):
        self.title_templates = [
            "{city}{duration}游｜{highlight}",
            "超详细！{city}{style}攻略",
            "{city}必打卡｜{poi_count}个宝藏地点",
            "人均{cost}💰{city}一日游攻略",
            "{city}旅行｜{weather}{style}路线"
        ]
        
        self.emoji_pool = {
            'food': ['🍜', '🥘', '🍱', '🥟', '🍲', '😋'],
            'scenery': ['🏞️', '🌅', '🏔️', '🌊', '🌸', '📸'],
            'transport': ['🚇', '🚌', '🚗', '🚶‍♀️', '🛵'],
            'money': ['💰', '💸', '💳', '💵'],
            'time': ['⏰', '🕐', '⏱️', '📅'],
            'tips': ['⚠️', '💡', '✨', '👍', '📝'],
            'location': ['📍', '🗺️', '🧭'],
            'positive': ['👍', '💯', '✨', '🌟', '❤️', '😍']
        }
    
    def generate_draft(self, storyline: Dict, brand_tone: str = "治愈", 
                      constraints: Dict = None) -> Dict:
        """
        生成小红书文案
        
        Args:
            storyline: 故事线数据
            brand_tone: 品牌调性
            constraints: 约束条件
            
        Returns:
            小红书文案
        """
        try:
            if constraints is None:
                constraints = {
                    'emoji_density': 0.3,  # 表情符号密度
                    'paragraph_length': 80,  # 每段字数
                    'hashtag_count': 12  # 话题标签数量
                }
            
            # 生成标题
            title = self._generate_title(storyline, brand_tone)
            
            # 生成正文
            body = self._generate_body(storyline, brand_tone, constraints)
            
            # 生成话题标签
            hashtags = self._generate_hashtags(storyline, constraints['hashtag_count'])
            
            # 生成POI信息
            poi_info = self._format_poi_info(storyline.get('pois', []))
            
            return {
                'title': title,
                'body': body,
                'hashtags': hashtags,
                'poi': poi_info,
                'metadata': {
                    'word_count': len(body.replace(' ', '')),
                    'emoji_count': len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', body)),
                    'paragraph_count': len(body.split('\n\n')),
                    'brand_tone': brand_tone,
                    'generated_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"文案生成失败: {str(e)}")
            return self._get_fallback_draft(storyline)
    
    def _generate_title(self, storyline: Dict, brand_tone: str) -> str:
        """生成标题"""
        try:
            city = storyline.get('city', '某城市')
            duration = storyline.get('duration_estimate', '一日')
            sections = storyline.get('sections', [])
            costs = storyline.get('costs', {})
            
            # 提取亮点
            highlight = "必打卡"
            if sections:
                food_sections = [s for s in sections if s['type'] == 'food']
                if food_sections:
                    highlight = "觅食之旅"
                elif any(s['type'] == 'hidden' for s in sections):
                    highlight = "小众路线"
            
            # 选择模板
            template = random.choice(self.title_templates)
            
            title = template.format(
                city=city,
                duration=duration,
                highlight=highlight,
                poi_count=len(storyline.get('pois', [])),
                cost=costs.get('estimated', 200),
                style=brand_tone,
                weather="☀️"
            )
            
            # 确保标题不超过20字
            if len(title) > 20:
                title = f"{city}{duration}游攻略"
            
            return title
            
        except Exception:
            return "城市一日游攻略"
    
    def _generate_body(self, storyline: Dict, brand_tone: str, constraints: Dict) -> str:
        """生成正文"""
        try:
            sections = storyline.get('sections', [])
            tips = storyline.get('tips', [])
            
            paragraphs = []
            
            # 开场白
            opening = self._generate_opening(storyline, brand_tone)
            paragraphs.append(opening)
            
            # 路线亮点
            if sections:
                route_para = self._generate_route_paragraph(sections, brand_tone)
                paragraphs.append(route_para)
            
            # 美食推荐
            food_sections = [s for s in sections if s['type'] == 'food']
            if food_sections:
                food_para = self._generate_food_paragraph(food_sections, brand_tone)
                paragraphs.append(food_para)
            
            # 踩雷避坑
            warning_tips = [t for t in tips if t['type'] == 'warning']
            if warning_tips:
                warning_para = self._generate_warning_paragraph(warning_tips, brand_tone)
                paragraphs.append(warning_para)
            
            # 实用信息
            practical_para = self._generate_practical_paragraph(storyline, brand_tone)
            paragraphs.append(practical_para)
            
            # 互动引导
            interaction = self._generate_interaction_guide(brand_tone)
            paragraphs.append(interaction)
            
            # 添加表情符号
            body = '\n\n'.join(paragraphs)
            body = self._add_emojis(body, constraints['emoji_density'])
            
            return body
            
        except Exception as e:
            logger.error(f"正文生成失败: {str(e)}")
            return "今天的城市探索之旅真的超级充实！"
    
    def _generate_opening(self, storyline: Dict, brand_tone: str) -> str:
        """生成开场白"""
        city = storyline.get('city', '这座城市')
        duration = storyline.get('duration_estimate', '一天')
        
        if brand_tone == "治愈":
            return f"在{city}度过了超级治愈的{duration}，每一个角落都让人心动不已"
        elif brand_tone == "专业":
            return f"详细记录{city}{duration}游的完整攻略，信息量巨大建议收藏"
        else:
            return f"{city}{duration}游实测，有惊喜也有踩雷，真实分享给大家"
    
    def _generate_route_paragraph(self, sections: List[Dict], brand_tone: str) -> str:
        """生成路线段落"""
        route_items = []
        for i, section in enumerate(sections[:4], 1):
            title = section['title']
            time = section['start_time']
            route_items.append(f"{time} {title}")
        
        route_text = " → ".join(route_items)
        return f"📍路线安排：{route_text}，时间安排刚刚好不会太赶"
    
    def _generate_food_paragraph(self, food_sections: List[Dict], brand_tone: str) -> str:
        """生成美食段落"""
        if brand_tone == "治愈":
            return "🍜美食推荐：每一口都是满满的幸福感，特别是那家小店的招牌菜"
        else:
            return "🍜觅食收获：找到了几家性价比超高的本地美食，味道正宗价格实惠"
    
    def _generate_warning_paragraph(self, warning_tips: List[Dict], brand_tone: str) -> str:
        """生成避坑段落"""
        warning_content = warning_tips[0]['content'] if warning_tips else "周末人比较多"
        return f"⚠️踩雷避坑：{warning_content}，大家去的时候要注意"
    
    def _generate_practical_paragraph(self, storyline: Dict, brand_tone: str) -> str:
        """生成实用信息段落"""
        costs = storyline.get('costs', {})
        duration = storyline.get('duration_estimate', '一天')
        
        cost_text = f"人均{costs.get('estimated', 200)}元" if costs else "费用适中"
        return f"💰实用信息：{cost_text}，{duration}时间刚好，建议穿舒适的鞋子"
    
    def _generate_interaction_guide(self, brand_tone: str) -> str:
        """生成互动引导"""
        guides = [
            "你们还有什么想了解的吗？评论区告诉我",
            "有去过的小伙伴吗？分享一下你们的体验吧",
            "还有什么好玩的地方推荐吗？求分享"
        ]
        return random.choice(guides)
    
    def _add_emojis(self, text: str, density: float) -> str:
        """添加表情符号"""
        try:
            sentences = re.split(r'[。！？]', text)
            emoji_count = max(1, int(len(sentences) * density))
            
            # 为随机句子添加相关表情
            for _ in range(emoji_count):
                if not sentences:
                    break
                    
                sentence_idx = random.randint(0, len(sentences) - 1)
                sentence = sentences[sentence_idx]
                
                # 根据内容选择表情
                emoji = self._select_emoji_for_content(sentence)
                sentences[sentence_idx] = sentence + emoji
            
            return '。'.join(sentences)
            
        except Exception:
            return text
    
    def _select_emoji_for_content(self, content: str) -> str:
        """根据内容选择表情符号"""
        for category, emojis in self.emoji_pool.items():
            if category == 'food' and any(word in content for word in ['吃', '美食', '餐厅']):
                return random.choice(emojis)
            elif category == 'scenery' and any(word in content for word in ['景', '美丽', '拍照']):
                return random.choice(emojis)
            elif category == 'money' and any(word in content for word in ['元', '费用', '价格']):
                return random.choice(emojis)
        
        return random.choice(self.emoji_pool['positive'])
    
    def _generate_hashtags(self, storyline: Dict, count: int) -> List[str]:
        """生成话题标签"""
        hashtags = []
        
        # 城市相关
        city = storyline.get('city', '')
        if city:
            hashtags.extend([f'#{city}', f'#{city}旅行', f'#{city}攻略'])
        
        # 基础标签
        base_tags = ['#一日游', '#旅行攻略', '#城市探索', '#周末去哪里']
        hashtags.extend(base_tags)
        
        # 主题标签
        sections = storyline.get('sections', [])
        if any(s['type'] == 'food' for s in sections):
            hashtags.extend(['#美食探店', '#觅食'])
        if any(s['type'] == 'hidden' for s in sections):
            hashtags.extend(['#小众景点', '#隐藏宝藏'])
        
        # 费用相关
        costs = storyline.get('costs', {})
        if costs.get('estimated', 0) < 150:
            hashtags.append('#穷游')
        elif costs.get('estimated', 0) < 300:
            hashtags.append('#性价比')
        
        # 风格标签
        hashtags.extend(['#旅行日记', '#打卡', '#拍照'])
        
        # 去重并限制数量
        unique_hashtags = list(dict.fromkeys(hashtags))  # 保持顺序去重
        return unique_hashtags[:count]
    
    def _format_poi_info(self, pois: List[Dict]) -> List[Dict]:
        """格式化POI信息"""
        formatted_pois = []
        
        for poi in pois:
            formatted_pois.append({
                'name': poi.get('name', ''),
                'category': poi.get('category', 'attraction'),
                'city': poi.get('city', ''),
                'rating': 4.5,  # 默认评分
                'tips': '值得一去'
            })
        
        return formatted_pois
    
    def _get_fallback_draft(self, storyline: Dict) -> Dict:
        """获取默认文案"""
        city = storyline.get('city', '某城市')
        return {
            'title': f'{city}一日游攻略',
            'body': f'今天在{city}度过了充实的一天，分享给大家一些实用的攻略',
            'hashtags': [f'#{city}', '#旅行攻略', '#一日游'],
            'poi': []
        }


# 全局服务实例
photo_ranking_service = None
storyline_generator = None
draft_generator = None

def get_photo_ranking_service() -> PhotoRankingService:
    """获取照片排序服务实例"""
    global photo_ranking_service
    if photo_ranking_service is None:
        photo_ranking_service = PhotoRankingService()
    return photo_ranking_service

def get_storyline_generator() -> StorylineGenerator:
    """获取故事线生成器实例"""
    global storyline_generator
    if storyline_generator is None:
        storyline_generator = StorylineGenerator()
    return storyline_generator

def get_draft_generator() -> XiaohongshuDraftGenerator:
    """获取文案生成器实例"""
    global draft_generator
    if draft_generator is None:
        draft_generator = XiaohongshuDraftGenerator()
    return draft_generator
