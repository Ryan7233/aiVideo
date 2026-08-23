"""
文案个性化服务
记住作者风格词库、口癖、常用表情、固定栏目
"""

import json
import logging
import re
from typing import Dict, List
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class PersonalizedWritingService:
    """个性化文案生成服务"""
    
    def __init__(self, user_profile_dir: str = "data/user_profiles"):
        self.user_profile_dir = Path(user_profile_dir)
        self.user_profile_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认风格模板
        self.default_styles = {
            '治愈': {
                'tone': 'gentle',
                'emoji_style': 'nature',
                'sentence_style': 'flowing',
                'common_phrases': ['心动不已', '治愈满满', '岁月静好', '温柔以待'],
                'emoji_pool': ['🌸', '🌿', '✨', '💫', '🌙', '☀️', '🍃', '🌺']
            },
            '专业': {
                'tone': 'informative',
                'emoji_style': 'minimal',
                'sentence_style': 'structured',
                'common_phrases': ['值得推荐', '性价比高', '综合评价', '实用攻略'],
                'emoji_pool': ['📍', '💰', '⏰', '🚇', '📝', '⭐', '👍', '📊']
            },
            '踩雷': {
                'tone': 'warning',
                'emoji_style': 'alert',
                'sentence_style': 'direct',
                'common_phrases': ['千万注意', '避雷指南', '血泪教训', '真实体验'],
                'emoji_pool': ['⚠️', '❌', '😤', '💔', '🚫', '😅', '🤦‍♀️', '💸']
            }
        }
        
        # 个性化特征权重
        self.personalization_weights = {
            'vocabulary_consistency': 0.25,    # 词汇一致性
            'emoji_pattern': 0.20,            # 表情符号模式
            'sentence_structure': 0.20,       # 句式结构
            'topic_preference': 0.15,         # 话题偏好
            'tone_consistency': 0.10,         # 语调一致性
            'format_template': 0.10           # 格式模板
        }
    
    def learn_user_style(self, user_id: str, content_samples: List[Dict]) -> Dict:
        """
        学习用户写作风格
        
        Args:
            user_id: 用户ID
            content_samples: 用户历史内容样本
            
        Returns:
            学习到的风格档案
        """
        try:
            logger.info(f"开始学习用户 {user_id} 的写作风格，样本数量: {len(content_samples)}")
            
            # 1. 提取词汇特征
            vocabulary_profile = self._analyze_vocabulary_patterns(content_samples)
            
            # 2. 分析表情符号使用模式
            emoji_profile = self._analyze_emoji_patterns(content_samples)
            
            # 3. 分析句式结构
            structure_profile = self._analyze_sentence_structures(content_samples)
            
            # 4. 分析话题偏好
            topic_profile = self._analyze_topic_preferences(content_samples)
            
            # 5. 分析语调风格
            tone_profile = self._analyze_tone_patterns(content_samples)
            
            # 6. 提取格式模板
            format_profile = self._extract_format_templates(content_samples)
            
            # 7. 构建完整的用户档案
            user_profile = {
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'sample_count': len(content_samples),
                'vocabulary': vocabulary_profile,
                'emoji': emoji_profile,
                'structure': structure_profile,
                'topics': topic_profile,
                'tone': tone_profile,
                'format': format_profile,
                'confidence_score': self._calculate_confidence_score(content_samples)
            }
            
            # 8. 保存用户档案
            self._save_user_profile(user_id, user_profile)
            
            logger.info(f"用户风格学习完成，置信度: {user_profile['confidence_score']:.2f}")
            return user_profile
            
        except Exception as e:
            logger.error(f"用户风格学习失败: {e}")
            return {}
    
    def generate_personalized_content(self, user_id: str, content_data: Dict, 
                                    style_override: str = None) -> Dict:
        """
        生成个性化内容
        
        Args:
            user_id: 用户ID
            content_data: 内容数据（故事线、关键信息等）
            style_override: 风格覆盖（可选）
            
        Returns:
            个性化的内容
        """
        try:
            logger.info(f"开始为用户 {user_id} 生成个性化内容")
            
            # 1. 加载用户档案
            user_profile = self._load_user_profile(user_id)
            
            # 2. 确定使用的风格
            target_style = style_override or self._determine_best_style(user_profile, content_data)
            
            # 3. 生成个性化标题
            personalized_title = self._generate_personalized_title(
                content_data, user_profile, target_style
            )
            
            # 4. 生成个性化正文
            personalized_body = self._generate_personalized_body(
                content_data, user_profile, target_style
            )
            
            # 5. 生成个性化话题标签
            personalized_hashtags = self._generate_personalized_hashtags(
                content_data, user_profile, target_style
            )
            
            # 6. 添加个性化互动引导
            personalized_cta = self._generate_personalized_cta(
                user_profile, target_style
            )
            
            result = {
                'title': personalized_title,
                'body': personalized_body,
                'hashtags': personalized_hashtags,
                'cta': personalized_cta,
                'style_used': target_style,
                'personalization_confidence': user_profile.get('confidence_score', 0.5),
                'metadata': {
                    'user_id': user_id,
                    'generated_at': datetime.now().isoformat(),
                    'personalization_features': self._get_applied_features(user_profile)
                }
            }
            
            logger.info(f"个性化内容生成完成，风格: {target_style}")
            return result
            
        except Exception as e:
            logger.error(f"个性化内容生成失败: {e}")
            # 回退到默认生成
            return self._generate_default_content(content_data, style_override or '治愈')
    
    def _analyze_vocabulary_patterns(self, samples: List[Dict]) -> Dict:
        """分析词汇模式"""
        try:
            all_words = []
            phrase_patterns = []
            
            for sample in samples:
                text = sample.get('body', '') + ' ' + sample.get('title', '')
                words = re.findall(r'\b[\u4e00-\u9fff]+\b', text)  # 中文词汇
                all_words.extend(words)
                
                # 提取短语模式
                phrases = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
                phrase_patterns.extend(phrases)
            
            # 统计词频
            word_freq = Counter(all_words)
            phrase_freq = Counter(phrase_patterns)
            
            # 提取特征词汇
            signature_words = [word for word, freq in word_freq.most_common(20) if freq >= 2]
            signature_phrases = [phrase for phrase, freq in phrase_freq.most_common(15) if freq >= 2]
            
            return {
                'signature_words': signature_words,
                'signature_phrases': signature_phrases,
                'total_vocabulary': len(set(all_words)),
                'vocabulary_diversity': len(set(all_words)) / max(len(all_words), 1),
                'common_word_patterns': word_freq.most_common(10)
            }
            
        except Exception as e:
            logger.error(f"词汇模式分析失败: {e}")
            return {}
    
    def _analyze_emoji_patterns(self, samples: List[Dict]) -> Dict:
        """分析表情符号模式"""
        try:
            all_emojis = []
            emoji_positions = {'start': 0, 'middle': 0, 'end': 0}
            
            for sample in samples:
                text = sample.get('body', '') + ' ' + sample.get('title', '')
                
                # 提取表情符号
                emojis = re.findall(r'[😀-🙏🌀-🗿🚀-🛿🇀-🇿]+', text)
                all_emojis.extend(emojis)
                
                # 分析位置模式
                if emojis:
                    text_length = len(text)
                    for emoji in emojis:
                        pos = text.find(emoji) / text_length
                        if pos < 0.2:
                            emoji_positions['start'] += 1
                        elif pos > 0.8:
                            emoji_positions['end'] += 1
                        else:
                            emoji_positions['middle'] += 1
            
            emoji_freq = Counter(all_emojis)
            
            return {
                'favorite_emojis': [emoji for emoji, freq in emoji_freq.most_common(10)],
                'emoji_frequency': len(all_emojis) / max(len(samples), 1),
                'position_preference': max(emoji_positions, key=emoji_positions.get),
                'emoji_diversity': len(set(all_emojis)),
                'usage_pattern': emoji_freq.most_common(5)
            }
            
        except Exception as e:
            logger.error(f"表情符号模式分析失败: {e}")
            return {}
    
    def _analyze_sentence_structures(self, samples: List[Dict]) -> Dict:
        """分析句式结构"""
        try:
            sentence_lengths = []
            punctuation_patterns = []
            structure_types = {'declarative': 0, 'exclamatory': 0, 'interrogative': 0}
            
            for sample in samples:
                text = sample.get('body', '')
                
                # 分句
                sentences = re.split(r'[。！？\n]', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                for sentence in sentences:
                    sentence_lengths.append(len(sentence))
                    
                    # 分析句型
                    if sentence.endswith('！') or '!' in sentence:
                        structure_types['exclamatory'] += 1
                    elif sentence.endswith('？') or '?' in sentence:
                        structure_types['interrogative'] += 1
                    else:
                        structure_types['declarative'] += 1
                    
                    # 标点模式
                    punctuation = re.findall(r'[，。！？、；：]', sentence)
                    punctuation_patterns.extend(punctuation)
            
            avg_sentence_length = sum(sentence_lengths) / max(len(sentence_lengths), 1)
            punctuation_freq = Counter(punctuation_patterns)
            
            return {
                'avg_sentence_length': avg_sentence_length,
                'preferred_sentence_type': max(structure_types, key=structure_types.get),
                'punctuation_style': punctuation_freq.most_common(3),
                'sentence_variety': len(set(sentence_lengths)) / max(len(sentence_lengths), 1),
                'structure_distribution': structure_types
            }
            
        except Exception as e:
            logger.error(f"句式结构分析失败: {e}")
            return {}
    
    def _analyze_topic_preferences(self, samples: List[Dict]) -> Dict:
        """分析话题偏好"""
        try:
            topic_keywords = {
                'food': ['美食', '好吃', '餐厅', '菜', '味道', '吃'],
                'scenery': ['风景', '景色', '美景', '山', '海', '天空'],
                'experience': ['体验', '感受', '觉得', '感觉', '印象'],
                'recommendation': ['推荐', '建议', '值得', '必须', '一定'],
                'cost': ['价格', '费用', '花费', '便宜', '贵', '性价比'],
                'transportation': ['交通', '地铁', '公交', '打车', '步行'],
                'time': ['时间', '早上', '下午', '晚上', '周末', '节假日']
            }
            
            topic_scores = defaultdict(int)
            
            for sample in samples:
                text = sample.get('body', '') + ' ' + sample.get('title', '')
                
                for topic, keywords in topic_keywords.items():
                    for keyword in keywords:
                        topic_scores[topic] += text.count(keyword)
            
            # 计算话题偏好分布
            total_mentions = sum(topic_scores.values())
            topic_distribution = {}
            if total_mentions > 0:
                for topic, count in topic_scores.items():
                    topic_distribution[topic] = count / total_mentions
            
            return {
                'preferred_topics': sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:5],
                'topic_distribution': topic_distribution,
                'topic_diversity': len([t for t in topic_scores.values() if t > 0])
            }
            
        except Exception as e:
            logger.error(f"话题偏好分析失败: {e}")
            return {}
    
    def _analyze_tone_patterns(self, samples: List[Dict]) -> Dict:
        """分析语调模式"""
        try:
            tone_indicators = {
                'enthusiastic': ['太棒了', '超级', '特别', '非常', '极其', '！！'],
                'gentle': ['温柔', '轻柔', '慢慢', '静静', '悄悄'],
                'professional': ['分析', '评估', '建议', '综合', '客观'],
                'casual': ['哈哈', '嘻嘻', '呀', '啦', '呢', '哦'],
                'warning': ['注意', '小心', '避免', '千万', '警告']
            }
            
            tone_scores = defaultdict(int)
            
            for sample in samples:
                text = sample.get('body', '') + ' ' + sample.get('title', '')
                
                for tone, indicators in tone_indicators.items():
                    for indicator in indicators:
                        tone_scores[tone] += text.count(indicator)
            
            # 确定主要语调
            primary_tone = max(tone_scores, key=tone_scores.get) if tone_scores else 'neutral'
            
            return {
                'primary_tone': primary_tone,
                'tone_scores': dict(tone_scores),
                'tone_consistency': max(tone_scores.values()) / max(sum(tone_scores.values()), 1)
            }
            
        except Exception as e:
            logger.error(f"语调模式分析失败: {e}")
            return {}
    
    def _extract_format_templates(self, samples: List[Dict]) -> Dict:
        """提取格式模板"""
        try:
            format_patterns = []
            
            for sample in samples:
                body = sample.get('body', '')
                
                # 提取结构模式
                lines = body.split('\n')
                structure = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith('📍') or '地点' in line:
                        structure.append('location')
                    elif line.startswith('💰') or '价格' in line or '费用' in line:
                        structure.append('cost')
                    elif line.startswith('⏰') or '时间' in line:
                        structure.append('time')
                    elif line.startswith('🍜') or '美食' in line:
                        structure.append('food')
                    elif line.startswith('⚠️') or '注意' in line or '避雷' in line:
                        structure.append('warning')
                    else:
                        structure.append('content')
                
                format_patterns.append(structure)
            
            # 找出最常见的格式模式
            pattern_freq = Counter([tuple(pattern) for pattern in format_patterns])
            most_common_pattern = pattern_freq.most_common(1)[0][0] if pattern_freq else []
            
            return {
                'preferred_structure': list(most_common_pattern),
                'structure_consistency': pattern_freq.most_common(1)[0][1] / max(len(samples), 1) if pattern_freq else 0,
                'format_variations': len(pattern_freq)
            }
            
        except Exception as e:
            logger.error(f"格式模板提取失败: {e}")
            return {}
    
    def _calculate_confidence_score(self, samples: List[Dict]) -> float:
        """计算学习置信度"""
        try:
            # 基于样本数量和质量计算置信度
            sample_count = len(samples)
            
            # 样本数量因子
            count_factor = min(sample_count / 10.0, 1.0)
            
            # 样本质量因子（基于内容长度和完整性）
            quality_scores = []
            for sample in samples:
                title_len = len(sample.get('title', ''))
                body_len = len(sample.get('body', ''))
                hashtag_count = len(sample.get('hashtags', []))
                
                quality = min((title_len + body_len) / 200.0, 1.0) * 0.7 + min(hashtag_count / 5.0, 1.0) * 0.3
                quality_scores.append(quality)
            
            quality_factor = sum(quality_scores) / max(len(quality_scores), 1)
            
            # 综合置信度
            confidence = count_factor * 0.6 + quality_factor * 0.4
            return min(confidence, 1.0)
            
        except Exception:
            return 0.5
    
    def _save_user_profile(self, user_id: str, profile: Dict):
        """保存用户档案"""
        try:
            profile_path = self.user_profile_dir / f"{user_id}_profile.json"
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
            logger.info(f"用户档案已保存: {profile_path}")
        except Exception as e:
            logger.error(f"保存用户档案失败: {e}")
    
    def _load_user_profile(self, user_id: str) -> Dict:
        """加载用户档案"""
        try:
            profile_path = self.user_profile_dir / f"{user_id}_profile.json"
            if profile_path.exists():
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                logger.info(f"用户档案已加载: {user_id}")
                return profile
            else:
                logger.info(f"用户档案不存在，使用默认配置: {user_id}")
                return {}
        except Exception as e:
            logger.error(f"加载用户档案失败: {e}")
            return {}
    
    def _determine_best_style(self, user_profile: Dict, content_data: Dict) -> str:
        """确定最佳风格"""
        if not user_profile:
            return '治愈'
        
        # 基于用户档案的语调偏好
        tone_profile = user_profile.get('tone', {})
        primary_tone = tone_profile.get('primary_tone', 'neutral')
        
        # 映射语调到风格
        tone_to_style = {
            'enthusiastic': '治愈',
            'professional': '专业',
            'warning': '踩雷',
            'gentle': '治愈',
            'casual': '治愈'
        }
        
        return tone_to_style.get(primary_tone, '治愈')
    
    def _generate_personalized_title(self, content_data: Dict, user_profile: Dict, style: str) -> str:
        """生成个性化标题"""
        try:
            # 基础标题模板
            base_templates = {
                '治愈': ['{city}治愈之旅', '在{city}的美好时光', '{city}慢生活体验'],
                '专业': ['{city}深度攻略', '{city}实用指南', '超详细{city}攻略'],
                '踩雷': ['{city}避雷指南', '{city}踩雷实录', '千万别在{city}这样做']
            }
            
            city = content_data.get('city', '这里')
            templates = base_templates.get(style, base_templates['治愈'])
            
            # 应用用户个性化
            if user_profile:
                vocabulary = user_profile.get('vocabulary', {})
                signature_phrases = vocabulary.get('signature_phrases', [])
                
                # 尝试融入用户常用短语
                if signature_phrases:
                    personalized_template = f"{signature_phrases[0]}！{city}攻略"
                    return personalized_template
            
            # 使用默认模板
            import random
            return random.choice(templates).format(city=city)
            
        except Exception as e:
            logger.error(f"个性化标题生成失败: {e}")
            return f"{content_data.get('city', '这里')}攻略"
    
    def _generate_personalized_body(self, content_data: Dict, user_profile: Dict, style: str) -> str:
        """生成个性化正文"""
        try:
            # 获取故事线数据
            storyline = content_data.get('storyline', {})
            sections = storyline.get('sections', [])
            tips = storyline.get('tips', [])
            costs = storyline.get('costs', {})
            
            body_parts = []
            
            # 开场白（个性化）
            if user_profile:
                tone = user_profile.get('tone', {}).get('primary_tone', 'neutral')
                if tone == 'enthusiastic':
                    opening = "今天的体验真的太棒了！每一个瞬间都让人心动不已"
                elif tone == 'gentle':
                    opening = "静静地走过这些地方，内心充满了温暖和感动"
                elif tone == 'professional':
                    opening = "经过实地体验，为大家整理了这份详细攻略"
                else:
                    opening = "分享一下今天的美好经历"
            else:
                opening = "分享一下今天的美好经历"
            
            body_parts.append(opening)
            
            # 路线安排（使用用户偏好的表情符号）
            if sections and user_profile:
                emoji_profile = user_profile.get('emoji', {})
                favorite_emojis = emoji_profile.get('favorite_emojis', ['📍'])
                location_emoji = favorite_emojis[0] if favorite_emojis else '📍'
                
                route_text = f"{location_emoji}路线安排："
                for i, section in enumerate(sections[:4]):
                    route_text += f" {section.get('timestamp', f'{i*20:02d}:{0:02d}')} {section.get('title', '精彩时刻')}"
                    if i < len(sections) - 1:
                        route_text += " →"
                route_text += "，时间安排刚刚好不会太赶"
                body_parts.append(route_text)
            
            # 美食推荐（个性化语言）
            if user_profile:
                vocabulary = user_profile.get('vocabulary', {})
                food_phrases = [p for p in vocabulary.get('signature_phrases', []) if '好吃' in p or '美味' in p]
                if food_phrases:
                    food_text = f"🍜美食推荐：{food_phrases[0]}，特别是那家小店的招牌菜"
                else:
                    food_text = "🍜美食推荐：每一口都是满满的幸福感，特别是那家小店的招牌菜"
            else:
                food_text = "🍜美食推荐：每一口都是满满的幸福感，特别是那家小店的招牌菜"
            
            body_parts.append(food_text)
            
            # 避雷提醒
            if tips:
                warning_text = f"⚠️踩雷避坑：{tips[0]}，大家去的时候要注意"
                body_parts.append(warning_text)
            
            # 实用信息
            cost_info = costs.get('total', '200元左右')
            practical_text = f"💰实用信息：人均{cost_info}，一天时间刚好，建议穿舒适的鞋子"
            body_parts.append(practical_text)
            
            return '\n\n'.join(body_parts)
            
        except Exception as e:
            logger.error(f"个性化正文生成失败: {e}")
            return "今天的体验很不错，推荐大家去试试！"
    
    def _generate_personalized_hashtags(self, content_data: Dict, user_profile: Dict, style: str) -> List[str]:
        """生成个性化话题标签"""
        try:
            base_hashtags = []
            
            # 基础标签
            city = content_data.get('city', '')
            if city:
                base_hashtags.extend([f'#{city}', f'#{city}旅行', f'#{city}攻略'])
            
            # 风格相关标签
            style_hashtags = {
                '治愈': ['#治愈系', '#慢生活', '#美好时光', '#岁月静好'],
                '专业': ['#实用攻略', '#深度游', '#旅行指南', '#性价比'],
                '踩雷': ['#避雷指南', '#真实体验', '#踩雷预警', '#血泪教训']
            }
            
            base_hashtags.extend(style_hashtags.get(style, style_hashtags['治愈']))
            
            # 通用标签
            base_hashtags.extend(['#一日游', '#旅行攻略', '#城市探索', '#周末去哪里'])
            
            # 个性化标签（基于用户话题偏好）
            if user_profile:
                topic_prefs = user_profile.get('topics', {}).get('preferred_topics', [])
                for topic, score in topic_prefs[:3]:
                    if topic == 'food':
                        base_hashtags.extend(['#美食探店', '#觅食'])
                    elif topic == 'scenery':
                        base_hashtags.extend(['#美景', '#风景'])
                    elif topic == 'cost':
                        base_hashtags.extend(['#性价比', '#省钱攻略'])
            
            # 去重并限制数量
            unique_hashtags = list(dict.fromkeys(base_hashtags))[:12]
            return unique_hashtags
            
        except Exception as e:
            logger.error(f"个性化标签生成失败: {e}")
            return ['#旅行', '#攻略', '#推荐']
    
    def _generate_personalized_cta(self, user_profile: Dict, style: str) -> str:
        """生成个性化互动引导"""
        try:
            if user_profile:
                tone = user_profile.get('tone', {}).get('primary_tone', 'neutral')
                emoji_profile = user_profile.get('emoji', {})
                favorite_emojis = emoji_profile.get('favorite_emojis', ['😋'])
                
                if tone == 'enthusiastic':
                    cta = f"还有什么好玩的地方推荐吗{favorite_emojis[0] if favorite_emojis else '😋'}！快来分享吧"
                elif tone == 'gentle':
                    cta = f"如果你也喜欢这样的地方，欢迎留言交流{favorite_emojis[0] if favorite_emojis else '💫'}"
                elif tone == 'professional':
                    cta = "有什么问题欢迎在评论区讨论，我会及时回复"
                else:
                    cta = f"还有什么好玩的地方推荐吗{favorite_emojis[0] if favorite_emojis else '😋'}。求分享"
            else:
                cta = "还有什么好玩的地方推荐吗😋。求分享"
            
            return cta
            
        except Exception as e:
            logger.error(f"个性化互动引导生成失败: {e}")
            return "欢迎分享你的想法！"
    
    def _get_applied_features(self, user_profile: Dict) -> List[str]:
        """获取应用的个性化特征"""
        features = []
        
        if user_profile.get('vocabulary', {}).get('signature_phrases'):
            features.append('个性化词汇')
        if user_profile.get('emoji', {}).get('favorite_emojis'):
            features.append('表情符号偏好')
        if user_profile.get('tone', {}).get('primary_tone'):
            features.append('语调风格')
        if user_profile.get('topics', {}).get('preferred_topics'):
            features.append('话题偏好')
        
        return features
    
    def _generate_default_content(self, content_data: Dict, style: str) -> Dict:
        """生成默认内容（回退方案）"""
        city = content_data.get('city', '这里')
        
        return {
            'title': f'{city}一日游攻略',
            'body': '今天的体验很不错，分享给大家一些实用的信息和建议。',
            'hashtags': [f'#{city}', '#旅行攻略', '#一日游'],
            'cta': '还有什么想了解的吗？',
            'style_used': style,
            'personalization_confidence': 0.0,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'is_fallback': True
            }
        }


# 全局服务实例
personalized_writing_service = None

def get_personalized_writing_service() -> PersonalizedWritingService:
    """获取个性化文案服务实例"""
    global personalized_writing_service
    if personalized_writing_service is None:
        personalized_writing_service = PersonalizedWritingService()
    return personalized_writing_service
