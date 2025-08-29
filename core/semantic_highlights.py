"""
语义高光检测服务
基于关键词、情绪、动作权重的精彩度打分系统
"""

import re
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class SemanticHighlightDetector:
    """语义高光检测器"""
    
    def __init__(self):
        # 关键词权重配置
        self.keyword_weights = {
            # 情感强烈词汇
            'emotion_high': {
                'words': ['amazing', 'incredible', 'fantastic', 'awesome', 'beautiful', 'stunning', 
                         'wow', 'omg', '太棒了', '惊艳', '绝了', '爱了', '震撼', '完美', '绝美'],
                'weight': 2.5,
                'category': 'emotion'
            },
            'emotion_medium': {
                'words': ['good', 'nice', 'great', 'cool', 'lovely', 'pretty', 
                         '好', '不错', '很棒', '挺好', '喜欢', '满意'],
                'weight': 1.5,
                'category': 'emotion'
            },
            
            # 动作关键词
            'action_discovery': {
                'words': ['found', 'discovered', 'see', 'look', 'check out', 'try', 
                         '发现', '找到', '看到', '尝试', '体验', '探索'],
                'weight': 1.8,
                'category': 'action'
            },
            'action_eating': {
                'words': ['eat', 'taste', 'try', 'delicious', 'yummy', 'flavor',
                         '吃', '尝', '品尝', '好吃', '美味', '香'],
                'weight': 2.0,
                'category': 'action'
            },
            'action_visiting': {
                'words': ['visit', 'go to', 'arrive', 'reach', 'enter',
                         '去', '到', '来到', '参观', '游览', '逛'],
                'weight': 1.6,
                'category': 'action'
            },
            
            # 描述性强烈词汇
            'description_intense': {
                'words': ['huge', 'massive', 'tiny', 'enormous', 'spectacular', 'breathtaking',
                         '巨大', '超大', '很小', '壮观', '震撼', '惊人'],
                'weight': 1.7,
                'category': 'description'
            },
            
            # 推荐相关
            'recommendation': {
                'words': ['recommend', 'must try', 'must see', 'worth it', 'definitely',
                         '推荐', '必须', '一定要', '值得', '建议'],
                'weight': 2.2,
                'category': 'recommendation'
            },
            
            # 时间敏感
            'temporal_urgent': {
                'words': ['now', 'right now', 'immediately', 'quickly', 'hurry',
                         '现在', '马上', '立刻', '赶紧', '快'],
                'weight': 1.9,
                'category': 'temporal'
            },
            
            # 比较级
            'comparison': {
                'words': ['best', 'better', 'worst', 'worse', 'different', 'unique',
                         '最好', '更好', '最差', '不同', '独特', '特别'],
                'weight': 1.6,
                'category': 'comparison'
            }
        }
        
        # 情绪强度检测
        self.emotion_patterns = {
            'excitement': {
                'patterns': [r'!{2,}', r'\b(wow|omg|amazing)\b', r'[哇呀]{2,}', r'太.*了'],
                'weight': 2.5
            },
            'surprise': {
                'patterns': [r'\?{2,}', r'what\?', r'really\?', r'seriously\?', r'什么\?', r'真的\?'],
                'weight': 2.0
            },
            'emphasis': {
                'patterns': [r'\b(very|really|so|such)\b', r'非常', r'特别', r'超级', r'巨'],
                'weight': 1.5
            },
            'negation_strong': {
                'patterns': [r"don't|can't|won't", r'不能', r'不要', r'别'],
                'weight': 1.8
            }
        }
        
        # 语音特征权重（基于ASR置信度和语速）
        self.speech_features = {
            'high_confidence': {'threshold': 0.9, 'weight': 1.3},
            'medium_confidence': {'threshold': 0.7, 'weight': 1.0},
            'low_confidence': {'threshold': 0.5, 'weight': 0.7},
            'fast_speech': {'words_per_second': 4.0, 'weight': 1.4},
            'slow_speech': {'words_per_second': 1.5, 'weight': 0.9}
        }
        
        # 上下文增强因子
        self.context_boosters = {
            'first_mention': 1.5,  # 第一次提到某个话题
            'repeated_emphasis': 1.3,  # 重复强调
            'topic_transition': 1.2,  # 话题转换
            'conclusion': 1.4  # 总结性语句
        }
    
    def detect_highlights(self, transcript_segments: List[Dict], 
                         context: Dict = None) -> List[Dict]:
        """
        检测语义高光时刻
        
        Args:
            transcript_segments: ASR转录片段
            context: 上下文信息（主题、风格等）
            
        Returns:
            高光片段列表，按精彩度排序
        """
        try:
            logger.info(f"开始语义高光检测，共 {len(transcript_segments)} 个片段")
            
            # 1. 预处理和特征提取
            enhanced_segments = self._preprocess_segments(transcript_segments)
            
            # 2. 关键词权重分析
            enhanced_segments = self._analyze_keyword_weights(enhanced_segments)
            
            # 3. 情绪强度检测
            enhanced_segments = self._detect_emotion_intensity(enhanced_segments)
            
            # 4. 语音特征分析
            enhanced_segments = self._analyze_speech_features(enhanced_segments)
            
            # 5. 上下文增强
            enhanced_segments = self._apply_context_boosters(enhanced_segments, context)
            
            # 6. 计算综合精彩度分数
            highlight_segments = self._calculate_highlight_scores(enhanced_segments)
            
            # 7. 筛选和排序高光时刻
            final_highlights = self._filter_and_rank_highlights(highlight_segments, context)
            
            logger.info(f"检测到 {len(final_highlights)} 个高光时刻")
            return final_highlights
            
        except Exception as e:
            logger.error(f"语义高光检测失败: {e}")
            return []
    
    def _preprocess_segments(self, segments: List[Dict]) -> List[Dict]:
        """预处理转录片段"""
        enhanced_segments = []
        
        for i, segment in enumerate(segments):
            enhanced = segment.copy()
            
            # 基础信息
            text = segment.get('text', '').strip()
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            duration = end_time - start_time
            
            # 文本预处理
            enhanced['processed_text'] = text.lower()
            enhanced['word_count'] = len(text.split())
            enhanced['char_count'] = len(text)
            enhanced['duration'] = duration
            enhanced['words_per_second'] = enhanced['word_count'] / max(duration, 0.1)
            enhanced['segment_index'] = i
            
            # 初始化分析结果
            enhanced['keyword_scores'] = {}
            enhanced['emotion_scores'] = {}
            enhanced['speech_features'] = {}
            enhanced['context_boosts'] = {}
            enhanced['highlight_score'] = 0.0
            
            enhanced_segments.append(enhanced)
        
        return enhanced_segments
    
    def _analyze_keyword_weights(self, segments: List[Dict]) -> List[Dict]:
        """分析关键词权重"""
        try:
            for segment in segments:
                text = segment['processed_text']
                keyword_scores = {}
                total_keyword_score = 0.0
                
                # 检查每个关键词类别
                for category, config in self.keyword_weights.items():
                    words = config['words']
                    weight = config['weight']
                    category_type = config['category']
                    
                    # 计算匹配的关键词数量和权重
                    matches = 0
                    for word in words:
                        if word.lower() in text:
                            matches += 1
                    
                    if matches > 0:
                        # 考虑重复出现的加权
                        category_score = matches * weight
                        # 避免过度加权
                        category_score = min(category_score, weight * 3)
                        
                        keyword_scores[category] = {
                            'matches': matches,
                            'weight': weight,
                            'score': category_score,
                            'type': category_type
                        }
                        
                        total_keyword_score += category_score
                
                segment['keyword_scores'] = keyword_scores
                segment['total_keyword_score'] = total_keyword_score
            
            return segments
            
        except Exception as e:
            logger.error(f"关键词权重分析失败: {e}")
            return segments
    
    def _detect_emotion_intensity(self, segments: List[Dict]) -> List[Dict]:
        """检测情绪强度"""
        try:
            for segment in segments:
                text = segment.get('text', '')  # 使用原始文本保持标点
                emotion_scores = {}
                total_emotion_score = 0.0
                
                # 检查情绪模式
                for emotion_type, config in self.emotion_patterns.items():
                    patterns = config['patterns']
                    weight = config['weight']
                    
                    matches = 0
                    for pattern in patterns:
                        matches += len(re.findall(pattern, text, re.IGNORECASE))
                    
                    if matches > 0:
                        emotion_score = matches * weight
                        emotion_scores[emotion_type] = {
                            'matches': matches,
                            'weight': weight,
                            'score': emotion_score
                        }
                        total_emotion_score += emotion_score
                
                # 特殊情绪检测：语调变化（基于标点和重复字符）
                exclamation_count = text.count('!')
                question_count = text.count('?')
                caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
                
                if exclamation_count > 0:
                    emotion_scores['exclamation'] = {
                        'matches': exclamation_count,
                        'score': exclamation_count * 1.5
                    }
                    total_emotion_score += exclamation_count * 1.5
                
                if question_count > 0:
                    emotion_scores['questioning'] = {
                        'matches': question_count,
                        'score': question_count * 1.2
                    }
                    total_emotion_score += question_count * 1.2
                
                if caps_ratio > 0.3:  # 大写字母比例高
                    emotion_scores['emphasis_caps'] = {
                        'ratio': caps_ratio,
                        'score': caps_ratio * 2.0
                    }
                    total_emotion_score += caps_ratio * 2.0
                
                segment['emotion_scores'] = emotion_scores
                segment['total_emotion_score'] = total_emotion_score
            
            return segments
            
        except Exception as e:
            logger.error(f"情绪强度检测失败: {e}")
            return segments
    
    def _analyze_speech_features(self, segments: List[Dict]) -> List[Dict]:
        """分析语音特征"""
        try:
            for segment in segments:
                speech_features = {}
                speech_score = 1.0  # 基础分数
                
                # 语速分析
                words_per_second = segment.get('words_per_second', 2.0)
                
                if words_per_second >= self.speech_features['fast_speech']['words_per_second']:
                    speech_features['speech_rate'] = 'fast'
                    speech_score *= self.speech_features['fast_speech']['weight']
                elif words_per_second <= self.speech_features['slow_speech']['words_per_second']:
                    speech_features['speech_rate'] = 'slow'
                    speech_score *= self.speech_features['slow_speech']['weight']
                else:
                    speech_features['speech_rate'] = 'normal'
                
                # ASR置信度分析（如果可用）
                confidence = segment.get('avg_logprob', 0)  # Whisper的置信度
                if confidence:
                    # 转换为0-1范围的置信度
                    normalized_confidence = max(0, min(1, (confidence + 1) / 1))
                    
                    if normalized_confidence >= self.speech_features['high_confidence']['threshold']:
                        speech_features['confidence'] = 'high'
                        speech_score *= self.speech_features['high_confidence']['weight']
                    elif normalized_confidence >= self.speech_features['medium_confidence']['threshold']:
                        speech_features['confidence'] = 'medium'
                        speech_score *= self.speech_features['medium_confidence']['weight']
                    else:
                        speech_features['confidence'] = 'low'
                        speech_score *= self.speech_features['low_confidence']['weight']
                
                # 语音长度分析
                duration = segment.get('duration', 0)
                if duration > 10:  # 长片段可能包含更多信息
                    speech_features['length'] = 'long'
                    speech_score *= 1.2
                elif duration < 2:  # 短片段可能是重要的快速反应
                    speech_features['length'] = 'short'
                    speech_score *= 1.1
                else:
                    speech_features['length'] = 'medium'
                
                segment['speech_features'] = speech_features
                segment['speech_score'] = speech_score
            
            return segments
            
        except Exception as e:
            logger.error(f"语音特征分析失败: {e}")
            return segments
    
    def _apply_context_boosters(self, segments: List[Dict], context: Dict = None) -> List[Dict]:
        """应用上下文增强因子"""
        try:
            # 分析话题和重复模式
            topic_mentions = defaultdict(list)
            all_texts = []
            
            for i, segment in enumerate(segments):
                text = segment['processed_text']
                all_texts.append(text)
                
                # 提取关键话题
                topics = self._extract_topics(text)
                for topic in topics:
                    topic_mentions[topic].append(i)
            
            # 应用增强因子
            for i, segment in enumerate(segments):
                context_boosts = {}
                boost_multiplier = 1.0
                
                text = segment['processed_text']
                topics = self._extract_topics(text)
                
                # 第一次提到某个话题
                for topic in topics:
                    if topic_mentions[topic][0] == i:
                        context_boosts['first_mention'] = topic
                        boost_multiplier *= self.context_boosters['first_mention']
                
                # 重复强调
                word_counts = Counter(text.split())
                repeated_words = [word for word, count in word_counts.items() if count > 1 and len(word) > 3]
                if repeated_words:
                    context_boosts['repeated_emphasis'] = repeated_words
                    boost_multiplier *= self.context_boosters['repeated_emphasis']
                
                # 话题转换（与前一个片段话题不同）
                if i > 0:
                    prev_topics = self._extract_topics(segments[i-1]['processed_text'])
                    if topics and prev_topics and not set(topics).intersection(set(prev_topics)):
                        context_boosts['topic_transition'] = True
                        boost_multiplier *= self.context_boosters['topic_transition']
                
                # 总结性语句检测
                conclusion_patterns = [
                    r'\b(overall|in conclusion|finally|to sum up|总的来说|总结|最后)\b',
                    r'\b(recommend|suggest|建议|推荐)\b'
                ]
                for pattern in conclusion_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        context_boosts['conclusion'] = True
                        boost_multiplier *= self.context_boosters['conclusion']
                        break
                
                segment['context_boosts'] = context_boosts
                segment['context_boost_multiplier'] = boost_multiplier
            
            return segments
            
        except Exception as e:
            logger.error(f"上下文增强失败: {e}")
            return segments
    
    def _extract_topics(self, text: str) -> List[str]:
        """从文本中提取话题"""
        topics = []
        
        topic_keywords = {
            'food': ['eat', 'food', 'restaurant', 'delicious', 'taste', '吃', '美食', '餐厅', '好吃'],
            'place': ['place', 'location', 'here', 'there', 'visit', '地方', '这里', '那里', '参观'],
            'experience': ['experience', 'feel', 'think', 'amazing', '体验', '感觉', '觉得', '惊艳'],
            'recommendation': ['recommend', 'suggest', 'should', 'must', '推荐', '建议', '应该', '必须']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _calculate_highlight_scores(self, segments: List[Dict]) -> List[Dict]:
        """计算综合精彩度分数"""
        try:
            for segment in segments:
                # 基础分数权重
                weights = {
                    'keyword': 0.35,
                    'emotion': 0.30,
                    'speech': 0.20,
                    'context': 0.15
                }
                
                # 获取各项分数
                keyword_score = segment.get('total_keyword_score', 0)
                emotion_score = segment.get('total_emotion_score', 0)
                speech_score = segment.get('speech_score', 1.0)
                context_multiplier = segment.get('context_boost_multiplier', 1.0)
                
                # 归一化分数（避免过高的分数）
                normalized_keyword = min(keyword_score / 10.0, 1.0)
                normalized_emotion = min(emotion_score / 8.0, 1.0)
                normalized_speech = min(speech_score / 2.0, 1.0)
                
                # 计算基础综合分数
                base_score = (
                    normalized_keyword * weights['keyword'] +
                    normalized_emotion * weights['emotion'] +
                    normalized_speech * weights['speech']
                )
                
                # 应用上下文增强
                final_score = base_score * context_multiplier
                
                # 确保分数在合理范围内
                final_score = min(final_score, 3.0)
                
                segment['highlight_score'] = final_score
                segment['score_breakdown'] = {
                    'keyword': normalized_keyword,
                    'emotion': normalized_emotion,
                    'speech': normalized_speech,
                    'context_multiplier': context_multiplier,
                    'base_score': base_score,
                    'final_score': final_score
                }
            
            return segments
            
        except Exception as e:
            logger.error(f"精彩度分数计算失败: {e}")
            return segments
    
    def _filter_and_rank_highlights(self, segments: List[Dict], context: Dict = None) -> List[Dict]:
        """筛选和排序高光时刻"""
        try:
            # 设置筛选阈值
            min_score = 0.3  # 最低精彩度分数
            min_duration = 1.0  # 最短时长
            
            # 根据上下文调整阈值
            if context:
                style = context.get('style', '')
                if style == '专业':
                    min_score = 0.4  # 专业风格要求更高
                elif style == '治愈':
                    min_score = 0.25  # 治愈风格更宽松
            
            # 筛选高光片段
            highlights = []
            for segment in segments:
                score = segment.get('highlight_score', 0)
                duration = segment.get('duration', 0)
                
                if score >= min_score and duration >= min_duration:
                    # 添加高光类型标签
                    highlight_type = self._classify_highlight_type(segment)
                    segment['highlight_type'] = highlight_type
                    
                    # 生成高光描述
                    segment['highlight_reason'] = self._generate_highlight_reason(segment)
                    
                    highlights.append(segment)
            
            # 按分数排序
            highlights.sort(key=lambda x: x['highlight_score'], reverse=True)
            
            # 去重和合并相邻的高光时刻
            merged_highlights = self._merge_adjacent_highlights(highlights)
            
            # 限制数量（避免过多高光）
            max_highlights = min(len(merged_highlights), 10)
            final_highlights = merged_highlights[:max_highlights]
            
            # 添加排名
            for i, highlight in enumerate(final_highlights):
                highlight['highlight_rank'] = i + 1
            
            return final_highlights
            
        except Exception as e:
            logger.error(f"高光筛选排序失败: {e}")
            return segments
    
    def _classify_highlight_type(self, segment: Dict) -> str:
        """分类高光类型"""
        keyword_scores = segment.get('keyword_scores', {})
        emotion_scores = segment.get('emotion_scores', {})
        
        # 基于主要得分来源分类
        if any('emotion_high' in k for k in keyword_scores.keys()):
            return 'emotional_peak'
        elif any('action_eating' in k for k in keyword_scores.keys()):
            return 'food_highlight'
        elif any('recommendation' in k for k in keyword_scores.keys()):
            return 'recommendation'
        elif any('action_discovery' in k for k in keyword_scores.keys()):
            return 'discovery_moment'
        elif 'exclamation' in emotion_scores:
            return 'excitement'
        elif any('comparison' in k for k in keyword_scores.keys()):
            return 'comparison'
        else:
            return 'general_highlight'
    
    def _generate_highlight_reason(self, segment: Dict) -> str:
        """生成高光原因描述"""
        reasons = []
        
        keyword_scores = segment.get('keyword_scores', {})
        emotion_scores = segment.get('emotion_scores', {})
        context_boosts = segment.get('context_boosts', {})
        
        # 关键词原因
        if keyword_scores:
            top_keyword = max(keyword_scores.items(), key=lambda x: x[1]['score'])
            category = top_keyword[0]
            if 'emotion' in category:
                reasons.append('强烈情感表达')
            elif 'action' in category:
                reasons.append('重要行为动作')
            elif 'recommendation' in category:
                reasons.append('推荐建议')
        
        # 情绪原因
        if emotion_scores:
            if 'exclamation' in emotion_scores:
                reasons.append('感叹语气')
            if 'excitement' in emotion_scores:
                reasons.append('兴奋表达')
        
        # 上下文原因
        if context_boosts:
            if 'first_mention' in context_boosts:
                reasons.append('首次提及')
            if 'conclusion' in context_boosts:
                reasons.append('总结性语句')
        
        return '、'.join(reasons) if reasons else '综合评分高'
    
    def _merge_adjacent_highlights(self, highlights: List[Dict]) -> List[Dict]:
        """合并相邻的高光时刻"""
        if not highlights:
            return []
        
        try:
            merged = []
            current_group = [highlights[0]]
            
            for i in range(1, len(highlights)):
                prev_end = current_group[-1]['end']
                curr_start = highlights[i]['start']
                
                # 如果间隔小于3秒，合并
                if curr_start - prev_end <= 3.0:
                    current_group.append(highlights[i])
                else:
                    # 完成当前组的合并
                    if len(current_group) > 1:
                        merged_highlight = self._merge_highlight_group(current_group)
                        merged.append(merged_highlight)
                    else:
                        merged.append(current_group[0])
                    
                    # 开始新组
                    current_group = [highlights[i]]
            
            # 处理最后一组
            if len(current_group) > 1:
                merged_highlight = self._merge_highlight_group(current_group)
                merged.append(merged_highlight)
            else:
                merged.append(current_group[0])
            
            return merged
            
        except Exception as e:
            logger.error(f"合并相邻高光失败: {e}")
            return highlights
    
    def _merge_highlight_group(self, group: List[Dict]) -> Dict:
        """合并一组高光片段"""
        merged = group[0].copy()
        
        # 合并时间范围
        merged['start'] = min(seg['start'] for seg in group)
        merged['end'] = max(seg['end'] for seg in group)
        merged['duration'] = merged['end'] - merged['start']
        
        # 合并文本
        merged['text'] = ' '.join(seg['text'] for seg in group)
        merged['processed_text'] = ' '.join(seg['processed_text'] for seg in group)
        
        # 合并分数（取平均值）
        merged['highlight_score'] = np.mean([seg['highlight_score'] for seg in group])
        
        # 合并原因
        reasons = []
        for seg in group:
            reason = seg.get('highlight_reason', '')
            if reason and reason not in reasons:
                reasons.append(reason)
        merged['highlight_reason'] = '、'.join(reasons)
        
        # 标记为合并片段
        merged['is_merged'] = True
        merged['merged_count'] = len(group)
        
        return merged


# 全局服务实例
semantic_highlight_detector = None

def get_semantic_highlight_detector() -> SemanticHighlightDetector:
    """获取语义高光检测器实例"""
    global semantic_highlight_detector
    if semantic_highlight_detector is None:
        semantic_highlight_detector = SemanticHighlightDetector()
    return semantic_highlight_detector
