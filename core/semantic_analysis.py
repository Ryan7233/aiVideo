"""
语义分析模块
提供基于ASR转录结果的智能分析功能，包括关键词提取、情感分析、主题相关性评分等
"""

import math
import logging
from typing import List, Dict, Tuple, Set
from collections import Counter

from core.text_tokenizer import content_tokens, scan_lexicon, split_sentences, tokenize

from core.degradation import mark_degraded

logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    """语义分析器"""
    
    def __init__(self):
        """初始化语义分析器"""
        self.stop_words = self._load_stop_words()
        self.emotion_keywords = self._load_emotion_keywords()
        self.topic_keywords = self._load_topic_keywords()
        self.importance_modifiers = self._load_importance_modifiers()
        # Flatten the per-polarity dictionaries once so a single longest-match
        # scan can resolve overlapping entries such as 好 / 很好 / 不好.
        self._emotion_lexicon: Dict[str, Tuple[str, float]] = {}
        for polarity, entries in self.emotion_keywords.items():
            for term, weight in entries.items():
                self._emotion_lexicon[term] = (polarity, weight)
        self._emotion_weights = {term: weight for term, (_, weight) in self._emotion_lexicon.items()}

        logger.info("语义分析器初始化完成")
    
    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        # 英文常见停用词
        english_stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'i', 'you', 'we', 'they', 'this',
            'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
            'should', 'now', 'am', 'or', 'but', 'if', 'then', 'else'
        }
        
        # 中文常见停用词
        chinese_stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有',
            '看', '好', '自己', '这', '那', '个', '们', '他', '她', '它', '我们',
            '你们', '他们', '她们', '它们', '这个', '那个', '这些', '那些', '什么',
            '怎么', '为什么', '哪里', '哪个', '如何', '因为', '所以', '但是', '然后',
            '现在', '已经', '还是', '可以', '应该', '能够', '需要', '想要', '知道',
            '觉得', '认为', '发现', '开始', '结束', '继续', '停止', '完成'
        }

        # ASR transcripts are full of these; without them filler-only speech
        # scores as legitimate content.
        chinese_filler_words = {
            '嗯', '呃', '啊', '哦', '噢', '呀', '吧', '呢', '嘛', '哈', '哎',
            '就是', '然后', '那个', '这个', '就是说', '其实', '反正', '真的',
            '一下', '一点', '有点', '比较', '可能', '大概', '差不多', '怎么说',
            '你知道', '对不对', '是不是', '这样子', '那么'
        }

        return english_stop_words | chinese_stop_words | chinese_filler_words
    
    def _load_emotion_keywords(self) -> Dict[str, Dict[str, float]]:
        """加载情感关键词字典"""
        return {
            'positive': {
                # 积极情感词汇
                'amazing': 0.9, 'awesome': 0.8, 'excellent': 0.8, 'fantastic': 0.9,
                'great': 0.7, 'good': 0.6, 'wonderful': 0.8, 'perfect': 0.9,
                'love': 0.8, 'like': 0.6, 'enjoy': 0.7, 'happy': 0.7,
                'excited': 0.8, 'thrilled': 0.9, 'delighted': 0.8, 'pleased': 0.7,
                'success': 0.8, 'achievement': 0.7, 'victory': 0.8, 'win': 0.7,
                'beautiful': 0.7, 'brilliant': 0.8, 'incredible': 0.8, 'outstanding': 0.8,
                # 中文积极词汇
                '好': 0.6, '很好': 0.7, '非常好': 0.8, '棒': 0.7, '太棒了': 0.9,
                '喜欢': 0.7, '爱': 0.8, '开心': 0.7, '高兴': 0.7, '兴奋': 0.8,
                '成功': 0.8, '胜利': 0.8, '完美': 0.9, '优秀': 0.8, '杰出': 0.8
            },
            'negative': {
                # 消极情感词汇
                'terrible': -0.8, 'awful': -0.8, 'horrible': -0.9, 'bad': -0.6,
                'worst': -0.9, 'hate': -0.8, 'dislike': -0.6, 'angry': -0.7,
                'sad': -0.7, 'disappointed': -0.7, 'frustrated': -0.7, 'annoyed': -0.6,
                'problem': -0.5, 'issue': -0.5, 'trouble': -0.6, 'difficult': -0.5,
                'hard': -0.4, 'challenging': -0.3, 'struggle': -0.6, 'fail': -0.7,
                'failure': -0.8, 'wrong': -0.5, 'mistake': -0.5, 'error': -0.5,
                # 中文消极词汇
                '不好': -0.6, '糟糕': -0.8, '差': -0.6, '讨厌': -0.7, '生气': -0.7,
                '难过': -0.7, '失望': -0.7, '问题': -0.5, '困难': -0.5, '失败': -0.8
            },
            'neutral': {
                # 中性词汇
                'okay': 0.0, 'fine': 0.0, 'normal': 0.0, 'regular': 0.0,
                'standard': 0.0, 'typical': 0.0, 'usual': 0.0, 'common': 0.0,
                # 中文中性词汇
                '还行': 0.0, '一般': 0.0, '普通': 0.0, '正常': 0.0
            }
        }
    
    def _load_topic_keywords(self) -> Dict[str, List[str]]:
        """加载主题关键词"""
        return {
            'technology': [
                'ai', 'artificial', 'intelligence', 'machine', 'learning', 'deep',
                'neural', 'network', 'algorithm', 'data', 'science', 'computer',
                'software', 'programming', 'code', 'development', 'tech', 'digital',
                'internet', 'web', 'app', 'application', 'system', 'platform',
                '人工智能', '机器学习', '深度学习', '算法', '数据', '编程', '代码',
                '技术', '科技', '互联网', '应用', '系统', '平台'
            ],
            'business': [
                'business', 'company', 'market', 'marketing', 'sales', 'revenue',
                'profit', 'growth', 'strategy', 'management', 'leadership', 'team',
                'project', 'client', 'customer', 'service', 'product', 'brand',
                'investment', 'finance', 'money', 'cost', 'budget', 'economy',
                '商业', '公司', '市场', '营销', '销售', '收入', '利润', '增长',
                '策略', '管理', '领导', '团队', '项目', '客户', '服务', '产品'
            ],
            'education': [
                'education', 'learning', 'study', 'student', 'teacher', 'school',
                'university', 'college', 'course', 'lesson', 'training', 'knowledge',
                'skill', 'tutorial', 'guide', 'instruction', 'research', 'academic',
                '教育', '学习', '学生', '老师', '学校', '大学', '课程', '培训',
                '知识', '技能', '教程', '指导', '研究', '学术'
            ],
            'entertainment': [
                'entertainment', 'fun', 'game', 'play', 'music', 'movie', 'video',
                'show', 'comedy', 'humor', 'joke', 'laugh', 'enjoy', 'party',
                'celebration', 'festival', 'event', 'performance', 'art', 'creative',
                '娱乐', '有趣', '游戏', '音乐', '电影', '视频', '节目', '喜剧',
                '幽默', '笑话', '派对', '庆祝', '节日', '表演', '艺术', '创意'
            ],
            'health': [
                'health', 'fitness', 'exercise', 'workout', 'diet', 'nutrition',
                'medical', 'doctor', 'hospital', 'medicine', 'treatment', 'therapy',
                'wellness', 'mental', 'physical', 'body', 'mind', 'stress', 'relax',
                '健康', '健身', '锻炼', '饮食', '营养', '医疗', '医生', '医院',
                '药物', '治疗', '身体', '心理', '压力', '放松'
            ],
            'food': [
                'food', 'restaurant', 'cafe', 'coffee', 'latte', 'dessert', 'brunch',
                'taste', 'delicious', 'menu', 'dish', 'cuisine', 'bakery', 'snack',
                '美食', '好吃', '好喝', '餐厅', '咖啡', '拿铁', '甜品', '探店',
                '菜单', '口味', '味道', '食材', '烘焙', '早午餐', '小吃', '打卡'
            ],
            'travel': [
                'travel', 'trip', 'journey', 'hotel', 'flight', 'itinerary', 'city',
                'scenery', 'attraction', 'checkin', 'weekend', 'vacation', 'resort',
                '旅行', '旅游', '攻略', '行程', '酒店', '民宿', '机票', '景点',
                '风景', '路线', '周末', '度假', '出行', '拍照', '出片'
            ],
            'beauty': [
                'beauty', 'makeup', 'skincare', 'cosmetic', 'lipstick', 'foundation',
                'serum', 'moisturizer', 'fragrance', 'outfit', 'fashion', 'style',
                '美妆', '护肤', '化妆', '口红', '粉底', '精华', '面膜', '香水',
                '穿搭', '时尚', '妆容', '成分', '平价', '种草'
            ],
            'lifestyle': [
                'lifestyle', 'home', 'decor', 'routine', 'habit', 'organize', 'cozy',
                'vlog', 'daily', 'share', 'recommend', 'review', 'budget',
                '生活', '日常', '家居', '布置', '收纳', '习惯', '好物', '推荐',
                '分享', '测评', '性价比', '踩雷', '避雷', '氛围感'
            ]
        }
    
    def _load_importance_modifiers(self) -> Dict[str, float]:
        """加载重要性修饰词"""
        return {
            # 强调词
            'very': 1.2, 'extremely': 1.5, 'incredibly': 1.4, 'absolutely': 1.3,
            'totally': 1.2, 'completely': 1.3, 'really': 1.1, 'truly': 1.2,
            'highly': 1.2, 'super': 1.3, 'ultra': 1.4, 'mega': 1.3,
            # 疑问和否定
            'not': -0.8, 'never': -0.9, 'no': -0.7, 'nothing': -0.8,
            'nobody': -0.7, 'none': -0.7, 'neither': -0.6, 'barely': -0.5,
            # 中文修饰词
            '非常': 1.3, '特别': 1.2, '很': 1.1, '超': 1.3, '极': 1.4,
            '不': -0.8, '没': -0.7, '从不': -0.9, '绝不': -0.9
        }
    
    def _modifier_factor(self, sentence: str) -> float:
        """Turn the emphasis/negation words in one sentence into a multiplier."""
        factor = 1.0
        for _, weight in scan_lexicon(sentence, self.importance_modifiers):
            # Positive entries amplify (很 -> 1.1); negative entries are
            # negations, which dampen importance rather than invert it.
            factor *= weight if weight > 0 else max(0.2, 1.0 + weight)
        return max(0.2, min(factor, 2.0))

    def extract_keywords(self, text: str, top_k: int = 10) -> List[Dict[str, any]]:
        """
        提取关键词

        Args:
            text: 输入文本
            top_k: 返回前k个关键词

        Returns:
            关键词列表，包含词汇、频率、重要性分数
        """
        try:
            words = content_tokens(tokenize(text), self.stop_words)
            if not words:
                return []

            total_words = len(words)
            word_freq = Counter(words)

            # Each sentence carries its own emphasis; a word inherits the
            # strongest emphasis of any sentence it appears in.
            modifiers: Dict[str, float] = {}
            for sentence in split_sentences(text):
                factor = self._modifier_factor(sentence)
                for token in set(content_tokens(tokenize(sentence), self.stop_words)):
                    modifiers[token] = max(modifiers.get(token, 0.0), factor)

            keywords = []
            for word, freq in word_freq.items():
                tf = freq / total_words
                # Single-document IDF: log(1 + N/f) so a hapax still scores
                # above zero. The old log(N/f) collapsed to 0 whenever every
                # token was unique, which is the normal case for one segment.
                idf = math.log(1 + total_words / freq)
                importance_modifier = modifiers.get(word, 1.0)

                keywords.append({
                    'word': word,
                    'frequency': freq,
                    'tf': tf,
                    'score': tf * idf * importance_modifier,
                    'importance_modifier': importance_modifier
                })

            keywords.sort(key=lambda x: x['score'], reverse=True)

            return keywords[:top_k]

        except Exception as e:
            logger.error(f"关键词提取失败: {str(e)}")
            return []

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        情感分析

        Args:
            text: 输入文本

        Returns:
            情感分析结果：positive, negative, neutral scores
        """
        try:
            sentiment_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            total_sentiment_words = 0

            # Longest-match scan instead of token lookup: Chinese sentiment
            # words are substrings of a continuous run, and 不好 has to win
            # over 好 rather than both firing.
            for term, weight in scan_lexicon(text, self._emotion_weights):
                polarity, _ = self._emotion_lexicon[term]
                sentiment_scores[polarity] += weight
                total_sentiment_words += 1

            # 归一化分数
            if total_sentiment_words > 0:
                for sentiment_type in sentiment_scores:
                    sentiment_scores[sentiment_type] /= total_sentiment_words

            # 计算总体情感倾向
            overall_sentiment = (
                sentiment_scores['positive'] +
                sentiment_scores['negative'] +
                sentiment_scores['neutral']
            )

            # 计算情感强度
            sentiment_intensity = abs(sentiment_scores['positive']) + abs(sentiment_scores['negative'])

            return {
                'positive': max(0, sentiment_scores['positive']),
                'negative': abs(min(0, sentiment_scores['negative'])),
                'neutral': sentiment_scores['neutral'],
                'overall': overall_sentiment,
                'intensity': sentiment_intensity,
                'dominant': (
                    max(sentiment_scores.items(), key=lambda x: abs(x[1]))[0]
                    if sentiment_intensity > 0 else 'neutral'
                )
            }

        except Exception as e:
            logger.error(f"情感分析失败: {str(e)}")
            return mark_degraded({'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'overall': 0.0, 'intensity': 0.0, 'dominant': 'neutral'}, e, logger=logger, context='analyze_sentiment')

    def analyze_topic_relevance(self, text: str) -> Dict[str, float]:
        """
        主题相关性分析

        Args:
            text: 输入文本

        Returns:
            各主题的相关性分数
        """
        try:
            topic_scores = {}

            for topic, keywords in self.topic_keywords.items():
                if not keywords:
                    topic_scores[topic] = 0.0
                    continue

                lexicon = {keyword: 1.0 for keyword in keywords}
                matched = {term for term, _ in scan_lexicon(text, lexicon)}
                # Saturating rather than matched/len(keywords): a bucket holds
                # ~37 synonyms, so dividing by all of them capped an obviously
                # on-topic segment at about 0.08.
                topic_scores[topic] = min(1.0, len(matched) / 3.0)

            return topic_scores

        except Exception as e:
            logger.error(f"主题相关性分析失败: {str(e)}")
            return {topic: 0.0 for topic in self.topic_keywords.keys()}

    def calculate_content_quality_score(self, text: str, duration: float = 0) -> Dict[str, float]:
        """
        计算内容质量综合分数
        
        Args:
            text: 文本内容
            duration: 时长（秒）
            
        Returns:
            内容质量评分
        """
        try:
            # 基础指标
            tokens = tokenize(text)
            meaningful = content_tokens(tokens, self.stop_words)
            word_count = len(tokens)
            sentence_count = max(1, len(split_sentences(text)))

            # 情感分析
            sentiment = self.analyze_sentiment(text)

            # 主题相关性
            topic_relevance = self.analyze_topic_relevance(text)
            max_topic_score = max(topic_relevance.values()) if topic_relevance else 0

            # 计算各维度分数
            scores = {}

            # 1. 内容丰富度 (0-1)
            if duration > 0:
                words_per_second = word_count / duration
                scores['content_density'] = min(words_per_second / 3.0, 1.0)  # 假设3词/秒为满分
            else:
                scores['content_density'] = min(word_count / 100.0, 1.0)  # 假设100词为满分

            # 2. 词汇多样性 (0-1)
            # Two halves: how much of the speech is actually content rather
            # than stop words and filler, and how varied that content is.
            # The old formula was keyword_diversity / (word_count * 0.1),
            # which saturated at 1.0 for every input.
            if word_count > 0 and meaningful:
                content_ratio = len(meaningful) / word_count
                type_token_ratio = len(set(meaningful)) / len(meaningful)
                scores['vocabulary_diversity'] = min(
                    1.0, content_ratio * 0.5 + type_token_ratio * 0.5
                )
            else:
                scores['vocabulary_diversity'] = 0.0

            # 3. 情感强度 (0-1)
            scores['emotional_intensity'] = min(sentiment['intensity'], 1.0)

            # 4. 主题相关性 (0-1)
            scores['topic_relevance'] = max_topic_score

            # 5. 结构完整性 (0-1)
            if word_count > 0:
                avg_sentence_length = word_count / sentence_count
                scores['structure_quality'] = min(avg_sentence_length / 15.0, 1.0)  # 假设15词/句为理想
            else:
                scores['structure_quality'] = 0.0

            # 6. 积极性 (0-1)
            scores['positivity'] = sentiment['positive']

            # 综合分数计算
            weights = {
                'content_density': 0.25,
                'vocabulary_diversity': 0.20,
                'emotional_intensity': 0.15,
                'topic_relevance': 0.20,
                'structure_quality': 0.10,
                'positivity': 0.10
            }
            
            overall_score = sum(scores[key] * weights[key] for key in scores)
            
            return {
                'overall_score': overall_score,
                'content_density': scores['content_density'],
                'vocabulary_diversity': scores['vocabulary_diversity'],
                'emotional_intensity': scores['emotional_intensity'],
                'topic_relevance': scores['topic_relevance'],
                'structure_quality': scores['structure_quality'],
                'positivity': scores['positivity'],
                'word_count': word_count,
                'sentence_count': sentence_count,
                'dominant_topic': (
                    max(topic_relevance.items(), key=lambda x: x[1])[0]
                    if max_topic_score > 0 else 'unknown'
                ),
                'dominant_sentiment': sentiment['dominant']
            }
            
        except Exception as e:
            logger.error(f"内容质量分析失败: {str(e)}")
            return mark_degraded({'overall_score': 0.0, 'content_density': 0.0, 'vocabulary_diversity': 0.0, 'emotional_intensity': 0.0, 'topic_relevance': 0.0, 'structure_quality': 0.0, 'positivity': 0.0, 'word_count': 0, 'sentence_count': 0, 'dominant_topic': 'unknown', 'dominant_sentiment': 'neutral'}, e, logger=logger, context='calculate_content_quality_score')
    
    def analyze_transcript_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        分析转录段落
        
        Args:
            segments: Whisper转录段落列表
            
        Returns:
            带有语义分析结果的段落列表
        """
        try:
            analyzed_segments = []
            
            for segment in segments:
                text = segment.get('text', '').strip()
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                duration = end_time - start_time
                
                if not text:
                    continue
                
                # 执行语义分析
                keywords = self.extract_keywords(text, top_k=3)
                sentiment = self.analyze_sentiment(text)
                topic_relevance = self.analyze_topic_relevance(text)
                quality_score = self.calculate_content_quality_score(text, duration)
                
                # 构建分析结果
                analyzed_segment = {
                    **segment,  # 保留原始段落信息
                    'semantic_analysis': {
                        'keywords': keywords,
                        'sentiment': sentiment,
                        'topic_relevance': topic_relevance,
                        'quality_score': quality_score,
                        'duration': duration
                    }
                }
                
                analyzed_segments.append(analyzed_segment)
            
            return analyzed_segments
            
        except Exception as e:
            logger.error(f"转录段落分析失败: {str(e)}")
            return segments  # 返回原始段落


# 全局语义分析器实例
semantic_analyzer = None

def get_semantic_analyzer() -> SemanticAnalyzer:
    """获取全局语义分析器实例"""
    global semantic_analyzer
    
    if semantic_analyzer is None:
        semantic_analyzer = SemanticAnalyzer()
    
    return semantic_analyzer


def analyze_transcription_semantics(transcription_result: Dict) -> Dict:
    """
    分析转录结果的语义信息
    
    Args:
        transcription_result: Whisper转录结果
        
    Returns:
        包含语义分析的转录结果
    """
    analyzer = get_semantic_analyzer()
    
    # 分析完整文本
    full_text = transcription_result.get('full_text', '')
    duration = transcription_result.get('duration', 0)
    
    # 完整文本分析
    full_text_analysis = {
        'keywords': analyzer.extract_keywords(full_text, top_k=10),
        'sentiment': analyzer.analyze_sentiment(full_text),
        'topic_relevance': analyzer.analyze_topic_relevance(full_text),
        'quality_score': analyzer.calculate_content_quality_score(full_text, duration)
    }
    
    # 段落级分析
    segments = transcription_result.get('segments', [])
    analyzed_segments = analyzer.analyze_transcript_segments(segments)
    
    # 构建增强的转录结果
    enhanced_result = {
        **transcription_result,
        'semantic_analysis': full_text_analysis,
        'segments': analyzed_segments
    }
    
    return enhanced_result
