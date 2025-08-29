"""
基于ASR结果的智能选段模块
结合语义分析和视频内容分析，提供更智能的片段选择
"""

import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from .smart_clipping import SmartClippingEngine
from .semantic_analysis import get_semantic_analyzer, analyze_transcription_semantics

logger = logging.getLogger(__name__)


class ASRSmartClippingEngine:
    """基于ASR的智能切片引擎"""
    
    def __init__(self):
        """初始化ASR智能切片引擎"""
        self.smart_engine = SmartClippingEngine()
        self.semantic_analyzer = get_semantic_analyzer()
        
        # 评分权重配置
        self.scoring_weights = {
            'visual_quality': 0.30,      # 视觉质量（场景、运动、音频）
            'content_quality': 0.25,     # 内容质量（语义分析）
            'sentiment_score': 0.15,     # 情感强度
            'keyword_relevance': 0.15,   # 关键词相关性
            'topic_coherence': 0.10,     # 主题连贯性
            'timing_preference': 0.05    # 时间位置偏好
        }
        
        logger.info("ASR智能切片引擎初始化完成")
    
    def analyze_video_with_asr(self, video_path: str, transcription_result: Dict, 
                              max_duration: int = 900) -> Dict:
        """
        结合ASR结果分析视频内容
        
        Args:
            video_path: 视频文件路径
            transcription_result: ASR转录结果
            max_duration: 最大分析时长
            
        Returns:
            综合分析结果
        """
        try:
            logger.info(f"开始ASR增强视频分析: {video_path}")
            
            # 1. 基础视频内容分析
            visual_analysis = self.smart_engine.analyze_video_content(video_path, max_duration)
            
            # 2. 语义分析
            semantic_analysis = analyze_transcription_semantics(transcription_result)
            
            # 3. 时间对齐分析
            time_aligned_analysis = self._align_visual_and_semantic_analysis(
                visual_analysis, semantic_analysis
            )
            
            return {
                'video_path': video_path,
                'duration': visual_analysis['duration'],
                'visual_analysis': visual_analysis,
                'semantic_analysis': semantic_analysis,
                'time_aligned_analysis': time_aligned_analysis,
                'timestamp': visual_analysis.get('timestamp', 0)
            }
            
        except Exception as e:
            logger.error(f"ASR增强视频分析失败: {str(e)}")
            # 回退到基础分析
            return self.smart_engine.analyze_video_content(video_path, max_duration)
    
    def _align_visual_and_semantic_analysis(self, visual_analysis: Dict, 
                                          semantic_analysis: Dict) -> Dict:
        """
        对齐视觉分析和语义分析的时间轴
        
        Args:
            visual_analysis: 视觉分析结果
            semantic_analysis: 语义分析结果
            
        Returns:
            时间对齐的综合分析
        """
        try:
            duration = visual_analysis.get('duration', 0)
            scene_changes = visual_analysis.get('scene_changes', [])
            audio_energy = visual_analysis.get('audio_energy', [])
            segments = semantic_analysis.get('segments', [])
            
            # 创建时间网格（每秒一个点）
            time_grid = []
            
            for second in range(int(duration)):
                time_point = {
                    'timestamp': second,
                    'visual_features': {},
                    'semantic_features': {},
                    'combined_score': 0.0
                }
                
                # 匹配视觉特征
                # 场景变化
                scene_score = 0.0
                for scene in scene_changes:
                    if abs(scene['timestamp'] - second) <= 1.0:
                        scene_score = max(scene_score, scene['score'])
                time_point['visual_features']['scene_score'] = scene_score
                
                # 音频能量
                audio_score = 0.0
                for audio in audio_energy:
                    if abs(audio['timestamp'] - second) <= 1.0:
                        audio_score = max(audio_score, audio['energy'])
                time_point['visual_features']['audio_energy'] = audio_score
                
                # 匹配语义特征
                semantic_score = 0.0
                sentiment_score = 0.0
                keyword_count = 0
                
                for segment in segments:
                    seg_start = segment.get('start', 0)
                    seg_end = segment.get('end', 0)
                    
                    if seg_start <= second <= seg_end:
                        sem_analysis = segment.get('semantic_analysis', {})
                        quality_score = sem_analysis.get('quality_score', {})
                        sentiment = sem_analysis.get('sentiment', {})
                        keywords = sem_analysis.get('keywords', [])
                        
                        semantic_score = max(semantic_score, quality_score.get('overall_score', 0))
                        sentiment_score = max(sentiment_score, sentiment.get('intensity', 0))
                        keyword_count = max(keyword_count, len(keywords))
                
                time_point['semantic_features'] = {
                    'content_quality': semantic_score,
                    'sentiment_intensity': sentiment_score,
                    'keyword_density': min(keyword_count / 3.0, 1.0)  # 归一化到0-1
                }
                
                # 计算综合分数
                time_point['combined_score'] = self._calculate_combined_score(
                    time_point['visual_features'], 
                    time_point['semantic_features'],
                    second / duration  # 时间位置因子
                )
                
                time_grid.append(time_point)
            
            return {
                'time_grid': time_grid,
                'peak_moments': self._identify_peak_moments(time_grid),
                'quality_segments': self._identify_quality_segments(time_grid)
            }
            
        except Exception as e:
            logger.error(f"时间对齐分析失败: {str(e)}")
            return {}
    
    def _calculate_combined_score(self, visual_features: Dict, semantic_features: Dict, 
                                time_position: float) -> float:
        """计算综合评分"""
        try:
            # 视觉质量分数
            visual_score = (
                visual_features.get('scene_score', 0) * 0.4 +
                visual_features.get('audio_energy', 0) * 0.6
            )
            
            # 语义质量分数
            semantic_score = (
                semantic_features.get('content_quality', 0) * 0.5 +
                semantic_features.get('sentiment_intensity', 0) * 0.3 +
                semantic_features.get('keyword_density', 0) * 0.2
            )
            
            # 时间位置偏好（轻微偏好前半部分）
            time_preference = 1.0 - time_position * 0.2
            
            # 综合计算
            combined_score = (
                visual_score * self.scoring_weights['visual_quality'] +
                semantic_score * self.scoring_weights['content_quality'] +
                time_preference * self.scoring_weights['timing_preference']
            )
            
            return max(0.0, min(1.0, combined_score))
            
        except Exception:
            return 0.0
    
    def _identify_peak_moments(self, time_grid: List[Dict]) -> List[Dict]:
        """识别峰值时刻"""
        try:
            peak_moments = []
            
            # 找到分数的局部最大值
            for i in range(1, len(time_grid) - 1):
                current_score = time_grid[i]['combined_score']
                prev_score = time_grid[i-1]['combined_score']
                next_score = time_grid[i+1]['combined_score']
                
                # 局部最大值且超过阈值
                if (current_score > prev_score and 
                    current_score > next_score and 
                    current_score > 0.2):
                    
                    peak_moments.append({
                        'timestamp': time_grid[i]['timestamp'],
                        'score': current_score,
                        'visual_features': time_grid[i]['visual_features'],
                        'semantic_features': time_grid[i]['semantic_features'],
                        'type': 'peak_moment'
                    })
            
            # 按分数排序
            peak_moments.sort(key=lambda x: x['score'], reverse=True)
            
            return peak_moments[:10]  # 返回前10个峰值时刻
            
        except Exception as e:
            logger.error(f"峰值时刻识别失败: {str(e)}")
            return []
    
    def _identify_quality_segments(self, time_grid: List[Dict]) -> List[Dict]:
        """识别高质量连续片段"""
        try:
            quality_segments = []
            current_segment = None
            threshold = 0.2  # 进一步降低阈值，更容易找到片段
            
            for time_point in time_grid:
                score = time_point['combined_score']
                timestamp = time_point['timestamp']
                
                if score >= threshold:
                    if current_segment is None:
                        # 开始新片段
                        current_segment = {
                            'start': timestamp,
                            'end': timestamp,
                            'scores': [score],
                            'avg_score': score
                        }
                    else:
                        # 延续当前片段
                        current_segment['end'] = timestamp
                        current_segment['scores'].append(score)
                        current_segment['avg_score'] = sum(current_segment['scores']) / len(current_segment['scores'])
                else:
                    if current_segment is not None:
                        # 结束当前片段
                        duration = current_segment['end'] - current_segment['start'] + 1
                        if duration >= 3:  # 至少3秒的片段
                            current_segment['duration'] = duration
                            quality_segments.append(current_segment)
                        current_segment = None
            
            # 处理最后一个片段
            if current_segment is not None:
                duration = current_segment['end'] - current_segment['start'] + 1
                if duration >= 3:
                    current_segment['duration'] = duration
                    quality_segments.append(current_segment)
            
            # 按平均分数排序
            quality_segments.sort(key=lambda x: x['avg_score'], reverse=True)
            
            return quality_segments
            
        except Exception as e:
            logger.error(f"质量片段识别失败: {str(e)}")
            return []
    
    def select_best_segments_with_asr(self, video_path: str, transcription_result: Dict,
                                    min_duration: int, max_duration: int, 
                                    count: int = 1) -> List[Dict]:
        """
        基于ASR结果选择最佳视频片段
        
        Args:
            video_path: 视频文件路径
            transcription_result: ASR转录结果
            min_duration: 最小片段时长
            max_duration: 最大片段时长
            count: 需要选择的片段数量
            
        Returns:
            最佳片段列表
        """
        try:
            logger.info(f"开始ASR增强智能选段: {video_path}")
            
            # 1. 综合分析
            analysis = self.analyze_video_with_asr(video_path, transcription_result)
            
            # 2. 获取候选片段
            time_aligned = analysis.get('time_aligned_analysis', {})
            quality_segments = time_aligned.get('quality_segments', [])
            peak_moments = time_aligned.get('peak_moments', [])
            
            # 3. 生成候选片段
            candidates = []
            
            # 基于质量片段生成候选
            for segment in quality_segments:
                start_time = segment['start']
                duration = min(max(segment['duration'], min_duration), max_duration)
                end_time = start_time + duration
                
                candidates.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'score': segment['avg_score'],
                    'type': 'quality_segment',
                    'source': segment
                })
            
            # 基于峰值时刻生成候选
            for peak in peak_moments[:5]:  # 前5个峰值
                center_time = peak['timestamp']
                # 围绕峰值时刻生成不同长度的片段
                for duration in [min_duration, (min_duration + max_duration) // 2, max_duration]:
                    start_time = max(0, center_time - duration // 2)
                    end_time = start_time + duration
                    
                    candidates.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'score': peak['score'],
                        'type': 'peak_moment',
                        'source': peak
                    })
            
            # 如果没有找到高质量候选，生成基于转录内容的候选片段
            if len(candidates) < count:
                logger.info(f"质量片段不足({len(candidates)} < {count})，基于转录内容生成候选片段")
                segments = transcription_result.get('segments', [])
                
                for i, seg in enumerate(segments):
                    if len(candidates) >= count * 3:  # 生成足够的候选
                        break
                        
                    seg_start = seg.get('start', 0)
                    seg_text = seg.get('text', '').strip()
                    
                    if len(seg_text) > 10:  # 只考虑有意义的片段
                        # 生成以这个片段为中心的候选
                        for duration in [min_duration, max_duration]:
                            start_time = max(0, seg_start - duration // 4)
                            end_time = start_time + duration
                            
                            if end_time <= analysis.get('duration', 0):
                                candidates.append({
                                    'start_time': start_time,
                                    'end_time': end_time,
                                    'duration': duration,
                                    'score': 0.3,  # 基础分数
                                    'type': 'content_based',
                                    'source': {'segment_id': i, 'text': seg_text[:50]}
                                })
            
            # 4. 过滤和排序候选片段
            valid_candidates = []
            video_duration = analysis.get('duration', 0)
            
            for candidate in candidates:
                # 检查片段有效性
                if (candidate['end_time'] <= video_duration and 
                    min_duration <= candidate['duration'] <= max_duration):
                    
                    # 检查是否包含有意义的内容
                    content_score = self._evaluate_segment_content(
                        candidate, transcription_result
                    )
                    candidate['content_score'] = content_score
                    candidate['final_score'] = (candidate['score'] + content_score) / 2
                    
                    valid_candidates.append(candidate)
            
            # 5. 选择非重叠的最佳片段
            selected_segments = self._select_non_overlapping_segments(
                valid_candidates, count
            )
            
            # 6. 格式化输出
            final_segments = []
            for segment in selected_segments:
                final_segments.append({
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'duration': segment['duration'],
                    'total_score': segment['final_score'],
                    'visual_score': segment['score'],
                    'content_score': segment['content_score'],
                    'type': segment['type'],
                    'start_hms': self._seconds_to_hms(segment['start_time']),
                    'end_hms': self._seconds_to_hms(segment['end_time']),
                    'asr_enhanced': True
                })
            
            logger.info(f"ASR增强选段完成，选择了 {len(final_segments)} 个片段")
            return final_segments
            
        except Exception as e:
            logger.error(f"ASR增强选段失败: {str(e)}")
            # 回退到基础智能选段
            return self.smart_engine.select_best_segments(
                video_path, min_duration, max_duration, count
            )
    
    def _evaluate_segment_content(self, segment: Dict, transcription_result: Dict) -> float:
        """评估片段的内容质量"""
        try:
            start_time = segment['start_time']
            end_time = segment['end_time']
            
            segments = transcription_result.get('segments', [])
            relevant_segments = []
            
            # 找到时间范围内的转录片段
            for seg in segments:
                seg_start = seg.get('start', 0)
                seg_end = seg.get('end', 0)
                
                # 检查时间重叠
                if not (seg_end < start_time or seg_start > end_time):
                    relevant_segments.append(seg)
            
            if not relevant_segments:
                return 0.0
            
            # 计算内容质量分数
            total_score = 0.0
            total_weight = 0.0
            
            for seg in relevant_segments:
                semantic_analysis = seg.get('semantic_analysis', {})
                quality_score = semantic_analysis.get('quality_score', {})
                
                # 片段权重基于时间重叠度
                seg_start = max(seg.get('start', 0), start_time)
                seg_end = min(seg.get('end', 0), end_time)
                overlap_duration = max(0, seg_end - seg_start)
                weight = overlap_duration / segment['duration']
                
                score = quality_score.get('overall_score', 0)
                total_score += score * weight
                total_weight += weight
            
            return total_score / max(total_weight, 0.001)
            
        except Exception as e:
            logger.error(f"片段内容评估失败: {str(e)}")
            return 0.0
    
    def _select_non_overlapping_segments(self, candidates: List[Dict], count: int) -> List[Dict]:
        """选择非重叠的最佳片段"""
        if not candidates:
            return []
        
        # 按最终分数排序
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        selected = []
        used_ranges = []
        
        for candidate in candidates:
            start, end = candidate['start_time'], candidate['end_time']
            
            # 检查是否与已选片段重叠
            overlap = False
            for used_start, used_end in used_ranges:
                if not (end <= used_start or start >= used_end):
                    overlap = True
                    break
            
            if not overlap:
                selected.append(candidate)
                used_ranges.append((start, end))
                
                if len(selected) >= count:
                    break
        
        return selected
    
    def _seconds_to_hms(self, seconds: int) -> str:
        """将秒转换为HH:MM:SS格式"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# 全局ASR智能切片引擎实例
asr_smart_engine = None

def get_asr_smart_engine() -> ASRSmartClippingEngine:
    """获取全局ASR智能切片引擎实例"""
    global asr_smart_engine
    
    if asr_smart_engine is None:
        asr_smart_engine = ASRSmartClippingEngine()
    
    return asr_smart_engine


def select_segments_with_asr(video_path: str, transcription_result: Dict,
                           min_duration: int, max_duration: int, 
                           count: int = 1) -> List[Dict]:
    """
    便捷函数：基于ASR结果选择最佳视频片段
    
    Args:
        video_path: 视频文件路径
        transcription_result: ASR转录结果
        min_duration: 最小片段时长
        max_duration: 最大片段时长
        count: 片段数量
        
    Returns:
        最佳片段列表
    """
    engine = get_asr_smart_engine()
    return engine.select_best_segments_with_asr(
        video_path, transcription_result, min_duration, max_duration, count
    )
