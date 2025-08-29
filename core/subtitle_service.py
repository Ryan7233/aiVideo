"""
字幕生成与断句服务
支持SRT/ASS格式，表情符号，双语字幕
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import timedelta
import random

logger = logging.getLogger(__name__)


class SubtitleGenerator:
    """字幕生成器"""
    
    def __init__(self):
        self.emoji_mapping = {
            # 情感表情
            '开心': ['😊', '😄', '🥰', '😍'],
            '惊讶': ['😱', '😲', '🤩', '😯'],
            '喜欢': ['❤️', '💕', '👍', '🥰'],
            '美味': ['😋', '🤤', '👌', '💯'],
            '美丽': ['✨', '🌟', '💖', '😍'],
            '累': ['😴', '😪', '🥱'],
            '贵': ['💸', '💰', '😭'],
            '便宜': ['💰', '👍', '😊'],
            # 物品表情
            '食物': ['🍜', '🍱', '🥘', '🍲'],
            '风景': ['🏞️', '🌅', '📸', '🌸'],
            '建筑': ['🏛️', '🏰', '🕌', '🗼'],
            '交通': ['🚇', '🚌', '🚗', '🚶‍♀️']
        }
        
        self.punctuation_pause = {
            '，': 0.3,
            '。': 0.5,
            '！': 0.4,
            '？': 0.4,
            '；': 0.4,
            '：': 0.3
        }
    
    def generate_subtitles(self, clips: List[Dict], transcript_mmss: List[Dict], 
                          style: str = "口语") -> Dict:
        """
        生成字幕文件
        
        Args:
            clips: 视频片段信息
            transcript_mmss: 带时间戳的转录文本
            style: 字幕风格（口语/书面/可爱）
            
        Returns:
            字幕数据和文件路径
        """
        try:
            subtitles_data = []
            
            for clip_idx, clip in enumerate(clips):
                clip_start = clip.get('start_time_seconds', 0)
                clip_end = clip.get('end_time_seconds', 0)
                
                # 找到时间范围内的转录片段
                relevant_segments = self._find_relevant_segments(
                    transcript_mmss, clip_start, clip_end
                )
                
                # 生成该片段的字幕
                clip_subtitles = self._generate_clip_subtitles(
                    relevant_segments, clip_start, style, clip_idx
                )
                
                subtitles_data.append({
                    'clip_index': clip_idx,
                    'clip_start': clip_start,
                    'clip_end': clip_end,
                    'subtitles': clip_subtitles,
                    'subtitle_count': len(clip_subtitles)
                })
            
            # 生成字幕文件
            srt_files = self._generate_srt_files(subtitles_data)
            ass_files = self._generate_ass_files(subtitles_data, style)
            
            return {
                'subtitles_data': subtitles_data,
                'srt_files': srt_files,
                'ass_files': ass_files,
                'total_clips': len(clips),
                'total_subtitles': sum(len(sub['subtitles']) for sub in subtitles_data),
                'style': style
            }
            
        except Exception as e:
            logger.error(f"字幕生成失败: {str(e)}")
            return {'error': str(e)}
    
    def _find_relevant_segments(self, transcript_mmss: List[Dict], 
                               start_time: float, end_time: float) -> List[Dict]:
        """找到时间范围内的转录片段"""
        relevant = []
        
        for segment in transcript_mmss:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            # 检查时间重叠
            if not (seg_end < start_time or seg_start > end_time):
                # 调整时间戳为相对于片段开始的时间
                adjusted_segment = segment.copy()
                adjusted_segment['start'] = max(0, seg_start - start_time)
                adjusted_segment['end'] = min(end_time - start_time, seg_end - start_time)
                relevant.append(adjusted_segment)
        
        return relevant
    
    def _generate_clip_subtitles(self, segments: List[Dict], clip_start: float, 
                                style: str, clip_idx: int) -> List[Dict]:
        """生成单个片段的字幕"""
        subtitles = []
        subtitle_id = 1
        
        for segment in segments:
            text = segment.get('text', '').strip()
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            
            if not text:
                continue
            
            # 根据风格处理文本
            processed_text = self._process_text_by_style(text, style)
            
            # 断句处理
            sentences = self._split_sentences(processed_text)
            
            # 为每个句子生成字幕
            sentence_duration = (end_time - start_time) / max(len(sentences), 1)
            
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                
                sub_start = start_time + i * sentence_duration
                sub_end = min(start_time + (i + 1) * sentence_duration, end_time)
                
                # 添加表情符号
                if style == "可爱":
                    sentence = self._add_emoji_to_text(sentence)
                
                subtitles.append({
                    'id': subtitle_id,
                    'start_time': sub_start,
                    'end_time': sub_end,
                    'start_timecode': self._seconds_to_timecode(sub_start),
                    'end_timecode': self._seconds_to_timecode(sub_end),
                    'text': sentence,
                    'original_text': text,
                    'style': style
                })
                
                subtitle_id += 1
        
        return subtitles
    
    def _process_text_by_style(self, text: str, style: str) -> str:
        """根据风格处理文本"""
        if style == "书面":
            # 书面语处理：规范化表达
            text = re.sub(r'嗯+', '嗯', text)
            text = re.sub(r'啊+', '啊', text)
            text = re.sub(r'呃+', '', text)
            text = re.sub(r'\s+', ' ', text)
        elif style == "口语":
            # 保持口语化特征
            pass
        elif style == "可爱":
            # 可爱风格：添加语气词
            text = re.sub(r'([。！？])$', r'\1~', text)
        
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """智能断句"""
        # 按标点符号分句
        sentences = re.split(r'([。！？；])', text)
        
        # 重新组合句子和标点
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ''
            
            if sentence.strip():
                combined = sentence + punctuation
                
                # 如果句子太长，进一步分割
                if len(combined) > 20:
                    sub_sentences = self._split_long_sentence(combined)
                    result.extend(sub_sentences)
                else:
                    result.append(combined)
        
        # 处理最后一个句子（如果没有标点）
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            last_sentence = sentences[-1]
            if len(last_sentence) > 20:
                result.extend(self._split_long_sentence(last_sentence))
            else:
                result.append(last_sentence)
        
        return [s.strip() for s in result if s.strip()]
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """分割长句子"""
        # 按逗号分割
        parts = sentence.split('，')
        
        result = []
        current_part = ""
        
        for part in parts:
            if len(current_part + part) <= 15:
                current_part += part + "，"
            else:
                if current_part:
                    result.append(current_part.rstrip('，'))
                current_part = part + "，"
        
        if current_part:
            result.append(current_part.rstrip('，'))
        
        return result
    
    def _add_emoji_to_text(self, text: str) -> str:
        """为文本添加表情符号"""
        try:
            # 根据关键词添加表情
            for keyword, emojis in self.emoji_mapping.items():
                if any(word in text for word in keyword.split()):
                    emoji = random.choice(emojis)
                    text = text + emoji
                    break
            
            return text
        except Exception:
            return text
    
    def _seconds_to_timecode(self, seconds: float) -> str:
        """将秒转换为时间码格式"""
        td = timedelta(seconds=seconds)
        hours = int(td.seconds // 3600)
        minutes = int((td.seconds % 3600) // 60)
        secs = int(td.seconds % 60)
        milliseconds = int(td.microseconds // 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _generate_srt_files(self, subtitles_data: List[Dict]) -> List[str]:
        """生成SRT字幕文件"""
        srt_files = []
        
        for clip_data in subtitles_data:
            clip_idx = clip_data['clip_index']
            subtitles = clip_data['subtitles']
            
            srt_content = []
            
            for subtitle in subtitles:
                srt_content.append(f"{subtitle['id']}")
                srt_content.append(f"{subtitle['start_timecode']} --> {subtitle['end_timecode']}")
                srt_content.append(subtitle['text'])
                srt_content.append("")  # 空行
            
            # 保存文件
            srt_filename = f"output_data/subtitles_clip_{clip_idx:02d}.srt"
            Path("output_data").mkdir(parents=True, exist_ok=True)
            
            with open(srt_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(srt_content))
            
            srt_files.append(srt_filename)
        
        return srt_files
    
    def _generate_ass_files(self, subtitles_data: List[Dict], style: str) -> List[str]:
        """生成ASS高级字幕文件"""
        ass_files = []
        
        # ASS样式定义
        style_definitions = {
            "口语": {
                "fontsize": "20",
                "color": "&H00FFFFFF",  # 白色
                "outline": "2",
                "shadow": "0"
            },
            "书面": {
                "fontsize": "18",
                "color": "&H00FFFF00",  # 黄色
                "outline": "1",
                "shadow": "1"
            },
            "可爱": {
                "fontsize": "22",
                "color": "&H00FF69B4",  # 粉色
                "outline": "2",
                "shadow": "0"
            }
        }
        
        current_style = style_definitions.get(style, style_definitions["口语"])
        
        for clip_data in subtitles_data:
            clip_idx = clip_data['clip_index']
            subtitles = clip_data['subtitles']
            
            ass_content = [
                "[Script Info]",
                "Title: AI Video Clipper Subtitles",
                "ScriptType: v4.00+",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                f"Style: Default,Arial,{current_style['fontsize']},{current_style['color']},&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,{current_style['outline']},{current_style['shadow']},2,10,10,10,1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
            ]
            
            for subtitle in subtitles:
                start_time = self._seconds_to_ass_time(subtitle['start_time'])
                end_time = self._seconds_to_ass_time(subtitle['end_time'])
                text = subtitle['text']
                
                ass_content.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}")
            
            # 保存文件
            ass_filename = f"output_data/subtitles_clip_{clip_idx:02d}.ass"
            
            with open(ass_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(ass_content))
            
            ass_files.append(ass_filename)
        
        return ass_files
    
    def _seconds_to_ass_time(self, seconds: float) -> str:
        """将秒转换为ASS时间格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds % 1) * 100)
        
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


class CoverSuggestionService:
    """封面建议服务"""
    
    def __init__(self):
        self.title_styles = {
            "简约": {
                "font": "Arial",
                "color": "#FFFFFF",
                "background": "rgba(0,0,0,0.5)",
                "position": "bottom"
            },
            "活泼": {
                "font": "Comic Sans MS",
                "color": "#FF69B4",
                "background": "rgba(255,255,255,0.8)",
                "position": "top"
            },
            "专业": {
                "font": "Helvetica",
                "color": "#333333",
                "background": "rgba(255,255,255,0.9)",
                "position": "center"
            }
        }
    
    def suggest_cover(self, clips: List[Dict], photos_topk: List[Dict], 
                     title: str = "") -> Dict:
        """
        生成封面建议
        
        Args:
            clips: 视频片段信息
            photos_topk: 排序后的照片
            title: 标题文本
            
        Returns:
            封面建议信息
        """
        try:
            suggestions = []
            
            # 从视频帧中选择封面
            for clip in clips[:3]:  # 只考虑前3个片段
                frame_suggestions = self._analyze_video_frames(clip, title)
                suggestions.extend(frame_suggestions)
            
            # 从照片中选择封面
            for photo in photos_topk[:5]:  # 只考虑前5张照片
                photo_suggestion = self._analyze_photo_cover(photo, title)
                suggestions.append(photo_suggestion)
            
            # 按评分排序
            suggestions.sort(key=lambda x: x['score'], reverse=True)
            
            # 选择最佳建议
            best_suggestion = suggestions[0] if suggestions else self._get_fallback_cover()
            
            return {
                'best_cover': best_suggestion,
                'alternatives': suggestions[1:4],  # 提供3个备选
                'title_overlay': self._generate_title_overlay(title, best_suggestion),
                'color_palette': self._extract_color_palette(best_suggestion),
                'total_suggestions': len(suggestions)
            }
            
        except Exception as e:
            logger.error(f"封面建议生成失败: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_video_frames(self, clip: Dict, title: str) -> List[Dict]:
        """分析视频帧作为封面"""
        suggestions = []
        
        start_time = clip.get('start_time_seconds', 0)
        duration = clip.get('duration', 0)
        clip_path = clip.get('output_path', '')
        
        # 选择关键帧时间点
        key_moments = [
            start_time + duration * 0.1,   # 开始后10%
            start_time + duration * 0.3,   # 30%处
            start_time + duration * 0.5,   # 中间
            start_time + duration * 0.7,   # 70%处
        ]
        
        for i, moment in enumerate(key_moments):
            suggestions.append({
                'type': 'video_frame',
                'source': clip_path,
                'timecode': self._seconds_to_timecode(moment - start_time),
                'timestamp': moment,
                'score': self._calculate_frame_score(moment, duration, i),
                'description': f'视频第{i+1}个关键帧',
                'extraction_command': f'ffmpeg -i "{clip_path}" -ss {moment-start_time:.2f} -vframes 1 -q:v 2'
            })
        
        return suggestions
    
    def _analyze_photo_cover(self, photo: Dict, title: str) -> Dict:
        """分析照片作为封面"""
        return {
            'type': 'photo',
            'source': photo['path'],
            'photo_id': photo['photo_id'],
            'score': photo['score'] * 0.9,  # 照片分数稍微降低
            'description': f'精选照片 (排名#{photo["rank"]})',
            'tags': photo.get('tags', []),
            'aesthetic_score': photo.get('aesthetic_score', 0.5)
        }
    
    def _calculate_frame_score(self, timestamp: float, total_duration: float, 
                              frame_index: int) -> float:
        """计算视频帧评分"""
        # 基础分数
        base_score = 0.6
        
        # 位置偏好（中间位置分数更高）
        position_factor = 1.0 - abs(0.5 - timestamp / total_duration)
        base_score += position_factor * 0.3
        
        # 避免开头和结尾
        if timestamp < total_duration * 0.1 or timestamp > total_duration * 0.9:
            base_score -= 0.2
        
        # 多样性加分
        diversity_bonus = frame_index * 0.05
        base_score += diversity_bonus
        
        return min(base_score, 1.0)
    
    def _generate_title_overlay(self, title: str, cover_suggestion: Dict) -> Dict:
        """生成标题叠加建议"""
        # 根据封面类型选择风格
        if cover_suggestion['type'] == 'photo':
            style_name = "简约"
        else:
            style_name = "活泼"
        
        style = self.title_styles[style_name]
        
        # 处理标题文本
        short_title = title[:12] + "..." if len(title) > 12 else title
        
        return {
            'text': short_title,
            'full_text': title,
            'style': style_name,
            'font': style['font'],
            'color': style['color'],
            'background': style['background'],
            'position': style['position'],
            'font_size': self._calculate_font_size(short_title),
            'shadow': True,
            'animation': 'fade_in'
        }
    
    def _calculate_font_size(self, text: str) -> int:
        """根据文本长度计算字体大小"""
        base_size = 36
        if len(text) > 10:
            return base_size - 4
        elif len(text) > 6:
            return base_size
        else:
            return base_size + 4
    
    def _extract_color_palette(self, cover_suggestion: Dict) -> Dict:
        """提取颜色调色板（简化版）"""
        # 简化版颜色提取
        default_palettes = {
            'warm': ['#FF6B6B', '#FFE66D', '#FF8E53'],
            'cool': ['#4ECDC4', '#45B7D1', '#96CEB4'],
            'neutral': ['#95A5A6', '#BDC3C7', '#ECF0F1']
        }
        
        # 根据封面类型选择调色板
        if cover_suggestion.get('tags'):
            if any(tag in ['风景', '自然'] for tag in cover_suggestion['tags']):
                palette_name = 'cool'
            elif any(tag in ['美食', '人物'] for tag in cover_suggestion['tags']):
                palette_name = 'warm'
            else:
                palette_name = 'neutral'
        else:
            palette_name = 'neutral'
        
        return {
            'name': palette_name,
            'primary': default_palettes[palette_name][0],
            'secondary': default_palettes[palette_name][1],
            'accent': default_palettes[palette_name][2],
            'colors': default_palettes[palette_name]
        }
    
    def _get_fallback_cover(self) -> Dict:
        """获取默认封面建议"""
        return {
            'type': 'default',
            'source': 'default_cover.jpg',
            'score': 0.3,
            'description': '默认封面',
            'timecode': '00:00:00'
        }
    
    def _seconds_to_timecode(self, seconds: float) -> str:
        """将秒转换为时间码"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# 全局服务实例
subtitle_generator = None
cover_service = None

def get_subtitle_generator() -> SubtitleGenerator:
    """获取字幕生成器实例"""
    global subtitle_generator
    if subtitle_generator is None:
        subtitle_generator = SubtitleGenerator()
    return subtitle_generator

def get_cover_service() -> CoverSuggestionService:
    """获取封面建议服务实例"""
    global cover_service
    if cover_service is None:
        cover_service = CoverSuggestionService()
    return cover_service
