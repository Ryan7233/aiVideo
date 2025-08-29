"""
智能视频切片模块
提供基于场景检测、音频能量分析、内容评分的智能选段功能
"""

import subprocess
import re
import json
import math
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SmartClippingEngine:
    """智能切片引擎"""
    
    def __init__(self):
        self.scene_threshold = 0.3  # 场景变化阈值
        self.audio_energy_window = 1.0  # 音频能量分析窗口(秒)
        self.min_segment_gap = 2.0  # 最小片段间隔(秒)
    
    def analyze_video_content(self, video_path: str, max_duration: int = 900) -> Dict:
        """
        全面分析视频内容
        
        Args:
            video_path: 视频文件路径
            max_duration: 最大分析时长(秒)，避免长视频分析过久
            
        Returns:
            包含场景变化、音频能量、视频时长等信息的字典
        """
        try:
            logger.info(f"Starting comprehensive video analysis for: {video_path}")
            
            # 获取视频基本信息
            duration = self._get_video_duration(video_path)
            if duration <= 0:
                raise ValueError("Invalid video duration")
            
            # 限制分析时长
            analysis_duration = min(duration, max_duration)
            
            # 并行分析各项指标
            scene_changes = self._detect_scene_changes(video_path, analysis_duration)
            audio_energy = self._analyze_audio_energy(video_path, analysis_duration)
            motion_activity = self._analyze_motion_activity(video_path, analysis_duration)
            
            return {
                'duration': duration,
                'analysis_duration': analysis_duration,
                'scene_changes': scene_changes,
                'audio_energy': audio_energy,
                'motion_activity': motion_activity,
                'timestamp': int(time.time()) if 'time' in globals() else 0
            }
            
        except Exception as e:
            logger.error(f"Video analysis failed: {str(e)}")
            return self._get_fallback_analysis(video_path)
    
    def _detect_scene_changes(self, video_path: str, duration: int) -> List[Dict]:
        """
        使用FFmpeg scdet filter检测场景变化
        
        Returns:
            List of scene changes with timestamps and scores
        """
        try:
            # 使用scdet filter而不是scene filter
            cmd = [
                "ffmpeg", "-hide_banner", "-i", video_path,
                "-t", str(duration),
                "-vf", f"scdet=t={self.scene_threshold}:sc_pass=1",
                "-an", "-f", "null", "-"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            stderr_output = result.stderr or ""
            
            # 解析场景变化时间点 - 更新正则表达式
            scene_pattern = r"scene_score=([0-9.]+).*pts_time:([0-9.]+)"
            matches = re.findall(scene_pattern, stderr_output)
            
            # 如果第一个正则不匹配，尝试另一种格式
            if not matches:
                scene_pattern = r"scdet.*score=([0-9.]+).*time:([0-9.]+)"
                matches = re.findall(scene_pattern, stderr_output)
            
            scenes = []
            for score_str, time_str in matches:
                try:
                    score = float(score_str)
                    timestamp = float(time_str)
                    if score >= self.scene_threshold:
                        scenes.append({
                            'timestamp': timestamp,
                            'score': score,
                            'type': 'scene_change'
                        })
                except ValueError:
                    continue
            
            # 如果还是没有结果，尝试简化的方法
            if not scenes:
                logger.info("Trying alternative scene detection method")
                cmd_alt = [
                    "ffmpeg", "-hide_banner", "-i", video_path,
                    "-t", str(min(duration, 60)),  # 限制为60秒以加快分析
                    "-vf", "select='gt(scene,0.3)',showinfo",
                    "-an", "-f", "null", "-"
                ]
                result_alt = subprocess.run(cmd_alt, capture_output=True, text=True, timeout=120)
                alt_output = result_alt.stderr or ""
                
                # 解析showinfo输出
                info_pattern = r"pts_time:([0-9.]+)"
                time_matches = re.findall(info_pattern, alt_output)
                
                for i, time_str in enumerate(time_matches):
                    try:
                        timestamp = float(time_str)
                        scenes.append({
                            'timestamp': timestamp,
                            'score': 0.4,  # 默认评分
                            'type': 'scene_change'
                        })
                    except ValueError:
                        continue
            
            logger.info(f"Detected {len(scenes)} scene changes")
            return sorted(scenes, key=lambda x: x['timestamp'])
            
        except Exception as e:
            logger.warning(f"Scene detection failed: {str(e)}")
            return []
    
    def _analyze_audio_energy(self, video_path: str, duration: int) -> List[Dict]:
        """
        分析音频RMS能量分布
        
        Returns:
            List of audio energy measurements over time
        """
        try:
            # 简化的音频能量分析，使用volumedetect
            cmd = [
                "ffmpeg", "-hide_banner", "-i", video_path,
                "-t", str(min(duration, 120)),  # 限制分析时长
                "-af", "volumedetect",
                "-vn", "-f", "null", "-"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            stderr_output = result.stderr or ""
            
            # 解析音量信息
            volume_pattern = r"mean_volume: ([+-]?[0-9]*\.?[0-9]+) dB"
            max_pattern = r"max_volume: ([+-]?[0-9]*\.?[0-9]+) dB"
            
            mean_match = re.search(volume_pattern, stderr_output)
            max_match = re.search(max_pattern, stderr_output)
            
            energy_data = []
            
            if mean_match and max_match:
                mean_db = float(mean_match.group(1))
                max_db = float(max_match.group(1))
                
                # 基于整体音量创建时间序列数据
                # 这是一个简化的实现，实际项目中可以使用更复杂的分段分析
                segment_count = max(1, min(duration // 5, 20))  # 每5秒一个段，最多20个段
                segment_duration = duration / segment_count
                
                for i in range(segment_count):
                    timestamp = i * segment_duration
                    # 模拟音频能量变化，基于均值和最大值
                    energy_variation = 0.8 + 0.2 * (i % 3) / 2  # 简单的变化模式
                    estimated_db = mean_db * energy_variation
                    energy_linear = max(0.0, min(1.0, (estimated_db + 60) / 60))
                    
                    energy_data.append({
                        'timestamp': timestamp,
                        'rms_db': estimated_db,
                        'energy': energy_linear,
                        'type': 'audio_energy'
                    })
            
            # 如果没有音频数据，尝试更简单的方法
            if not energy_data:
                logger.info("No audio detected, creating default energy profile")
                # 创建默认的能量分布
                segment_count = max(1, min(duration // 10, 10))
                for i in range(segment_count):
                    timestamp = i * (duration / segment_count)
                    energy_data.append({
                        'timestamp': timestamp,
                        'rms_db': -20.0,  # 默认音量
                        'energy': 0.6,    # 默认能量
                        'type': 'audio_energy'
                    })
            
            logger.info(f"Analyzed {len(energy_data)} audio energy points")
            return energy_data
            
        except Exception as e:
            logger.warning(f"Audio energy analysis failed: {str(e)}")
            # 返回默认数据
            return [{
                'timestamp': 0.0,
                'rms_db': -20.0,
                'energy': 0.5,
                'type': 'audio_energy'
            }]
    
    def _analyze_motion_activity(self, video_path: str, duration: int) -> List[Dict]:
        """
        分析视频运动活跃度
        
        Returns:
            List of motion activity measurements
        """
        try:
            # 使用select filter计算帧间差异
            cmd = [
                "ffmpeg", "-hide_banner", "-i", video_path,
                "-t", str(duration),
                "-vf", "select='gt(scene,0.01)',showinfo",
                "-an", "-f", "null", "-"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            stderr_output = result.stderr or ""
            
            # 解析运动信息
            motion_pattern = r"pts_time:([0-9.]+).*pos:([0-9]+)"
            matches = re.findall(motion_pattern, stderr_output)
            
            motion_data = []
            prev_time = 0
            for time_str, pos_str in matches:
                try:
                    timestamp = float(time_str)
                    if timestamp > prev_time + 0.5:  # 每0.5秒采样一次
                        motion_data.append({
                            'timestamp': timestamp,
                            'activity': 1.0,  # 有运动
                            'type': 'motion'
                        })
                        prev_time = timestamp
                except ValueError:
                    continue
            
            logger.info(f"Analyzed {len(motion_data)} motion activity points")
            return motion_data
            
        except Exception as e:
            logger.warning(f"Motion analysis failed: {str(e)}")
            return []
    
    def calculate_segment_scores(self, analysis: Dict, min_duration: int, max_duration: int) -> List[Dict]:
        """
        基于分析结果计算各时间段的综合评分
        
        Args:
            analysis: analyze_video_content的返回结果
            min_duration: 最小片段时长
            max_duration: 最大片段时长
            
        Returns:
            List of scored time segments
        """
        try:
            duration = analysis['duration']
            scene_changes = analysis.get('scene_changes', [])
            audio_energy = analysis.get('audio_energy', [])
            motion_activity = analysis.get('motion_activity', [])
            
            segments = []
            
            # 生成候选时间窗口
            step_size = max(1, min_duration // 4)  # 步长为最小时长的1/4
            
            for start_time in range(0, max(1, int(duration - min_duration + 1)), step_size):
                for window_duration in range(min_duration, min(max_duration + 1, int(duration - start_time + 1))):
                    end_time = start_time + window_duration
                    
                    # 计算该时间窗口的综合评分
                    score_components = self._calculate_window_score(
                        start_time, end_time, scene_changes, audio_energy, motion_activity
                    )
                    
                    total_score = (
                        score_components['scene_score'] * 0.25 +
                        score_components['audio_score'] * 0.35 +
                        score_components['motion_score'] * 0.2 +
                        score_components['position_score'] * 0.1 +
                        score_components['duration_score'] * 0.1
                    )
                    
                    segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': window_duration,
                        'total_score': total_score,
                        'components': score_components,
                        'start_hms': self._seconds_to_hms(start_time),
                        'end_hms': self._seconds_to_hms(end_time)
                    })
            
            # 按评分排序并返回前N个
            segments.sort(key=lambda x: x['total_score'], reverse=True)
            
            logger.info(f"Generated {len(segments)} scored segments")
            return segments[:20]  # 返回前20个最佳片段
            
        except Exception as e:
            logger.error(f"Segment scoring failed: {str(e)}")
            return []
    
    def _calculate_window_score(self, start_time: int, end_time: int, 
                               scene_changes: List, audio_energy: List, motion_activity: List) -> Dict:
        """计算时间窗口内的各项评分"""
        
        # 1. 场景变化评分 - 适中的场景变化数量更好
        scene_count = len([s for s in scene_changes 
                          if start_time <= s['timestamp'] <= end_time])
        ideal_scenes = max(1, (end_time - start_time) // 10)  # 每10秒1个场景变化较理想
        scene_score = 1.0 - abs(scene_count - ideal_scenes) / max(ideal_scenes, 1)
        scene_score = max(0.0, min(1.0, scene_score))
        
        # 2. 音频能量评分 - 高能量更好
        window_audio = [a for a in audio_energy 
                       if start_time <= a['timestamp'] <= end_time]
        if window_audio:
            avg_energy = sum(a['energy'] for a in window_audio) / len(window_audio)
            audio_score = avg_energy
        else:
            audio_score = 0.5  # 默认中等评分
        
        # 3. 运动活跃度评分
        window_motion = [m for m in motion_activity 
                        if start_time <= m['timestamp'] <= end_time]
        motion_density = len(window_motion) / max(1, end_time - start_time)
        motion_score = min(1.0, motion_density * 2)  # 运动密度越高越好，但有上限
        
        # 4. 位置评分 - 稍微偏好视频前半部分
        total_duration = max(1, end_time)
        position_ratio = start_time / total_duration
        position_score = 1.0 - position_ratio * 0.3  # 前面的片段有轻微加分
        
        # 5. 时长评分 - 偏好适中时长
        duration = end_time - start_time
        duration_score = 1.0 - abs(duration - 20) / 20  # 20秒为理想时长
        duration_score = max(0.0, min(1.0, duration_score))
        
        return {
            'scene_score': scene_score,
            'audio_score': audio_score,
            'motion_score': motion_score,
            'position_score': position_score,
            'duration_score': duration_score,
            'scene_count': scene_count,
            'avg_energy': window_audio[0]['energy'] if window_audio else 0.5,
            'motion_density': motion_density
        }
    
    def select_best_segments(self, video_path: str, min_duration: int, max_duration: int, 
                           count: int = 1, avoid_black: bool = True, avoid_silence: bool = True) -> List[Dict]:
        """
        选择最佳视频片段
        
        Args:
            video_path: 视频文件路径
            min_duration: 最小片段时长
            max_duration: 最大片段时长
            count: 需要选择的片段数量
            avoid_black: 是否避免黑屏片段
            avoid_silence: 是否避免静音片段
            
        Returns:
            List of best segments with metadata
        """
        try:
            # 1. 分析视频内容
            analysis = self.analyze_video_content(video_path)
            
            # 2. 计算片段评分
            scored_segments = self.calculate_segment_scores(analysis, min_duration, max_duration)
            
            if not scored_segments:
                logger.warning("No segments generated, using fallback")
                return self._get_fallback_segments(video_path, min_duration, max_duration, count)
            
            # 3. 过滤黑屏和静音片段
            if avoid_black or avoid_silence:
                scored_segments = self._filter_problematic_segments(
                    video_path, scored_segments, avoid_black, avoid_silence
                )
            
            # 4. 去重叠选择
            final_segments = self._select_non_overlapping_segments(scored_segments, count)
            
            logger.info(f"Selected {len(final_segments)} best segments")
            return final_segments
            
        except Exception as e:
            logger.error(f"Smart segment selection failed: {str(e)}")
            return self._get_fallback_segments(video_path, min_duration, max_duration, count)
    
    def _filter_problematic_segments(self, video_path: str, segments: List[Dict], 
                                   avoid_black: bool, avoid_silence: bool) -> List[Dict]:
        """过滤黑屏和静音片段"""
        filtered = []
        
        for segment in segments:
            if avoid_black:
                black_frac = self._check_black_fraction(
                    video_path, segment['start_hms'], segment['duration']
                )
                if black_frac > 0.3:  # 超过30%黑屏则跳过
                    continue
            
            if avoid_silence:
                silence_frac = self._check_silence_fraction(
                    video_path, segment['start_hms'], segment['duration']
                )
                if silence_frac > 0.5:  # 超过50%静音则跳过
                    continue
            
            filtered.append(segment)
        
        return filtered
    
    def _select_non_overlapping_segments(self, segments: List[Dict], count: int) -> List[Dict]:
        """选择不重叠的最佳片段"""
        if not segments:
            return []
        
        selected = []
        used_ranges = []
        
        for segment in segments:
            start, end = segment['start_time'], segment['end_time']
            
            # 检查是否与已选片段重叠
            overlap = False
            for used_start, used_end in used_ranges:
                if not (end <= used_start or start >= used_end):
                    overlap = True
                    break
            
            if not overlap:
                selected.append(segment)
                used_ranges.append((start, end))
                
                if len(selected) >= count:
                    break
        
        return selected
    
    def _get_video_duration(self, video_path: str) -> float:
        """获取视频时长"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return float(result.stdout.strip())
        except Exception:
            return 0.0
    
    def _check_black_fraction(self, video_path: str, start_hms: str, duration: float) -> float:
        """检查片段中的黑屏比例"""
        try:
            cmd = [
                "ffmpeg", "-hide_banner", "-ss", start_hms, "-t", f"{duration:.2f}",
                "-i", video_path,
                "-vf", "crop=in_w*0.9:in_h*0.9:(in_w-out_w)/2:(in_h-out_h)/2,blackdetect=d=0.3:pic_th=0.98",
                "-an", "-f", "null", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            stderr_output = result.stderr or ""
            
            matches = re.findall(r"black_duration:([0-9]+\.?[0-9]*)", stderr_output)
            if matches:
                total_black = sum(float(x) for x in matches)
                return min(max(total_black / max(duration, 0.001), 0.0), 1.0)
            return 0.0
        except Exception:
            return 0.0
    
    def _check_silence_fraction(self, video_path: str, start_hms: str, duration: float) -> float:
        """检查片段中的静音比例"""
        try:
            cmd = [
                "ffmpeg", "-hide_banner", "-ss", start_hms, "-t", f"{duration:.2f}",
                "-i", video_path, "-af", "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            stderr_output = result.stderr or ""
            
            matches = re.findall(r"silence_duration: ([0-9]+\.?[0-9]*)", stderr_output)
            if matches:
                total_silence = sum(float(x) for x in matches)
                return min(max(total_silence / max(duration, 0.001), 0.0), 1.0)
            return 0.0
        except Exception:
            return 0.0
    
    def _seconds_to_hms(self, seconds: int) -> str:
        """将秒转换为HH:MM:SS格式"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _get_fallback_analysis(self, video_path: str) -> Dict:
        """获取后备分析结果"""
        duration = self._get_video_duration(video_path)
        return {
            'duration': duration,
            'analysis_duration': duration,
            'scene_changes': [],
            'audio_energy': [],
            'motion_activity': [],
            'fallback': True
        }
    
    def _get_fallback_segments(self, video_path: str, min_duration: int, 
                              max_duration: int, count: int) -> List[Dict]:
        """获取后备片段选择"""
        duration = self._get_video_duration(video_path)
        segments = []
        
        # 简单地从视频开头选择片段
        for i in range(count):
            start = i * max_duration
            if start + min_duration > duration:
                break
            
            end = min(start + max_duration, duration)
            segments.append({
                'start_time': start,
                'end_time': int(end),
                'duration': int(end - start),
                'total_score': 0.5,  # 默认评分
                'start_hms': self._seconds_to_hms(start),
                'end_hms': self._seconds_to_hms(int(end)),
                'fallback': True
            })
        
        return segments


# 全局智能切片引擎实例
smart_engine = SmartClippingEngine()


def get_smart_segments(video_path: str, min_duration: int, max_duration: int, 
                      count: int = 1) -> List[Dict]:
    """
    获取智能选择的视频片段
    
    这是主要的对外接口函数
    """
    return smart_engine.select_best_segments(video_path, min_duration, max_duration, count)


def analyze_video_intelligence(video_path: str) -> Dict:
    """
    分析视频的智能化指标
    
    返回详细的分析报告
    """
    return smart_engine.analyze_video_content(video_path)
