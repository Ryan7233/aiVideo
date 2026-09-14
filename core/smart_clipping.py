"""
智能视频切片模块
提供基于场景检测、音频能量分析、内容评分的智能选段功能
"""

import subprocess
import re
import tempfile
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

from core.concurrency import run_ffmpeg

from core.degradation import mark_degraded

logger = logging.getLogger(__name__)


class SmartClippingEngine:
    """智能切片引擎"""
    
    def __init__(self):
        self.scene_threshold = 0.3  # 场景变化阈值
        self.audio_energy_window = 1.0  # 音频能量分析窗口(秒)
        self.min_segment_gap = 2.0  # 最小片段间隔(秒)
        self.motion_sample_fps = 2  # 运动采样频率(帧/秒)
    
    def analyze_video_content(self, video_path: str, max_duration: Optional[int] = 900,
                              chunk_seconds: int = 300) -> Dict:
        """
        全面分析视频内容

        Scene changes, motion and audio energy share one FFmpeg decode per
        chunk. The previous implementation ran three separate commands,
        each decoding the whole file, and scraped the numbers back out of
        stderr with regexes that broke silently on new FFmpeg releases.

        Args:
            video_path: 视频文件路径
            max_duration: 最大分析时长(秒)，None 覆盖全片
            chunk_seconds: 每次解码的最长时长，失败区间单独报告

        Returns:
            包含场景变化、音频能量、视频时长等信息的字典
        """
        try:
            logger.info(f"Starting comprehensive video analysis for: {video_path}")

            duration = self._get_video_duration(video_path)
            if duration <= 0:
                raise ValueError("Invalid video duration")

            analysis_duration = duration if max_duration is None else min(duration, max_duration)
            if chunk_seconds <= 0:
                raise ValueError("chunk_seconds must be positive")
            measurements = {key: [] for key in ("scene_changes", "audio_energy", "motion_activity")}
            intervals, errors = [], []
            start = 0.0
            while start < analysis_duration:
                span = min(chunk_seconds, analysis_duration - start)
                try:
                    chunk = self._measure_streams(video_path, span, start=start)
                    for key in measurements:
                        measurements[key].extend(
                            {**point, "timestamp": point["timestamp"] + start}
                            for point in chunk[key] if 0 <= point["timestamp"] < span
                        )
                    intervals.append({"start": start, "end": start + span})
                except Exception as exc:
                    errors.append({"start": start, "end": start + span, "error": str(exc)})
                    logger.warning("Measurement chunk %.1f-%.1f failed: %s", start, start + span, exc)
                start += span

            return {
                'duration': duration,
                'analysis_duration': analysis_duration,
                'measured_intervals': intervals,
                'measurement_errors': errors,
                'measured_seconds': sum(item['end'] - item['start'] for item in intervals),
                'degraded': bool(errors),
                'scene_changes': measurements['scene_changes'],
                'audio_energy': measurements['audio_energy'],
                'motion_activity': measurements['motion_activity'],
                'timestamp': int(time.time())
            }

        except Exception as e:
            logger.error(f"Video analysis failed: {str(e)}")
            return mark_degraded(self._get_fallback_analysis(video_path), e, logger=logger, context='analyze_video_content')

    def _has_audio_stream(self, video_path: str) -> bool:
        """Check for an audio stream before wiring it into the filter graph."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=index", "-of", "csv=p=0", video_path],
                capture_output=True, text=True, timeout=30,
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    @staticmethod
    def _parse_metadata_file(path: Path, key: str) -> List[Tuple[float, float]]:
        """Read an FFmpeg ``metadata=print`` dump into (timestamp, value) pairs.

        The format is a ``frame:... pts_time:N`` header followed by one
        ``key=value`` line per requested key.
        """
        if not path.exists():
            return []
        points: List[Tuple[float, float]] = []
        timestamp: Optional[float] = None
        for line in path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("frame:"):
                match = re.search(r"pts_time:(-?[0-9.]+)", line)
                timestamp = float(match.group(1)) if match else None
            elif "=" in line and timestamp is not None:
                name, _, raw = line.partition("=")
                if name.strip() != key:
                    continue
                try:
                    points.append((timestamp, float(raw.strip())))
                except ValueError:
                    continue
        return points

    def _measure_streams(self, video_path: str, duration: float, start: float = 0.0) -> Dict[str, List[Dict]]:
        """Run one decode that emits scene, motion and audio metadata."""
        with tempfile.TemporaryDirectory(prefix="aivideo_analysis_") as tmp:
            tmp_path = Path(tmp)
            scene_file = tmp_path / "scenes.txt"
            motion_file = tmp_path / "motion.txt"
            audio_file = tmp_path / "audio.txt"

            chains = [
                "[0:v]split=2[v_scene][v_motion]",
                f"[v_scene]scdet=t={self.scene_threshold}:sc_pass=1,"
                f"metadata=print:file={self._escape_filter_path(scene_file)}[scene_out]",
                # YDIF is the mean luma difference between consecutive frames:
                # an actual motion magnitude, unlike the old pass which just
                # re-ran scene detection at a lower threshold and recorded 1.0.
                f"[v_motion]fps={self.motion_sample_fps},signalstats,"
                f"metadata=print:file={self._escape_filter_path(motion_file)}"
                f":key=lavfi.signalstats.YDIF[motion_out]",
            ]
            maps = ["-map", "[scene_out]", "-map", "[motion_out]"]

            has_audio = self._has_audio_stream(video_path)
            if has_audio:
                chains.append(
                    "[0:a]aresample=16000,asetnsamples=n=16000:p=0,"
                    "astats=metadata=1:reset=1,"
                    f"ametadata=print:file={self._escape_filter_path(audio_file)}"
                    ":key=lavfi.astats.Overall.RMS_level[audio_out]"
                )
                maps += ["-map", "[audio_out]"]

            cmd = [
                "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error",
                "-ss", f"{start:.3f}", "-i", video_path, "-t", f"{duration:.3f}",
                "-filter_complex", ";".join(chains),
                *maps, "-f", "null", "-",
            ]
            result = run_ffmpeg(cmd, timeout=900)
            if result.returncode != 0:
                raise RuntimeError(f"analysis pass failed: {(result.stderr or '')[-2000:]}")

            scene_changes = [
                {'timestamp': ts, 'score': score, 'type': 'scene_change'}
                for ts, score in self._parse_metadata_file(scene_file, "lavfi.scd.score")
            ]
            motion_activity = [
                {
                    'timestamp': ts,
                    # YDIF is 0-255; normal footage sits well under 20, so that
                    # is the point where motion counts as fully active.
                    'activity': max(0.0, min(1.0, ydif / 20.0)),
                    'ydif': ydif,
                    'type': 'motion',
                }
                for ts, ydif in self._parse_metadata_file(motion_file, "lavfi.signalstats.YDIF")
            ]
            audio_energy = [
                {
                    'timestamp': ts,
                    'rms_db': rms_db,
                    'energy': max(0.0, min(1.0, (rms_db + 60.0) / 60.0)),
                    'type': 'audio_energy',
                }
                for ts, rms_db in self._parse_metadata_file(audio_file, "lavfi.astats.Overall.RMS_level")
            ]

        logger.info(
            "Analysis pass: %d scene changes, %d motion samples, %d audio samples%s",
            len(scene_changes), len(motion_activity), len(audio_energy),
            "" if has_audio else " (no audio stream)",
        )
        return {
            'scene_changes': scene_changes,
            'motion_activity': motion_activity,
            'audio_energy': audio_energy,
        }

    @staticmethod
    def _escape_filter_path(path: Path) -> str:
        """Escape a path for use inside an FFmpeg filter argument."""
        return str(path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")

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
        # Mean measured motion magnitude in the window; the old sample count
        # only measured the sampling rate.
        motion_density = (
            sum(m.get('activity', 0.0) for m in window_motion) / len(window_motion)
            if window_motion else 0.0
        )
        motion_score = min(1.0, motion_density)
        
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
            return mark_degraded(self._get_fallback_segments(video_path, min_duration, max_duration, count), e, logger=logger, context='select_best_segments')
    
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
            result = run_ffmpeg(cmd, timeout=60)
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
            result = run_ffmpeg(cmd, timeout=60)
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
            'measured_intervals': [],
            'measured_seconds': 0.0,
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
