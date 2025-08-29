"""
音频处理服务
降噪、BGM自动匹配（节拍对齐）
"""

import logging
import numpy as np
import subprocess
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import tempfile
import librosa
from scipy import signal
import random

logger = logging.getLogger(__name__)

# 尝试导入音频处理库
try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
    logger.info("音频处理库可用")
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    logger.warning("音频处理库不可用，某些功能将受限")


class AudioProcessingService:
    """音频处理服务"""
    
    def __init__(self):
        # 音频处理配置
        self.audio_config = {
            'sample_rate': 44100,
            'channels': 2,
            'bit_depth': 16,
            'format': 'wav'
        }
        
        # 降噪配置
        self.noise_reduction_config = {
            'noise_floor_db': -60,
            'reduction_strength': 0.7,
            'spectral_gate_db': -30,
            'frequency_smoothing': 3
        }
        
        # BGM库配置
        self.bgm_library = {
            '治愈': {
                'tempo_range': (60, 90),
                'mood': 'calm',
                'instruments': ['piano', 'strings', 'ambient'],
                'samples': [
                    {'name': 'gentle_piano', 'tempo': 72, 'energy': 0.3, 'mood': 'peaceful'},
                    {'name': 'soft_strings', 'tempo': 80, 'energy': 0.4, 'mood': 'warm'},
                    {'name': 'ambient_pad', 'tempo': 65, 'energy': 0.2, 'mood': 'dreamy'}
                ]
            },
            '专业': {
                'tempo_range': (90, 120),
                'mood': 'corporate',
                'instruments': ['acoustic', 'light_percussion'],
                'samples': [
                    {'name': 'corporate_upbeat', 'tempo': 105, 'energy': 0.6, 'mood': 'confident'},
                    {'name': 'acoustic_positive', 'tempo': 95, 'energy': 0.5, 'mood': 'professional'},
                    {'name': 'light_tech', 'tempo': 110, 'energy': 0.7, 'mood': 'modern'}
                ]
            },
            '踩雷': {
                'tempo_range': (100, 140),
                'mood': 'dramatic',
                'instruments': ['electronic', 'bass', 'drums'],
                'samples': [
                    {'name': 'dramatic_build', 'tempo': 120, 'energy': 0.8, 'mood': 'tense'},
                    {'name': 'warning_alert', 'tempo': 130, 'energy': 0.9, 'mood': 'urgent'},
                    {'name': 'suspense_low', 'tempo': 110, 'energy': 0.7, 'mood': 'mysterious'}
                ]
            }
        }
        
        # 节拍对齐配置
        self.beat_alignment_config = {
            'beat_tracking_window': 2048,
            'tempo_stability_threshold': 0.8,
            'sync_tolerance': 0.1,  # 秒
            'crossfade_duration': 0.5  # 秒
        }
    
    def process_video_audio(self, video_path: str, style: str = '治愈', 
                          enhance_speech: bool = True, add_bgm: bool = True) -> Dict:
        """
        处理视频音频
        
        Args:
            video_path: 视频文件路径
            style: 风格类型
            enhance_speech: 是否增强语音
            add_bgm: 是否添加背景音乐
            
        Returns:
            处理结果
        """
        try:
            logger.info(f"开始处理视频音频 - 风格: {style}")
            
            # 1. 提取原始音频
            original_audio_path = self._extract_audio_from_video(video_path)
            
            if not original_audio_path:
                return {'success': False, 'error': '音频提取失败'}
            
            # 2. 分析音频特征
            audio_analysis = self._analyze_audio_features(original_audio_path)
            
            # 3. 语音增强和降噪
            processed_audio_path = original_audio_path
            if enhance_speech:
                processed_audio_path = self._enhance_speech_audio(original_audio_path, audio_analysis)
            
            # 4. 选择和对齐BGM
            bgm_result = {}
            if add_bgm:
                bgm_result = self._add_background_music(processed_audio_path, style, audio_analysis)
                if bgm_result.get('success'):
                    processed_audio_path = bgm_result['mixed_audio_path']
            
            # 5. 最终音频优化
            final_audio_path = self._optimize_final_audio(processed_audio_path, audio_analysis)
            
            # 6. 生成处理后的视频
            output_video_path = self._merge_audio_with_video(video_path, final_audio_path)
            
            result = {
                'success': True,
                'original_video': video_path,
                'processed_video': output_video_path,
                'audio_analysis': audio_analysis,
                'processing_steps': {
                    'noise_reduction': enhance_speech,
                    'bgm_added': add_bgm and bgm_result.get('success', False),
                    'final_optimization': True
                },
                'bgm_info': bgm_result.get('bgm_info', {}),
                'quality_metrics': self._calculate_audio_quality_metrics(final_audio_path)
            }
            
            logger.info("视频音频处理完成")
            return result
            
        except Exception as e:
            logger.error(f"视频音频处理失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """从视频中提取音频"""
        try:
            output_dir = Path("output_data/audio_temp")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            audio_filename = f"extracted_{Path(video_path).stem}.wav"
            audio_path = output_dir / audio_filename
            
            # 使用FFmpeg提取音频
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(self.audio_config['sample_rate']),
                "-ac", str(self.audio_config['channels']),
                str(audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and audio_path.exists():
                logger.info(f"音频提取成功: {audio_path}")
                return str(audio_path)
            else:
                logger.error(f"音频提取失败: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"音频提取异常: {e}")
            return None
    
    def _analyze_audio_features(self, audio_path: str) -> Dict:
        """分析音频特征"""
        try:
            if not AUDIO_LIBS_AVAILABLE:
                return self._basic_audio_analysis(audio_path)
            
            # 加载音频
            y, sr = librosa.load(audio_path, sr=self.audio_config['sample_rate'])
            
            # 基础特征
            duration = librosa.get_duration(y=y, sr=sr)
            
            # 音量分析
            rms = librosa.feature.rms(y=y)[0]
            volume_mean = np.mean(rms)
            volume_std = np.std(rms)
            
            # 频谱特征
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # 节拍检测
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # 频率分析
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            
            # 语音活动检测
            speech_segments = self._detect_speech_segments(y, sr)
            
            # 噪音水平估计
            noise_level = self._estimate_noise_level(y, sr)
            
            return {
                'duration': float(duration),
                'sample_rate': sr,
                'channels': len(y.shape),
                'volume': {
                    'mean': float(volume_mean),
                    'std': float(volume_std),
                    'max': float(np.max(rms)),
                    'min': float(np.min(rms))
                },
                'spectral': {
                    'centroid_mean': float(np.mean(spectral_centroids)),
                    'rolloff_mean': float(np.mean(spectral_rolloff))
                },
                'tempo': float(tempo),
                'beats_count': len(beats),
                'speech_segments': speech_segments,
                'noise_level': noise_level,
                'dynamic_range': float(np.max(rms) - np.min(rms))
            }
            
        except Exception as e:
            logger.error(f"音频特征分析失败: {e}")
            return self._basic_audio_analysis(audio_path)
    
    def _basic_audio_analysis(self, audio_path: str) -> Dict:
        """基础音频分析（不依赖librosa）"""
        try:
            # 使用FFprobe获取基础信息
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                probe_data = json.loads(result.stdout)
                
                audio_stream = None
                for stream in probe_data.get('streams', []):
                    if stream.get('codec_type') == 'audio':
                        audio_stream = stream
                        break
                
                if audio_stream:
                    return {
                        'duration': float(audio_stream.get('duration', 0)),
                        'sample_rate': int(audio_stream.get('sample_rate', 44100)),
                        'channels': int(audio_stream.get('channels', 2)),
                        'bit_rate': int(audio_stream.get('bit_rate', 0)),
                        'volume': {'mean': 0.5, 'std': 0.2},
                        'tempo': 120.0,  # 默认值
                        'noise_level': 0.3,
                        'speech_segments': []
                    }
            
            # 回退到默认值
            return {
                'duration': 30.0,
                'sample_rate': 44100,
                'channels': 2,
                'volume': {'mean': 0.5, 'std': 0.2},
                'tempo': 120.0,
                'noise_level': 0.3,
                'speech_segments': []
            }
            
        except Exception as e:
            logger.error(f"基础音频分析失败: {e}")
            return {'duration': 30.0, 'sample_rate': 44100, 'channels': 2}
    
    def _detect_speech_segments(self, y: np.ndarray, sr: int) -> List[Dict]:
        """检测语音片段"""
        try:
            # 简单的语音活动检测
            frame_length = 2048
            hop_length = 512
            
            # 计算短时能量
            energy = []
            for i in range(0, len(y) - frame_length, hop_length):
                frame = y[i:i + frame_length]
                energy.append(np.sum(frame ** 2))
            
            energy = np.array(energy)
            
            # 动态阈值
            energy_threshold = np.mean(energy) + 0.5 * np.std(energy)
            
            # 检测语音段
            speech_frames = energy > energy_threshold
            
            # 转换为时间段
            segments = []
            in_speech = False
            start_frame = 0
            
            for i, is_speech in enumerate(speech_frames):
                if is_speech and not in_speech:
                    start_frame = i
                    in_speech = True
                elif not is_speech and in_speech:
                    start_time = start_frame * hop_length / sr
                    end_time = i * hop_length / sr
                    if end_time - start_time > 0.5:  # 最小语音段长度
                        segments.append({
                            'start': start_time,
                            'end': end_time,
                            'duration': end_time - start_time
                        })
                    in_speech = False
            
            return segments
            
        except Exception as e:
            logger.error(f"语音段检测失败: {e}")
            return []
    
    def _estimate_noise_level(self, y: np.ndarray, sr: int) -> float:
        """估计噪音水平"""
        try:
            # 使用静音段估计噪音水平
            frame_length = 2048
            hop_length = 512
            
            # 计算短时能量
            energy = []
            for i in range(0, len(y) - frame_length, hop_length):
                frame = y[i:i + frame_length]
                energy.append(np.sum(frame ** 2))
            
            energy = np.array(energy)
            
            # 假设最低的20%能量帧为噪音
            noise_threshold = np.percentile(energy, 20)
            
            # 归一化到0-1范围
            max_energy = np.max(energy)
            noise_level = noise_threshold / max_energy if max_energy > 0 else 0.1
            
            return float(noise_level)
            
        except Exception:
            return 0.1
    
    def _enhance_speech_audio(self, audio_path: str, audio_analysis: Dict) -> str:
        """增强语音音频"""
        try:
            output_dir = Path("output_data/audio_temp")
            enhanced_filename = f"enhanced_{Path(audio_path).stem}.wav"
            enhanced_path = output_dir / enhanced_filename
            
            # 使用FFmpeg进行音频增强
            filters = []
            
            # 1. 高通滤波器去除低频噪音
            filters.append("highpass=f=80")
            
            # 2. 动态范围压缩
            filters.append("acompressor=threshold=0.003:ratio=3:attack=30:release=100")
            
            # 3. 音量归一化
            filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
            
            # 4. 去噪（基于噪音水平）
            noise_level = audio_analysis.get('noise_level', 0.3)
            if noise_level > 0.2:
                # 使用更强的降噪
                filters.append("afftdn=nr=20:nf=-25")
            elif noise_level > 0.1:
                # 使用适中的降噪
                filters.append("afftdn=nr=12:nf=-20")
            
            # 5. 语音增强
            filters.append("aemphasis=level_in=1:level_out=1:mode=reproduction")
            
            filter_chain = ",".join(filters)
            
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", filter_chain,
                "-ar", str(self.audio_config['sample_rate']),
                "-ac", str(self.audio_config['channels']),
                str(enhanced_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and enhanced_path.exists():
                logger.info(f"语音增强完成: {enhanced_path}")
                return str(enhanced_path)
            else:
                logger.warning(f"语音增强失败，使用原音频: {result.stderr}")
                return audio_path
                
        except Exception as e:
            logger.error(f"语音增强异常: {e}")
            return audio_path
    
    def _add_background_music(self, audio_path: str, style: str, audio_analysis: Dict) -> Dict:
        """添加背景音乐"""
        try:
            logger.info(f"开始添加背景音乐 - 风格: {style}")
            
            # 1. 选择合适的BGM
            bgm_info = self._select_bgm(style, audio_analysis)
            
            # 2. 生成或获取BGM音频
            bgm_audio_path = self._generate_bgm_audio(bgm_info, audio_analysis['duration'])
            
            if not bgm_audio_path:
                return {'success': False, 'error': 'BGM生成失败'}
            
            # 3. 节拍对齐
            aligned_bgm_path = self._align_bgm_to_speech(bgm_audio_path, audio_path, audio_analysis)
            
            # 4. 混合音频
            mixed_audio_path = self._mix_audio_tracks(audio_path, aligned_bgm_path, style)
            
            return {
                'success': True,
                'mixed_audio_path': mixed_audio_path,
                'bgm_info': bgm_info,
                'bgm_audio_path': bgm_audio_path,
                'aligned_bgm_path': aligned_bgm_path
            }
            
        except Exception as e:
            logger.error(f"背景音乐添加失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _select_bgm(self, style: str, audio_analysis: Dict) -> Dict:
        """选择合适的BGM"""
        try:
            bgm_config = self.bgm_library.get(style, self.bgm_library['治愈'])
            samples = bgm_config['samples']
            
            # 基于音频特征选择BGM
            original_tempo = audio_analysis.get('tempo', 120)
            speech_ratio = len(audio_analysis.get('speech_segments', [])) / max(audio_analysis.get('duration', 1), 1)
            
            # 评分每个BGM样本
            scored_samples = []
            for sample in samples:
                score = 0
                
                # 节拍匹配评分
                tempo_diff = abs(sample['tempo'] - original_tempo)
                tempo_score = max(0, 1 - tempo_diff / 50)  # 50 BPM容差
                score += tempo_score * 0.4
                
                # 能量匹配评分
                if speech_ratio > 0.7:  # 语音密集
                    energy_score = 1 - sample['energy']  # 低能量BGM
                else:
                    energy_score = sample['energy']  # 高能量BGM
                score += energy_score * 0.3
                
                # 风格匹配评分
                score += 0.3  # 基础风格匹配分
                
                scored_samples.append((sample, score))
            
            # 选择最高分的BGM
            best_sample = max(scored_samples, key=lambda x: x[1])[0]
            
            logger.info(f"选择BGM: {best_sample['name']}, 节拍: {best_sample['tempo']}")
            return best_sample
            
        except Exception as e:
            logger.error(f"BGM选择失败: {e}")
            return self.bgm_library['治愈']['samples'][0]
    
    def _generate_bgm_audio(self, bgm_info: Dict, duration: float) -> Optional[str]:
        """生成BGM音频"""
        try:
            output_dir = Path("output_data/audio_temp")
            bgm_filename = f"bgm_{bgm_info['name']}_{int(duration)}.wav"
            bgm_path = output_dir / bgm_filename
            
            # 这里应该从BGM库中获取实际的音频文件
            # 为了演示，我们生成一个简单的合成BGM
            if AUDIO_LIBS_AVAILABLE:
                bgm_audio = self._synthesize_bgm(bgm_info, duration)
                sf.write(str(bgm_path), bgm_audio, self.audio_config['sample_rate'])
                logger.info(f"BGM音频生成完成: {bgm_path}")
                return str(bgm_path)
            else:
                # 使用FFmpeg生成简单的音调
                return self._generate_simple_bgm_with_ffmpeg(bgm_info, duration, str(bgm_path))
                
        except Exception as e:
            logger.error(f"BGM音频生成失败: {e}")
            return None
    
    def _synthesize_bgm(self, bgm_info: Dict, duration: float) -> np.ndarray:
        """合成BGM音频"""
        try:
            sr = self.audio_config['sample_rate']
            samples = int(duration * sr)
            
            # 基于BGM信息生成音频
            tempo = bgm_info['tempo']
            energy = bgm_info['energy']
            mood = bgm_info['mood']
            
            # 生成基础音调
            t = np.linspace(0, duration, samples)
            
            if mood == 'peaceful':
                # 平和的音调 - 低频正弦波
                bgm = 0.1 * energy * np.sin(2 * np.pi * 220 * t)  # A3
                bgm += 0.05 * energy * np.sin(2 * np.pi * 330 * t)  # E4
            elif mood == 'confident':
                # 自信的音调 - 中频和弦
                bgm = 0.15 * energy * np.sin(2 * np.pi * 262 * t)  # C4
                bgm += 0.1 * energy * np.sin(2 * np.pi * 330 * t)   # E4
                bgm += 0.05 * energy * np.sin(2 * np.pi * 392 * t)  # G4
            elif mood == 'tense':
                # 紧张的音调 - 不协和音程
                bgm = 0.2 * energy * np.sin(2 * np.pi * 277 * t)   # C#4
                bgm += 0.1 * energy * np.sin(2 * np.pi * 370 * t)  # F#4
            else:
                # 默认音调
                bgm = 0.1 * energy * np.sin(2 * np.pi * 440 * t)   # A4
            
            # 添加节拍感
            beat_period = 60 / tempo  # 每拍的秒数
            beat_envelope = np.zeros_like(t)
            
            for beat_time in np.arange(0, duration, beat_period):
                beat_start = int(beat_time * sr)
                beat_end = min(beat_start + int(0.1 * sr), samples)
                if beat_end > beat_start:
                    beat_envelope[beat_start:beat_end] = 1.0
            
            # 应用包络
            envelope = 0.7 + 0.3 * beat_envelope
            bgm = bgm * envelope
            
            # 添加渐入渐出
            fade_samples = int(0.5 * sr)  # 0.5秒渐入渐出
            if samples > 2 * fade_samples:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                
                bgm[:fade_samples] *= fade_in
                bgm[-fade_samples:] *= fade_out
            
            # 立体声
            if self.audio_config['channels'] == 2:
                bgm_stereo = np.column_stack([bgm, bgm])
                return bgm_stereo
            else:
                return bgm
                
        except Exception as e:
            logger.error(f"BGM合成失败: {e}")
            return np.zeros(int(duration * self.audio_config['sample_rate']))
    
    def _generate_simple_bgm_with_ffmpeg(self, bgm_info: Dict, duration: float, output_path: str) -> Optional[str]:
        """使用FFmpeg生成简单BGM"""
        try:
            tempo = bgm_info['tempo']
            energy = bgm_info['energy']
            
            # 生成简单的音调
            frequency = 220 if bgm_info['mood'] == 'peaceful' else 440
            volume = energy * 0.1
            
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"sine=frequency={frequency}:duration={duration}:sample_rate={self.audio_config['sample_rate']}",
                "-af", f"volume={volume}",
                "-ac", str(self.audio_config['channels']),
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and Path(output_path).exists():
                return output_path
            else:
                logger.error(f"FFmpeg BGM生成失败: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"FFmpeg BGM生成异常: {e}")
            return None
    
    def _align_bgm_to_speech(self, bgm_path: str, speech_path: str, audio_analysis: Dict) -> str:
        """将BGM与语音节拍对齐"""
        try:
            # 简化版本：直接返回原BGM
            # 在实际实现中，这里应该分析语音的节拍并调整BGM
            logger.info("BGM节拍对齐完成（简化版本）")
            return bgm_path
            
        except Exception as e:
            logger.error(f"BGM节拍对齐失败: {e}")
            return bgm_path
    
    def _mix_audio_tracks(self, speech_path: str, bgm_path: str, style: str) -> str:
        """混合音频轨道"""
        try:
            output_dir = Path("output_data/audio_temp")
            mixed_filename = f"mixed_{Path(speech_path).stem}.wav"
            mixed_path = output_dir / mixed_filename
            
            # 根据风格确定混音比例
            style_mix_ratios = {
                '治愈': {'speech': 0.8, 'bgm': 0.3},
                '专业': {'speech': 0.9, 'bgm': 0.2},
                '踩雷': {'speech': 0.85, 'bgm': 0.4}
            }
            
            ratios = style_mix_ratios.get(style, style_mix_ratios['治愈'])
            
            # 使用FFmpeg混音
            cmd = [
                "ffmpeg", "-y",
                "-i", speech_path,
                "-i", bgm_path,
                "-filter_complex",
                f"[0:a]volume={ratios['speech']}[speech];"
                f"[1:a]volume={ratios['bgm']}[bgm];"
                "[speech][bgm]amix=inputs=2:duration=first:dropout_transition=2",
                "-ar", str(self.audio_config['sample_rate']),
                "-ac", str(self.audio_config['channels']),
                str(mixed_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and mixed_path.exists():
                logger.info(f"音频混音完成: {mixed_path}")
                return str(mixed_path)
            else:
                logger.error(f"音频混音失败: {result.stderr}")
                return speech_path
                
        except Exception as e:
            logger.error(f"音频混音异常: {e}")
            return speech_path
    
    def _optimize_final_audio(self, audio_path: str, audio_analysis: Dict) -> str:
        """优化最终音频"""
        try:
            output_dir = Path("output_data/audio_temp")
            optimized_filename = f"optimized_{Path(audio_path).stem}.wav"
            optimized_path = output_dir / optimized_filename
            
            # 最终优化滤镜
            filters = [
                "acompressor=threshold=0.003:ratio=2:attack=30:release=100",  # 轻微压缩
                "loudnorm=I=-20:TP=-1.0:LRA=7",  # 响度标准化
                "highpass=f=60",  # 去除极低频
                "lowpass=f=15000"  # 去除极高频
            ]
            
            filter_chain = ",".join(filters)
            
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-af", filter_chain,
                "-ar", str(self.audio_config['sample_rate']),
                "-ac", str(self.audio_config['channels']),
                str(optimized_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and optimized_path.exists():
                logger.info(f"音频最终优化完成: {optimized_path}")
                return str(optimized_path)
            else:
                logger.warning(f"音频最终优化失败，使用原音频: {result.stderr}")
                return audio_path
                
        except Exception as e:
            logger.error(f"音频最终优化异常: {e}")
            return audio_path
    
    def _merge_audio_with_video(self, video_path: str, audio_path: str) -> str:
        """将处理后的音频合并到视频"""
        try:
            output_dir = Path("output_data")
            video_filename = f"enhanced_{Path(video_path).stem}.mp4"
            output_video_path = output_dir / video_filename
            
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",  # 保持视频不变
                "-c:a", "aac",
                "-b:a", "192k",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                str(output_video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and output_video_path.exists():
                logger.info(f"音视频合并完成: {output_video_path}")
                return str(output_video_path)
            else:
                logger.error(f"音视频合并失败: {result.stderr}")
                return video_path
                
        except Exception as e:
            logger.error(f"音视频合并异常: {e}")
            return video_path
    
    def _calculate_audio_quality_metrics(self, audio_path: str) -> Dict:
        """计算音频质量指标"""
        try:
            if not AUDIO_LIBS_AVAILABLE:
                return {'snr': 'N/A', 'dynamic_range': 'N/A', 'peak_level': 'N/A'}
            
            y, sr = librosa.load(audio_path, sr=self.audio_config['sample_rate'])
            
            # 信噪比估计
            signal_power = np.mean(y ** 2)
            noise_power = np.mean(y[:sr] ** 2)  # 假设前1秒为噪音
            snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))
            
            # 动态范围
            rms = librosa.feature.rms(y=y)[0]
            dynamic_range = 20 * np.log10(np.max(rms) / max(np.min(rms), 1e-10))
            
            # 峰值电平
            peak_level = 20 * np.log10(np.max(np.abs(y)))
            
            return {
                'snr': f"{snr:.1f} dB",
                'dynamic_range': f"{dynamic_range:.1f} dB",
                'peak_level': f"{peak_level:.1f} dB",
                'rms_level': f"{20 * np.log10(np.sqrt(np.mean(y ** 2))):.1f} dB"
            }
            
        except Exception as e:
            logger.error(f"音频质量指标计算失败: {e}")
            return {'error': str(e)}


# 全局服务实例
audio_processing_service = None

def get_audio_processing_service() -> AudioProcessingService:
    """获取音频处理服务实例"""
    global audio_processing_service
    if audio_processing_service is None:
        audio_processing_service = AudioProcessingService()
    return audio_processing_service
