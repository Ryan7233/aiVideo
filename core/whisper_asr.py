"""
Whisper自动语音识别服务
提供高性能的语音转文字功能，支持多语言和字幕生成
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import subprocess
from core.runtime import MODEL_DIR, OUTPUT_DIR

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    WhisperModel = None

logger = logging.getLogger(__name__)


class WhisperASRService:
    """Whisper自动语音识别服务"""
    
    def __init__(self, model_size: str = "base", device: str = "auto", compute_type: str = "auto"):
        """
        初始化Whisper ASR服务
        
        Args:
            model_size: 模型大小 (tiny, base, small, medium, large, large-v2, large-v3)
            device: 设备类型 (cpu, cuda, auto)
            compute_type: 计算类型 (int8, int16, float16, float32, auto)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.model_loaded = False
        
        # 支持的语言代码
        self.supported_languages = {
            'zh': '中文', 'en': '英语', 'ja': '日语', 'ko': '韩语',
            'es': '西班牙语', 'fr': '法语', 'de': '德语', 'ru': '俄语',
            'ar': '阿拉伯语', 'hi': '印地语', 'pt': '葡萄牙语', 'it': '意大利语'
        }
        
        # 字幕格式支持
        self.subtitle_formats = ['srt', 'vtt', 'txt', 'json']
        
        logger.info(f"Whisper ASR服务初始化 - 模型: {model_size}, 设备: {device}")
    
    def _load_model(self) -> bool:
        """延迟加载Whisper模型"""
        if not WHISPER_AVAILABLE:
            logger.error("faster-whisper未安装，请运行: pip install faster-whisper")
            return False
        
        if self.model_loaded:
            return True
        
        try:
            logger.info(f"正在加载Whisper模型: {self.model_size}")
            start_time = time.time()
            
            # 自动选择设备和计算类型
            if self.device == "auto":
                try:
                    import torch
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    self.device = "cpu"
            
            if self.compute_type == "auto":
                self.compute_type = "float16" if self.device == "cuda" else "int8"
            
            # 加载模型
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(MODEL_DIR / "whisper")
            )
            
            load_time = time.time() - start_time
            self.model_loaded = True
            
            logger.info(f"✅ Whisper模型加载成功 - 耗时: {load_time:.2f}秒")
            logger.info(f"   设备: {self.device}, 计算类型: {self.compute_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Whisper模型加载失败: {str(e)}")
            return False
    
    def extract_audio_from_video(self, video_path: str, audio_path: str = None, 
                                sample_rate: int = 16000) -> str:
        """
        从视频文件中提取音频
        
        Args:
            video_path: 视频文件路径
            audio_path: 输出音频路径（可选）
            sample_rate: 采样率
            
        Returns:
            提取的音频文件路径
        """
        try:
            if not audio_path:
                video_stem = Path(video_path).stem
                audio_path = f"temp_audio_{video_stem}_{int(time.time())}.wav"
            
            # 使用FFmpeg提取音频
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", video_path,
                "-ar", str(sample_rate),  # 采样率
                "-ac", "1",               # 单声道
                "-c:a", "pcm_s16le",      # PCM编码
                "-y",                     # 覆盖输出文件
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"音频提取失败: {result.stderr}")
            
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"音频文件未生成: {audio_path}")
            
            logger.info(f"✅ 音频提取成功: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"❌ 音频提取失败: {str(e)}")
            raise
    
    def detect_language(self, audio_path: str) -> Tuple[str, float]:
        """
        检测音频语言
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            (语言代码, 置信度)
        """
        if not self._load_model():
            return "en", 0.0
        
        try:
            # 使用Whisper检测语言
            segments, info = self.model.transcribe(
                audio_path, 
                language=None,  # 自动检测
                task="transcribe",
                vad_filter=True,  # 启用语音活动检测
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            detected_language = info.language
            confidence = info.language_probability
            
            language_name = self.supported_languages.get(detected_language, detected_language)
            
            logger.info(f"🌐 检测到语言: {language_name} ({detected_language}) - 置信度: {confidence:.3f}")
            
            return detected_language, confidence
            
        except Exception as e:
            logger.error(f"❌ 语言检测失败: {str(e)}")
            return "en", 0.0
    
    def transcribe_audio(self, audio_path: str, language: str = None, 
                        task: str = "transcribe", **kwargs) -> Dict:
        """
        转录音频文件
        
        Args:
            audio_path: 音频文件路径
            language: 指定语言（None为自动检测）
            task: 任务类型 (transcribe/translate)
            **kwargs: 其他Whisper参数
            
        Returns:
            转录结果字典
        """
        if not self._load_model():
            raise RuntimeError("Whisper模型未加载")
        
        try:
            logger.info(f"🎤 开始转录音频: {audio_path}")
            start_time = time.time()
            
            # 默认参数
            default_params = {
                "beam_size": 5,
                "best_of": 5,
                "patience": 1,
                "length_penalty": 1,
                "repetition_penalty": 1,
                "no_repeat_ngram_size": 0,
                "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
                "prompt_reset_on_temperature": 0.5,
                "initial_prompt": None,
                "prefix": None,
                "suppress_blank": True,
                "suppress_tokens": [-1],
                "without_timestamps": False,
                "max_initial_timestamp": 1.0,
                "word_timestamps": False,
                "prepend_punctuations": "\"'([{-",
                "append_punctuations": "\"'.。,，!！?？:：\")]}、",
                "vad_filter": True,
                "vad_parameters": dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=400
                )
            }
            
            # 合并用户参数
            params = {**default_params, **kwargs}
            
            # 执行转录
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                **params
            )
            
            # 处理转录结果
            transcription_result = self._process_transcription_result(segments, info)
            
            processing_time = time.time() - start_time
            transcription_result['processing_time'] = processing_time
            
            logger.info(f"✅ 转录完成 - 耗时: {processing_time:.2f}秒")
            logger.info(f"   语言: {info.language} - 时长: {info.duration:.2f}秒")
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"❌ 音频转录失败: {str(e)}")
            raise
    
    def _process_transcription_result(self, segments, info) -> Dict:
        """处理转录结果"""
        result = {
            'language': info.language,
            'language_probability': info.language_probability,
            'duration': info.duration,
            'segments': [],
            'full_text': '',
            'word_count': 0,
            'segment_count': 0
        }
        
        full_text_parts = []
        
        for segment in segments:
            segment_data = {
                'id': segment.id,
                'seek': segment.seek,
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'tokens': segment.tokens,
                'temperature': segment.temperature,
                'avg_logprob': segment.avg_logprob,
                'compression_ratio': segment.compression_ratio,
                'no_speech_prob': segment.no_speech_prob
            }
            
            result['segments'].append(segment_data)
            full_text_parts.append(segment.text.strip())
        
        result['full_text'] = ' '.join(full_text_parts)
        result['word_count'] = len(result['full_text'].split())
        result['segment_count'] = len(result['segments'])
        
        return result
    
    def transcribe_video(self, video_path: str, language: str = None, 
                        cleanup_audio: bool = True, **kwargs) -> Dict:
        """
        直接转录视频文件
        
        Args:
            video_path: 视频文件路径
            language: 指定语言
            cleanup_audio: 是否清理临时音频文件
            **kwargs: 其他转录参数
            
        Returns:
            转录结果
        """
        temp_audio_path = None
        
        try:
            # 提取音频
            temp_audio_path = self.extract_audio_from_video(video_path)
            
            # 如果没有指定语言，先检测语言
            if language is None:
                detected_lang, confidence = self.detect_language(temp_audio_path)
                if confidence > 0.7:  # 置信度阈值
                    language = detected_lang
                    logger.info(f"🎯 使用检测到的语言: {language} (置信度: {confidence:.3f})")
            
            # 转录音频
            result = self.transcribe_audio(temp_audio_path, language=language, **kwargs)
            
            # 添加视频信息
            result['video_path'] = video_path
            result['audio_extracted'] = True
            
            return result
            
        finally:
            # 清理临时音频文件
            if cleanup_audio and temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                    logger.info(f"🗑️  临时音频文件已清理: {temp_audio_path}")
                except Exception as e:
                    logger.warning(f"⚠️  清理临时音频文件失败: {e}")
    
    def generate_subtitles(self, transcription_result: Dict, 
                          format: str = "srt", output_path: str = None) -> str:
        """
        生成字幕文件
        
        Args:
            transcription_result: 转录结果
            format: 字幕格式 (srt, vtt, txt, json)
            output_path: 输出文件路径
            
        Returns:
            字幕文件路径
        """
        if format not in self.subtitle_formats:
            raise ValueError(f"不支持的字幕格式: {format}")
        
        if not output_path:
            timestamp = int(time.time())
            output_path = f"subtitles_{timestamp}.{format}"
        
        try:
            segments = transcription_result.get('segments', [])
            
            if format == "srt":
                content = self._generate_srt(segments)
            elif format == "vtt":
                content = self._generate_vtt(segments)
            elif format == "txt":
                content = self._generate_txt(segments)
            elif format == "json":
                content = json.dumps(transcription_result, ensure_ascii=False, indent=2)
            
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"✅ 字幕文件生成成功: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ 字幕生成失败: {str(e)}")
            raise
    
    def _generate_srt(self, segments: List[Dict]) -> str:
        """生成SRT格式字幕"""
        srt_content = []
        
        for i, segment in enumerate(segments, 1):
            start_time = self._seconds_to_srt_time(segment['start'])
            end_time = self._seconds_to_srt_time(segment['end'])
            text = segment['text'].strip()
            
            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")  # 空行
        
        return "\n".join(srt_content)
    
    def _generate_vtt(self, segments: List[Dict]) -> str:
        """生成VTT格式字幕"""
        vtt_content = ["WEBVTT", ""]
        
        for segment in segments:
            start_time = self._seconds_to_vtt_time(segment['start'])
            end_time = self._seconds_to_vtt_time(segment['end'])
            text = segment['text'].strip()
            
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text)
            vtt_content.append("")
        
        return "\n".join(vtt_content)
    
    def _generate_txt(self, segments: List[Dict]) -> str:
        """生成纯文本格式"""
        txt_content = []
        
        for segment in segments:
            timestamp = self._seconds_to_readable_time(segment['start'])
            text = segment['text'].strip()
            txt_content.append(f"[{timestamp}] {text}")
        
        return "\n".join(txt_content)
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """将秒转换为SRT时间格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """将秒转换为VTT时间格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def _seconds_to_readable_time(self, seconds: float) -> str:
        """将秒转换为可读时间格式"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_size': self.model_size,
            'device': self.device,
            'compute_type': self.compute_type,
            'model_loaded': self.model_loaded,
            'whisper_available': WHISPER_AVAILABLE,
            'supported_languages': self.supported_languages,
            'subtitle_formats': self.subtitle_formats
        }
    
    def cleanup(self):
        """清理资源"""
        if self.model:
            del self.model
            self.model = None
            self.model_loaded = False
            logger.info("🧹 Whisper模型资源已清理")


# Cache one service per model/device pair so a previous request cannot silently
# force later requests to use the wrong Whisper model.
asr_services: Dict[Tuple[str, str], WhisperASRService] = {}

def get_asr_service(model_size: str = "base", device: str = "auto") -> WhisperASRService:
    """Get a cached ASR service for the requested model and device."""
    key = (model_size, device)
    if key not in asr_services:
        asr_services[key] = WhisperASRService(model_size=model_size, device=device)
    return asr_services[key]

def transcribe_video_file(video_path: str, language: str = None, 
                         subtitle_format: str = "srt", model_size: str = "base",
                         device: str = "auto", **kwargs) -> Dict:
    """
    便捷函数：转录视频文件并生成字幕
    
    Args:
        video_path: 视频文件路径
        language: 指定语言
        subtitle_format: 字幕格式
        **kwargs: 其他参数
        
    Returns:
        包含转录结果和字幕文件路径的字典
    """
    service = get_asr_service(model_size=model_size, device=device)
    
    # 转录视频
    result = service.transcribe_video(video_path, language=language, **kwargs)
    
    # 生成字幕文件
    if subtitle_format and subtitle_format != "none":
        video_stem = Path(video_path).stem
        subtitle_path = str(OUTPUT_DIR / f"{video_stem}_subtitles.{subtitle_format}")
        
        subtitle_file = service.generate_subtitles(
            result, 
            format=subtitle_format, 
            output_path=subtitle_path
        )
        
        result['subtitle_file'] = subtitle_file
    
    return result
