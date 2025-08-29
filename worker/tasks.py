import os
import json
import time
import subprocess
from pathlib import Path
from urllib.request import urlretrieve
from typing import List, Dict, Any

from worker.celery_app import celery_app
import logging

# Import ASR functionality
try:
    from core.whisper_asr import transcribe_video_file, get_asr_service
    from core.semantic_analysis import analyze_transcription_semantics
    from core.asr_smart_clipping import select_segments_with_asr
    ASR_AVAILABLE = True
    SEMANTIC_ANALYSIS_AVAILABLE = True
except ImportError:
    ASR_AVAILABLE = False
    SEMANTIC_ANALYSIS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import core processing functions from api.main
# We'll create a shared processing module to avoid circular imports
from core.settings import settings


@celery_app.task(bind=True, name="worker.tasks.process_video_async")
def process_video_async(self, url: str, min_sec: int = 15, max_sec: int = 25, want_asr: bool = False, output: str = ""):
    """
    Asynchronous video processing task.
    This moves the heavy processing from the API to background workers.
    """
    logger.info(f"Starting async video processing task {self.request.id} for URL: {url}")
    
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'status': 'Downloading video...', 'progress': 10})
        
        # Create directories
        Path("input_data/downloads").mkdir(parents=True, exist_ok=True)
        Path("output_data").mkdir(parents=True, exist_ok=True)
        
        # Download video (simplified version for now)
        ts = int(time.time())
        if url.startswith("file://"):
            # Local file
            src_path = url.replace("file://", "")
            dl_path = src_path
        else:
            # Download from URL
            dl_path = str(Path("input_data/downloads") / f"async_dl_{ts}.mp4")
            try:
                # Try yt-dlp CLI first
                cli_cmd = [
                    "yt-dlp", "-f", "bv*+ba/b", "--merge-output-format", "mp4",
                    "-o", dl_path, url
                ]
                result = subprocess.run(cli_cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    # Fallback to direct download
                    urlretrieve(url, dl_path)
            except Exception as e:
                logger.error(f"Download failed: {e}")
                raise Exception(f"Failed to download video: {str(e)}")
        
        self.update_state(state='PROGRESS', meta={'status': 'Processing video...', 'progress': 40})
        
        # Get video duration
        probe_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", dl_path]
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            total_duration = float(result.stdout.strip()) if result.stdout.strip() else 0
        except:
            total_duration = 0
            
        if total_duration == 0:
            raise Exception("Could not determine video duration")
        
        # Simple window selection (start from beginning for now)
        start_time = "00:00:00"
        duration = min(min_sec, int(total_duration))
        
        self.update_state(state='PROGRESS', meta={'status': 'Generating clip...', 'progress': 70})
        
        # Generate output path
        if not output:
            output = str(Path("output_data") / f"async_intro_{ts}.mp4")
        
        # FFmpeg command for 9:16 conversion
        fade_out_start = max(0.1, duration - 0.25)
        vf_filter = (
            "scale=1080:1920:force_original_aspect_ratio=decrease,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1:1,"
            f"fade=t=in:st=0:d=0.25,fade=t=out:st={fade_out_start:.2f}:d=0.25"
        )
        
        cmd = [
            "ffmpeg", "-y", "-hwaccel", "none",
            "-i", dl_path,
            "-ss", start_time, "-t", str(duration),
            "-vf", vf_filter,
            "-pix_fmt", "yuv420p",
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output
        ]
        
        logger.info(f"Running FFmpeg: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            raise Exception(f"Video processing failed: {result.stderr}")
        
        self.update_state(state='PROGRESS', meta={'status': 'Finalizing...', 'progress': 90})
        
        # Verify output file exists
        if not os.path.exists(output):
            raise Exception("Output file was not created")
        
        file_size = os.path.getsize(output)
        logger.info(f"Task {self.request.id} completed successfully. Output: {output} ({file_size} bytes)")
        
        return {
            "status": "completed",
            "output": output,
            "file_size": file_size,
            "duration": duration,
            "clips": [{"start": start_time, "end": f"00:00:{duration:02d}", "reason": "async_window"}]
        }
        
    except Exception as e:
        logger.error(f"Task {self.request.id} failed: {e}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name='asr_transcribe_async')
def asr_transcribe_async(self, url: str, language: str = None, subtitle_format: str = "srt", 
                        task: str = "transcribe", model_size: str = "base"):
    """
    异步ASR转录任务
    
    Args:
        url: 视频/音频文件URL或路径
        language: 指定语言代码
        subtitle_format: 字幕格式
        task: 任务类型 (transcribe/translate)
        model_size: 模型大小
    """
    if not ASR_AVAILABLE:
        self.update_state(state='FAILURE', meta={
            'error': 'ASR service not available. Please install faster-whisper.'
        })
        raise RuntimeError("ASR service not available")
    
    try:
        logger.info(f"Starting async ASR transcription task {self.request.id} for URL: {url}")
        
        # 更新任务状态
        self.update_state(state='PROGRESS', meta={'status': 'Preparing input...', 'progress': 10})
        
        # 准备输入文件
        ts = int(time.time())
        if url.startswith("file:"):
            input_path = url.replace("file://", "")
            if not Path(input_path).exists():
                raise FileNotFoundError(f"Local file not found: {input_path}")
        else:
            # 下载文件
            input_path = str(Path("input_data/downloads") / f"asr_async_{ts}.mp4")
            Path("input_data/downloads").mkdir(parents=True, exist_ok=True)
            urlretrieve(url, input_path)
        
        self.update_state(state='PROGRESS', meta={'status': 'Loading ASR model...', 'progress': 20})
        
        # 执行转录
        self.update_state(state='PROGRESS', meta={'status': 'Transcribing audio...', 'progress': 30})
        
        result = transcribe_video_file(
            input_path,
            language=language,
            subtitle_format=subtitle_format,
            task=task,
            model_size=model_size
        )
        
        self.update_state(state='PROGRESS', meta={'status': 'Finalizing results...', 'progress': 90})
        
        # 准备返回结果
        response = {
            'input_path': input_path,
            'language': result.get('language'),
            'language_probability': result.get('language_probability'),
            'duration': result.get('duration'),
            'full_text': result.get('full_text'),
            'word_count': result.get('word_count'),
            'segment_count': result.get('segment_count'),
            'processing_time': result.get('processing_time'),
            'subtitle_file': result.get('subtitle_file'),
            'subtitle_format': subtitle_format if subtitle_format != "none" else None,
            'segments_sample': result.get('segments', [])[:5],  # 只返回前5个段落作为样例
            'task_id': self.request.id
        }
        
        logger.info(f"ASR transcription task {self.request.id} completed successfully")
        
        return {
            'status': 'completed',
            'result': response,
            'message': f"转录完成 - 语言: {result.get('language')}, 文本长度: {result.get('word_count')}词"
        }
        
    except Exception as e:
        logger.error(f"ASR transcription task failed: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name='extract_audio_async')  
def extract_audio_async(self, url: str, sample_rate: int = 16000):
    """
    异步音频提取任务
    
    Args:
        url: 视频文件URL或路径
        sample_rate: 采样率
    """
    try:
        logger.info(f"Starting async audio extraction task {self.request.id} for URL: {url}")
        
        self.update_state(state='PROGRESS', meta={'status': 'Preparing input...', 'progress': 20})
        
        # 准备输入文件
        ts = int(time.time())
        if url.startswith("file:"):
            input_path = url.replace("file://", "")
            if not Path(input_path).exists():
                raise FileNotFoundError(f"Local file not found: {input_path}")
        else:
            input_path = str(Path("input_data/downloads") / f"audio_extract_async_{ts}.mp4")
            Path("input_data/downloads").mkdir(parents=True, exist_ok=True)
            urlretrieve(url, input_path)
        
        self.update_state(state='PROGRESS', meta={'status': 'Extracting audio...', 'progress': 50})
        
        # 提取音频
        from core.whisper_asr import get_asr_service
        asr_service = get_asr_service()
        
        video_stem = Path(input_path).stem
        audio_path = f"output_data/{video_stem}_audio_async_{ts}.wav"
        Path("output_data").mkdir(parents=True, exist_ok=True)
        
        extracted_audio = asr_service.extract_audio_from_video(
            input_path,
            audio_path=audio_path,
            sample_rate=sample_rate
        )
        
        self.update_state(state='PROGRESS', meta={'status': 'Getting audio info...', 'progress': 80})
        
        # 获取音频信息
        audio_info = {}
        try:
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
                "-show_streams", extracted_audio
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                probe_data = json.loads(result.stdout)
                if 'format' in probe_data:
                    audio_info = {
                        'duration': float(probe_data['format'].get('duration', 0)),
                        'size': int(probe_data['format'].get('size', 0)),
                        'bit_rate': probe_data['format'].get('bit_rate'),
                    }
                if 'streams' in probe_data and probe_data['streams']:
                    stream = probe_data['streams'][0]
                    audio_info.update({
                        'sample_rate': stream.get('sample_rate'),
                        'channels': stream.get('channels'),
                        'codec': stream.get('codec_name')
                    })
        except Exception as e:
            logger.warning(f"Failed to get audio info: {e}")
        
        logger.info(f"Audio extraction task {self.request.id} completed successfully")
        
        return {
            'status': 'completed',
            'result': {
                'input_path': input_path,
                'audio_path': extracted_audio,
                'sample_rate': sample_rate,
                'audio_info': audio_info,
                'task_id': self.request.id
            },
            'message': f"音频提取完成: {extracted_audio}"
        }
        
    except Exception as e:
        logger.error(f"Audio extraction task failed: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name="worker.tasks.asr_smart_clipping_async")
def asr_smart_clipping_async(self, url: str, min_sec: int = 15, max_sec: int = 25, 
                           count: int = 1, model_size: str = "base", language: str = None,
                           output_prefix: str = "asr_smart_clip"):
    """
    ASR增强智能切片异步任务
    结合语音识别和语义分析进行智能选段
    """
    if not ASR_AVAILABLE or not SEMANTIC_ANALYSIS_AVAILABLE:
        raise RuntimeError("ASR或语义分析功能不可用")
    
    logger.info(f"Starting ASR enhanced smart clipping task {self.request.id} for URL: {url}")
    
    try:
        # 任务状态更新
        self.update_state(state='PROGRESS', meta={'status': '准备处理...', 'progress': 5})
        
        # 创建目录
        Path("input_data/downloads").mkdir(parents=True, exist_ok=True)
        Path("output_data").mkdir(parents=True, exist_ok=True)
        
        # 处理输入文件
        ts = int(time.time())
        if url.startswith("file:"):
            local_path = url.replace("file://", "")
            if not Path(local_path).exists():
                raise FileNotFoundError("本地文件不存在")
            input_path = local_path
        else:
            input_path = str(Path("input_data/downloads") / f"asr_smart_{ts}.mp4")
            self.update_state(state='PROGRESS', meta={'status': '下载视频...', 'progress': 10})
            urlretrieve(url, input_path)
        
        # 1. ASR转录
        self.update_state(state='PROGRESS', meta={'status': '语音识别转录...', 'progress': 20})
        asr_service = get_asr_service(model_size=model_size)
        transcription_result = asr_service.transcribe_video(
            input_path,
            language=language,
            cleanup_audio=True
        )
        
        logger.info(f"ASR转录完成 - 语言: {transcription_result['language']}, 文本长度: {transcription_result['word_count']}词")
        
        # 2. 语义分析
        self.update_state(state='PROGRESS', meta={'status': '语义分析...', 'progress': 40})
        enhanced_transcription = analyze_transcription_semantics(transcription_result)
        
        # 3. ASR增强智能选段
        self.update_state(state='PROGRESS', meta={'status': '智能选段分析...', 'progress': 60})
        selected_segments = select_segments_with_asr(
            input_path,
            enhanced_transcription,
            min_sec,
            max_sec,
            count
        )
        
        if not selected_segments:
            raise ValueError("未找到符合条件的智能片段")
        
        # 4. 生成视频片段
        self.update_state(state='PROGRESS', meta={'status': '生成视频片段...', 'progress': 70})
        generated_clips = []
        
        for i, segment in enumerate(selected_segments):
            progress = 70 + (i + 1) / len(selected_segments) * 25
            self.update_state(
                state='PROGRESS', 
                meta={'status': f'生成片段 {i+1}/{len(selected_segments)}...', 'progress': int(progress)}
            )
            
            output_filename = f"{output_prefix}_{ts}_{i+1:02d}.mp4"
            output_path = f"output_data/{output_filename}"
            
            start_time = segment['start_hms']
            duration = segment['duration']
            
            # 使用pad-first策略生成9:16视频
            fade_out_start = max(0.1, duration - 0.25)
            vf_filters = (
                "scale=1080:1920:force_original_aspect_ratio=decrease,"
                "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1:1,"
                f"fade=t=in:st=0:d=0.25,fade=t=out:st={fade_out_start:.2f}:d=0.25"
            )
            
            cmd = [
                "ffmpeg", "-y", "-hwaccel", "none",
                "-i", input_path,
                "-ss", start_time, "-t", f"{duration:.2f}",
                "-vf", vf_filters,
                "-pix_fmt", "yuv420p",
                "-map", "0:v:0", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-shortest", "-movflags", "+faststart",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(f"视频处理失败: {result.stderr}")
            
            # 获取生成的视频信息
            file_size = Path(output_path).stat().st_size if Path(output_path).exists() else 0
            
            generated_clips.append({
                "clip_index": i + 1,
                "output_path": output_path,
                "start_time": start_time,
                "duration": duration,
                "file_size": file_size,
                "visual_score": segment['visual_score'],
                "content_score": segment['content_score'],
                "total_score": segment['total_score'],
                "selection_type": segment['type']
            })
        
        self.update_state(state='PROGRESS', meta={'status': '完成处理...', 'progress': 95})
        
        logger.info(f"ASR enhanced smart clipping task {self.request.id} completed successfully")
        
        return {
            'status': 'completed',
            'result': {
                'input_path': input_path,
                'transcription': {
                    "language": transcription_result["language"],
                    "language_probability": transcription_result["language_probability"],
                    "duration": transcription_result["duration"],
                    "word_count": transcription_result["word_count"],
                    "segment_count": transcription_result["segment_count"]
                },
                'semantic_analysis': {
                    'keywords': enhanced_transcription['semantic_analysis']['keywords'][:5],
                    'sentiment': enhanced_transcription['semantic_analysis']['sentiment'],
                    'quality_score': enhanced_transcription['semantic_analysis']['quality_score']['overall_score']
                },
                'selected_segments': len(selected_segments),
                'generated_clips': generated_clips,
                'processing_summary': {
                    "asr_enhanced": True,
                    "semantic_analysis": True,
                    "visual_analysis": True,
                    "selection_algorithm": "asr_smart_clipping"
                },
                'task_id': self.request.id
            },
            'message': f"ASR增强智能切片完成 - 生成了 {len(generated_clips)} 个高质量片段"
        }
        
    except Exception as e:
        logger.error(f"ASR enhanced smart clipping task failed: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True, name="worker.tasks.semantic_analysis_async")
def semantic_analysis_async(self, text: str, include_keywords: bool = True,
                          include_sentiment: bool = True, include_topics: bool = True,
                          include_quality: bool = True):
    """
    语义分析异步任务
    """
    if not SEMANTIC_ANALYSIS_AVAILABLE:
        raise RuntimeError("语义分析功能不可用")
    
    logger.info(f"Starting semantic analysis task {self.request.id}")
    
    try:
        self.update_state(state='PROGRESS', meta={'status': '开始语义分析...', 'progress': 10})
        
        from core.semantic_analysis import get_semantic_analyzer
        analyzer = get_semantic_analyzer()
        result = {}
        
        # 关键词提取
        if include_keywords:
            self.update_state(state='PROGRESS', meta={'status': '提取关键词...', 'progress': 25})
            result['keywords'] = analyzer.extract_keywords(text, top_k=10)
        
        # 情感分析
        if include_sentiment:
            self.update_state(state='PROGRESS', meta={'status': '情感分析...', 'progress': 50})
            result['sentiment'] = analyzer.analyze_sentiment(text)
        
        # 主题相关性
        if include_topics:
            self.update_state(state='PROGRESS', meta={'status': '主题分析...', 'progress': 75})
            result['topic_relevance'] = analyzer.analyze_topic_relevance(text)
        
        # 内容质量评分
        if include_quality:
            self.update_state(state='PROGRESS', meta={'status': '质量评分...', 'progress': 90})
            result['quality_score'] = analyzer.calculate_content_quality_score(text)
        
        logger.info(f"Semantic analysis task {self.request.id} completed successfully")
        
        return {
            'status': 'completed',
            'result': {
                "text_length": len(text),
                "word_count": len(text.split()),
                "analysis": result,
                'task_id': self.request.id
            },
            'message': "语义分析完成"
        }
        
    except Exception as e:
        logger.error(f"Semantic analysis task failed: {str(e)}")
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


