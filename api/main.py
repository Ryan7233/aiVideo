import os
import subprocess
import time
import asyncio
import hmac
from pathlib import Path
from typing import List, Dict, Any, Optional
import aiofiles
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator, model_validator
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
import uuid
from contextlib import asynccontextmanager

# Import configuration and validation functions
from core.concurrency import run_ffmpeg as _run_ffmpeg_gated
from core.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    AUDIO_BITRATE,
    MAX_FILE_SIZE,
    VIDEO_CRF,
    VIDEO_FPS,
    validate_video_extension,
)
from core.runtime import (
    INPUT_DIR,
    LOG_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    VIDEO_UPLOAD_DIR,
    ensure_runtime_directories,
    materialize_video_source as resolve_video_source,
    resolve_media_path,
    resolve_output_path,
    validate_remote_url,
)
from core.evaluation import evaluate_selection
from core import job_store
from core.smart_clipping import get_smart_segments, analyze_video_intelligence
from core.whisper_asr import get_asr_service
from core.semantic_analysis import get_semantic_analyzer
from core.video_workflow import process_multi_segment_video
from core import retention
from core.jobs import shutdown as shutdown_jobs
from routers.jobs import router as jobs_router

# Setup runtime and logging before mounting static directories.
ensure_runtime_directories()
logger.add(str(LOG_DIR / "api.log"), rotation="10 MB", level="INFO")
logger.add(str(LOG_DIR / "api.jsonl"), rotation="10 MB", level="INFO", serialize=True)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start the retention sweeper, and stop background workers on shutdown."""
    interval_hours = retention.sweep_interval_hours()
    task = None
    if interval_hours > 0:
        async def _sweep_loop() -> None:
            while True:
                try:
                    await asyncio.to_thread(retention.run_sweep)
                except Exception as exc:  # a failed sweep must not kill the loop
                    logger.warning(f"Retention sweep failed: {exc}")
                await asyncio.sleep(interval_hours * 3600)

        task = asyncio.create_task(_sweep_loop())
        logger.info(f"Retention sweeper started, every {interval_hours}h")
    else:
        logger.info("Retention sweeper disabled (RETENTION_SWEEP_INTERVAL_HOURS=0)")

    try:
        yield
    finally:
        if task:
            task.cancel()
        shutdown_jobs(wait=False)


app = FastAPI(
    title="AI Video Clipper API",
    description="智能短视频自动切片和文案生成服务",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID middleware for tracing
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response

app.add_middleware(RequestIDMiddleware)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Optionally protect API routes when AIVIDEO_API_KEY is configured."""

    async def dispatch(self, request, call_next):
        configured_key = os.getenv("AIVIDEO_API_KEY", "")
        public_prefixes = ("/static", "/docs", "/openapi.json", "/redoc")
        if configured_key and request.url.path not in {"/", "/health", "/info"}:
            if not request.url.path.startswith(public_prefixes):
                supplied_key = request.headers.get("X-API-Key", "")
                if not hmac.compare_digest(supplied_key, configured_key):
                    return JSONResponse(status_code=401, content={"detail": "无效或缺失的 API Key"})
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response


app.add_middleware(APIKeyMiddleware)

# Create logs directory if it doesn't exist
ensure_runtime_directories()

# Routers
app.include_router(jobs_router)

# Mount static files and frontend
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "frontend")), name="static")
# The payload reports artefacts as "output_data/<name>" -- a path relative to
# the data root, which is what the batch script prints. Mounting the directory
# under that same prefix means a client can fetch what it was handed without
# knowing the mapping. /output stays for anything already using it.
app.mount("/output_data", StaticFiles(directory=str(OUTPUT_DIR)), name="output_data")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
# The source itself, so the UI can play a job's video back next to its cut
# list -- after a reload the browser no longer has the uploaded File. Same
# API-key gate as the outputs; a swept file simply 404s.
app.mount("/input_data", StaticFiles(directory=str(INPUT_DIR)), name="input_data")

@app.post("/admin/retention/sweep", tags=["admin"])
async def trigger_retention_sweep() -> Dict[str, Any]:
    """Run the retention sweep now and report what was reclaimed."""
    return await asyncio.to_thread(retention.run_sweep)


@app.get("/", tags=["system"])
async def serve_frontend():
    """Serve the main frontend page"""
    return FileResponse(str(PROJECT_ROOT / "frontend" / "index.html"))

# Exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = [{key: value for key, value in item.items() if key != "ctx"} for item in exc.errors()]
    logger.error(f"Validation error: {errors}")
    return JSONResponse(
        status_code=422,
        content={"detail": errors, "message": "请求参数验证失败"},
    )

# --- Pydantic Models with Validation ---


class CutReq(BaseModel):
    src: str
    start: str
    end: str
    out: str
    
    @field_validator('src')
    @classmethod
    def validate_src_file(cls, v):
        path = resolve_media_path(v)
        if not validate_video_extension(str(path)):
            raise ValueError(f'不支持的视频格式: {v}')
        return str(path)
    
    @field_validator('start', 'end')
    @classmethod
    def validate_time_format(cls, v):
        import re
        if not re.match(r'^\d{2}:\d{2}(:\d{2})?$', v):
            raise ValueError(f'时间格式错误，应为 mm:ss 或 hh:mm:ss: {v}')
        return v

class BurnSubReq(BaseModel):
    src: str
    srt: str
    out: str
    
    @field_validator('src')
    @classmethod
    def validate_src_file(cls, v):
        return str(resolve_media_path(v))
    
    @field_validator('srt')
    @classmethod
    def validate_srt_file(cls, v):
        path = resolve_media_path(v)
        if path.suffix.lower() not in {'.srt', '.ass', '.vtt'}:
            raise ValueError('字幕文件格式必须是 srt/ass/vtt')
        return str(path)


# --- Utility Functions ---
def normalize_time(time_str: str) -> str:
    """Normalize time format to hh:mm:ss"""
    parts = time_str.split(':')
    if len(parts) == 2:
        return f"00:{time_str}"
    return time_str


def safe_run_ffmpeg(cmd: List[str], timeout: int = 300) -> Dict[str, Any]:
    """Safely run ffmpeg command with timeout and error handling"""
    try:
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        start_time = time.time()
        
        process = _run_ffmpeg_gated(cmd, timeout=timeout, encoding='utf-8')
        
        duration = time.time() - start_time
        logger.info(f"FFmpeg completed in {duration:.2f}s with return code: {process.returncode}")
        
        if process.returncode != 0:
            logger.error(f"FFmpeg failed: {process.stderr}")
            raise HTTPException(
                status_code=500, 
                detail=f"视频处理失败: {process.stderr}"
            )
        
        return {
            "code": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "duration": duration
        }
        
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg command timed out")
        raise HTTPException(status_code=504, detail="视频处理超时")
    except Exception as e:
        logger.error(f"FFmpeg execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"视频处理异常: {str(e)}")


# --- Gemini Mock Function with Error Handling ---


async def materialize_video_source(url: str, prefix: str) -> str:
    """Resolve a managed local video or securely download a public remote one."""
    path = await asyncio.to_thread(resolve_video_source, url, prefix, MAX_FILE_SIZE)
    return str(path)

# --- API Endpoints ---
@app.get("/info", tags=["system"])
async def root():
    """Basic service information."""
    return {"message": "AI Video Clipper API is running", "version": "1.0.0"}

@app.get("/health", tags=["system"])
async def health_check():
    """详细健康检查端点"""
    return {
        "status": "healthy", 
        "timestamp": int(time.time()),
        "version": "1.0.0",
        "services": {
            "api": "running",
            "smart_clipping": "available"
        }
    }


@app.post("/cut916", tags=["video"])
async def cut916(req: CutReq):
    """生成9:16竖屏视频"""
    try:
        logger.info(f"Cutting video: {req.src} ({req.start} - {req.end})")
        
        # Normalize time format
        start_time = normalize_time(req.start)
        end_time = normalize_time(req.end)
        output_path = resolve_output_path(req.out, f"clip_{uuid.uuid4().hex}.mp4")
        
        # Robust 9:16 pipeline using expressions (no FOAR option):
        # - If input is wider than 9:16, scale height to 1920 and width proportional; else scale width to 1080
        # - Then center crop to exactly 1080x1920; set pixel format and SAR for compatibility
        vf_filters = (
            r"scale="
            r"if(gte(iw/ih\,1080/1920)\,-2\,1080):"
            r"if(gte(iw/ih\,1080/1920)\,1920\,-2),"
            "crop=1080:1920,format=yuv420p,setsar=1:1"
        )

        cmd = [
            "ffmpeg", "-y",
            # Use output seeking (place -ss/-to after -i) to keep A/V sync and audio reliably
            "-i", req.src,
            "-ss", start_time,
            "-to", end_time,
            "-vf", vf_filters,
            "-r", str(VIDEO_FPS),
            "-pix_fmt", "yuv420p",
            # Explicitly map first video and (optional) first audio stream to avoid accidental drops
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", str(VIDEO_CRF),
            "-c:a", "aac",
            "-b:a", AUDIO_BITRATE,
            "-movflags", "+faststart",
            str(output_path)
        ]
        
        result = await asyncio.to_thread(safe_run_ffmpeg, cmd)
        result["out"] = str(output_path)

        logger.info(f"Successfully created 9:16 video: {output_path}")
        return result
        
    except Exception as e:
        logger.error(f"9:16 video cut error: {str(e)}")
        # Clean up partial output file if it exists
        try:
            resolve_output_path(req.out, "failed.mp4").unlink(missing_ok=True)
        except ValueError:
            pass
        raise HTTPException(status_code=500, detail=f"9:16视频生成失败: {str(e)}")

@app.post("/burnsub", tags=["video"])
async def burnsub(req: BurnSubReq):
    """烧录字幕到视频"""
    try:
        logger.info(f"Burning subtitles: {req.srt} -> {req.src}")
        
        output_path = resolve_output_path(req.out, f"subtitled_{uuid.uuid4().hex}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", req.src,
            "-vf", f"subtitles={req.srt}:force_style='Fontsize=28'",
            "-c:a", "copy",
            str(output_path)
        ]
        
        result = await asyncio.to_thread(safe_run_ffmpeg, cmd)
        result["out"] = str(output_path)

        logger.info(f"Successfully burned subtitles: {output_path}")
        return result
        
    except Exception as e:
        logger.error(f"Subtitle burning error: {str(e)}")
        # Clean up partial output file if it exists
        try:
            resolve_output_path(req.out, "failed.mp4").unlink(missing_ok=True)
        except ValueError:
            pass
        raise HTTPException(status_code=500, detail=f"字幕烧录失败: {str(e)}")


# --- New: Intro-style auto highlights from URL ---

class VideoAnalysisReq(BaseModel):
    url: str
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        if not v.strip():
            raise ValueError('URL不能为空')
        v = v.strip()
        if not v.startswith("http") and not v.startswith("file:"):
            raise ValueError('仅支持 http/https/file URL')
        return v

class ASRTranscribeReq(BaseModel):
    url: str
    language: Optional[str] = None  # 语言代码，None为自动检测
    subtitle_format: str = "srt"    # 字幕格式: srt, vtt, txt, json, none
    task: str = "transcribe"        # transcribe 或 translate
    model_size: str = "base"        # tiny, base, small, medium, large
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        if not v.strip():
            raise ValueError('URL不能为空')
        v = v.strip()
        if not v.startswith("http") and not v.startswith("file:"):
            raise ValueError('仅支持 http/https/file URL')
        return v
    
    @field_validator('subtitle_format')
    @classmethod
    def validate_subtitle_format(cls, v):
        valid_formats = ['srt', 'vtt', 'txt', 'json', 'none']
        if v not in valid_formats:
            raise ValueError(f'字幕格式必须是: {", ".join(valid_formats)}')
        return v
    
    @field_validator('task')
    @classmethod
    def validate_task(cls, v):
        if v not in ['transcribe', 'translate']:
            raise ValueError('任务类型必须是: transcribe 或 translate')
        return v
    
    @field_validator('model_size')
    @classmethod
    def validate_model_size(cls, v):
        valid_sizes = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
        if v not in valid_sizes:
            raise ValueError(f'模型大小必须是: {", ".join(valid_sizes)}')
        return v

class AudioExtractionReq(BaseModel):
    url: str
    sample_rate: int = 16000
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        if not v.strip():
            raise ValueError('URL不能为空')
        v = v.strip()
        if not v.startswith("http") and not v.startswith("file:"):
            raise ValueError('仅支持 http/https/file URL')
        return v
    
    @field_validator('sample_rate')
    @classmethod
    def validate_sample_rate(cls, v):
        if v not in [8000, 16000, 22050, 44100, 48000]:
            raise ValueError('采样率必须是: 8000, 16000, 22050, 44100, 48000')
        return v

class SemanticAnalysisReq(BaseModel):
    text: str
    include_keywords: bool = True
    include_sentiment: bool = True
    include_topics: bool = True
    include_quality: bool = True
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('文本内容不能为空')
        return v.strip()


# Pro功能API模型


# 新增API模型


# Canvas and batch limits for the image endpoints. Pillow allocates
# width * height * 3 bytes up front, so an unbounded request is a
# straightforward way to exhaust the box: 10^9 x 10^9 was accepted.
MAX_CANVAS_EDGE = 8192
MAX_CANVAS_PIXELS = 16_000_000      # ~4000x4000
MAX_COLLAGE_IMAGES = 50
MAX_TEXT_BLOCKS = 30


def _validate_canvas(width: int, height: int) -> None:
    if width * height > MAX_CANVAS_PIXELS:
        raise ValueError(
            f"画布像素数不能超过 {MAX_CANVAS_PIXELS}（当前 {width}x{height}）"
        )


@app.post("/analyze_video", tags=["video"])
async def analyze_video(req: VideoAnalysisReq):
    """分析视频内容，返回智能化分析结果"""
    try:
        dl_path = await materialize_video_source(req.url, "analysis")
        
        # 执行智能分析
        logger.info(f"Starting intelligent video analysis for: {dl_path}")
        analysis_result = await asyncio.to_thread(analyze_video_intelligence, dl_path)
        
        # 获取智能片段推荐
        smart_segments = await asyncio.to_thread(get_smart_segments, dl_path, 15, 30, 5)
        
        return {
            "status": "success",
            "video_path": dl_path,
            "analysis": analysis_result,
            "recommended_segments": smart_segments,
            "message": f"成功分析视频，发现 {len(smart_segments)} 个推荐片段"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"视频分析失败: {str(e)}")


@app.post("/asr/transcribe", tags=["asr"])
async def asr_transcribe(req: ASRTranscribeReq):
    """自动语音识别 - 转录视频/音频"""
    try:
        ts = int(time.time())
        input_path = await materialize_video_source(req.url, "asr_input")
        
        logger.info(f"🎤 开始ASR转录: {input_path}")
        
        # 获取ASR服务
        asr_service = get_asr_service(model_size=req.model_size)
        
        # 转录视频
        result = await asyncio.to_thread(
            asr_service.transcribe_video,
            input_path,
            req.language,
            True,
            task=req.task,
        )
        
        # 生成字幕文件
        subtitle_file = None
        if req.subtitle_format != "none":
            video_stem = Path(input_path).stem
            subtitle_path = str(OUTPUT_DIR / f"{video_stem}_asr_{ts}.{req.subtitle_format}")
            
            subtitle_file = asr_service.generate_subtitles(
                result,
                format=req.subtitle_format,
                output_path=subtitle_path
            )
        
        return {
            "status": "success",
            "input_path": input_path,
            "language": result["language"],
            "language_probability": result["language_probability"],
            "duration": result["duration"],
            "full_text": result["full_text"],
            "word_count": result["word_count"],
            "segment_count": result["segment_count"],
            "processing_time": result["processing_time"],
            "subtitle_file": subtitle_file,
            "subtitle_format": req.subtitle_format if req.subtitle_format != "none" else None,
            "segments": result["segments"][:10] if len(result["segments"]) > 10 else result["segments"],  # 限制返回的段落数量
            "message": f"转录完成 - 检测语言: {result['language']}, 文本长度: {result['word_count']}词"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ASR转录失败: {e}")
        raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")


@app.post("/asr/extract_audio", tags=["asr"])
async def extract_audio(req: AudioExtractionReq):
    """从视频中提取音频"""
    try:
        input_path = await materialize_video_source(req.url, "audio_extract")
        
        logger.info(f"🎵 开始提取音频: {input_path}")
        
        # 获取ASR服务
        asr_service = get_asr_service()
        
        # 提取音频
        video_stem = Path(input_path).stem
        audio_path = str(OUTPUT_DIR / f"{video_stem}_audio_{int(time.time())}.wav")
        
        extracted_audio = await asyncio.to_thread(
            asr_service.extract_audio_from_video,
            input_path,
            audio_path,
            req.sample_rate,
        )
        
        # 获取音频信息
        audio_info = {}
        try:
            # ffprobe is fast but not instant, and this is a coroutine: run it
            # in a thread, and never without a timeout.
            result = await asyncio.to_thread(
                subprocess.run,
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
                 "-show_streams", extracted_audio],
                capture_output=True, text=True, timeout=30,
            )
            
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
            logger.warning(f"获取音频信息失败: {e}")
        
        return {
            "status": "success",
            "input_path": input_path,
            "audio_path": extracted_audio,
            "sample_rate": req.sample_rate,
            "audio_info": audio_info,
            "message": f"音频提取完成: {extracted_audio}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"音频提取失败: {e}")
        raise HTTPException(status_code=500, detail=f"音频提取失败: {str(e)}")


@app.get("/asr/info", tags=["asr"])
async def asr_info():
    """获取ASR服务信息"""
    try:
        asr_service = get_asr_service()
        info = asr_service.get_model_info()
        
        return {
            "status": "success",
            "asr_info": info,
            "available_models": ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
            "supported_languages": info["supported_languages"],
            "subtitle_formats": info["subtitle_formats"],
            "message": "ASR服务信息获取成功"
        }
    except Exception as e:
        logger.error(f"获取ASR信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取ASR信息失败: {str(e)}")


@app.post("/semantic/analyze", tags=["semantic"])
async def semantic_analyze(req: SemanticAnalysisReq):
    """语义分析 - 分析文本的关键词、情感、主题等"""
    try:
        logger.info(f"开始语义分析，文本长度: {len(req.text)}")
        
        analyzer = get_semantic_analyzer()
        result = {}
        
        # 关键词提取
        if req.include_keywords:
            result['keywords'] = analyzer.extract_keywords(req.text, top_k=10)
        
        # 情感分析
        if req.include_sentiment:
            result['sentiment'] = analyzer.analyze_sentiment(req.text)
        
        # 主题相关性
        if req.include_topics:
            result['topic_relevance'] = analyzer.analyze_topic_relevance(req.text)
        
        # 内容质量评分
        if req.include_quality:
            result['quality_score'] = analyzer.calculate_content_quality_score(req.text)
        
        return {
            "status": "success",
            "text_length": len(req.text),
            "word_count": len(req.text.split()),
            "analysis": result,
            "message": "语义分析完成"
        }
        
    except Exception as e:
        logger.error(f"语义分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"语义分析失败: {str(e)}")


# 新增：智能多片段视频剪辑API
class MultiSegmentClippingReq(BaseModel):
    video_path: str
    subtitle_path: Optional[str] = None
    topic: str
    target_segments: int = 3
    segment_duration: Optional[float] = None
    total_duration: float = 60.0
    semantic_weight: float = 0.4
    visual_weight: float = 0.3
    audio_weight: float = 0.3
    selection_mode: str = "highlights"
    include_intro: Optional[bool] = None
    include_highlights: bool = True
    include_conclusion: Optional[bool] = None
    enable_content_analysis: bool = True
    asr_model_size: str = "base"
    asr_language: Optional[str] = None
    # The decision list is always produced. Rendering the 9:16 file is the
    # expensive half, and the half a dedicated editor does better, so a client
    # that omits this gets the cheap answer rather than a surprise transcode.
    render: bool = False

    @field_validator("selection_mode")
    @classmethod
    def validate_selection_mode(cls, value):
        if value not in {"highlights", "summary"}:
            raise ValueError("选段模式必须是 highlights 或 summary")
        return value

    @field_validator("video_path")
    @classmethod
    def validate_video_path(cls, value):
        """Accept a managed local path or a public URL.

        A remote URL is checked for scheme and address here and materialised
        by the workflow; validating it as a local path rejected every platform
        link before it reached the resolver.
        """
        value = str(value).strip()
        if not value:
            raise ValueError("视频路径不能为空")
        if value.lower().startswith(("http://", "https://")):
            return validate_remote_url(value)
        try:
            return str(resolve_media_path(value))
        except FileNotFoundError as exc:
            # Pydantic only converts ValueError into a 422; anything else
            # escapes as a 500, which is the wrong answer for a bad request.
            raise ValueError(str(exc)) from None

    @field_validator("subtitle_path")
    @classmethod
    def validate_subtitle_path(cls, value):
        if not value:
            return None
        try:
            return str(resolve_media_path(value))
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from None

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value):
        if not value.strip():
            raise ValueError("主题不能为空")
        return value.strip()

    @field_validator("target_segments")
    @classmethod
    def validate_target_segments(cls, value):
        if not 1 <= value <= 10:
            raise ValueError("目标片段数必须在 1-10 之间")
        return value

    @field_validator("segment_duration")
    @classmethod
    def validate_segment_duration(cls, value):
        if value is not None and not 5 <= value <= 30:
            raise ValueError("单片段时长必须在 5-30 秒之间")
        return value

    @field_validator("total_duration")
    @classmethod
    def validate_total_duration(cls, value):
        if not 5 <= value <= 300:
            raise ValueError("总时长必须在 5-300 秒之间")
        return value

    @field_validator("semantic_weight", "visual_weight", "audio_weight")
    @classmethod
    def validate_weight(cls, value):
        if not 0 <= value <= 1:
            raise ValueError("评分权重必须在 0-1 之间")
        return value

    @field_validator("asr_model_size")
    @classmethod
    def validate_asr_model_size(cls, value):
        if value not in {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}:
            raise ValueError("不支持的 Whisper 模型")
        return value

    @model_validator(mode="after")
    def validate_duration_budget(self):
        minimum_total = self.target_segments * 5
        if self.total_duration < minimum_total:
            raise ValueError(f"{self.target_segments} 个片段的总时长至少需要 {minimum_total} 秒")
        if self.segment_duration and self.segment_duration * self.target_segments > self.total_duration:
            raise ValueError("单片段时长乘以片段数不能超过总时长")
        return self


@app.post("/video/multi_segment_clipping", tags=["video"])
async def multi_segment_intelligent_clipping(req: MultiSegmentClippingReq):
    """Select measured audiovisual/semantic highlights and combine them."""
    try:
        logger.info("开始智能多片段剪辑: %s", req.video_path)
        return await asyncio.to_thread(process_multi_segment_video, req.model_dump())
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"智能多片段剪辑失败: {e}")
        raise HTTPException(status_code=500, detail=f"智能多片段剪辑失败: {str(e)}")


# 文件上传相关API

class SelectionReviewReq(BaseModel):
    """A reviewer's verdict on one job's segments."""

    job_id: str
    accepted: List[Dict[str, float]]

    @field_validator("accepted")
    @classmethod
    def validate_spans(cls, spans):
        for span in spans:
            if "start_time" not in span or "end_time" not in span:
                raise ValueError("每个区间需要 start_time 和 end_time")
            if not 0 <= float(span["start_time"]) < float(span["end_time"]):
                raise ValueError("标注必须满足 0 <= start_time < end_time")
        return spans


@app.post("/evaluate/selection", tags=["video"])
async def evaluate_selection_endpoint(req: SelectionReviewReq) -> Dict[str, Any]:
    """Score a finished job's segments against what the reviewer accepted.

    The page already holds the verdict, so making the reviewer save a file and
    hand it to a CLI just to see two numbers broke the loop at the handoff.
    This reads the segments from the job record rather than trusting the
    client to resend them, so the metrics describe what was really produced.
    """
    job = job_store.get_job(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务不存在: {req.job_id}")
    if job["status"] != job_store.SUCCEEDED:
        raise HTTPException(status_code=409, detail="只能评估已成功的任务")

    segments = ((job.get("result") or {}).get("selected_segments")) or []
    try:
        report = evaluate_selection(segments, req.accepted)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    # All suggestions kept and nothing added means the labels are a copy of
    # the output, and one-to-one IoU matching then returns 1.0 by
    # construction. Say so rather than reporting a perfect score.
    suggested = len(segments)
    trivial = len(req.accepted) == suggested and all(
        any(abs(float(a["start_time"]) - float(s["start_time"])) < 0.01
            and abs(float(a["end_time"]) - float(s["end_time"])) < 0.01
            for a in req.accepted)
        for s in segments
    )
    return {
        "job_id": req.job_id,
        "topic": (job.get("params") or {}).get("topic", ""),
        "evaluation": report,
        "trivial": trivial,
    }


@app.post("/upload/video", tags=["upload"])
async def upload_video(file: UploadFile = File(...)):
    """上传视频文件"""
    try:
        # 验证文件类型
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="请上传有效的视频文件")
        
        file_extension = Path(file.filename or "").suffix.lower() or ".mp4"
        if file_extension not in set(ALLOWED_VIDEO_EXTENSIONS):
            raise HTTPException(status_code=400, detail=f"不支持的视频扩展名: {file_extension}")
        unique_filename = f"video_{uuid.uuid4().hex}{file_extension}"
        file_path = VIDEO_UPLOAD_DIR / unique_filename

        size = 0
        try:
            async with aiofiles.open(file_path, "wb") as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_FILE_SIZE:
                        raise HTTPException(status_code=413, detail="视频文件大小不能超过配置限制")
                    await output.write(chunk)
        except Exception:
            file_path.unlink(missing_ok=True)
            raise
        
        logger.info(f"上传视频成功: {file.filename} -> {file_path}")
        
        return {
            "status": "success",
            "message": "视频上传成功",
            "file": {
                "original_name": file.filename,
                "saved_path": str(file_path),
                "size": size
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"视频上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    from core.config import API_HOST, API_PORT
    
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
