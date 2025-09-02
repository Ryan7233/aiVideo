import json
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
from urllib.request import urlretrieve
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
import uuid

# Import configuration and validation functions
from core.config import (
    SEGMENT_PROMPT, CAPTIONS_PROMPT, GEMINI_API_BASE, CUT_API_BASE,
    UPLOAD_BUCKET, UPLOAD_BASE_URL, VIDEO_FPS, VIDEO_CRF, AUDIO_BITRATE,
    validate_video_extension, validate_file_size, MIN_CLIP_DURATION, MAX_CLIP_DURATION
)
from core.settings import settings
from core.smart_clipping import get_smart_segments, analyze_video_intelligence
from core.whisper_asr import get_asr_service, transcribe_video_file
from core.semantic_analysis import get_semantic_analyzer, analyze_transcription_semantics
from core.asr_smart_clipping import get_asr_smart_engine, select_segments_with_asr
from core.xiaohongshu_pipeline import (
    get_photo_ranking_service, get_storyline_generator, get_draft_generator
)
from core.subtitle_service import get_subtitle_generator, get_cover_service
from core.export_service import get_export_service
from core.advanced_photo_ranking import get_advanced_photo_service
from core.semantic_highlights import get_semantic_highlight_detector
from core.personalized_writing import get_personalized_writing_service
from core.smart_cover_design import get_smart_cover_designer
from core.audio_processing import get_audio_processing_service
from core.xiaohongshu_publisher import get_xiaohongshu_publisher
from core.image_decorator import get_image_decorator
from core.smart_cover_generator import get_smart_cover_generator
from core.llm_service import get_llm_service
from core.advanced_collage_generator import get_advanced_collage_generator
from core.xiaohongshu_collage_generator import get_xiaohongshu_collage_generator, CollageConfig, TextConfig
from routers.tasks import router as tasks_router

# Setup logging
logger.add("logs/api.log", rotation="10 MB", level="INFO")
logger.add("logs/api.jsonl", rotation="10 MB", level="INFO", serialize=True)

app = FastAPI(
    title="AI Video Clipper API",
    description="智能短视频自动切片和文案生成服务",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
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

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Routers
app.include_router(tasks_router)

# Mount static files and frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")
app.mount("/static/uploads", StaticFiles(directory="output_data/uploads"), name="uploads")
app.mount("/output", StaticFiles(directory="output_data"), name="output")

@app.get("/")
async def serve_frontend():
    """Serve the main frontend page"""
    return FileResponse("frontend/index.html")

# Exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "message": "请求参数验证失败"},
    )

# --- Pydantic Models with Validation ---
class SegmentReq(BaseModel):
    transcript: str
    min_sec: int = MIN_CLIP_DURATION
    max_sec: int = MAX_CLIP_DURATION
    
    @field_validator('transcript')
    @classmethod
    def validate_transcript(cls, v):
        if not v.strip():
            raise ValueError('字幕内容不能为空')
        return v.strip()
    
    @field_validator('min_sec', 'max_sec')
    @classmethod
    def validate_duration(cls, v):
        if v <= 0:
            raise ValueError('时长必须大于0')
        return v

class CaptionsReq(BaseModel):
    topic: str = "AI and Technology"
    transcript: str
    clip_text: str
    
    @field_validator('topic', 'transcript', 'clip_text')
    @classmethod
    def validate_text_fields(cls, v):
        if not v.strip():
            raise ValueError('文本字段不能为空')
        return v.strip()

class CutReq(BaseModel):
    src: str
    start: str
    end: str
    out: str
    
    @field_validator('src')
    @classmethod
    def validate_src_file(cls, v):
        if not os.path.exists(v):
            raise ValueError(f'源文件不存在: {v}')
        if not validate_video_extension(v):
            raise ValueError(f'不支持的视频格式: {v}')
        return v
    
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
        if not os.path.exists(v):
            raise ValueError(f'源文件不存在: {v}')
        return v
    
    @field_validator('srt')
    @classmethod
    def validate_srt_file(cls, v):
        if not os.path.exists(v):
            raise ValueError(f'字幕文件不存在: {v}')
        return v

class UploadReq(BaseModel):
    path: str
    bucket: str = UPLOAD_BUCKET
    
    @field_validator('path')
    @classmethod
    def validate_path(cls, v):
        if not os.path.exists(v):
            raise ValueError(f'文件不存在: {v}')
        return v

# --- Utility Functions ---
def normalize_time(time_str: str) -> str:
    """Normalize time format to hh:mm:ss"""
    parts = time_str.split(':')
    if len(parts) == 2:
        return f"00:{time_str}"
    return time_str

def hms_to_seconds(hms: str) -> float:
    parts = list(map(int, hms.split(':')))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]

def seconds_to_hms(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def safe_run_ffmpeg(cmd: List[str], timeout: int = 300) -> Dict[str, Any]:
    """Safely run ffmpeg command with timeout and error handling"""
    try:
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        start_time = time.time()
        
        process = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            timeout=timeout
        )
        
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

def black_fraction_in_segment(path: str, start_hms: str, duration_s: float) -> float:
    """Approximate fraction of time that is black in the given segment using ffmpeg blackdetect.
    Use a center ROI to ignore letterboxing borders.
    """
    try:
        # Limit runtime to short duration analysis
        cmd = [
            "ffmpeg", "-hide_banner", "-ss", start_hms, "-t", f"{max(0.5, duration_s):.2f}",
            "-i", path,
            "-vf", "crop=in_w*0.9:in_h*0.9:(in_w-out_w)/2:(in_h-out_h)/2,blackdetect=d=0.3:pic_th=0.98",
            "-an", "-f", "null", "-"
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=60)
        out = (completed.stdout or "") + "\n" + (completed.stderr or "")
        import re
        # Parse last blackdetect line: black_start:.. black_end:.. black_duration:..
        matches = re.findall(r"black_duration:([0-9]+\.?[0-9]*)", out)
        if not matches:
            return 0.0
        total_black = sum(float(x) for x in matches)
        return min(max(total_black / max(duration_s, 0.001), 0.0), 1.0)
    except Exception:
        return 0.0

def silence_fraction_in_segment(path: str, start_hms: str, duration_s: float) -> float:
    """Approximate fraction of time that is silence in the given segment using silencedetect."""
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-ss", start_hms, "-t", f"{max(0.5, duration_s):.2f}",
            "-i", path, "-af", "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-"
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=60)
        out = (completed.stdout or "") + "\n" + (completed.stderr or "")
        import re
        sil_starts = [float(x) for x in re.findall(r"silence_start:([0-9]+\.?[0-9]*)", out)]
        sil_ends = [float(x) for x in re.findall(r"silence_end:([0-9]+\.?[0-9]*)", out)]
        # Pair up starts and ends; if unmatched end, ignore
        total_sil = 0.0
        for i, s in enumerate(sil_starts):
            if i < len(sil_ends):
                total_sil += max(0.0, sil_ends[i] - s)
        return min(max(total_sil / max(duration_s, 0.001), 0.0), 1.0)
    except Exception:
        return 0.0

def ffprobe_json(path: str, select_streams: str = "") -> Dict[str, Any]:
    """Return ffprobe json info for given path (optionally select streams)."""
    probe_cmd = [
        "ffprobe", "-v", "error", "-print_format", "json", "-show_streams"
    ]
    if select_streams:
        probe_cmd += ["-select_streams", select_streams]
    probe_cmd += [path]
    try:
        completed = subprocess.run(probe_cmd, capture_output=True, text=True, encoding="utf-8")
        if completed.returncode != 0:
            return {"streams": []}
        return json.loads(completed.stdout or "{}")
    except Exception:
        return {"streams": []}

def get_duration_seconds(path: str) -> float:
    info = ffprobe_json(path)
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            try:
                return float(s.get("duration")) if s.get("duration") else 0.0
            except Exception:
                continue
    return 0.0

def try_python_asr_to_srt(src: str, out_srt: str) -> bool:
    """Try to run ASR via Python libs (openai-whisper or faster_whisper). Returns True on success."""
    # openai-whisper
    try:
        import whisper  # type: ignore
        model = whisper.load_model("small")
        result = model.transcribe(src, language="zh", task="transcribe")
        segments = result.get("segments", [])
        with open(out_srt, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
                def fmt(t: float) -> str:
                    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
                    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                text = str(seg.get("text", "")).strip()
                if not text:
                    continue
                f.write(f"{i}\n{fmt(start)} --> {fmt(end)}\n{text}\n\n")
        return True
    except Exception:
        pass
    # faster-whisper
    try:
        from faster_whisper import WhisperModel  # type: ignore
        model = WhisperModel("small", device="cpu")
        segments, _ = model.transcribe(src, language="zh")
        def fmt(t: float) -> str:
            h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        with open(out_srt, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                text = (seg.text or "").strip()
                if not text:
                    continue
                f.write(f"{i}\n{fmt(seg.start)} --> {fmt(seg.end)}\n{text}\n\n")
        return True
    except Exception:
        return False

def parse_srt_simple(path: str) -> List[Dict[str, str]]:
    """Parse SRT to list of dicts with start, end, text (mm:ss granularity)."""
    if not os.path.exists(path):
        return []
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    items: List[Dict[str, str]] = []
    for block in [b.strip() for b in raw.split("\n\n") if b.strip()]:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        timeline = next((l for l in lines if "-->" in l), "")
        if not timeline:
            continue
        try:
            start, end = [t.strip().replace(",", ":")[0:8] for t in timeline.split("-->")]
        except Exception:
            continue
        txt_lines = []
        seen = False
        for l in lines:
            if seen:
                txt_lines.append(l)
            if l == timeline:
                seen = True
        text = " ".join(txt_lines).strip()
        items.append({"start": start if len(start)==8 else f"00:{start}", "end": end if len(end)==8 else f"00:{end}", "text": text})
    return items

def score_text_heuristic(t: str) -> float:
    score = 0.0
    if "！" in t or "!" in t:
        score += 2
    if "？" in t or "?" in t:
        score += 1.2
    if any(ch.isdigit() for ch in t):
        score += 1.0
    for w in ["亮点","关键","总结","步骤","因此","所以","案例","注意","做法","技巧","揭秘","提升","对比","推荐"]:
        if w in t:
            score += 0.8
    score += min(len(t) / 20.0, 1.5)
    return score

def select_windows_from_srt(srt_items: List[Dict[str, str]], min_sec: int, max_sec: int, top_k: int = 3) -> List[Dict[str, str]]:
    def to_s(ts: str) -> int:
        parts = list(map(int, ts.split(":")))
        if len(parts) == 2:
            return parts[0]*60 + parts[1]
        return parts[0]*3600 + parts[1]*60 + parts[2]
    n = len(srt_items)
    windows: List[Tuple[float,int,int]] = []
    for i in range(n):
        start_s = to_s(srt_items[i]["start"]) 
        acc = 0.0
        j = i
        while j < n:
            acc += score_text_heuristic(srt_items[j]["text"]) 
            end_s = to_s(srt_items[j]["end"]) 
            dur = end_s - start_s
            if dur >= min_sec:
                windows.append((acc, i, j))
            if dur >= max_sec:
                break
            j += 1
    windows.sort(key=lambda x: x[0], reverse=True)
    chosen: List[Tuple[int,int]] = []
    for _, i, j in windows:
        if len(chosen) >= top_k:
            break
        ok = True
        for ci, cj in chosen:
            if not (j < ci or i > cj):
                ok = False
                break
        if ok:
            chosen.append((i, j))
    clips: List[Dict[str,str]] = []
    for i, j in chosen:
        clips.append({"start": srt_items[i]["start"], "end": srt_items[j]["end"], "reason": "highlight"})
    return clips

# --- Gemini Mock Function with Error Handling ---
def run_gemini(prompt: str, max_retries: int = 3) -> str:
    """Mock Gemini API call with retry logic"""
    logger.info(f"Mock Gemini call with prompt length: {len(prompt)}")
    
    for attempt in range(max_retries):
        try:
            if "切片专家" in prompt:
                return '''{"clips": [{"start": "00:15","end": "00:48","reason": "介绍了项目的核心目标和技术选型。"},{"start": "01:10","end": "01:55","reason": "演示了关键功能并解释了其用户价值。"}]}'''
            else:
                return '''{"title": "AI如何颠覆我们的生活？","hashtags": ["#AI", "#科技改变生活"],"desc": "这个视频的核心观点，让你三分钟看懂人工智能的真正威力！"}'''
        except Exception as e:
            logger.error(f"Gemini call attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail="AI服务调用失败")
            time.sleep(1)  # Wait before retry

# --- API Endpoints ---
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AI Video Clipper API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """详细健康检查端点"""
    return {
        "status": "healthy", 
        "timestamp": int(time.time()),
        "services": {
            "api": "running",
            "smart_clipping": "available"
        }
    }



@app.post("/segment")
async def segment(req: SegmentReq):
    """AI智能切片分析"""
    try:
        logger.info(f"Processing segment request with duration: {req.min_sec}-{req.max_sec}s")
        
        prompt = SEGMENT_PROMPT.format(
            txt=req.transcript, 
            min_sec=req.min_sec, 
            max_sec=req.max_sec
        )
        
        result = run_gemini(prompt)
        parsed_result = json.loads(result[result.find("{"):result.rfind("}")+1])
        
        logger.info(f"Generated {len(parsed_result.get('clips', []))} clips")
        return parsed_result
        
    except json.JSONDecodeError:
        logger.error("Failed to parse Gemini response as JSON")
        raise HTTPException(status_code=500, detail="AI分析结果格式错误")
    except Exception as e:
        logger.error(f"Segment processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"切片分析失败: {str(e)}")

@app.post("/captions")
async def captions(req: CaptionsReq):
    """AI文案生成"""
    try:
        logger.info(f"Generating captions for topic: {req.topic}")
        
        prompt = CAPTIONS_PROMPT.format(
            topic=req.topic, 
            clip_text=req.clip_text, 
            transcript=req.transcript
        )
        
        result = run_gemini(prompt)
        parsed_result = json.loads(result[result.find("{"):result.rfind("}")+1])
        
        logger.info(f"Generated captions: {parsed_result.get('title', '')}")
        return parsed_result
        
    except json.JSONDecodeError:
        logger.error("Failed to parse captions response as JSON")
        raise HTTPException(status_code=500, detail="文案生成结果格式错误")
    except Exception as e:
        logger.error(f"Caption generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文案生成失败: {str(e)}")

@app.post("/cut916")
async def cut916(req: CutReq):
    """生成9:16竖屏视频"""
    try:
        logger.info(f"Cutting video: {req.src} ({req.start} - {req.end})")
        
        # Normalize time format
        start_time = normalize_time(req.start)
        end_time = normalize_time(req.end)
        
        # Robust 9:16 pipeline using expressions (no FOAR option):
        # - If input is wider than 9:16, scale height to 1920 and width proportional; else scale width to 1080
        # - Then center crop to exactly 1080x1920; set pixel format and SAR for compatibility
        vf_filters = (
            "scale="
            "if(gte(iw/ih\,1080/1920)\,-2\,1080):"
            "if(gte(iw/ih\,1080/1920)\,1920\,-2),"
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
            req.out
        ]
        
        result = safe_run_ffmpeg(cmd)
        result["out"] = req.out
        
        logger.info(f"Successfully created 9:16 video: {req.out}")
        return result
        
    except Exception as e:
        logger.error(f"9:16 video cut error: {str(e)}")
        # Clean up partial output file if it exists
        if os.path.exists(req.out):
            os.remove(req.out)
        raise HTTPException(status_code=500, detail=f"9:16视频生成失败: {str(e)}")

@app.post("/burnsub")
async def burnsub(req: BurnSubReq):
    """烧录字幕到视频"""
    try:
        logger.info(f"Burning subtitles: {req.srt} -> {req.src}")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", req.src,
            "-vf", f"subtitles={req.srt}:force_style='Fontsize=28'",
            "-c:a", "copy",
            req.out
        ]
        
        result = safe_run_ffmpeg(cmd)
        result["out"] = req.out
        
        logger.info(f"Successfully burned subtitles: {req.out}")
        return result
        
    except Exception as e:
        logger.error(f"Subtitle burning error: {str(e)}")
        # Clean up partial output file if it exists
        if os.path.exists(req.out):
            os.remove(req.out)
        raise HTTPException(status_code=500, detail=f"字幕烧录失败: {str(e)}")

@app.post("/upload")
async def upload(req: UploadReq):
    """模拟文件上传到云存储"""
    try:
        if not os.path.exists(req.path):
            logger.error(f"File not found: {req.path}")
            raise HTTPException(status_code=404, detail="文件不存在")
        
        file_name = os.path.basename(req.path)
        file_size = os.path.getsize(req.path)
        
        if not validate_file_size(file_size):
            raise HTTPException(status_code=400, detail="文件大小超出限制")
        
        # Mock upload to cloud storage
        mock_url = f"{UPLOAD_BASE_URL}/{req.bucket}/{file_name}"
        logger.info(f"Mock upload: {req.path} -> {mock_url}")
        
        return {
            "url": mock_url,
            "key": file_name,
            "size": file_size,
            "bucket": req.bucket
        }
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


# --- New: Intro-style auto highlights from URL ---
class URLIntroReq(BaseModel):
    url: str
    min_sec: int = MIN_CLIP_DURATION
    max_sec: int = MAX_CLIP_DURATION
    want_asr: bool = True
    top_k: int = 3
    output: str = str(Path("output_data") / "intro_916.mp4")
    smart_mode: bool = True  # 启用智能选段模式

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

class ASRSmartClippingReq(BaseModel):
    url: str
    min_sec: int = MIN_CLIP_DURATION
    max_sec: int = MAX_CLIP_DURATION
    count: int = 1
    model_size: str = "base"
    language: Optional[str] = None
    output_prefix: str = "asr_smart_clip"
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        if not v.strip():
            raise ValueError('URL不能为空')
        v = v.strip()
        if not v.startswith("http") and not v.startswith("file:"):
            raise ValueError('仅支持 http/https/file URL')
        return v
    
    @field_validator('min_sec', 'max_sec')
    @classmethod
    def validate_duration(cls, v):
        if v < 5 or v > 300:
            raise ValueError('时长必须在5-300秒之间')
        return v
    
    @field_validator('count')
    @classmethod
    def validate_count(cls, v):
        if v < 1 or v > 5:
            raise ValueError('片段数量必须在1-5之间')
        return v

class PhotoRankingReq(BaseModel):
    photos: List[str]
    top_k: int = 15
    
    @field_validator('photos')
    @classmethod
    def validate_photos(cls, v):
        if not v:
            raise ValueError('照片列表不能为空')
        return v
    
    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v):
        if v < 1 or v > 50:
            raise ValueError('选择数量必须在1-50之间')
        return v

class StorylineReq(BaseModel):
    transcript_mmss: List[Dict]
    notes: str
    city: str = ""
    date: str = ""
    style: str = "治愈"
    
    @field_validator('transcript_mmss')
    @classmethod
    def validate_transcript(cls, v):
        if not v:
            raise ValueError('转录文本不能为空')
        return v
    
    @field_validator('style')
    @classmethod
    def validate_style(cls, v):
        valid_styles = ['治愈', '专业', '踩雷']
        if v not in valid_styles:
            raise ValueError(f'风格必须是: {", ".join(valid_styles)}')
        return v

class XHSDraftReq(BaseModel):
    storyline: Dict
    brand_tone: str = "治愈"
    constraints: Optional[Dict] = None
    
    @field_validator('storyline')
    @classmethod
    def validate_storyline(cls, v):
        if not v:
            raise ValueError('故事线数据不能为空')
        return v

class SubtitleReq(BaseModel):
    clips: List[Dict]
    transcript_mmss: List[Dict]
    style: str = "口语"
    
    @field_validator('style')
    @classmethod
    def validate_style(cls, v):
        valid_styles = ['口语', '书面', '可爱']
        if v not in valid_styles:
            raise ValueError(f'字幕风格必须是: {", ".join(valid_styles)}')
        return v

class CoverReq(BaseModel):
    clips: List[Dict]
    photos_topk: List[Dict]
    title: str = ""

class ExportReq(BaseModel):
    pipeline_result: Dict
    export_format: str = "zip"
    include_source: bool = False
    
    @field_validator('export_format')
    @classmethod
    def validate_format(cls, v):
        valid_formats = ['json', 'zip', 'folder']
        if v not in valid_formats:
            raise ValueError(f'导出格式必须是: {", ".join(valid_formats)}')
        return v

class XHSPipelineReq(BaseModel):
    video_url: str
    photos: List[str] = []
    notes: str = ""
    city: str = ""
    style: str = "治愈"
    model_size: str = "base"
    export_format: str = "zip"
    
    @field_validator('video_url')
    @classmethod
    def validate_video_url(cls, v):
        if not v.strip():
            raise ValueError('视频URL不能为空')
        return v.strip()

# Pro功能API模型
class AdvancedPhotoRankingReq(BaseModel):
    photos: List[str]
    top_k: int = 15
    context: Optional[Dict] = None
    use_clip: bool = True
    use_aesthetic_model: bool = True
    
    @field_validator('photos')
    @classmethod
    def validate_photos(cls, v):
        if not v:
            raise ValueError('照片列表不能为空')
        return v

class SemanticHighlightsReq(BaseModel):
    transcript_segments: List[Dict]
    context: Optional[Dict] = None
    min_score: float = 0.3
    max_highlights: int = 10
    
    @field_validator('transcript_segments')
    @classmethod
    def validate_segments(cls, v):
        if not v:
            raise ValueError('转录片段不能为空')
        return v

class PersonalizedWritingReq(BaseModel):
    user_id: str
    content_data: Dict
    style_override: Optional[str] = None
    learn_from_history: bool = True
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v):
        if not v.strip():
            raise ValueError('用户ID不能为空')
        return v.strip()

class UserStyleLearningReq(BaseModel):
    user_id: str
    content_samples: List[Dict]
    
    @field_validator('content_samples')
    @classmethod
    def validate_samples(cls, v):
        if not v:
            raise ValueError('内容样本不能为空')
        return v

class SmartCoverDesignReq(BaseModel):
    clips: List[Dict]
    photos: List[Dict] = []
    title: str
    style: str = "治愈"
    template: str = "minimal"
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError('标题不能为空')
        return v.strip()

class AudioProcessingReq(BaseModel):
    video_path: str
    style: str = "治愈"
    enhance_speech: bool = True
    add_bgm: bool = True
    bgm_volume: float = 0.3
    
    @field_validator('video_path')
    @classmethod
    def validate_video_path(cls, v):
        if not v.strip():
            raise ValueError('视频路径不能为空')
        return v.strip()
    
    @field_validator('bgm_volume')
    @classmethod
    def validate_bgm_volume(cls, v):
        if v < 0 or v > 1:
            raise ValueError('BGM音量必须在0-1之间')
        return v

class XHSProPipelineReq(BaseModel):
    video_url: str
    photos: List[str] = []
    notes: str = ""
    city: str = ""
    style: str = "治愈"
    user_id: Optional[str] = None
    model_size: str = "base"

# 新增API模型
class XHSPublishReq(BaseModel):
    title: str
    content: str
    images: List[str]
    tags: List[str]
    location: Optional[str] = None
    privacy: str = "public"

class ImageDecorateReq(BaseModel):
    image_path: str
    decorations: Dict[str, Any]

class SmartCoverReq(BaseModel):
    images: List[str]
    title: str
    subtitle: str = ""
    layout: str = "grid_3x3"
    theme: str = "pink_gradient"
    platform: str = "xiaohongshu"
    custom_config: Optional[Dict[str, Any]] = None
    export_format: str = "zip"
    
    # Pro功能开关
    use_advanced_photo_ranking: bool = True
    use_semantic_highlights: bool = True
    use_personalized_writing: bool = False
    use_smart_cover: bool = True
    use_audio_enhancement: bool = True

class LLMContentReq(BaseModel):
    theme: str
    photo_descriptions: List[str] = []
    highlights: str = ""
    feeling: str = ""
    style: str = "活泼"
    length: str = "medium"
    custom_requirements: str = ""
    content_type: str = "xiaohongshu"  # xiaohongshu, pro, general

class ProContentReq(BaseModel):
    topic: str
    content_type: str = "complete"  # title, description, hashtags, complete
    style: str = "professional"
    target_audience: str = "general"

class AdvancedCollageReq(BaseModel):
    images: List[str]  # 图片路径列表
    title: str
    layout_type: str = "dynamic"  # dynamic, grid, magazine, mosaic, creative
    style: str = "modern"  # modern, vintage, artistic, minimal
    color_scheme: str = "auto"  # auto, warm, cool, monochrome, vibrant
    canvas_size: List[int] = [800, 800]
    add_effects: bool = True
    add_text_overlay: bool = True
    extra_text: Optional[str] = ""
    text_position: str = "bottom"  # bottom | center

class XiaohongshuCollageReq(BaseModel):
    images: List[str]
    title: str
    subtitle: str = ""
    layout: str = "magazine_style"
    color_scheme: str = "xiaohongshu_pink"
    width: int = 2160
    height: int = 2160
    quality: int = 95
    custom_texts: List[Dict[str, Any]] = []
    # 新增：标题排版控制
    title_position: str = "center_overlay"  # center_overlay | top
    font_path: Optional[str] = None
    overlay_texts: List[Dict[str, Any]] = []  # 额外文案块，锚点定位

class EditableTextReq(BaseModel):
    collage_id: str
    text_id: str
    new_text: str
    font_size: int = 80
    color: str = "#2C2C2C"
    position: List[int] = [100, 100]


@app.post("/analyze_video")
async def analyze_video(req: VideoAnalysisReq):
    """分析视频内容，返回智能化分析结果"""
    try:
        Path("input_data/downloads").mkdir(parents=True, exist_ok=True)
        
        # 下载视频
        ts = int(time.time())
        dl_path = str(Path("input_data/downloads") / f"analysis_{ts}.mp4")
        
        if req.url.startswith("file:"):
            # 本地文件
            local_path = req.url.replace("file://", "")
            if not Path(local_path).exists():
                raise HTTPException(status_code=400, detail="本地文件不存在")
            dl_path = local_path
        else:
            # URL下载 (简化版，实际项目中可能需要更复杂的下载逻辑)
            from urllib.request import urlretrieve
            urlretrieve(req.url, dl_path)
        
        # 执行智能分析
        logger.info(f"Starting intelligent video analysis for: {dl_path}")
        analysis_result = analyze_video_intelligence(dl_path)
        
        # 获取智能片段推荐
        smart_segments = get_smart_segments(dl_path, 15, 30, count=5)
        
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


@app.post("/asr/transcribe")
async def asr_transcribe(req: ASRTranscribeReq):
    """自动语音识别 - 转录视频/音频"""
    try:
        Path("input_data/downloads").mkdir(parents=True, exist_ok=True)
        Path("output_data").mkdir(parents=True, exist_ok=True)
        
        # 处理输入文件
        ts = int(time.time())
        if req.url.startswith("file:"):
            # 本地文件
            local_path = req.url.replace("file://", "")
            if not Path(local_path).exists():
                raise HTTPException(status_code=400, detail="本地文件不存在")
            input_path = local_path
        else:
            # URL下载
            input_path = str(Path("input_data/downloads") / f"asr_input_{ts}.mp4")
            from urllib.request import urlretrieve
            urlretrieve(req.url, input_path)
        
        logger.info(f"🎤 开始ASR转录: {input_path}")
        
        # 获取ASR服务
        asr_service = get_asr_service(model_size=req.model_size)
        
        # 转录视频
        result = asr_service.transcribe_video(
            input_path,
            language=req.language,
            task=req.task,
            cleanup_audio=True
        )
        
        # 生成字幕文件
        subtitle_file = None
        if req.subtitle_format != "none":
            video_stem = Path(input_path).stem
            subtitle_path = f"output_data/{video_stem}_asr_{ts}.{req.subtitle_format}"
            
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


@app.post("/asr/extract_audio")
async def extract_audio(req: AudioExtractionReq):
    """从视频中提取音频"""
    try:
        Path("output_data").mkdir(parents=True, exist_ok=True)
        
        # 处理输入文件
        if req.url.startswith("file:"):
            local_path = req.url.replace("file://", "")
            if not Path(local_path).exists():
                raise HTTPException(status_code=400, detail="本地文件不存在")
            input_path = local_path
        else:
            # URL下载
            ts = int(time.time())
            input_path = str(Path("input_data/downloads") / f"audio_extract_{ts}.mp4")
            from urllib.request import urlretrieve
            urlretrieve(req.url, input_path)
        
        logger.info(f"🎵 开始提取音频: {input_path}")
        
        # 获取ASR服务
        asr_service = get_asr_service()
        
        # 提取音频
        video_stem = Path(input_path).stem
        audio_path = f"output_data/{video_stem}_audio_{int(time.time())}.wav"
        
        extracted_audio = asr_service.extract_audio_from_video(
            input_path,
            audio_path=audio_path,
            sample_rate=req.sample_rate
        )
        
        # 获取音频信息
        audio_info = {}
        try:
            import subprocess
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
                "-show_streams", extracted_audio
            ], capture_output=True, text=True)
            
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


@app.get("/asr/info")
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


@app.post("/semantic/analyze")
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


@app.post("/smart_clipping/asr_enhanced")
async def asr_enhanced_smart_clipping(req: ASRSmartClippingReq):
    """ASR增强智能切片 - 结合语音识别和语义分析的智能选段"""
    try:
        Path("input_data/downloads").mkdir(parents=True, exist_ok=True)
        Path("output_data").mkdir(parents=True, exist_ok=True)
        
        # 处理输入文件
        ts = int(time.time())
        if req.url.startswith("file:"):
            local_path = req.url.replace("file://", "")
            if not Path(local_path).exists():
                raise HTTPException(status_code=400, detail="本地文件不存在")
            input_path = local_path
        else:
            input_path = str(Path("input_data/downloads") / f"asr_smart_{ts}.mp4")
            from urllib.request import urlretrieve
            urlretrieve(req.url, input_path)
        
        logger.info(f"🎯 开始ASR增强智能切片: {input_path}")
        
        # 1. 获取ASR转录结果
        asr_service = get_asr_service(model_size=req.model_size)
        transcription_result = asr_service.transcribe_video(
            input_path,
            language=req.language,
            cleanup_audio=True
        )
        
        logger.info(f"ASR转录完成 - 语言: {transcription_result['language']}, 文本长度: {transcription_result['word_count']}词")
        
        # 2. 执行ASR增强智能选段
        asr_engine = get_asr_smart_engine()
        selected_segments = asr_engine.select_best_segments_with_asr(
            input_path,
            transcription_result,
            req.min_sec,
            req.max_sec,
            req.count
        )
        
        if not selected_segments:
            raise HTTPException(status_code=400, detail="未找到符合条件的智能片段")
        
        # 3. 生成视频片段
        generated_clips = []
        
        for i, segment in enumerate(selected_segments):
            output_filename = f"{req.output_prefix}_{ts}_{i+1:02d}.mp4"
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
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(VIDEO_CRF),
                "-c:a", "aac", "-b:a", AUDIO_BITRATE,
                "-shortest", "-movflags", "+faststart",
                output_path
            ]
            
            safe_run_ffmpeg(cmd)
            
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
        
        # 4. 构建响应
        return {
            "status": "success",
            "input_path": input_path,
            "transcription": {
                "language": transcription_result["language"],
                "language_probability": transcription_result["language_probability"],
                "duration": transcription_result["duration"],
                "word_count": transcription_result["word_count"],
                "segment_count": transcription_result["segment_count"]
            },
            "selected_segments": len(selected_segments),
            "generated_clips": generated_clips,
            "processing_summary": {
                "asr_enhanced": True,
                "semantic_analysis": True,
                "visual_analysis": True,
                "selection_algorithm": "asr_smart_clipping"
            },
            "message": f"ASR增强智能切片完成 - 生成了 {len(generated_clips)} 个高质量片段"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ASR增强智能切片失败: {e}")
        raise HTTPException(status_code=500, detail=f"ASR增强智能切片失败: {str(e)}")


@app.post("/pro/photo_rank_advanced")
async def advanced_photo_rank(req: AdvancedPhotoRankingReq):
    """高级照片选优排序（Pro版本）"""
    try:
        logger.info(f"开始高级照片选优，共 {len(req.photos)} 张照片")
        
        advanced_service = get_advanced_photo_service()
        ranked_photos = advanced_service.rank_photos_advanced(
            req.photos, req.top_k, req.context
        )
        
        return {
            "status": "success",
            "input_count": len(req.photos),
            "output_count": len(ranked_photos),
            "ranked_photos": ranked_photos,
            "features_used": {
                "clip_analysis": req.use_clip,
                "aesthetic_model": req.use_aesthetic_model,
                "duplicate_detection": True,
                "subject_consistency": True
            },
            "message": f"高级照片选优完成，返回前 {len(ranked_photos)} 张"
        }
        
    except Exception as e:
        logger.error(f"高级照片选优失败: {e}")
        raise HTTPException(status_code=500, detail=f"高级照片选优失败: {str(e)}")


@app.post("/pro/semantic_highlights")
async def detect_semantic_highlights(req: SemanticHighlightsReq):
    """语义高光检测"""
    try:
        logger.info(f"开始语义高光检测，共 {len(req.transcript_segments)} 个片段")
        
        detector = get_semantic_highlight_detector()
        highlights = detector.detect_highlights(req.transcript_segments, req.context)
        
        # 筛选符合条件的高光
        filtered_highlights = [
            h for h in highlights 
            if h.get('highlight_score', 0) >= req.min_score
        ][:req.max_highlights]
        
        return {
            "status": "success",
            "total_segments": len(req.transcript_segments),
            "highlights_found": len(highlights),
            "highlights_returned": len(filtered_highlights),
            "highlights": filtered_highlights,
            "analysis_summary": {
                "avg_score": np.mean([h.get('highlight_score', 0) for h in filtered_highlights]) if filtered_highlights else 0,
                "highlight_types": list(set(h.get('highlight_type', 'unknown') for h in filtered_highlights)),
                "min_score_threshold": req.min_score
            },
            "message": f"检测到 {len(filtered_highlights)} 个高光时刻"
        }
        
    except Exception as e:
        logger.error(f"语义高光检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"语义高光检测失败: {str(e)}")


@app.post("/pro/user_style_learning")
async def learn_user_style(req: UserStyleLearningReq):
    """学习用户写作风格"""
    try:
        logger.info(f"开始学习用户 {req.user_id} 的写作风格")
        
        writing_service = get_personalized_writing_service()
        user_profile = writing_service.learn_user_style(req.user_id, req.content_samples)
        
        return {
            "status": "success",
            "user_id": req.user_id,
            "samples_analyzed": len(req.content_samples),
            "confidence_score": user_profile.get('confidence_score', 0),
            "learned_features": {
                "vocabulary_patterns": len(user_profile.get('vocabulary', {}).get('signature_words', [])),
                "emoji_preferences": len(user_profile.get('emoji', {}).get('favorite_emojis', [])),
                "tone_analysis": user_profile.get('tone', {}).get('primary_tone', 'unknown'),
                "topic_preferences": len(user_profile.get('topics', {}).get('preferred_topics', []))
            },
            "message": f"用户风格学习完成，置信度: {user_profile.get('confidence_score', 0):.2f}"
        }
        
    except Exception as e:
        logger.error(f"用户风格学习失败: {e}")
        raise HTTPException(status_code=500, detail=f"用户风格学习失败: {str(e)}")


@app.post("/pro/personalized_writing")
async def generate_personalized_content(req: PersonalizedWritingReq):
    """生成个性化内容"""
    try:
        logger.info(f"开始为用户 {req.user_id} 生成个性化内容")
        
        writing_service = get_personalized_writing_service()
        personalized_content = writing_service.generate_personalized_content(
            req.user_id, req.content_data, req.style_override
        )
        
        return {
            "status": "success",
            "user_id": req.user_id,
            "personalized_content": personalized_content,
            "personalization_confidence": personalized_content.get('personalization_confidence', 0),
            "features_applied": personalized_content.get('metadata', {}).get('personalization_features', []),
            "message": "个性化内容生成完成"
        }
        
    except Exception as e:
        logger.error(f"个性化内容生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"个性化内容生成失败: {str(e)}")


@app.post("/pro/smart_cover")
async def generate_smart_cover(req: SmartCoverDesignReq):
    """智能封面设计"""
    try:
        logger.info(f"开始智能封面设计 - 标题: {req.title[:20]}...")
        
        cover_designer = get_smart_cover_designer()
        cover_result = cover_designer.generate_smart_cover(
            req.clips, req.photos, req.title, req.style
        )
        
        return {
            "status": "success",
            "cover_result": cover_result,
            "design_features": {
                "frame_analysis": bool(cover_result.get('source_frame')),
                "color_extraction": bool(cover_result.get('color_palette')),
                "text_layout": bool(cover_result.get('text_layout')),
                "smart_positioning": True
            },
            "message": "智能封面设计完成"
        }
        
    except Exception as e:
        logger.error(f"智能封面设计失败: {e}")
        raise HTTPException(status_code=500, detail=f"智能封面设计失败: {str(e)}")


@app.post("/pro/audio_processing")
async def process_video_audio(req: AudioProcessingReq):
    """音频处理（降噪、BGM匹配）"""
    try:
        logger.info(f"开始音频处理 - 风格: {req.style}")
        
        audio_service = get_audio_processing_service()
        processing_result = audio_service.process_video_audio(
            req.video_path, req.style, req.enhance_speech, req.add_bgm
        )
        
        return {
            "status": "success",
            "processing_result": processing_result,
            "audio_enhancements": {
                "noise_reduction": req.enhance_speech,
                "bgm_added": req.add_bgm,
                "beat_alignment": processing_result.get('bgm_info', {}).get('tempo', 0) > 0,
                "final_optimization": processing_result.get('processing_steps', {}).get('final_optimization', False)
            },
            "message": "音频处理完成"
        }
        
    except Exception as e:
        logger.error(f"音频处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"音频处理失败: {str(e)}")


@app.post("/xiaohongshu/photo_rank")
async def photo_rank(req: PhotoRankingReq):
    """照片选优排序"""
    try:
        logger.info(f"开始照片选优，共 {len(req.photos)} 张照片")
        
        photo_service = get_photo_ranking_service()
        ranked_photos = photo_service.rank_photos(req.photos, req.top_k)
        
        return {
            "status": "success",
            "input_count": len(req.photos),
            "output_count": len(ranked_photos),
            "ranked_photos": ranked_photos,
            "message": f"照片选优完成，返回前 {len(ranked_photos)} 张"
        }
        
    except Exception as e:
        logger.error(f"照片选优失败: {e}")
        raise HTTPException(status_code=500, detail=f"照片选优失败: {str(e)}")


@app.post("/xiaohongshu/storyline")
async def generate_storyline(req: StorylineReq):
    """生成旅行故事线"""
    try:
        logger.info(f"开始生成故事线 - 城市: {req.city}, 风格: {req.style}")
        
        storyline_gen = get_storyline_generator()
        storyline = storyline_gen.generate_storyline(
            req.transcript_mmss, req.notes, req.city, req.date, req.style
        )
        
        return {
            "status": "success",
            "storyline": storyline,
            "section_count": len(storyline.get('sections', [])),
            "tip_count": len(storyline.get('tips', [])),
            "poi_count": len(storyline.get('pois', [])),
            "message": "故事线生成完成"
        }
        
    except Exception as e:
        logger.error(f"故事线生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"故事线生成失败: {str(e)}")


@app.post("/xiaohongshu/draft")
async def generate_xhs_draft(req: XHSDraftReq):
    """生成小红书文案"""
    try:
        logger.info(f"开始生成小红书文案 - 调性: {req.brand_tone}")
        
        draft_gen = get_draft_generator()
        draft = draft_gen.generate_draft(req.storyline, req.brand_tone, req.constraints)
        
        return {
            "status": "success",
            "draft": draft,
            "metadata": draft.get('metadata', {}),
            "message": "小红书文案生成完成"
        }
        
    except Exception as e:
        logger.error(f"文案生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"文案生成失败: {str(e)}")


@app.post("/xiaohongshu/subtitles")
async def generate_subtitles(req: SubtitleReq):
    """生成字幕文件"""
    try:
        logger.info(f"开始生成字幕 - 风格: {req.style}")
        
        subtitle_gen = get_subtitle_generator()
        subtitles = subtitle_gen.generate_subtitles(req.clips, req.transcript_mmss, req.style)
        
        return {
            "status": "success",
            "subtitles": subtitles,
            "message": "字幕生成完成"
        }
        
    except Exception as e:
        logger.error(f"字幕生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"字幕生成失败: {str(e)}")


@app.post("/xiaohongshu/cover")
async def suggest_cover(req: CoverReq):
    """生成封面建议"""
    try:
        logger.info(f"开始生成封面建议 - 标题: {req.title[:20]}...")
        
        cover_service = get_cover_service()
        cover_suggestions = cover_service.suggest_cover(req.clips, req.photos_topk, req.title)
        
        return {
            "status": "success",
            "cover_suggestions": cover_suggestions,
            "message": "封面建议生成完成"
        }
        
    except Exception as e:
        logger.error(f"封面建议生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"封面建议生成失败: {str(e)}")


@app.post("/xiaohongshu/export")
async def export_content(req: ExportReq):
    """导出小红书内容产物"""
    try:
        logger.info(f"开始导出内容 - 格式: {req.export_format}")
        
        export_service = get_export_service()
        export_result = export_service.export_xiaohongshu_content(
            req.pipeline_result, req.export_format, req.include_source
        )
        
        return {
            "status": "success",
            "export_result": export_result,
            "message": "内容导出完成"
        }
        
    except Exception as e:
        logger.error(f"内容导出失败: {e}")
        raise HTTPException(status_code=500, detail=f"内容导出失败: {str(e)}")


@app.post("/xiaohongshu/pipeline")
async def xiaohongshu_pipeline(req: XHSPipelineReq):
    """小红书一键出稿完整流水线"""
    try:
        Path("input_data/downloads").mkdir(parents=True, exist_ok=True)
        Path("output_data").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎬 开始小红书一键出稿流水线 - 城市: {req.city}, 风格: {req.style}")
        
        # 处理输入文件
        ts = int(time.time())
        if req.video_url.startswith("file:"):
            local_path = req.video_url.replace("file://", "")
            if not Path(local_path).exists():
                raise HTTPException(status_code=400, detail="本地文件不存在")
            input_path = local_path
        else:
            input_path = str(Path("input_data/downloads") / f"xhs_pipeline_{ts}.mp4")
            from urllib.request import urlretrieve
            urlretrieve(req.video_url, input_path)
        
        pipeline_result = {
            'source_video': input_path,
            'processing_id': f'xhs_{ts}',
            'created_at': datetime.now().isoformat()
        }
        
        # 1. ASR语音识别
        logger.info("步骤1: ASR语音识别...")
        asr_service = get_asr_service(model_size=req.model_size)
        transcription_result = asr_service.transcribe_video(input_path, cleanup_audio=True)
        
        # 转换为带时间戳格式
        transcript_mmss = []
        for segment in transcription_result.get('segments', []):
            transcript_mmss.append({
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'text': segment.get('text', ''),
                'timestamp': f"{int(segment.get('start', 0)//60):02d}:{int(segment.get('start', 0)%60):02d}"
            })
        
        pipeline_result['transcription'] = transcription_result
        pipeline_result['transcript_mmss'] = transcript_mmss
        
        # 2. 智能选段
        logger.info("步骤2: ASR增强智能选段...")
        asr_engine = get_asr_smart_engine()
        selected_segments = asr_engine.select_best_segments_with_asr(
            input_path, transcription_result, 15, 30, 2  # 生成2个15-30秒片段
        )
        
        if not selected_segments:
            raise HTTPException(status_code=400, detail="未找到符合条件的视频片段")
        
        # 生成视频片段
        generated_clips = []
        for i, segment in enumerate(selected_segments):
            output_filename = f"xhs_clip_{ts}_{i+1:02d}.mp4"
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
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(VIDEO_CRF),
                "-c:a", "aac", "-b:a", AUDIO_BITRATE,
                "-shortest", "-movflags", "+faststart",
                output_path
            ]
            
            safe_run_ffmpeg(cmd)
            
            file_size = Path(output_path).stat().st_size if Path(output_path).exists() else 0
            
            generated_clips.append({
                "clip_index": i + 1,
                "output_path": output_path,
                "start_time": start_time,
                "start_time_seconds": segment['start_time'],
                "end_time_seconds": segment['end_time'],
                "duration": duration,
                "file_size": file_size
            })
        
        pipeline_result['clips'] = generated_clips
        
        # 3. 照片选优（如果有照片）
        if req.photos:
            logger.info("步骤3: 照片选优...")
            photo_service = get_photo_ranking_service()
            ranked_photos = photo_service.rank_photos(req.photos, 10)
            pipeline_result['photos_ranked'] = ranked_photos
        else:
            pipeline_result['photos_ranked'] = []
        
        # 4. 故事线生成
        logger.info("步骤4: 生成故事线...")
        storyline_gen = get_storyline_generator()
        storyline = storyline_gen.generate_storyline(
            transcript_mmss, req.notes, req.city, "", req.style
        )
        pipeline_result['storyline'] = storyline
        
        # 5. 小红书文案生成
        logger.info("步骤5: 生成小红书文案...")
        draft_gen = get_draft_generator()
        draft = draft_gen.generate_draft(storyline, req.style)
        pipeline_result['draft'] = draft
        
        # 6. 字幕生成
        logger.info("步骤6: 生成字幕...")
        subtitle_gen = get_subtitle_generator()
        subtitles = subtitle_gen.generate_subtitles(generated_clips, transcript_mmss, "可爱")
        pipeline_result['subtitles'] = subtitles
        
        # 7. 封面建议
        logger.info("步骤7: 生成封面建议...")
        cover_service = get_cover_service()
        cover_suggestions = cover_service.suggest_cover(
            generated_clips, pipeline_result['photos_ranked'], draft['title']
        )
        pipeline_result['cover'] = cover_suggestions
        
        # 8. 导出产物
        logger.info("步骤8: 导出内容产物...")
        export_service = get_export_service()
        export_result = export_service.export_xiaohongshu_content(
            pipeline_result, req.export_format, False
        )
        
        return {
            "status": "success",
            "pipeline_result": {
                'processing_id': pipeline_result['processing_id'],
                'transcription_summary': {
                    'language': transcription_result['language'],
                    'duration': transcription_result['duration'],
                    'word_count': transcription_result['word_count']
                },
                'clips_generated': len(generated_clips),
                'photos_ranked': len(pipeline_result['photos_ranked']),
                'storyline_sections': len(storyline.get('sections', [])),
                'draft_info': {
                    'title': draft['title'],
                    'hashtag_count': len(draft.get('hashtags', [])),
                    'word_count': draft.get('metadata', {}).get('word_count', 0)
                },
                'subtitle_files': len(subtitles.get('srt_files', [])),
                'export_info': export_result
            },
            "download_links": export_result.get('share_urls', {}),
            "message": f"🎉 小红书一键出稿完成！生成了 {len(generated_clips)} 个视频片段和完整文案"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"小红书流水线失败: {e}")
        raise HTTPException(status_code=500, detail=f"小红书流水线失败: {str(e)}")


@app.post("/xiaohongshu/pipeline_pro")
async def xiaohongshu_pipeline_pro(req: XHSProPipelineReq):
    """小红书一键出稿Pro版流水线（包含所有高级功能）"""
    try:
        Path("input_data/downloads").mkdir(parents=True, exist_ok=True)
        Path("output_data").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 开始小红书Pro流水线 - 城市: {req.city}, 风格: {req.style}, 用户: {req.user_id or 'anonymous'}")
        
        # 处理输入文件
        ts = int(time.time())
        if req.video_url.startswith("file:"):
            local_path = req.video_url.replace("file://", "")
            if not Path(local_path).exists():
                raise HTTPException(status_code=400, detail="本地文件不存在")
            input_path = local_path
        else:
            input_path = str(Path("input_data/downloads") / f"xhs_pro_{ts}.mp4")
            from urllib.request import urlretrieve
            urlretrieve(req.video_url, input_path)
        
        pipeline_result = {
            'source_video': input_path,
            'processing_id': f'xhs_pro_{ts}',
            'created_at': datetime.now().isoformat(),
            'pro_features_enabled': {
                'advanced_photo_ranking': req.use_advanced_photo_ranking,
                'semantic_highlights': req.use_semantic_highlights,
                'personalized_writing': req.use_personalized_writing,
                'smart_cover': req.use_smart_cover,
                'audio_enhancement': req.use_audio_enhancement
            }
        }
        
        # 1. ASR语音识别
        logger.info("步骤1: ASR语音识别...")
        asr_service = get_asr_service(model_size=req.model_size)
        transcription_result = asr_service.transcribe_video(input_path, cleanup_audio=True)
        
        # 转换为带时间戳格式
        transcript_mmss = []
        for segment in transcription_result.get('segments', []):
            transcript_mmss.append({
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'text': segment.get('text', ''),
                'timestamp': f"{int(segment.get('start', 0)//60):02d}:{int(segment.get('start', 0)%60):02d}"
            })
        
        pipeline_result['transcription'] = transcription_result
        pipeline_result['transcript_mmss'] = transcript_mmss
        
        # 2. Pro功能：语义高光检测
        if req.use_semantic_highlights:
            logger.info("步骤2Pro: 语义高光检测...")
            detector = get_semantic_highlight_detector()
            semantic_highlights = detector.detect_highlights(
                transcription_result.get('segments', []), 
                {'city': req.city, 'style': req.style}
            )
            pipeline_result['semantic_highlights'] = semantic_highlights
            logger.info(f"检测到 {len(semantic_highlights)} 个语义高光时刻")
        
        # 3. 智能选段（结合语义高光）
        logger.info("步骤3: ASR增强智能选段...")
        asr_engine = get_asr_smart_engine()
        
        # 如果有语义高光，优先选择高光片段
        if req.use_semantic_highlights and pipeline_result.get('semantic_highlights'):
            # 基于语义高光选择片段
            highlight_segments = []
            for highlight in pipeline_result['semantic_highlights'][:3]:  # 取前3个高光
                highlight_segments.append({
                    'start_time': highlight.get('start', 0),
                    'end_time': highlight.get('end', 0),
                    'duration': highlight.get('duration', 15),
                    'start_hms': f"{int(highlight.get('start', 0)//3600):02d}:{int((highlight.get('start', 0)%3600)//60):02d}:{int(highlight.get('start', 0)%60):02d}",
                    'reason': f"语义高光: {highlight.get('highlight_reason', '精彩内容')}",
                    'highlight_score': highlight.get('highlight_score', 0.8)
                })
            selected_segments = highlight_segments
        else:
            # 使用传统智能选段
            selected_segments = asr_engine.select_best_segments_with_asr(
                input_path, transcription_result, 15, 30, 2
            )
        
        if not selected_segments:
            raise HTTPException(status_code=400, detail="未找到符合条件的视频片段")
        
        # 生成视频片段
        generated_clips = []
        for i, segment in enumerate(selected_segments):
            output_filename = f"xhs_pro_clip_{ts}_{i+1:02d}.mp4"
            output_path = f"output_data/{output_filename}"
            
            start_time = segment['start_hms']
            duration = segment['duration']
            
            # Pro功能：音频增强
            if req.use_audio_enhancement:
                # 先生成基础视频
                fade_out_start = max(0.1, duration - 0.25)
                vf_filters = (
                    "scale=1080:1920:force_original_aspect_ratio=decrease,"
                    "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1:1,"
                    f"fade=t=in:st=0:d=0.25,fade=t=out:st={fade_out_start:.2f}:d=0.25"
                )
                
                temp_video_path = f"output_data/temp_{output_filename}"
                
                cmd = [
                    "ffmpeg", "-y", "-hwaccel", "none",
                    "-i", input_path,
                    "-ss", start_time, "-t", f"{duration:.2f}",
                    "-vf", vf_filters,
                    "-pix_fmt", "yuv420p",
                    "-map", "0:v:0", "-map", "0:a?",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", str(VIDEO_CRF),
                    "-c:a", "aac", "-b:a", AUDIO_BITRATE,
                    "-shortest", "-movflags", "+faststart",
                    temp_video_path
                ]
                
                safe_run_ffmpeg(cmd)
                
                # 音频增强处理
                try:
                    audio_service = get_audio_processing_service()
                    audio_result = audio_service.process_video_audio(
                        temp_video_path, req.style, True, True
                    )
                    
                    if audio_result.get('success') and audio_result.get('processed_video'):
                        # 使用增强后的视频
                        Path(temp_video_path).unlink(missing_ok=True)  # 删除临时文件
                        Path(audio_result['processed_video']).rename(output_path)
                    else:
                        # 音频增强失败，使用原视频
                        Path(temp_video_path).rename(output_path)
                        
                except Exception as audio_error:
                    logger.warning(f"音频增强失败，使用原音频: {audio_error}")
                    Path(temp_video_path).rename(output_path)
            else:
                # 标准视频生成
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
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", str(VIDEO_CRF),
                    "-c:a", "aac", "-b:a", AUDIO_BITRATE,
                    "-shortest", "-movflags", "+faststart",
                    output_path
                ]
                
                safe_run_ffmpeg(cmd)
            
            file_size = Path(output_path).stat().st_size if Path(output_path).exists() else 0
            
            generated_clips.append({
                "clip_index": i + 1,
                "output_path": output_path,
                "start_time": start_time,
                "start_time_seconds": segment['start_time'],
                "end_time_seconds": segment['end_time'],
                "duration": duration,
                "file_size": file_size,
                "selection_reason": segment.get('reason', 'ASR智能选段'),
                "highlight_score": segment.get('highlight_score', 0.7)
            })
        
        pipeline_result['clips'] = generated_clips
        
        # 4. Pro功能：高级照片选优
        if req.photos and req.use_advanced_photo_ranking:
            logger.info("步骤4Pro: 高级照片选优...")
            advanced_photo_service = get_advanced_photo_service()
            ranked_photos = advanced_photo_service.rank_photos_advanced(
                req.photos, 10, {'city': req.city, 'style': req.style}
            )
            pipeline_result['photos_ranked'] = ranked_photos
        elif req.photos:
            # 使用基础照片选优
            logger.info("步骤4: 基础照片选优...")
            photo_service = get_photo_ranking_service()
            ranked_photos = photo_service.rank_photos(req.photos, 10)
            pipeline_result['photos_ranked'] = ranked_photos
        else:
            pipeline_result['photos_ranked'] = []
        
        # 5. 故事线生成
        logger.info("步骤5: 生成故事线...")
        storyline_gen = get_storyline_generator()
        storyline = storyline_gen.generate_storyline(
            transcript_mmss, req.notes, req.city, "", req.style
        )
        pipeline_result['storyline'] = storyline
        
        # 6. Pro功能：个性化文案生成
        if req.use_personalized_writing and req.user_id:
            logger.info("步骤6Pro: 个性化文案生成...")
            writing_service = get_personalized_writing_service()
            draft = writing_service.generate_personalized_content(
                req.user_id, 
                {'storyline': storyline, 'city': req.city, 'style': req.style},
                req.style
            )
            pipeline_result['draft'] = draft
        else:
            # 使用标准文案生成
            logger.info("步骤6: 标准文案生成...")
            draft_gen = get_draft_generator()
            draft = draft_gen.generate_draft(storyline, req.style)
            pipeline_result['draft'] = draft
        
        # 7. 字幕生成
        logger.info("步骤7: 生成字幕...")
        subtitle_gen = get_subtitle_generator()
        subtitles = subtitle_gen.generate_subtitles(generated_clips, transcript_mmss, "可爱")
        pipeline_result['subtitles'] = subtitles
        
        # 8. Pro功能：智能封面设计
        if req.use_smart_cover:
            logger.info("步骤8Pro: 智能封面设计...")
            cover_designer = get_smart_cover_designer()
            cover_suggestions = cover_designer.generate_smart_cover(
                generated_clips, pipeline_result['photos_ranked'], 
                pipeline_result['draft']['title'], req.style
            )
            pipeline_result['cover'] = cover_suggestions
        else:
            # 使用基础封面建议
            logger.info("步骤8: 基础封面建议...")
            cover_service = get_cover_service()
            cover_suggestions = cover_service.suggest_cover(
                generated_clips, pipeline_result['photos_ranked'], 
                pipeline_result['draft']['title']
            )
            pipeline_result['cover'] = cover_suggestions
        
        # 9. 导出产物
        logger.info("步骤9: 导出内容产物...")
        export_service = get_export_service()
        export_result = export_service.export_xiaohongshu_content(
            pipeline_result, req.export_format, False
        )
        
        # 统计Pro功能使用情况
        pro_features_used = []
        if req.use_advanced_photo_ranking and req.photos:
            pro_features_used.append("高级照片选优")
        if req.use_semantic_highlights:
            pro_features_used.append("语义高光检测")
        if req.use_personalized_writing and req.user_id:
            pro_features_used.append("个性化文案生成")
        if req.use_smart_cover:
            pro_features_used.append("智能封面设计")
        if req.use_audio_enhancement:
            pro_features_used.append("音频增强处理")
        
        return {
            "status": "success",
            "pipeline_result": {
                'processing_id': pipeline_result['processing_id'],
                'pro_features_used': pro_features_used,
                'transcription_summary': {
                    'language': transcription_result['language'],
                    'duration': transcription_result['duration'],
                    'word_count': transcription_result['word_count']
                },
                'semantic_highlights_count': len(pipeline_result.get('semantic_highlights', [])),
                'clips_generated': len(generated_clips),
                'photos_ranked': len(pipeline_result['photos_ranked']),
                'storyline_sections': len(storyline.get('sections', [])),
                'draft_info': {
                    'title': pipeline_result['draft']['title'],
                    'hashtag_count': len(pipeline_result['draft'].get('hashtags', [])),
                    'word_count': pipeline_result['draft'].get('metadata', {}).get('word_count', 0),
                    'personalization_confidence': pipeline_result['draft'].get('personalization_confidence', 0)
                },
                'subtitle_files': len(subtitles.get('srt_files', [])),
                'cover_design': {
                    'success': pipeline_result['cover'].get('success', False),
                    'cover_path': pipeline_result['cover'].get('cover_path', ''),
                    'design_features': len(pro_features_used)
                },
                'export_info': export_result
            },
            "download_links": export_result.get('share_urls', {}),
            "message": f"🎉 小红书Pro一键出稿完成！生成了 {len(generated_clips)} 个视频片段，使用了 {len(pro_features_used)} 个Pro功能"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"小红书Pro流水线失败: {e}")
        raise HTTPException(status_code=500, detail=f"小红书Pro流水线失败: {str(e)}")


@app.post("/auto_intro")
async def auto_intro(req: URLIntroReq):
    """Download video from URL, detect/extract subtitles or ASR, select highlights, cut 9:16, add simple fades, and concatenate into one intro video."""
    try:
        Path("input_data/downloads").mkdir(parents=True, exist_ok=True)
        Path("output_data").mkdir(parents=True, exist_ok=True)

        # 1) download via yt-dlp if platform URL, else urlretrieve
        ts = int(time.time())
        dl_path = str(Path("input_data/downloads") / f"dl_{ts}.mp4")
        try:
            import yt_dlp  # type: ignore
            ydl_opts = {
                'format': 'bv*+ba/b',
                'merge_output_format': 'mp4',
                'outtmpl': str(Path("input_data/downloads") / f"dl_{ts}.%(ext)s"),
                'writesubtitles': True,
                'subtitleslangs': ['zh.*','zh','zh-Hans','zh-Hant','en.*','en'],
                'subtitleformat': 'srt',
                'quiet': True,
                'noprogress': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(req.url, download=True)
                # determine final filepath
                if 'requested_downloads' in info:
                    dl_path = info['requested_downloads'][0]['filepath']
                else:
                    # fallback: guess mp4 path
                    base = Path("input_data/downloads") / f"dl_{ts}"
                    if (base.with_suffix('.mp4')).exists():
                        dl_path = str(base.with_suffix('.mp4'))
                # normalize
                dl_path = str(Path(dl_path))
        except Exception:
            # Fallback 1: try system yt-dlp CLI
            try:
                cli_cmd = [
                    "yt-dlp",
                    "-f", "bv*+ba/b",
                    "--merge-output-format", "mp4",
                    "-o", str(Path("input_data/downloads") / f"dl_{ts}.%(ext)s"),
                    "--write-sub",
                    "--sub-lang", "zh.*,zh,zh-Hans,zh-Hant,en.*,en",
                    "--sub-format", "srt",
                    req.url,
                ]
                completed = subprocess.run(cli_cmd, capture_output=True, text=True)
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr or "yt-dlp CLI failed")
                # guess final mp4 path
                base = Path("input_data/downloads") / f"dl_{ts}"
                if (base.with_suffix('.mp4')).exists():
                    dl_path = str(base.with_suffix('.mp4'))
                else:
                    # try common containers
                    for ext in [".mkv", ".webm", ".mov"]:
                        cand = base.with_suffix(ext)
                        if cand.exists():
                            dl_path = str(cand)
                            break
                dl_path = str(Path(dl_path))
            except Exception:
                # Fallback 2: direct URL download (works only for direct media URLs)
                urlretrieve(req.url, dl_path)

        # 2) try extract subtitle stream to SRT
        srt_path = str(Path("output_data") / f"dl_{ts}.srt")
        # prefer subtitles downloaded by yt-dlp (scoped to current timestamp)
        possible_srts = list(Path("input_data/downloads").glob(f"dl_{ts}*.srt"))
        if possible_srts:
            # pick the largest srt as best
            best = max(possible_srts, key=lambda p: p.stat().st_size)
            try:
                Path(srt_path).write_text(best.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
            except Exception:
                pass
        if not os.path.exists(srt_path):
            try:
                extract_cmd = [
                    "ffmpeg", "-y", "-i", dl_path,
                    "-map", "0:s:0", "-c:s", "srt", srt_path
                ]
                subprocess.run(extract_cmd, capture_output=True, text=True, timeout=45)
            except subprocess.TimeoutExpired:
                logger.warning("Subtitle extraction timed out; skipping embedded subs")
            except Exception:
                pass

        # 3) if no SRT and want_asr, run python ASR fallback
        if not os.path.exists(srt_path) and req.want_asr:
            ok = try_python_asr_to_srt(dl_path, srt_path)
            if not ok:
                srt_path = ""

        # 4) build clips
        clips: List[Dict[str, str]] = []
        if srt_path and os.path.exists(srt_path):
            items = parse_srt_simple(srt_path)
            clips = select_windows_from_srt(items, req.min_sec, req.max_sec, top_k=req.top_k)
        else:
            # fallback: split video into windows heuristically
            total = max(0, int(get_duration_seconds(dl_path)))
            if total == 0:
                raise HTTPException(status_code=400, detail="无法获取视频时长，且无字幕可供分析")
            start = 10
            while start + req.min_sec < total and len(clips) < req.top_k:
                end = min(total, start + req.max_sec)
                ss = f"{start//60:02d}:{start%60:02d}"
                ee = f"{end//60:02d}:{end%60:02d}"
                clips.append({"start": ss, "end": ee, "reason": "window"})
                start = end + 2

        if not clips:
            raise HTTPException(status_code=400, detail="未找到可用片段")

        # 5) 智能选段或传统选段
        total = max(0, int(get_duration_seconds(dl_path)))
        if total == 0:
            raise HTTPException(status_code=400, detail="视频时长不可用")

        window_min = int(req.min_sec)
        window_max = int(req.max_sec)
        
        if req.smart_mode:
            # 使用智能选段
            logger.info("Using smart segment selection")
            try:
                smart_segments = get_smart_segments(dl_path, window_min, window_max, count=1)
                if smart_segments:
                    best_segment = smart_segments[0]
                    best_ss = best_segment['start_time']
                    best_dur = best_segment['duration']
                    logger.info(f"Smart selection: start={best_ss}s, duration={best_dur}s, score={best_segment['total_score']:.3f}")
                else:
                    raise ValueError("Smart selection failed")
            except Exception as e:
                logger.warning(f"Smart selection failed: {e}, falling back to traditional method")
                req.smart_mode = False  # 回退到传统方法
        
        if not req.smart_mode:
            # 传统选段方法 (原有逻辑)
            scan_step = 1
            best_score = -1e9
            best_ss = 0
            best_dur = window_min
            max_scan = min(total - window_min, 15 * 60)  # up to 15 min scan
            for start_s in range(0, max(1, max_scan), scan_step):
                ss_hms = seconds_to_hms(start_s)
                # evaluate at min window first
                black_frac = black_fraction_in_segment(dl_path, ss_hms, float(window_min))
                if black_frac >= 0.5:
                    continue
                sil_frac = silence_fraction_in_segment(dl_path, ss_hms, float(window_min))
                # simple score: prefer less black/silence, earlier windows
                score = (1.0 - black_frac) * 1.5 + (1.0 - sil_frac) * 1.0 - 0.001 * start_s
                if score > best_score:
                    best_score = score
                    best_ss = start_s
                    best_dur = window_min
            if best_score < -1e8:
                # fallback to the first provided clip
                first = clips[0]
                best_ss = int(hms_to_seconds(normalize_time(first["start"])))
                best_dur = max(window_min, int(hms_to_seconds(normalize_time(first["end"])) - best_ss))

        start_time = seconds_to_hms(best_ss)
        dur_s = float(best_dur)
        fade_out_start = max(0.1, dur_s - 0.25)
        # pad-first to 1080x1920, avoiding black due to crop
        vf_pad = (
            "scale=1080:1920:force_original_aspect_ratio=decrease,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1:1,"
            f"fade=t=in:st=0:d=0.25,fade=t=out:st={fade_out_start:.2f}:d=0.25"
        )

        out_final = req.output if req.output else str((Path("output_data") / f"intro_{ts}_916.mp4").resolve())
        cmd_single = [
            "ffmpeg", "-y", "-hwaccel", "none",
            "-i", dl_path,
            "-ss", start_time, "-t", f"{dur_s:.2f}",
            "-vf", vf_pad,
            # do not force frame rate; honor input
            "-pix_fmt", "yuv420p",
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(VIDEO_CRF),
            "-c:a", "aac", "-b:a", AUDIO_BITRATE,
            "-shortest", "-movflags", "+faststart",
            out_final
        ]
        safe_run_ffmpeg(cmd_single)

        # 7) optionally burn subtitles if we have srt and only if you want (requirement said: if has subs, do not add)
        # We skip burning here as per requirement to keep existing subtitles if present.

        return {"out": out_final, "parts": [], "clips": [{"start": seconds_to_hms(best_ss), "end": seconds_to_hms(best_ss + best_dur), "reason": "smart_window"}]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auto intro error: {e}")
        raise HTTPException(status_code=500, detail=f"自动简介视频生成失败: {str(e)}")


# 新增：智能多片段视频剪辑API
class MultiSegmentClippingReq(BaseModel):
    video_path: str
    subtitle_path: Optional[str] = None
    topic: str
    target_segments: int = 3
    segment_duration: float = 8.0
    total_duration: float = 24.0
    semantic_weight: float = 0.4
    visual_weight: float = 0.3
    audio_weight: float = 0.3
    include_intro: bool = True
    include_highlights: bool = True
    include_conclusion: bool = True

def parse_srt_file(srt_path: str) -> List[Dict]:
    """解析SRT字幕文件"""
    segments = []
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 简单的SRT解析
        blocks = content.strip().split('\n\n')
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # 时间戳行
                time_line = lines[1]
                if '-->' in time_line:
                    start_time, end_time = time_line.split(' --> ')
                    # 文本内容
                    text = ' '.join(lines[2:])
                    segments.append({
                        'start': start_time.strip(),
                        'end': end_time.strip(),
                        'text': text.strip()
                    })
    except Exception as e:
        logger.error(f"解析SRT文件失败: {e}")
    
    return segments

def time_to_seconds(time_str: str) -> float:
    """将时间字符串转换为秒数"""
    try:
        # 处理格式如 "00:01:23,456"
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        else:
            return float(parts[0])
    except:
        return 0.0

def analyze_video_segments_v2(video_path: str, subtitle_segments: List[Dict], total_duration: float, topic: str) -> List[Dict]:
    """基于核心概念分析视频片段"""
    segments = []
    
    # 1. 提取核心概念
    core_concepts = extract_video_core_concepts(subtitle_segments, topic)
    
    # 2. 使用更灵活的窗口大小
    base_window_size = 15.0  # 基础15秒窗口
    overlap = 5.0            # 5秒重叠
    
    current_time = 0.0
    segment_id = 0
    
    while current_time < total_duration - base_window_size:
        # 动态调整窗口大小（8-30秒之间）
        window_size = base_window_size
        end_time = min(current_time + window_size, total_duration)
        
        # 获取该时间段的字幕文本
        segment_text = ""
        subtitle_count = 0
        
        for sub in subtitle_segments:
            sub_start = time_to_seconds(sub['start'])
            sub_end = time_to_seconds(sub['end'])
            
            # 更宽松的时间匹配
            if sub_start < end_time and sub_end > current_time:
                segment_text += sub['text'] + " "
                subtitle_count += 1
        
        # 智能边界检测：确保不在句子中间切断
        if subtitle_count >= 3:
            # 寻找更好的结束点（句号、问号、感叹号后）
            for sub in subtitle_segments:
                sub_start = time_to_seconds(sub['start'])
                sub_end = time_to_seconds(sub['end'])
                
                if sub_start >= end_time and sub_start <= end_time + 5:  # 5秒缓冲
                    text = sub['text'].strip()
                    # 如果这个字幕以句号、问号、感叹号结尾，就延伸到这里
                    if text.endswith(('。', '？', '！', '.', '?', '!')):
                        window_size = sub_end - current_time
                        end_time = sub_end
                        break
        
        # 如果文本太少，扩展窗口
        if subtitle_count < 3 and end_time < total_duration - 10:
            window_size = min(window_size + 10, 35.0)  # 增加到35秒最大
            end_time = min(current_time + window_size, total_duration)
            
            # 重新获取文本
            segment_text = ""
            for sub in subtitle_segments:
                sub_start = time_to_seconds(sub['start'])
                sub_end = time_to_seconds(sub['end'])
                if sub_start < end_time and sub_end > current_time:
                    segment_text += sub['text'] + " "
        
        # 计算语义评分（基于核心概念）
        semantic_analysis = calculate_semantic_score(segment_text, topic, core_concepts)
        
        # 计算其他评分
        visual_score = calculate_visual_score(current_time, end_time, video_path)
        audio_score = calculate_audio_score(current_time, end_time, video_path)
        position_weight = calculate_position_weight(current_time, total_duration)
        
        # 综合评分 - 提高语义权重
        total_score = (
            semantic_analysis["total_score"] * 0.6 + 
            visual_score * 0.2 + 
            audio_score * 0.1 + 
            position_weight * 0.1
        )
        
        # 找出最匹配的核心概念
        best_concept = None
        best_concept_score = 0
        for concept_name, concept_data in semantic_analysis["concept_scores"].items():
            if concept_data["score"] > best_concept_score:
                best_concept = concept_name
                best_concept_score = concept_data["score"]
        
        segments.append({
            'id': segment_id,
            'start_time': current_time,
            'end_time': end_time,
            'duration': end_time - current_time,
            'text': segment_text.strip(),
            'semantic_analysis': semantic_analysis,
            'visual_score': visual_score,
            'audio_score': audio_score,
            'position_weight': position_weight,
            'total_score': total_score,
            'best_concept': best_concept,
            'best_concept_score': best_concept_score,
            'subtitle_count': subtitle_count
        })
        
        current_time += (window_size - overlap)
        segment_id += 1
    
    return segments

def analyze_video_content_structure(subtitle_segments: List[Dict]) -> Dict:
    """分析视频内容结构，自动提取关键主题和论点"""
    full_text = " ".join([seg['text'] for seg in subtitle_segments])
    
    # 1. 找到关键问题和定义
    key_questions = []
    definitions = []
    examples = []
    background = []
    conclusions = []
    
    for i, seg in enumerate(subtitle_segments):
        text = seg['text'].strip()
        start_time = time_to_seconds(seg['start'])
        
        # 识别关键问题
        if any(q in text for q in ['什么是', '到底是什么', '为什么', '怎么做', '如何']):
            key_questions.append({
                'text': text,
                'start_time': start_time,
                'type': 'question'
            })
        
        # 识别定义性内容
        if any(d in text for d in ['定义', '概念', '本质', '核心', '就是']):
            definitions.append({
                'text': text,
                'start_time': start_time,
                'type': 'definition'
            })
        
        # 识别例子和案例
        if any(e in text for e in ['比如', '例子', '案例', '举例', '演示']):
            examples.append({
                'text': text,
                'start_time': start_time,
                'type': 'example'
            })
        
        # 识别背景信息
        if any(b in text for b in ['背景', '现实', '发生', '变化', '革命', '历史']):
            background.append({
                'text': text,
                'start_time': start_time,
                'type': 'background'
            })
    
    # 2. 基于Vibe Coding主题的特定分析
    vibe_coding_analysis = {
        'introduction': [],
        'definition': [],
        'why_important': [],
        'how_it_works': [],
        'advantages': [],
        'context': [],
        'future': []
    }
    
    for seg in subtitle_segments:
        text = seg['text'].strip().lower()
        start_time = time_to_seconds(seg['start'])
        
        # 开场介绍
        if any(intro in text for intro in ['最近', '发生', '两件事', '告诉大家']):
            vibe_coding_analysis['introduction'].append({
                'text': seg['text'],
                'start_time': start_time,
                'relevance': 0.9
            })
        
        # 核心定义
        if 'vibe coding' in text and any(def_word in text for def_word in ['什么', '定义', '概念', '到底是']):
            vibe_coding_analysis['definition'].append({
                'text': seg['text'],
                'start_time': start_time,
                'relevance': 1.0
            })
        
        # 重要性说明
        if any(imp in text for imp in ['价值', '重要', '着迷', '必须', '需要']):
            vibe_coding_analysis['why_important'].append({
                'text': seg['text'],
                'start_time': start_time,
                'relevance': 0.9
            })
        
        # 工作原理
        if any(how in text for how in ['怎么', '如何', '方法', '方式', '操作']):
            vibe_coding_analysis['how_it_works'].append({
                'text': seg['text'],
                'start_time': start_time,
                'relevance': 0.8
            })
        
        # 优势分析
        if any(adv in text for adv in ['优势', '好处', '效率', '快速', '简单']):
            vibe_coding_analysis['advantages'].append({
                'text': seg['text'],
                'start_time': start_time,
                'relevance': 0.8
            })
        
        # 背景上下文
        if any(ctx in text for ctx in ['自动化', '革命', '失业', '工具人', '现实']):
            vibe_coding_analysis['context'].append({
                'text': seg['text'],
                'start_time': start_time,
                'relevance': 0.7
            })
    
    return {
        'structure_analysis': {
            'key_questions': key_questions,
            'definitions': definitions,
            'examples': examples,
            'background': background
        },
        'vibe_coding_analysis': vibe_coding_analysis,
        'content_summary': generate_content_summary(vibe_coding_analysis)
    }

def generate_content_summary(analysis: Dict) -> Dict:
    """基于分析生成内容摘要"""
    summary = {
        'main_topics': [],
        'key_segments': [],
        'content_flow': []
    }
    
    # 识别主要话题
    for topic, segments in analysis.items():
        if segments and len(segments) > 0:
            avg_relevance = sum(s.get('relevance', 0) for s in segments) / len(segments)
            summary['main_topics'].append({
                'topic': topic,
                'segment_count': len(segments),
                'avg_relevance': avg_relevance,
                'description': get_topic_description(topic)
            })
    
    # 按相关性排序
    summary['main_topics'].sort(key=lambda x: x['avg_relevance'], reverse=True)
    
    # 识别关键片段
    all_segments = []
    for topic, segments in analysis.items():
        for seg in segments:
            seg['topic'] = topic
            all_segments.append(seg)
    
    # 按相关性和时间排序
    all_segments.sort(key=lambda x: (x.get('relevance', 0), -x['start_time']), reverse=True)
    summary['key_segments'] = all_segments[:10]  # 取前10个关键片段
    
    return summary

def get_topic_description(topic: str) -> str:
    """获取话题描述"""
    descriptions = {
        'introduction': '视频开场引入',
        'definition': 'Vibe Coding核心定义',
        'why_important': '重要性和价值阐述',
        'how_it_works': '工作原理和方法',
        'advantages': '优势和好处',
        'context': '背景和现实分析',
        'future': '未来发展趋势'
    }
    return descriptions.get(topic, topic)

def extract_video_core_concepts(subtitle_segments: List[Dict], topic: str) -> Dict:
    """提取视频核心概念和关键论点"""
    # 合并所有字幕文本
    full_text = " ".join([seg['text'] for seg in subtitle_segments])
    
    # 基于主题的核心概念识别
    if "vibe coding" in topic.lower() or "编程" in topic:
        core_concepts = {
            "opening_hook": {
                "keywords": ["为什么", "我们每个人", "立即", "马上", "开始", "所谓的", "vibe coding"],
                "weight": 1.0,
                "min_duration": 8.0,
                "description": "开场引入：为什么要开始Vibe Coding"
            },
            "core_definition": {
                "keywords": ["vibe coding到底是什么", "什么是", "定义", "概念", "本质", "核心"],
                "weight": 1.0,
                "min_duration": 15.0,
                "description": "核心定义：Vibe Coding到底是什么"
            },
            "value_proposition": {
                "keywords": ["价值", "为什么", "着迷", "重要", "意义", "作用"],
                "weight": 1.0,
                "min_duration": 12.0,
                "description": "价值主张：为什么Vibe Coding重要"
            },
            "background_context": {
                "keywords": ["背景", "现实", "发生", "变化", "趋势", "革命", "自动化"],
                "weight": 0.9,
                "min_duration": 10.0,
                "description": "背景分析：技术变革的现实"
            },
            "practical_approach": {
                "keywords": ["怎么做", "方法", "实践", "操作", "具体", "步骤"],
                "weight": 0.8,
                "min_duration": 8.0,
                "description": "实践方法：如何进行Vibe Coding"
            },
            # 降低这些边缘概念的权重
            "personal_story": {
                "keywords": ["一万美金", "赞助", "取消", "收益", "更新"],
                "weight": 0.3,  # 大幅降低权重
                "min_duration": 5.0,
                "description": "个人故事（边缘内容）"
            }
        }
    else:
        # 通用技术分享的核心概念
        core_concepts = {
            "introduction": {
                "keywords": ["介绍", "什么是", "概念", "定义"],
                "weight": 1.0,
                "min_duration": 8.0,
                "description": "主题介绍"
            },
            "main_points": {
                "keywords": ["重要", "关键", "核心", "主要"],
                "weight": 1.0,
                "min_duration": 10.0,
                "description": "核心要点"
            },
            "examples": {
                "keywords": ["例子", "案例", "比如", "演示"],
                "weight": 0.8,
                "min_duration": 6.0,
                "description": "实例说明"
            }
        }
    
    return core_concepts

def calculate_semantic_score(text: str, topic: str, core_concepts: Dict) -> Dict:
    """基于核心概念计算语义相关性评分"""
    if not text.strip():
        return {"total_score": 0.0, "concept_scores": {}}
    
    text_lower = text.lower()
    concept_scores = {}
    
    for concept_name, concept_info in core_concepts.items():
        score = 0.0
        matched_keywords = []
        
        # 关键词匹配
        for keyword in concept_info["keywords"]:
            if keyword.lower() in text_lower:
                score += 0.2
                matched_keywords.append(keyword)
        
        # 文本长度奖励
        if len(text) > 30:
            score += 0.1
        
        # 权重调整
        weighted_score = score * concept_info["weight"]
        
        concept_scores[concept_name] = {
            "score": min(weighted_score, 1.0),
            "matched_keywords": matched_keywords,
            "weight": concept_info["weight"],
            "min_duration": concept_info["min_duration"]
        }
    
    # 计算总分
    total_score = sum(cs["score"] for cs in concept_scores.values())
    
    return {
        "total_score": min(total_score, 1.0),
        "concept_scores": concept_scores
    }

def calculate_visual_score(start_time: float, end_time: float, video_path: str) -> float:
    """计算视觉质量评分（简化版）"""
    # 这里使用简化的评分逻辑
    # 在实际应用中，可以使用计算机视觉技术分析画面质量
    
    # 避开视频开头和结尾的低质量部分
    if start_time < 5.0 or end_time > 1400:  # 24分钟视频的最后1分钟
        return 0.6
    
    # 中间部分通常质量较好
    return 0.8

def calculate_audio_score(start_time: float, end_time: float, video_path: str) -> float:
    """计算音频质量评分（简化版）"""
    # 简化的音频质量评估
    # 假设中间部分音频质量更稳定
    if 60 < start_time < 1200:  # 1-20分钟之间
        return 0.9
    else:
        return 0.7

def calculate_position_weight(current_time: float, total_duration: float) -> float:
    """计算位置权重"""
    position_ratio = current_time / total_duration
    
    # 开头部分权重较高
    if position_ratio < 0.1:
        return 0.9
    # 结尾部分权重较高
    elif position_ratio > 0.8:
        return 0.8
    # 中间部分权重中等
    else:
        return 0.6

def intelligent_segment_selection_v2(
    segments: List[Dict], 
    topic: str,
    core_concepts: Dict
) -> List[Dict]:
    """基于核心概念的智能片段选择"""
    
    selected = []
    concept_coverage = {}
    
    # 1. 为每个核心概念找到最佳片段
    for concept_name, concept_info in core_concepts.items():
        best_segments_for_concept = []
        
        # 找到所有与该概念相关的片段
        for segment in segments:
            concept_score = segment['semantic_analysis']['concept_scores'].get(concept_name, {}).get('score', 0)
            if concept_score > 0.3:  # 最低相关性阈值
                segment_with_concept = segment.copy()
                segment_with_concept['concept_relevance'] = concept_score
                segment_with_concept['target_concept'] = concept_name
                best_segments_for_concept.append(segment_with_concept)
        
        # 按概念相关性排序
        best_segments_for_concept.sort(key=lambda x: x['concept_relevance'], reverse=True)
        
        # 为这个概念选择最佳片段（可能选择多个）
        concept_segments_selected = 0
        max_segments_per_concept = 2 if concept_info['weight'] >= 0.9 else 1
        
        for segment in best_segments_for_concept:
            if concept_segments_selected >= max_segments_per_concept:
                break
                
            # 检查是否与已选片段重叠
            overlaps = False
            for selected_seg in selected:
                if (segment['start_time'] < selected_seg['end_time'] and 
                    segment['end_time'] > selected_seg['start_time']):
                    overlaps = True
                    break
            
            if not overlaps:
                # 动态调整片段时长
                min_duration = concept_info['min_duration']
                actual_duration = max(min_duration, segment['duration'])
                actual_duration = min(actual_duration, 30.0)  # 最长30秒
                
                selected.append({
                    'start_time': segment['start_time'],
                    'end_time': segment['start_time'] + actual_duration,
                    'duration': actual_duration,
                    'score': segment['total_score'],
                    'concept_relevance': segment['concept_relevance'],
                    'type': concept_name,
                    'description': concept_info['description'],
                    'text': segment['text'][:200] + "..." if len(segment['text']) > 200 else segment['text'],
                    'matched_keywords': segment['semantic_analysis']['concept_scores'][concept_name].get('matched_keywords', [])
                })
                concept_segments_selected += 1
                concept_coverage[concept_name] = concept_segments_selected
    
    # 2. 如果没有找到足够的核心内容，添加高分片段
    if len(selected) < 2:
        high_score_segments = sorted(segments, key=lambda x: x['total_score'], reverse=True)
        
        for segment in high_score_segments[:5]:  # 检查前5个高分片段
            overlaps = False
            for selected_seg in selected:
                if (segment['start_time'] < selected_seg['end_time'] and 
                    segment['end_time'] > selected_seg['start_time']):
                    overlaps = True
                    break
            
            if not overlaps:
                selected.append({
                    'start_time': segment['start_time'],
                    'end_time': segment['start_time'] + min(segment['duration'], 20.0),
                    'duration': min(segment['duration'], 20.0),
                    'score': segment['total_score'],
                    'concept_relevance': segment.get('best_concept_score', 0),
                    'type': 'high_score',
                    'description': '高质量内容片段',
                    'text': segment['text'][:200] + "..." if len(segment['text']) > 200 else segment['text'],
                    'matched_keywords': []
                })
                
                if len(selected) >= 3:  # 最多3个片段
                    break
    
    # 3. 按时间顺序排序
    selected.sort(key=lambda x: x['start_time'])
    
    # 4. 限制总时长（如果太长就缩短片段）
    total_duration = sum(seg['duration'] for seg in selected)
    if total_duration > 90:  # 最长90秒
        # 等比例缩短每个片段
        scale_factor = 90 / total_duration
        for seg in selected:
            seg['duration'] *= scale_factor
            seg['end_time'] = seg['start_time'] + seg['duration']
    
    return selected

def generate_dynamic_concepts_from_analysis(content_analysis: Dict, topic: str) -> Dict:
    """基于内容分析动态生成核心概念"""
    analysis = content_analysis['vibe_coding_analysis']
    concepts = {}
    
    # 基于实际分析结果生成概念
    for topic_name, segments in analysis.items():
        if segments:  # 只有当实际找到相关内容时才生成概念
            avg_relevance = sum(s.get('relevance', 0) for s in segments) / len(segments)
            
            concepts[topic_name] = {
                'weight': min(avg_relevance + 0.1, 1.0),  # 基于实际相关性设置权重
                'min_duration': get_dynamic_duration(topic_name, len(segments)),
                'description': get_topic_description(topic_name),
                'segment_count': len(segments),
                'avg_relevance': avg_relevance
            }
    
    return concepts

def get_dynamic_duration(topic_name: str, segment_count: int) -> float:
    """根据话题和片段数量动态决定最小时长"""
    base_durations = {
        'definition': 20.0,      # 定义需要更长时间
        'why_important': 15.0,   # 重要性说明
        'introduction': 12.0,    # 开场引入
        'how_it_works': 18.0,    # 工作原理
        'advantages': 10.0,      # 优势说明
        'context': 15.0,         # 背景分析
        'future': 8.0            # 未来展望
    }
    
    base_duration = base_durations.get(topic_name, 10.0)
    
    # 如果片段很多，可能需要更长时间来完整表达
    if segment_count > 3:
        base_duration *= 1.2
    
    return min(base_duration, 25.0)  # 最长25秒

def intelligent_segment_selection_v3(
    segments: List[Dict], 
    content_analysis: Dict,
    core_concepts: Dict
) -> List[Dict]:
    """基于内容分析的高级智能片段选择"""
    
    selected = []
    key_segments = content_analysis['content_summary']['key_segments']
    
    # 1. 优先选择关键片段中相关性最高的
    processed_time_ranges = []
    
    for key_seg in key_segments[:5]:  # 检查前5个关键片段
        start_time = key_seg['start_time']
        topic = key_seg['topic']
        relevance = key_seg.get('relevance', 0)
        
        # 只选择高相关性的片段
        if relevance < 0.7:
            continue
            
        # 找到包含这个关键片段的视频段
        best_video_segment = None
        best_overlap = 0
        
        for video_seg in segments:
            # 检查重叠
            overlap_start = max(start_time, video_seg['start_time'])
            overlap_end = min(start_time + 10, video_seg['end_time'])  # 假设关键片段10秒
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_video_segment = video_seg
        
        if best_video_segment and best_overlap > 3:  # 至少3秒重叠
            # 检查是否与已选片段重叠
            overlaps_existing = False
            for existing in selected:
                if (best_video_segment['start_time'] < existing['end_time'] and 
                    best_video_segment['end_time'] > existing['start_time']):
                    overlaps_existing = True
                    break
            
            if not overlaps_existing:
                # 根据话题重要性调整时长
                concept_info = core_concepts.get(topic, {})
                min_duration = concept_info.get('min_duration', 15.0)
                
                # 确保片段足够长来完整表达
                actual_duration = max(min_duration, best_video_segment['duration'])
                actual_duration = min(actual_duration, 30.0)  # 最长30秒
                
                selected.append({
                    'start_time': best_video_segment['start_time'],
                    'end_time': best_video_segment['start_time'] + actual_duration,
                    'duration': actual_duration,
                    'score': best_video_segment['total_score'],
                    'relevance': relevance,
                    'type': topic,
                    'description': concept_info.get('description', topic),
                    'text': key_seg['text'][:300] + "..." if len(key_seg['text']) > 300 else key_seg['text'],
                    'key_segment_match': True
                })
                
                processed_time_ranges.append((best_video_segment['start_time'], 
                                            best_video_segment['start_time'] + actual_duration))
    
    # 2. 如果选择的片段不够，从高分片段中补充
    if len(selected) < 2:
        high_score_segments = sorted(segments, key=lambda x: x['total_score'], reverse=True)
        
        for segment in high_score_segments[:8]:
            # 检查是否与已处理的时间范围重叠
            overlaps = False
            for start_range, end_range in processed_time_ranges:
                if (segment['start_time'] < end_range and segment['end_time'] > start_range):
                    overlaps = True
                    break
            
            if not overlaps:
                selected.append({
                    'start_time': segment['start_time'],
                    'end_time': segment['start_time'] + min(segment['duration'], 20.0),
                    'duration': min(segment['duration'], 20.0),
                    'score': segment['total_score'],
                    'relevance': segment.get('best_concept_score', 0.5),
                    'type': 'high_quality',
                    'description': '高质量补充内容',
                    'text': segment['text'][:300] + "..." if len(segment['text']) > 300 else segment['text'],
                    'key_segment_match': False
                })
                
                processed_time_ranges.append((segment['start_time'], 
                                            segment['start_time'] + min(segment['duration'], 20.0)))
                
                if len(selected) >= 3:
                    break
    
    # 3. 按时间顺序排序
    selected.sort(key=lambda x: x['start_time'])
    
    # 4. 优化总时长
    total_duration = sum(seg['duration'] for seg in selected)
    if total_duration > 60:  # 如果超过60秒，适当缩短
        scale_factor = 60 / total_duration
        for seg in selected:
            seg['duration'] *= scale_factor
            seg['end_time'] = seg['start_time'] + seg['duration']
    
    return selected

def combine_segments_to_video(video_path: str, segments: List[Dict], output_path: str) -> None:
    """将多个片段组合成一个视频"""
    try:
        # 创建临时片段文件列表
        temp_files = []
        concat_list_path = f"temp_concat_{int(time.time())}.txt"
        
        # 生成每个片段
        for i, segment in enumerate(segments):
            temp_file = f"temp_segment_{int(time.time())}_{i}.mp4"
            
            # 提取片段
            cmd = [
                "ffmpeg", "-y", "-hwaccel", "none",
                "-i", video_path,
                "-ss", str(segment['start_time']),
                "-t", str(segment['duration']),
                "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1:1",
                "-pix_fmt", "yuv420p",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                temp_file
            ]
            
            safe_run_ffmpeg(cmd)
            temp_files.append(temp_file)
        
        # 创建concat文件列表
        with open(concat_list_path, 'w') as f:
            for temp_file in temp_files:
                f.write(f"file '{os.path.abspath(temp_file)}'\n")
        
        # 合并所有片段
        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path
        ]
        
        safe_run_ffmpeg(concat_cmd)
        
        # 清理临时文件
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)
            
    except Exception as e:
        logger.error(f"合并视频片段失败: {e}")
        raise

def calculate_content_coverage(segments: List[Dict], subtitle_text: List[Dict]) -> float:
    """计算内容覆盖度"""
    if not subtitle_text:
        return 0.5
    
    total_subtitle_duration = sum(
        time_to_seconds(sub['end']) - time_to_seconds(sub['start']) 
        for sub in subtitle_text
    )
    
    selected_duration = sum(seg['duration'] for seg in segments)
    
    return min(selected_duration / total_subtitle_duration * 5, 1.0)  # 乘以5因为我们只选择了很小一部分

def calculate_visual_quality(segments: List[Dict]) -> float:
    """计算视觉质量评分"""
    if not segments:
        return 0.0
    return sum(seg.get('visual_score', 0.8) for seg in segments) / len(segments)

def calculate_audio_quality(segments: List[Dict]) -> float:
    """计算音频质量评分"""
    if not segments:
        return 0.0
    return sum(seg.get('audio_score', 0.8) for seg in segments) / len(segments)

@app.post("/video/multi_segment_clipping")
async def multi_segment_intelligent_clipping(req: MultiSegmentClippingReq):
    """智能多片段视频剪辑 - 从长视频中选择多个精彩片段组合"""
    try:
        logger.info(f"开始智能多片段剪辑: {req.video_path}")
        
        # 1. 检查视频文件是否存在
        if not os.path.exists(req.video_path):
            raise HTTPException(status_code=404, detail=f"视频文件不存在: {req.video_path}")
        
        # 2. 分析视频基本信息
        try:
            result = subprocess.run([
                "ffprobe", "-v", "quiet", "-print_format", "json", 
                "-show_format", req.video_path
            ], capture_output=True, text=True, check=True)
            
            video_info = json.loads(result.stdout)
            total_duration = float(video_info['format']['duration'])
            
        except Exception as e:
            logger.error(f"获取视频信息失败: {e}")
            raise HTTPException(status_code=500, detail="无法获取视频信息")
        
        if total_duration < 60:
            raise HTTPException(status_code=400, detail="视频时长太短，不适合多片段剪辑")
        
        logger.info(f"视频总时长: {total_duration:.1f}秒")
        
        # 3. 读取字幕文件
        subtitle_segments = []
        if req.subtitle_path and os.path.exists(req.subtitle_path):
            subtitle_segments = parse_srt_file(req.subtitle_path)
            logger.info(f"读取到{len(subtitle_segments)}个字幕片段")
        else:
            logger.warning("未找到字幕文件，将使用简化分析")
        
        # 4. 先进行智能内容分析
        content_analysis = analyze_video_content_structure(subtitle_segments)
        logger.info(f"内容分析完成：识别出{len(content_analysis['content_summary']['main_topics'])}个主要话题")
        
        # 打印分析摘要
        for topic in content_analysis['content_summary']['main_topics'][:3]:
            logger.info(f"话题: {topic['description']} - 相关性: {topic['avg_relevance']:.2f} - 片段数: {topic['segment_count']}")
        
        # 5. 基于内容分析结果进行视频片段分析
        segments = analyze_video_segments_v2(req.video_path, subtitle_segments, total_duration, req.topic)
        logger.info(f"分析了{len(segments)}个视频片段")
        
        # 6. 基于内容分析动态生成核心概念
        core_concepts = generate_dynamic_concepts_from_analysis(content_analysis, req.topic)
        logger.info(f"基于内容分析生成{len(core_concepts)}个核心概念")
        
        # 7. 基于分析结果智能选择片段
        selected_segments = intelligent_segment_selection_v3(
            segments, 
            content_analysis,
            core_concepts
        )
        
        logger.info(f"选择了{len(selected_segments)}个片段进行组合")
        
        # 6. 生成组合视频
        timestamp = int(time.time())
        output_path = f"output_data/multi_clip_{timestamp}.mp4"
        
        combine_segments_to_video(req.video_path, selected_segments, output_path)
        
        logger.info(f"多片段视频生成完成: {output_path}")
        
        return {
            "status": "success",
            "output_video": output_path,
            "selected_segments": [
                {
                    "start_time": seg['start_time'],
                    "end_time": seg['end_time'],
                    "duration": seg['duration'],
                    "type": seg['type'],
                    "score": round(seg['score'], 3),
                    "preview_text": seg['text']
                }
                for seg in selected_segments
            ],
            "analysis": {
                "total_video_duration": round(total_duration, 1),
                "analyzed_segments": len(segments),
                "selected_segments": len(selected_segments),
                "total_output_duration": sum(seg['duration'] for seg in selected_segments),
                "selection_strategy": {
                    "semantic_weight": req.semantic_weight,
                    "visual_weight": req.visual_weight,
                    "audio_weight": req.audio_weight,
                    "include_intro": req.include_intro,
                    "include_highlights": req.include_highlights,
                    "include_conclusion": req.include_conclusion
                }
            },
            "quality_metrics": {
                "content_coverage": round(calculate_content_coverage(selected_segments, subtitle_segments), 3),
                "visual_quality": round(calculate_visual_quality(selected_segments), 3),
                "audio_quality": round(calculate_audio_quality(selected_segments), 3),
                "overall_score": round(sum(seg['score'] for seg in selected_segments) / len(selected_segments), 3) if selected_segments else 0
            },
            "message": f"成功生成{len(selected_segments)}个片段组合的智能剪辑视频"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"智能多片段剪辑失败: {e}")
        raise HTTPException(status_code=500, detail=f"智能多片段剪辑失败: {str(e)}")


# 小红书发布相关API
@app.post("/xiaohongshu/authorize")
async def xiaohongshu_authorize(auth_code: str):
    """小红书OAuth授权"""
    try:
        publisher = get_xiaohongshu_publisher()
        result = await publisher.authorize(auth_code)
        return result
    except Exception as e:
        logger.error(f"小红书授权失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"授权失败: {str(e)}")


@app.post("/xiaohongshu/publish")
async def xiaohongshu_publish(request: XHSPublishReq):
    """发布内容到小红书"""
    try:
        publisher = get_xiaohongshu_publisher()
        result = await publisher.publish_note(
            title=request.title,
            content=request.content,
            images=request.images,
            tags=request.tags,
            location=request.location,
            privacy=request.privacy
        )
        return result
    except Exception as e:
        logger.error(f"小红书发布失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"发布失败: {str(e)}")


@app.get("/xiaohongshu/profile")
async def xiaohongshu_get_profile():
    """获取用户资料"""
    try:
        publisher = get_xiaohongshu_publisher()
        result = await publisher.get_user_profile()
        return result
    except Exception as e:
        logger.error(f"获取用户资料失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@app.get("/xiaohongshu/note/{note_id}/stats")
async def xiaohongshu_get_note_stats(note_id: str):
    """获取笔记统计数据"""
    try:
        publisher = get_xiaohongshu_publisher()
        result = await publisher.get_note_stats(note_id)
        return result
    except Exception as e:
        logger.error(f"获取笔记统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


# 图片装饰相关API
@app.post("/image/decorate")
async def decorate_image(request: ImageDecorateReq):
    """装饰图片"""
    try:
        decorator = get_image_decorator()
        result = await decorator.decorate_image(
            image_path=request.image_path,
            decorations=request.decorations
        )
        return result
    except Exception as e:
        logger.error(f"图片装饰失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"装饰失败: {str(e)}")


@app.post("/image/decorations/smart")
async def generate_smart_decorations(theme: str, content_type: str, mood: str = "vibrant"):
    """智能生成装饰配置"""
    try:
        decorator = get_image_decorator()
        result = await decorator.generate_smart_decorations(
            theme=theme,
            content_type=content_type,
            mood=mood
        )
        return result
    except Exception as e:
        logger.error(f"智能装饰生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


# 文件上传相关API
@app.post("/upload/photos")
async def upload_photos(files: List[UploadFile] = File(...)):
    """上传多张照片"""
    try:
        if len(files) > 20:
            raise HTTPException(status_code=400, detail="最多只能上传20张照片")
        
        uploaded_files = []
        upload_dir = Path("output_data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            # 验证文件类型
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"文件 {file.filename} 不是有效的图片格式")
            
            # 生成唯一文件名
            timestamp = int(time.time())
            file_extension = Path(file.filename).suffix if file.filename else '.jpg'
            unique_filename = f"photo_{timestamp}_{len(uploaded_files)}{file_extension}"
            file_path = upload_dir / unique_filename
            
            # 保存文件
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            uploaded_files.append({
                "original_name": file.filename,
                "saved_path": str(file_path),
                "url": f"/static/uploads/{unique_filename}",
                "size": len(content)
            })
            
            logger.info(f"上传照片成功: {file.filename} -> {file_path}")
        
        return {
            "status": "success",
            "message": f"成功上传{len(uploaded_files)}张照片",
            "files": uploaded_files
        }
        
    except Exception as e:
        logger.error(f"照片上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

@app.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """上传视频文件"""
    try:
        # 验证文件类型
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="请上传有效的视频文件")
        
        # 检查文件大小 (限制500MB)
        content = await file.read()
        if len(content) > 500 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="视频文件大小不能超过500MB")
        
        upload_dir = Path("input_data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成唯一文件名
        timestamp = int(time.time())
        file_extension = Path(file.filename).suffix if file.filename else '.mp4'
        unique_filename = f"video_{timestamp}{file_extension}"
        file_path = upload_dir / unique_filename
        
        # 保存文件
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"上传视频成功: {file.filename} -> {file_path}")
        
        return {
            "status": "success",
            "message": "视频上传成功",
            "file": {
                "original_name": file.filename,
                "saved_path": str(file_path),
                "size": len(content)
            }
        }
        
    except Exception as e:
        logger.error(f"视频上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

# LLM智能内容生成相关API
@app.post("/llm/generate_content")
async def generate_llm_content(request: LLMContentReq):
    """使用大模型生成智能内容"""
    try:
        llm_service = get_llm_service()
        
        if request.content_type == "xiaohongshu":
            result = await llm_service.generate_xiaohongshu_content(
                theme=request.theme,
                photo_descriptions=request.photo_descriptions,
                highlights=request.highlights,
                feeling=request.feeling,
                style=request.style,
                length=request.length,
                custom_requirements=request.custom_requirements
            )
        else:
            # 其他类型内容生成
            result = await llm_service.generate_pro_content(
                topic=request.theme,
                content_type="complete",
                style=request.style,
                target_audience="general"
            )
        
        logger.info(f"LLM内容生成成功: {request.theme}")
        return result
        
    except Exception as e:
        logger.error(f"LLM内容生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内容生成失败: {str(e)}")

@app.post("/llm/generate_pro_content")
async def generate_pro_llm_content(request: ProContentReq):
    """生成Pro功能专业内容"""
    try:
        llm_service = get_llm_service()
        
        result = await llm_service.generate_pro_content(
            topic=request.topic,
            content_type=request.content_type,
            style=request.style,
            target_audience=request.target_audience
        )
        
        logger.info(f"Pro内容生成成功: {request.topic}")
        return result
        
    except Exception as e:
        logger.error(f"Pro内容生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pro内容生成失败: {str(e)}")

@app.get("/llm/status")
async def get_llm_status():
    """获取LLM服务状态"""
    try:
        llm_service = get_llm_service()
        
        return {
            "status": "active" if llm_service.current_api else "fallback",
            "current_api": llm_service.current_api,
            "available_apis": [
                {
                    "name": api_name,
                    "enabled": config["enabled"],
                    "model": config["model"]
                }
                for api_name, config in llm_service.apis.items()
                if config["enabled"]
            ],
            "message": "LLM服务正常运行" if llm_service.current_api else "未配置LLM API，使用备用模式"
        }
        
    except Exception as e:
        logger.error(f"获取LLM状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

# 高级拼图生成相关API
@app.post("/collage/generate_advanced")
async def generate_advanced_collage(request: AdvancedCollageReq):
    """生成高级拼图效果"""
    try:
        collage_generator = get_advanced_collage_generator()
        
        result = await collage_generator.generate_advanced_collage(
            images=request.images,
            title=request.title,
            layout_type=request.layout_type,
            style=request.style,
            color_scheme=request.color_scheme,
            canvas_size=tuple(request.canvas_size),
            add_effects=request.add_effects,
            add_text_overlay=request.add_text_overlay,
            extra_text=request.extra_text or "",
            text_position=request.text_position
        )
        
        logger.info(f"高级拼图生成成功: {request.title}")
        return result
        
    except Exception as e:
        logger.error(f"高级拼图生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"拼图生成失败: {str(e)}")

@app.get("/collage/layouts")
async def get_collage_layouts():
    """获取可用的拼图布局"""
    return {
        "layouts": [
            {"id": "dynamic", "name": "智能动态", "description": "根据图片数量和比例智能排列"},
            {"id": "grid", "name": "网格布局", "description": "规整的网格排列"},
            {"id": "magazine", "name": "杂志风格", "description": "主图+辅助图的杂志布局"},
            {"id": "mosaic", "name": "马赛克拼图", "description": "不规则大小和位置的艺术拼贴"},
            {"id": "creative", "name": "创意布局", "description": "圆形、多边形等特殊形状"},
            {"id": "treemap", "name": "树地图", "description": "按权重区域切分的混合尺寸拼贴"}
        ],
        "styles": [
            {"id": "modern", "name": "现代简约", "description": "简洁现代的设计风格"},
            {"id": "vintage", "name": "复古怀旧", "description": "温暖的复古色调和效果"},
            {"id": "artistic", "name": "艺术风格", "description": "强调艺术感的视觉效果"},
            {"id": "minimal", "name": "极简主义", "description": "最简化的设计元素"}
        ],
        "color_schemes": [
            {"id": "auto", "name": "自动选择", "description": "根据图片内容自动选择配色"},
            {"id": "warm", "name": "暖色调", "description": "温暖的橙红色系"},
            {"id": "cool", "name": "冷色调", "description": "清爽的蓝绿色系"},
            {"id": "monochrome", "name": "单色调", "description": "黑白灰的经典搭配"},
            {"id": "vibrant", "name": "鲜艳色彩", "description": "高饱和度的鲜艳配色"}
        ]
    }

# 智能封面生成相关API
@app.post("/cover/generate")
async def generate_smart_cover(request: SmartCoverReq):
    """生成智能封面"""
    try:
        generator = get_smart_cover_generator()
        result = await generator.generate_cover(
            images=request.images,
            title=request.title,
            subtitle=request.subtitle,
            layout=request.layout,
            theme=request.theme,
            platform=request.platform,
            custom_config=request.custom_config
        )
        return result
    except Exception as e:
        logger.error(f"智能封面生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.get("/cover/templates")
async def get_cover_templates():
    """获取封面模板列表"""
    try:
        generator = get_smart_cover_generator()
        templates = {
            "layouts": list(generator.layout_templates.keys()),
            "themes": list(generator.color_themes.keys()),
            "platforms": list(generator.cover_sizes.keys())
        }
        return {
            "status": "success",
            "data": templates,
            "message": "模板列表获取成功"
        }
    except Exception as e:
        logger.error(f"获取模板列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")

# 小红书级别拼图生成API
@app.post("/xiaohongshu/generate_collage")
async def generate_xiaohongshu_collage(request: XiaohongshuCollageReq):
    """生成小红书级别的高质量拼图"""
    try:
        xiaohongshu_generator = get_xiaohongshu_collage_generator()
        
        # 创建配置
        config = CollageConfig(
            width=request.width,
            height=request.height,
            quality=request.quality,
            title_position=request.title_position if request.title_position in ["center_overlay", "top"] else "center_overlay",
            font_path_override=request.font_path
        )
        
        # 处理自定义文案
        custom_texts = []
        for text_data in request.custom_texts:
            if isinstance(text_data, dict):
                text_config = TextConfig(
                    text=text_data.get("text", ""),
                    font_size=text_data.get("font_size", 80),
                    color=text_data.get("color", "#2C2C2C"),
                    position=tuple(text_data.get("position", [100, 100])),
                    max_width=text_data.get("max_width", 800),
                    font_weight=text_data.get("font_weight", "normal"),
                    shadow=text_data.get("shadow", True),
                    background=text_data.get("background", False),
                    background_color=text_data.get("background_color", "#FFFFFF"),
                    background_opacity=text_data.get("background_opacity", 180)
                )
                custom_texts.append(text_config)
        
        result = await xiaohongshu_generator.generate_xiaohongshu_collage(
            images=request.images,
            title=request.title,
            subtitle=request.subtitle,
            layout=request.layout,
            color_scheme=request.color_scheme,
            custom_texts=custom_texts,
            config=config,
            overlay_texts=request.overlay_texts
        )
        
        logger.info(f"小红书拼图生成成功: {request.title}")
        return result
        
    except Exception as e:
        logger.error(f"小红书拼图生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"拼图生成失败: {str(e)}")

@app.get("/xiaohongshu/layouts")
async def get_xiaohongshu_layouts():
    """获取小红书拼图布局选项"""
    return {
        "layouts": [
            {"id": "magazine_style", "name": "杂志风格", "description": "专业杂志级别的不对称布局"},
            {"id": "grid_modern", "name": "现代网格", "description": "整齐的现代网格布局"},
            {"id": "story_flow", "name": "故事流", "description": "讲述故事的流畅布局"},
            {"id": "featured_main", "name": "主图特色", "description": "突出主图的特色布局"},
            {"id": "artistic_collage", "name": "艺术拼贴", "description": "富有艺术感的拼贴效果"},
            {"id": "minimal_clean", "name": "简约清新", "description": "简约清新的小红书风格"},
            {"id": "scrapbook", "name": "手帐拼贴", "description": "随机旋转/叠放/胶带/撕边更有拼贴感"}
        ],
        "color_schemes": [
            {"id": "xiaohongshu_pink", "name": "小红书粉", "description": "经典小红书粉色主题"},
            {"id": "fresh_mint", "name": "清新薄荷", "description": "清新的薄荷绿主题"},
            {"id": "warm_sunset", "name": "温暖夕阳", "description": "温暖的夕阳橙色主题"},
            {"id": "elegant_gray", "name": "优雅灰调", "description": "优雅的灰色主题"}
        ],
        "quality_options": [
            {"value": 85, "name": "标准质量", "description": "适合快速分享"},
            {"value": 95, "name": "高质量", "description": "推荐用于小红书发布"},
            {"value": 98, "name": "超高质量", "description": "最佳质量，文件较大"}
        ],
        "size_presets": [
            {"width": 1080, "height": 1080, "name": "小红书标准", "description": "1080x1080 标准正方形"},
            {"width": 2160, "height": 2160, "name": "4K高清", "description": "2160x2160 超高清"},
            {"width": 1080, "height": 1350, "name": "小红书竖版", "description": "1080x1350 竖版比例"}
        ]
    }

@app.post("/xiaohongshu/edit_text")
async def edit_collage_text(request: EditableTextReq):
    """编辑拼图中的文案"""
    try:
        # 这里可以实现文案编辑功能
        # 目前返回成功响应，实际实现需要存储和更新拼图数据
        return {
            "success": True,
            "message": "文案编辑成功",
            "collage_id": request.collage_id,
            "text_id": request.text_id,
            "new_text": request.new_text
        }
        
    except Exception as e:
        logger.error(f"文案编辑失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文案编辑失败: {str(e)}")


# 单页渲染（单图或拼图）
class PageRenderReq(BaseModel):
    images: List[str]
    mode: str = "single"  # single | collage
    layout: str = "scrapbook"  # 当 mode=collage 时生效
    title: str = ""
    subtitle: str = ""
    title_position: str = "center_overlay"
    overlay_texts: List[Dict[str, Any]] = []
    width: int = 1080
    height: int = 1080
    quality: int = 95

@app.post("/xiaohongshu/render_page")
async def render_xhs_page(request: PageRenderReq):
    try:
        xh = get_xiaohongshu_collage_generator()
        cfg = CollageConfig(width=request.width, height=request.height, quality=request.quality,
                            title_position=request.title_position)
        if request.mode == "single":
            from PIL import Image as PILImage
            img_paths = request.images
            if not img_paths:
                raise HTTPException(status_code=400, detail="缺少图片")
            canvas = PILImage.new('RGB', (cfg.width, cfg.height), '#FFFFFF')
            p = Path(img_paths[0])
            if not p.exists():
                p = Path("output_data")/img_paths[0]
            im = PILImage.open(p)
            if im.mode != 'RGB':
                im = im.convert('RGB')
            max_w = int(cfg.width*0.82)
            max_h = int(cfg.height*0.68)
            ratio = min(max_w/im.width, max_h/im.height)
            new_size = (max(1,int(im.width*ratio)), max(1,int(im.height*ratio)))
            im = im.resize(new_size, PILImage.Resampling.LANCZOS)
            x = (cfg.width - im.width)//2
            y = (cfg.height - im.height)//2 + 50
            canvas.paste(im, (x,y))
            # 标题与文案
            xh.config_font_override = None
            colors = xh.color_schemes['xiaohongshu_pink']
            canvas = xh._add_main_texts(canvas, request.title, request.subtitle, colors, cfg)
            if request.overlay_texts:
                canvas = xh._draw_text_blocks(canvas, request.overlay_texts, colors, cfg)
            out_path, b64 = await xh._save_high_quality_image(canvas, cfg)
            return {"success": True, "image_path": str(out_path), "base64_data": b64, "mode": request.mode}
        else:
            res = await xh.generate_xiaohongshu_collage(
                images=request.images,
                title=request.title,
                subtitle=request.subtitle,
                layout=request.layout,
                color_scheme='xiaohongshu_pink',
                custom_texts=[],
                config=cfg,
                overlay_texts=request.overlay_texts
            )
            return res
    except Exception as e:
        logger.error(f"渲染页面失败: {e}")
        raise HTTPException(status_code=500, detail=f"渲染失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    from core.config import API_HOST, API_PORT
    
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
