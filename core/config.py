import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
GEMINI_API_BASE = os.getenv("GEMINI_API_BASE", "http://localhost:8080")
CUT_API_BASE = os.getenv("CUT_API_BASE", "http://localhost:8081")

# Video Processing Configuration
MIN_CLIP_DURATION = int(os.getenv("MIN_CLIP_DURATION", 25))
MAX_CLIP_DURATION = int(os.getenv("MAX_CLIP_DURATION", 60))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", 30))
VIDEO_CRF = int(os.getenv("VIDEO_CRF", 23))
AUDIO_BITRATE = os.getenv("AUDIO_BITRATE", "128k")

# Storage Configuration
UPLOAD_BUCKET = os.getenv("UPLOAD_BUCKET", "clips")
UPLOAD_BASE_URL = os.getenv("UPLOAD_BASE_URL", "https://storage.example.com")

# Security Configuration
ALLOWED_VIDEO_EXTENSIONS = os.getenv("ALLOWED_VIDEO_EXTENSIONS", ".mp4,.avi,.mov,.mkv").split(",")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 500 * 1024 * 1024))  # 500MB in bytes

# AI Prompts
SEGMENT_PROMPT = """你是短视频切片专家。基于带时间戳字幕，给出{min_sec}~{max_sec}秒的切片区间。
输出纯JSON：{{"clips":[{{"start":"mm:ss","end":"mm:ss","reason":"简述"}}]}}
字幕：
<<<
{txt}
>>>"""

CAPTIONS_PROMPT = '''你是短视频文案专家。根据主题与片段内容，生成抖音风格文案：
输出纯JSON：{"title":"...", "hashtags":["#话题1","#话题2"], "desc":"80字内简介"}
主题：{topic}
片段：{clip_text}
整体字幕摘要：{transcript}'''

# Validation schemas
def validate_video_extension(filename: str) -> bool:
    """Validate if file extension is allowed"""
    if not filename:
        return False
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_VIDEO_EXTENSIONS

def validate_file_size(file_size: int) -> bool:
    """Validate if file size is within limits"""
    return file_size <= MAX_FILE_SIZE
