import os

from dotenv import load_dotenv

from core.env import env_int, env_list, env_str

# Load environment variables
load_dotenv()

# Bind to loopback by default. The container images set API_HOST=0.0.0.0
# explicitly; a developer running start_server.py on a shared network should
# not be publishing an unauthenticated upload and transcoding endpoint.
DEFAULT_API_HOST = "127.0.0.1"

# API Configuration
API_HOST = env_str("API_HOST", DEFAULT_API_HOST)
API_PORT = env_int("API_PORT", 8000)
GEMINI_API_BASE = env_str("GEMINI_API_BASE", "http://localhost:8080")
CUT_API_BASE = env_str("CUT_API_BASE", "http://localhost:8081")

# Video Processing Configuration
MIN_CLIP_DURATION = env_int("MIN_CLIP_DURATION", 25)
MAX_CLIP_DURATION = env_int("MAX_CLIP_DURATION", 60)
VIDEO_FPS = env_int("VIDEO_FPS", 30)
VIDEO_CRF = env_int("VIDEO_CRF", 23)
AUDIO_BITRATE = env_str("AUDIO_BITRATE", "128k")

# Storage Configuration
UPLOAD_BUCKET = env_str("UPLOAD_BUCKET", "clips")
UPLOAD_BASE_URL = env_str("UPLOAD_BASE_URL", "https://storage.example.com")

# Celery/Queue Configuration
CELERY_BROKER_URL = env_str("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0")
CELERY_RESULT_BACKEND = env_str("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/1")

# Security Configuration
ALLOWED_VIDEO_EXTENSIONS = env_list("ALLOWED_VIDEO_EXTENSIONS", ".mp4,.avi,.mov,.mkv")
MAX_FILE_SIZE = env_int("MAX_FILE_SIZE", 500 * 1024 * 1024)  # 500MB in bytes

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
