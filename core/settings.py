from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # External services
    gemini_api_base: str = "http://localhost:8080"
    cut_api_base: str = "http://localhost:8081"

    # Video processing
    min_clip_duration: int = 25
    max_clip_duration: int = 60
    video_fps: int = 30
    video_crf: int = 23
    audio_bitrate: str = "128k"

    # Storage
    upload_bucket: str = "clips"
    upload_base_url: AnyHttpUrl | str = "https://storage.example.com"

    # Celery/Queue
    celery_broker_url: str = "redis://127.0.0.1:6379/0"
    celery_result_backend: str = "redis://127.0.0.1:6379/1"

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"  # 忽略未定义的额外字段
    }


settings = Settings()

