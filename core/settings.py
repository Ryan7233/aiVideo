from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")

    # External services
    gemini_api_base: str = Field("http://localhost:8080", env="GEMINI_API_BASE")
    cut_api_base: str = Field("http://localhost:8081", env="CUT_API_BASE")

    # Video processing
    min_clip_duration: int = Field(25, env="MIN_CLIP_DURATION")
    max_clip_duration: int = Field(60, env="MAX_CLIP_DURATION")
    video_fps: int = Field(30, env="VIDEO_FPS")
    video_crf: int = Field(23, env="VIDEO_CRF")
    audio_bitrate: str = Field("128k", env="AUDIO_BITRATE")

    # Storage
    upload_bucket: str = Field("clips", env="UPLOAD_BUCKET")
    upload_base_url: AnyHttpUrl | str = Field("https://storage.example.com", env="UPLOAD_BASE_URL")

    # Celery/Queue
    celery_broker_url: str = Field("redis://127.0.0.1:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field("redis://127.0.0.1:6379/1", env="CELERY_RESULT_BACKEND")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"  # 忽略未定义的额外字段
    }


settings = Settings()


