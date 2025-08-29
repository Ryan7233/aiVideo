from celery import Celery
from core.settings import settings

celery_app = Celery(
    "ai_video_clipper",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["worker.tasks"]
)

celery_app.conf.update(
    # Task configuration
    task_track_started=True,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    result_expires=3600,  # 1 hour
    timezone='Asia/Shanghai',
    enable_utc=True,
    
    # Task routing
    task_routes={
        "worker.tasks.*": {"queue": "celery"},
    },
    
    # Task limits
    task_time_limit=60 * 30,  # 30 minutes hard limit
    task_soft_time_limit=60 * 25,  # 25 minutes soft limit
    worker_prefetch_multiplier=1,  # Only take one task at a time
    
    # Retry configuration
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_reject_on_worker_lost=True,
)


