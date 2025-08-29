from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from loguru import logger

try:
    from worker.celery_app import celery_app
    from worker.tasks import (
        process_video_async, asr_transcribe_async, extract_audio_async,
        asr_smart_clipping_async, semantic_analysis_async
    )
    TASKS_AVAILABLE = True
except Exception as e:  # allow API to boot without worker
    logger.warning(f"Celery worker not available: {e}")
    celery_app = None
    process_video_async = None
    asr_transcribe_async = None
    extract_audio_async = None
    asr_smart_clipping_async = None
    semantic_analysis_async = None
    TASKS_AVAILABLE = False


router = APIRouter(prefix="/tasks", tags=["Tasks"]) 


class EnqueueVideoReq(BaseModel):
    url: str
    min_sec: int = 15
    max_sec: int = 25
    want_asr: bool = False
    output: Optional[str] = None
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "file:///Users/yanran/Desktop/123.mp4",
                "min_sec": 15,
                "max_sec": 25,
                "want_asr": False,
                "output": "output_data/my_clip.mp4"
            }
        }
    }

class ASRTranscribeRequest(BaseModel):
    url: str
    language: Optional[str] = None
    subtitle_format: str = "srt"
    task: str = "transcribe"
    model_size: str = "base"
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "file:///path/to/video.mp4",
                "language": "zh",
                "subtitle_format": "srt",
                "task": "transcribe",
                "model_size": "base"
            }
        }
    }

class AudioExtractionRequest(BaseModel):
    url: str
    sample_rate: int = 16000
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "file:///path/to/video.mp4",
                "sample_rate": 16000
            }
        }
    }

class ASRSmartClippingRequest(BaseModel):
    url: str
    min_sec: int = 15
    max_sec: int = 25
    count: int = 1
    model_size: str = "base"
    language: Optional[str] = None
    output_prefix: str = "asr_smart_clip"
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "url": "file:///Users/yanran/Desktop/123.mp4",
                "min_sec": 15,
                "max_sec": 25,
                "count": 1,
                "model_size": "base",
                "language": None,
                "output_prefix": "asr_smart_clip"
            }
        }
    }

class SemanticAnalysisRequest(BaseModel):
    text: str
    include_keywords: bool = True
    include_sentiment: bool = True
    include_topics: bool = True
    include_quality: bool = True
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "This is a sample text for semantic analysis. It contains various topics and emotions.",
                "include_keywords": True,
                "include_sentiment": True,
                "include_topics": True,
                "include_quality": True
            }
        }
    }


@router.post("/enqueue")
async def enqueue_video_processing(req: EnqueueVideoReq) -> Dict[str, Any]:
    """
    Enqueue a video processing task for background processing.
    Returns a task_id that can be used to check status.
    """
    if not celery_app:
        raise HTTPException(status_code=503, detail="Task queue service not available")
    
    try:
        task = process_video_async.delay(
            url=req.url,
            min_sec=req.min_sec,
            max_sec=req.max_sec,
            want_asr=req.want_asr,
            output=req.output or ""
        )
        logger.info(f"Enqueued video processing task: {task.id}")
        return {
            "task_id": task.id,
            "status": "PENDING",
            "message": "Task has been queued for processing"
        }
    except Exception as e:
        logger.error(f"Failed to enqueue task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to enqueue task: {str(e)}")


@router.get("/status/{task_id}")
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status and result of a video processing task.
    """
    if not celery_app:
        raise HTTPException(status_code=503, detail="Task queue service not available")
    
    try:
        result = celery_app.AsyncResult(task_id)
        
        if result.state == 'PENDING':
            response = {
                "task_id": task_id,
                "status": result.state,
                "message": "Task is waiting to be processed"
            }
        elif result.state == 'PROGRESS':
            response = {
                "task_id": task_id,
                "status": result.state,
                "progress": result.info.get('progress', 0),
                "message": result.info.get('status', 'Processing...')
            }
        elif result.state == 'SUCCESS':
            response = {
                "task_id": task_id,
                "status": result.state,
                "result": result.result
            }
        elif result.state == 'FAILURE':
            response = {
                "task_id": task_id,
                "status": result.state,
                "error": str(result.info)
            }
        else:
            response = {
                "task_id": task_id,
                "status": result.state,
                "message": "Unknown task state"
            }
            
        return response
        
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@router.delete("/cancel/{task_id}")
async def cancel_task(task_id: str) -> Dict[str, Any]:
    """
    Cancel a running or pending task.
    """
    if not celery_app:
        raise HTTPException(status_code=503, detail="Task queue service not available")
    
    try:
        celery_app.control.revoke(task_id, terminate=True)
        logger.info(f"Cancelled task: {task_id}")
        return {
            "task_id": task_id,
            "message": "Task cancellation requested"
        }
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")


@router.post("/asr/enqueue")
async def enqueue_asr_transcription(req: ASRTranscribeRequest) -> Dict[str, Any]:
    """
    提交ASR转录任务到后台队列
    """
    if not celery_app or not asr_transcribe_async:
        raise HTTPException(status_code=503, detail="Task queue service unavailable")
    
    try:
        logger.info(f"Enqueuing ASR transcription task for: {req.url}")
        
        # 提交异步任务
        task = asr_transcribe_async.delay(
            url=req.url,
            language=req.language,
            subtitle_format=req.subtitle_format,
            task=req.task,
            model_size=req.model_size
        )
        
        task_id = task.id
        logger.info(f"ASR task enqueued with ID: {task_id}")
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "ASR transcription task has been queued for processing",
            "parameters": {
                "url": req.url,
                "language": req.language,
                "subtitle_format": req.subtitle_format,
                "task": req.task,
                "model_size": req.model_size
            }
        }
    except Exception as e:
        logger.error(f"Error enqueuing ASR task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to enqueue ASR task: {str(e)}")


@router.post("/audio/enqueue")
async def enqueue_audio_extraction(req: AudioExtractionRequest) -> Dict[str, Any]:
    """
    提交音频提取任务到后台队列
    """
    if not celery_app or not extract_audio_async:
        raise HTTPException(status_code=503, detail="Task queue service unavailable")
    
    try:
        logger.info(f"Enqueuing audio extraction task for: {req.url}")
        
        # 提交异步任务
        task = extract_audio_async.delay(
            url=req.url,
            sample_rate=req.sample_rate
        )
        
        task_id = task.id
        logger.info(f"Audio extraction task enqueued with ID: {task_id}")
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Audio extraction task has been queued for processing",
            "parameters": {
                "url": req.url,
                "sample_rate": req.sample_rate
            }
        }
    except Exception as e:
        logger.error(f"Error enqueuing audio extraction task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to enqueue audio extraction task: {str(e)}")


@router.post("/smart_clipping/enqueue")
async def enqueue_asr_smart_clipping(req: ASRSmartClippingRequest) -> Dict[str, Any]:
    """
    提交ASR增强智能切片任务到后台队列
    """
    if not celery_app or not asr_smart_clipping_async:
        raise HTTPException(status_code=503, detail="Task queue service unavailable")
    
    try:
        logger.info(f"Enqueuing ASR smart clipping task for: {req.url}")
        
        # 提交异步任务
        task = asr_smart_clipping_async.delay(
            url=req.url,
            min_sec=req.min_sec,
            max_sec=req.max_sec,
            count=req.count,
            model_size=req.model_size,
            language=req.language,
            output_prefix=req.output_prefix
        )
        
        task_id = task.id
        logger.info(f"ASR smart clipping task enqueued with ID: {task_id}")
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "ASR enhanced smart clipping task has been queued for processing",
            "parameters": {
                "url": req.url,
                "min_sec": req.min_sec,
                "max_sec": req.max_sec,
                "count": req.count,
                "model_size": req.model_size,
                "language": req.language,
                "output_prefix": req.output_prefix
            }
        }
    except Exception as e:
        logger.error(f"Error enqueuing ASR smart clipping task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to enqueue ASR smart clipping task: {str(e)}")


@router.post("/semantic/enqueue")
async def enqueue_semantic_analysis(req: SemanticAnalysisRequest) -> Dict[str, Any]:
    """
    提交语义分析任务到后台队列
    """
    if not celery_app or not semantic_analysis_async:
        raise HTTPException(status_code=503, detail="Task queue service unavailable")
    
    try:
        logger.info(f"Enqueuing semantic analysis task, text length: {len(req.text)}")
        
        # 提交异步任务
        task = semantic_analysis_async.delay(
            text=req.text,
            include_keywords=req.include_keywords,
            include_sentiment=req.include_sentiment,
            include_topics=req.include_topics,
            include_quality=req.include_quality
        )
        
        task_id = task.id
        logger.info(f"Semantic analysis task enqueued with ID: {task_id}")
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Semantic analysis task has been queued for processing",
            "parameters": {
                "text_length": len(req.text),
                "word_count": len(req.text.split()),
                "include_keywords": req.include_keywords,
                "include_sentiment": req.include_sentiment,
                "include_topics": req.include_topics,
                "include_quality": req.include_quality
            }
        }
    except Exception as e:
        logger.error(f"Error enqueuing semantic analysis task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to enqueue semantic analysis task: {str(e)}")


