"""Background job submission for the long-running media pipelines.

Clipping a video takes minutes. Running it inside the request meant the client
had to hold an HTTP connection open for the whole thing, and a dropped
connection threw the work away. Callers now submit a job and poll it.

Two backends: a local thread pool (the default, so a single-machine install
needs no Redis) and Celery (``JOB_BACKEND=celery``) for a real deployment.
Both write through the same :mod:`core.job_store` table, so the polling API
does not care which one ran the work.
"""

from __future__ import annotations

import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

from core import job_store
from core.concurrency import MAX_CONCURRENT_MEDIA_JOBS

logger = logging.getLogger(__name__)

JobHandler = Callable[[Dict[str, Any], str], Dict[str, Any]]

_handlers: Dict[str, JobHandler] = {}
_executor: Optional[ThreadPoolExecutor] = None


def backend_name() -> str:
    """Which backend submissions go to. Read per call so tests can flip it."""
    return os.getenv("JOB_BACKEND", "thread").strip().lower()


def register_job_handler(kind: str, handler: JobHandler) -> None:
    """Register the callable that runs jobs of ``kind``."""
    _handlers[kind] = handler
    logger.debug("Registered job handler: %s", kind)


def registered_kinds() -> list[str]:
    return sorted(_handlers)


def get_handler(kind: str) -> JobHandler:
    try:
        return _handlers[kind]
    except KeyError:
        raise KeyError(f"未注册的任务类型: {kind}") from None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        # Jobs spend most of their time inside FFmpeg, which has its own gate;
        # this bound just stops unbounded thread growth under a flood.
        _executor = ThreadPoolExecutor(
            max_workers=max(2, MAX_CONCURRENT_MEDIA_JOBS),
            thread_name_prefix="aivideo-job",
        )
    return _executor


def run_job(kind: str, params: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """Execute one job and record the outcome. Never raises."""
    if job_store.is_cancelled(job_id):
        logger.info("Job %s was cancelled before it started", job_id)
        return {"status": "cancelled"}

    job_store.mark_running(job_id)
    try:
        handler = get_handler(kind)
        result = handler(params, job_id)
    except Exception as exc:
        logger.exception("Job %s (%s) failed", job_id, kind)
        job_store.mark_failed(job_id, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}")
        return {"status": "failed", "error": str(exc)}

    # A cancel that landed mid-run wins; do not resurrect the job.
    if job_store.is_cancelled(job_id):
        logger.info("Job %s finished but had been cancelled; result discarded", job_id)
        return {"status": "cancelled"}

    job_store.mark_succeeded(job_id, result)
    return {"status": "succeeded", "result": result}


def submit(kind: str, params: Dict[str, Any]) -> str:
    """Create a job record and hand the work to the configured backend."""
    get_handler(kind)  # fail fast on an unknown kind, before creating a record
    backend = backend_name()
    job_id = job_store.create_job(kind, params, backend=backend)

    if backend == "celery":
        from worker.tasks import run_registered_job

        async_result = run_registered_job.delay(kind, params, job_id)
        job_store.mark_running(job_id, external_id=async_result.id)
        logger.info("Dispatched job %s to Celery task %s", job_id, async_result.id)
    else:
        _get_executor().submit(run_job, kind, params, job_id)
        logger.info("Queued job %s on the local thread pool", job_id)

    return job_id


def shutdown(wait: bool = False) -> None:
    """Release the thread pool (used by tests and clean shutdowns)."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=wait)
        _executor = None


def _multi_segment_handler(params: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    from core.video_workflow import process_multi_segment_video

    job_store.set_progress(job_id, {"step": "clipping"})
    return process_multi_segment_video(params)


register_job_handler("multi_segment_clipping", _multi_segment_handler)
