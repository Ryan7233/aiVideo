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
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

from core import job_store
from core.concurrency import MAX_CONCURRENT_MEDIA_JOBS

from core.env import env_str

logger = logging.getLogger(__name__)

JobHandler = Callable[[Dict[str, Any], str], Dict[str, Any]]

class JobDispatchError(RuntimeError):
    """Raised when a job could not be handed to its backend.

    Carries the job id so the caller can report it: the record exists and is
    marked failed, which is what an operator needs to see a broker outage.
    """

    def __init__(self, message: str, job_id: str):
        super().__init__(message)
        self.job_id = job_id


_handlers: Dict[str, JobHandler] = {}
_executor: Optional[ThreadPoolExecutor] = None


def backend_name() -> str:
    """Which backend submissions go to. Read per call so tests can flip it."""
    return env_str("JOB_BACKEND", "thread").lower()


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
    """Execute one job and record the outcome. Never raises.

    Claiming the job and checking it was not cancelled are the same atomic
    write; a read-then-write here could start work a cancel had already
    landed on.
    """
    if not job_store.mark_running(job_id):
        current = (job_store.get_job(job_id) or {}).get("status", "unknown")
        logger.info("Job %s not started; it is already %s", job_id, current)
        return {"status": current}

    try:
        handler = get_handler(kind)
        result = handler(params, job_id)
    except Exception as exc:
        logger.exception("Job %s (%s) failed", job_id, kind)
        if not job_store.mark_failed(
            job_id, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"
        ):
            # A cancel landed while the work was in flight; it wins.
            return {"status": job_store.CANCELLED}
        return {"status": "failed", "error": str(exc)}

    if not job_store.mark_succeeded(job_id, result):
        logger.info("Job %s finished but had been cancelled; result discarded", job_id)
        return {"status": job_store.CANCELLED}
    return {"status": "succeeded", "result": result}


def submit(kind: str, params: Dict[str, Any]) -> str:
    """Create a job record and hand the work to the configured backend."""
    get_handler(kind)  # fail fast on an unknown kind, before creating a record
    backend = backend_name()
    job_id = job_store.create_job(kind, params, backend=backend)

    try:
        if backend == "celery":
            from worker.tasks import run_registered_job

            async_result = run_registered_job.delay(kind, params, job_id)
            # Record the dispatch id only. The worker owns the status: a task
            # that failed or finished before this line must not be pulled back
            # to "running", which would leave the poller waiting forever.
            job_store.set_external_id(job_id, async_result.id)
            logger.info("Dispatched job %s to Celery task %s", job_id, async_result.id)
        else:
            _get_executor().submit(run_job, kind, params, job_id)
            logger.info("Queued job %s on the local thread pool", job_id)
    except Exception as exc:
        # Dispatch failed -- an unreachable broker, a shut-down pool. Nothing
        # will ever pick this row up, so close it out here rather than leaving
        # a pending record that accumulates every time the broker flaps.
        logger.exception("Failed to dispatch job %s (%s)", job_id, kind)
        job_store.mark_failed(job_id, f"任务投递失败（{backend}）: {type(exc).__name__}: {exc}")
        raise JobDispatchError(f"任务投递失败: {exc}", job_id=job_id) from exc

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
