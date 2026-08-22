"""Submit-and-poll API for the long-running pipelines.

These sit alongside the original synchronous endpoints rather than replacing
them, so existing clients keep working while new ones move to job polling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core import job_store
from core.jobs import backend_name, registered_kinds, submit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobSubmission(BaseModel):
    """A job request: the kind of work plus the payload its endpoint expects."""

    kind: str = Field(..., description="任务类型，见 GET /jobs/kinds")
    params: Dict[str, Any] = Field(default_factory=dict, description="对应同步接口的请求体")


class JobSummary(BaseModel):
    id: str
    kind: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None


class JobDetail(JobSummary):
    params: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    backend: Optional[str] = None


@router.get("/kinds")
async def list_job_kinds() -> Dict[str, Any]:
    """List the job types this deployment can run."""
    return {"kinds": registered_kinds(), "backend": backend_name()}


@router.post("", status_code=202)
async def create_job(submission: JobSubmission) -> Dict[str, Any]:
    """Queue a job and return immediately with its id."""
    try:
        job_id = submit(submission.kind, submission.params)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception as exc:
        logger.exception("Failed to submit job")
        raise HTTPException(status_code=500, detail=f"任务提交失败: {exc}") from None

    return {
        "job_id": job_id,
        "status": job_store.PENDING,
        "poll_url": f"/jobs/{job_id}",
    }


@router.get("")
async def list_jobs(
    limit: int = Query(50, ge=1, le=200),
    kind: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Recent jobs, newest first."""
    jobs = job_store.list_jobs(limit=limit, kind=kind, status=status)
    # Keep the listing light; params and results can be large.
    summaries = [
        {key: job[key] for key in
         ("id", "kind", "status", "created_at", "started_at", "finished_at", "progress")}
        for job in jobs
    ]
    return {"jobs": summaries}


@router.get("/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    """Full job record, including the result once it has finished."""
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
    job["done"] = job["status"] in job_store.TERMINAL_STATES
    return job


@router.delete("/{job_id}")
async def cancel_job(job_id: str) -> Dict[str, Any]:
    """Best-effort cancel.

    A job that has not started yet will not start. A job already running is
    marked cancelled and its result is discarded, but the FFmpeg work in
    flight is not killed.
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
    if job["status"] in job_store.TERMINAL_STATES:
        return {"job_id": job_id, "cancelled": False, "status": job["status"],
                "message": "任务已结束，无法取消"}

    cancelled = job_store.mark_cancelled(job_id)
    return {"job_id": job_id, "cancelled": cancelled, "status": job_store.CANCELLED}
