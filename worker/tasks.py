"""The one Celery task: run a job from the shared registry.

Everything else here was a parallel task system -- enqueue/status/cancel
endpoints and their own ASR, audio and clipping tasks -- built before
core.jobs existed and duplicating it. /jobs replaced it.
"""

import logging

from worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="worker.tasks.run_registered_job")
def run_registered_job(self, kind: str, params: dict, job_id: str):
    """Execute a registered job and record the outcome in the job store."""
    import importlib

    # Importing api.main registers the job handlers. Done here rather than at
    # module import so the worker only pays for it when a job arrives, and via
    # importlib so it is not mistaken for an unused import.
    importlib.import_module("api.main")
    from core.jobs import run_job

    return run_job(kind, params, job_id)
