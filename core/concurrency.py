"""A cap on how many FFmpeg processes can run at once.

Blocking media work is handed to ``asyncio.to_thread``, which uses the default
executor -- roughly ``min(32, cpu_count + 4)`` threads. Each clipping request
spawns an analysis decode plus one encode per segment, so a handful of
concurrent requests was enough to put dozens of FFmpeg processes on the box and
starve everything, including the health check.

This gate makes surplus requests queue instead.

Scope: the semaphore is per process. With several Celery workers the effective
cap is (workers x MAX_CONCURRENT_MEDIA_JOBS), so size it against the host
rather than against one container. A cluster-wide limit would need the counter
in Redis; that is worth doing only once more than one machine is involved.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from contextlib import contextmanager
from typing import Iterator, List, Optional

from core.env import env_float, env_int

logger = logging.getLogger(__name__)


def _default_slots() -> int:
    cpus = os.cpu_count() or 2
    return max(1, cpus // 2)


MAX_CONCURRENT_MEDIA_JOBS = max(1, env_int("MAX_CONCURRENT_MEDIA_JOBS", _default_slots()))
MEDIA_SLOT_TIMEOUT = env_float("MEDIA_SLOT_TIMEOUT", 1800.0)

_slots = threading.BoundedSemaphore(MAX_CONCURRENT_MEDIA_JOBS)


class MediaCapacityError(RuntimeError):
    """Raised when a media slot could not be acquired within the timeout."""


@contextmanager
def media_slot(timeout: Optional[float] = None) -> Iterator[None]:
    """Hold one of the media slots for the duration of the block."""
    wait = MEDIA_SLOT_TIMEOUT if timeout is None else timeout
    if not _slots.acquire(timeout=wait):
        raise MediaCapacityError(
            f"媒体处理槽位在 {wait:.0f} 秒内未能获取，当前并发上限为 {MAX_CONCURRENT_MEDIA_JOBS}"
        )
    try:
        yield
    finally:
        _slots.release()


def run_ffmpeg(cmd: List[str], timeout: int = 300, **kwargs) -> subprocess.CompletedProcess:
    """Run one FFmpeg command while holding a media slot."""
    with media_slot():
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kwargs)
