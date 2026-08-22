"""The media gate must actually bound concurrent FFmpeg processes."""

import threading
import time

import pytest

from core import concurrency
from core.concurrency import MediaCapacityError, media_slot


def test_slot_count_is_at_least_one():
    assert concurrency.MAX_CONCURRENT_MEDIA_JOBS >= 1


def test_concurrent_holders_never_exceed_the_cap(monkeypatch):
    cap = 2
    monkeypatch.setattr(concurrency, "_slots", threading.BoundedSemaphore(cap))

    live = 0
    peak = 0
    lock = threading.Lock()

    def worker():
        nonlocal live, peak
        with media_slot(timeout=10):
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.05)
            with lock:
                live -= 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert peak <= cap, f"{peak} concurrent holders with a cap of {cap}"
    assert peak > 1, "the gate serialised everything; it should allow the full cap"


def test_slot_is_released_when_the_body_raises(monkeypatch):
    monkeypatch.setattr(concurrency, "_slots", threading.BoundedSemaphore(1))

    with pytest.raises(ValueError):
        with media_slot(timeout=1):
            raise ValueError("boom")

    # If the slot leaked, this acquisition would time out.
    with media_slot(timeout=1):
        pass


def test_waiting_past_the_timeout_raises(monkeypatch):
    monkeypatch.setattr(concurrency, "_slots", threading.BoundedSemaphore(1))

    with media_slot(timeout=1):
        with pytest.raises(MediaCapacityError):
            with media_slot(timeout=0.1):
                pass
