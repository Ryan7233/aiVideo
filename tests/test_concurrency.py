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


def test_every_ffmpeg_invocation_goes_through_the_gate():
    """The README claims a cap on concurrent FFmpeg processes.

    That was only true for the main clipping path; the black/silence probes,
    cover frame extraction and subtitle extraction still called subprocess.run
    directly, so the cap could be exceeded by whatever those added. ffprobe is
    deliberately not gated: it reads metadata in milliseconds, and queueing it
    behind long encodes would add contention for no benefit.
    """
    import ast
    import pathlib

    offenders = []
    scanned = [
        *pathlib.Path("core").glob("*.py"),
        *pathlib.Path("worker").glob("*.py"),
        *pathlib.Path("routers").glob("*.py"),
        pathlib.Path("api/main.py"),
    ]
    for path in scanned:
        if path.name == "concurrency.py":
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "run"
                    and isinstance(func.value, ast.Name) and func.value.id == "subprocess"):
                continue
            # Look at the command being built just above the call.
            window = "\n".join(lines[max(0, node.lineno - 12):node.lineno])
            if '"ffmpeg"' in window or "'ffmpeg'" in window:
                offenders.append(f"{path}:{node.lineno}")

    assert not offenders, (
        "these run FFmpeg without holding a media slot; use "
        f"core.concurrency.run_ffmpeg: {offenders}"
    )
