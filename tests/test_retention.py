"""The retention sweep must reclaim old artifacts without eating live ones."""

import os
import time

import pytest

from core import job_store, retention


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A managed runtime tree, with uploads nested inside output_data."""
    output = tmp_path / "output_data"
    photos = output / "uploads"
    videos = tmp_path / "input_data" / "uploads"
    downloads = tmp_path / "input_data" / "downloads"
    for directory in (output, photos, videos, downloads):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(retention, "OUTPUT_DIR", output)
    monkeypatch.setattr(retention, "PHOTO_UPLOAD_DIR", photos)
    monkeypatch.setattr(retention, "VIDEO_UPLOAD_DIR", videos)
    monkeypatch.setattr(retention, "DOWNLOAD_DIR", downloads)
    job_store.reset_for_tests(tmp_path / "jobs.db")
    return {"output": output, "photos": photos, "videos": videos, "downloads": downloads}


def _write(path, *, age_days=0.0, size=64):
    path.write_bytes(b"x" * size)
    if age_days:
        old = time.time() - age_days * 86400
        os.utime(path, (old, old))
    return path


def test_old_outputs_are_removed_and_fresh_ones_kept(tree, monkeypatch):
    monkeypatch.setenv("OUTPUT_RETENTION_DAYS", "7")
    stale = _write(tree["output"] / "old_clip.mp4", age_days=30)
    fresh = _write(tree["output"] / "new_clip.mp4", age_days=1)

    report = retention.run_sweep()

    assert not stale.exists()
    assert fresh.exists()
    assert report["removed_files"] == 1
    assert report["freed_bytes"] == 64


def test_uploads_keep_their_own_longer_window(tree, monkeypatch):
    """Uploads live under output_data but must not inherit its short TTL."""
    monkeypatch.setenv("OUTPUT_RETENTION_DAYS", "7")
    monkeypatch.setenv("UPLOAD_RETENTION_DAYS", "30")
    upload = _write(tree["photos"] / "user_photo.jpg", age_days=10)

    retention.run_sweep()

    assert upload.exists(), "a 10-day-old upload was deleted by the 7-day output rule"


def test_uploads_past_their_own_window_are_removed(tree, monkeypatch):
    monkeypatch.setenv("UPLOAD_RETENTION_DAYS", "30")
    upload = _write(tree["photos"] / "ancient.jpg", age_days=60)

    retention.run_sweep()

    assert not upload.exists()


def test_zero_days_disables_a_rule(tree, monkeypatch):
    monkeypatch.setenv("OUTPUT_RETENTION_DAYS", "0")
    monkeypatch.setenv("UPLOAD_RETENTION_DAYS", "0")
    kept = _write(tree["output"] / "ancient.mp4", age_days=999)

    assert retention.run_sweep()["removed_files"] == 0
    assert kept.exists()


def test_downloads_are_swept(tree, monkeypatch):
    monkeypatch.setenv("OUTPUT_RETENTION_DAYS", "7")
    stale = _write(tree["downloads"] / "source.mp4", age_days=30)

    retention.run_sweep()

    assert not stale.exists()


def test_finished_job_rows_are_pruned(tree, monkeypatch):
    import sqlite3

    monkeypatch.setenv("JOB_RETENTION_DAYS", "1")
    done = job_store.create_job("multi_segment_clipping", {})
    job_store.mark_succeeded(done, {})
    pending = job_store.create_job("multi_segment_clipping", {})
    with sqlite3.connect(str(job_store.DB_PATH)) as connection:
        connection.execute(
            "UPDATE jobs SET finished_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (done,)
        )

    report = retention.run_sweep()

    assert report["removed_jobs"] == 1
    assert job_store.get_job(done) is None
    assert job_store.get_job(pending) is not None


def test_zero_job_retention_keeps_everything(tree, monkeypatch):
    """0 disables the rule; it used to mean "cutoff = now" and wipe the table."""
    monkeypatch.setenv("JOB_RETENTION_DAYS", "0")
    done = job_store.create_job("multi_segment_clipping", {})
    job_store.mark_succeeded(done, {})

    assert retention.run_sweep()["removed_jobs"] == 0
    assert job_store.get_job(done) is not None


def test_sweep_reports_errors_instead_of_raising(tree, monkeypatch):
    monkeypatch.setenv("OUTPUT_RETENTION_DAYS", "1")
    _write(tree["output"] / "locked.mp4", age_days=30)

    def boom(self):
        raise OSError("permission denied")

    monkeypatch.setattr("pathlib.Path.unlink", boom)
    report = retention.run_sweep()

    assert report["removed_files"] == 0
    assert report["errors"]


def test_directories_are_never_removed(tree, monkeypatch):
    monkeypatch.setenv("OUTPUT_RETENTION_DAYS", "1")
    nested = tree["output"] / "collages"
    nested.mkdir()
    old = time.time() - 999 * 86400
    os.utime(nested, (old, old))

    retention.run_sweep()

    assert nested.is_dir()


def test_sweeper_starts_and_stops_with_the_app(monkeypatch):
    """The lifespan hook must actually schedule the sweep, then clean up."""
    from fastapi.testclient import TestClient

    from api.main import app

    calls = []
    monkeypatch.setenv("RETENTION_SWEEP_INTERVAL_HOURS", "6")
    monkeypatch.setattr(retention, "run_sweep", lambda: calls.append(1) or {})

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        deadline = time.time() + 5
        while not calls and time.time() < deadline:
            time.sleep(0.05)

    assert calls, "the retention sweeper never ran"


def test_sweeper_can_be_disabled(monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app

    calls = []
    monkeypatch.setenv("RETENTION_SWEEP_INTERVAL_HOURS", "0")
    monkeypatch.setattr(retention, "run_sweep", lambda: calls.append(1) or {})

    with TestClient(app) as client:
        client.get("/health")
        time.sleep(0.3)

    assert not calls
