"""End-to-end coverage for submit-and-poll.

The pipelines used to run inside the request, so a dropped connection or a
restart lost the work with no record it ever happened. These tests exercise the
thread backend, which is what a single-machine install uses.
"""

import shutil
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core import job_store, jobs
from core.runtime import INPUT_DIR, OUTPUT_DIR


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("JOB_BACKEND", "thread")
    job_store.reset_for_tests(tmp_path / "jobs.db")
    yield
    jobs.shutdown(wait=True)


@pytest.fixture
def client():
    return TestClient(app)


def _wait_for(job_id, client, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/jobs/{job_id}").json()
        if payload["done"]:
            return payload
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


class TestJobApi:
    def test_kinds_are_advertised(self, client):
        payload = client.get("/jobs/kinds").json()
        assert "multi_segment_clipping" in payload["kinds"]
        assert "xiaohongshu_pipeline" in payload["kinds"]
        assert payload["backend"] == "thread"

    def test_unknown_kind_is_rejected_without_creating_a_record(self, client):
        response = client.post("/jobs", json={"kind": "nope", "params": {}})
        assert response.status_code == 400
        assert client.get("/jobs").json()["jobs"] == []

    def test_missing_job_is_404(self, client):
        assert client.get("/jobs/does-not-exist").status_code == 404

    def test_submission_returns_immediately_with_a_poll_url(self, client):
        response = client.post(
            "/jobs",
            json={"kind": "multi_segment_clipping", "params": {"video_path": "missing.mp4"}},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["poll_url"] == f"/jobs/{body['job_id']}"

    def test_failure_is_recorded_rather_than_raised(self, client):
        """A failing job still leaves a durable record with the error."""
        job_id = client.post(
            "/jobs",
            json={"kind": "multi_segment_clipping", "params": {"video_path": "missing.mp4"}},
        ).json()["job_id"]

        payload = _wait_for(job_id, client, timeout=30)
        assert payload["status"] == job_store.FAILED
        assert payload["error"]
        assert payload["result"] is None

    def test_listing_filters_by_status(self, client):
        job_id = client.post(
            "/jobs",
            json={"kind": "multi_segment_clipping", "params": {"video_path": "missing.mp4"}},
        ).json()["job_id"]
        _wait_for(job_id, client, timeout=30)

        failed = client.get("/jobs", params={"status": job_store.FAILED}).json()["jobs"]
        assert [job["id"] for job in failed] == [job_id]
        assert client.get("/jobs", params={"status": job_store.SUCCEEDED}).json()["jobs"] == []

    def test_cancel_marks_the_record_and_is_idempotent(self, client):
        job_id = job_store.create_job("multi_segment_clipping", {})
        first = client.delete(f"/jobs/{job_id}").json()
        assert first["cancelled"] is True
        assert first["status"] == job_store.CANCELLED

        second = client.delete(f"/jobs/{job_id}").json()
        assert second["cancelled"] is False


class TestJobStore:
    def test_record_survives_a_fresh_connection(self):
        job_id = job_store.create_job("multi_segment_clipping", {"topic": "咖啡店探店"})
        job_store.mark_succeeded(job_id, {"output_video": "a.mp4"})

        reloaded = job_store.get_job(job_id)
        assert reloaded["params"]["topic"] == "咖啡店探店"
        assert reloaded["result"]["output_video"] == "a.mp4"

    def test_cancelled_job_is_not_resurrected_by_a_late_success(self):
        job_id = job_store.create_job("multi_segment_clipping", {})
        job_store.mark_cancelled(job_id)

        outcome = jobs.run_job("multi_segment_clipping", {}, job_id)

        assert outcome["status"] == "cancelled"
        assert job_store.get_job(job_id)["status"] == job_store.CANCELLED

    def test_retention_only_drops_finished_jobs(self):
        live = job_store.create_job("multi_segment_clipping", {})
        done = job_store.create_job("multi_segment_clipping", {})
        job_store.mark_succeeded(done, {})

        assert job_store.delete_jobs_older_than(days=365) == 0
        assert job_store.delete_jobs_older_than(days=-1) == 1
        assert job_store.get_job(live) is not None
        assert job_store.get_job(done) is None


@pytest.mark.integration
def test_clipping_job_runs_to_completion(client):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required")

    source = INPUT_DIR / "job_e2e.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15:duration=24",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=24",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source)],
        check=True, capture_output=True,
    )

    output = None
    try:
        job_id = client.post("/jobs", json={
            "kind": "multi_segment_clipping",
            "params": {
                "video_path": str(source),
                "topic": "测试",
                "target_segments": 2,
                "total_duration": 16,
                "enable_content_analysis": False,
            },
        }).json()["job_id"]

        payload = _wait_for(job_id, client)
        assert payload["status"] == job_store.SUCCEEDED, payload.get("error")
        result = payload["result"]
        assert len(result["selected_segments"]) == 2
        output = OUTPUT_DIR / result["output_video"].split("/")[-1]
        assert output.exists()
    finally:
        source.unlink(missing_ok=True)
        if output:
            output.unlink(missing_ok=True)


class TestPipelineJobHandlers:
    """The XHS pipelines as background jobs.

    The pipeline body itself is unchanged; what is new is the job wiring around
    it -- param validation, source resolution and progress reporting -- so that
    is what these cover, with the (minutes-long, model-downloading) body stubbed.
    """

    def test_handler_validates_params_and_resolves_the_source(self, monkeypatch, client):
        from api import main

        source = INPUT_DIR / "pipeline_job.mp4"
        source.write_bytes(b"placeholder")
        seen = {}

        def fake_body(req, input_path, ts):
            seen["input_path"] = input_path
            seen["city"] = req.city
            return {"status": "success", "pipeline_result": {"processing_id": f"xhs_{ts}"}}

        monkeypatch.setattr(main, "_run_xiaohongshu_pipeline", fake_body)
        try:
            job_id = client.post("/jobs", json={
                "kind": "xiaohongshu_pipeline",
                "params": {"video_url": str(source), "city": "上海", "style": "轻松"},
            }).json()["job_id"]

            payload = _wait_for(job_id, client, timeout=30)
            assert payload["status"] == job_store.SUCCEEDED, payload.get("error")
            assert seen["input_path"] == str(source.resolve())
            assert seen["city"] == "上海"
            assert payload["result"]["status"] == "success"
        finally:
            source.unlink(missing_ok=True)

    def test_bad_params_fail_the_job_instead_of_crashing_the_worker(self, client):
        job_id = client.post("/jobs", json={
            "kind": "xiaohongshu_pipeline",
            "params": {"city": "上海"},  # video_url missing
        }).json()["job_id"]

        payload = _wait_for(job_id, client, timeout=30)
        assert payload["status"] == job_store.FAILED
        assert "video_url" in payload["error"]

    def test_progress_is_reported_while_running(self, monkeypatch, client):
        from api import main

        source = INPUT_DIR / "pipeline_progress.mp4"
        source.write_bytes(b"placeholder")
        observed = []

        def fake_body(req, input_path, ts):
            observed.append(job_store.get_job(job_id_holder["id"])["progress"])
            return {"status": "success"}

        monkeypatch.setattr(main, "_run_xiaohongshu_pipeline", fake_body)
        job_id_holder = {}
        try:
            job_id_holder["id"] = client.post("/jobs", json={
                "kind": "xiaohongshu_pipeline",
                "params": {"video_url": str(source)},
            }).json()["job_id"]

            _wait_for(job_id_holder["id"], client, timeout=30)
            assert observed == [{"step": "processing"}]
        finally:
            source.unlink(missing_ok=True)


class TestStatusTransitions:
    """A terminal status is final.

    The API dispatched to Celery and then called mark_running(). A worker that
    finished or failed first had its result overwritten back to "running", so
    the poller waited forever; a cancel could likewise be overwritten by a late
    failure. Every transition is now guarded in SQL rather than by a
    read-then-write, because the API process and the worker race on one row.
    """

    def test_finished_job_cannot_be_pulled_back_to_running(self):
        job_id = job_store.create_job("multi_segment_clipping", {}, backend="celery")
        job_store.mark_succeeded(job_id, {"output_video": "a.mp4"})

        assert job_store.mark_running(job_id) is False
        job = job_store.get_job(job_id)
        assert job["status"] == job_store.SUCCEEDED
        assert job["result"]["output_video"] == "a.mp4"

    def test_failed_job_cannot_be_pulled_back_to_running(self):
        job_id = job_store.create_job("multi_segment_clipping", {})
        job_store.mark_failed(job_id, "boom")

        assert job_store.mark_running(job_id) is False
        assert job_store.get_job(job_id)["status"] == job_store.FAILED

    def test_cancel_beats_a_late_failure(self):
        job_id = job_store.create_job("multi_segment_clipping", {})
        job_store.mark_cancelled(job_id)

        assert job_store.mark_failed(job_id, "worker died") is False
        assert job_store.get_job(job_id)["status"] == job_store.CANCELLED

    def test_cancel_beats_a_late_success(self):
        job_id = job_store.create_job("multi_segment_clipping", {})
        job_store.mark_cancelled(job_id)

        assert job_store.mark_succeeded(job_id, {"output_video": "a.mp4"}) is False
        job = job_store.get_job(job_id)
        assert job["status"] == job_store.CANCELLED
        assert job["result"] is None

    def test_a_cancelled_job_never_starts(self):
        job_id = job_store.create_job("multi_segment_clipping", {})
        job_store.mark_cancelled(job_id)
        ran = []

        jobs.register_job_handler("_transition_probe", lambda params, jid: ran.append(1) or {})
        try:
            outcome = jobs.run_job("_transition_probe", {}, job_id)
        finally:
            jobs._handlers.pop("_transition_probe", None)

        assert ran == [], "work started on a job that was already cancelled"
        assert outcome["status"] == job_store.CANCELLED

    def test_dispatch_records_the_task_id_without_touching_status(self):
        """submit() must not own the status; the executor does."""
        job_id = job_store.create_job("multi_segment_clipping", {}, backend="celery")
        job_store.mark_succeeded(job_id, {})

        job_store.set_external_id(job_id, "celery-task-123")

        job = job_store.get_job(job_id)
        assert job["external_id"] == "celery-task-123"
        assert job["status"] == job_store.SUCCEEDED

    def test_running_is_idempotent(self):
        job_id = job_store.create_job("multi_segment_clipping", {})
        assert job_store.mark_running(job_id) is True
        assert job_store.mark_running(job_id) is True
        assert job_store.get_job(job_id)["status"] == job_store.RUNNING


class TestDispatchFailure:
    """A broker that is down must not leave orphan records.

    submit() created the row and then dispatched. When the broker was
    unreachable the exception propagated, leaving a row stuck in "pending"
    with no finished_at and no error -- one more every time the broker
    flapped, and the caller never learned the id.
    """

    @pytest.fixture
    def broken_broker(self, monkeypatch):
        monkeypatch.setenv("JOB_BACKEND", "celery")

        class Unreachable:
            def delay(self, *args, **kwargs):
                raise ConnectionError("Error 61 connecting to localhost:6379.")

        import worker.tasks

        monkeypatch.setattr(worker.tasks, "run_registered_job", Unreachable())

    def test_failed_dispatch_is_recorded_not_orphaned(self, broken_broker):
        with pytest.raises(jobs.JobDispatchError) as caught:
            jobs.submit("multi_segment_clipping", {"video_path": "x.mp4"})

        job = job_store.get_job(caught.value.job_id)
        assert job["status"] == job_store.FAILED
        assert job["finished_at"] is not None
        assert "投递失败" in job["error"]
        assert "ConnectionError" in job["error"]

    def test_no_pending_rows_are_left_behind(self, broken_broker):
        for _ in range(3):
            with pytest.raises(jobs.JobDispatchError):
                jobs.submit("multi_segment_clipping", {"video_path": "x.mp4"})

        assert job_store.list_jobs(status=job_store.PENDING) == []
        assert len(job_store.list_jobs(status=job_store.FAILED)) == 3

    def test_the_error_carries_the_job_id(self, broken_broker):
        with pytest.raises(jobs.JobDispatchError) as caught:
            jobs.submit("multi_segment_clipping", {})
        assert job_store.get_job(caught.value.job_id) is not None

    def test_route_returns_503_with_the_job_id(self, broken_broker, client):
        response = client.post(
            "/jobs", json={"kind": "multi_segment_clipping", "params": {}}
        )
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["job_id"]
        assert job_store.get_job(detail["job_id"])["status"] == job_store.FAILED

    def test_a_broken_thread_pool_is_handled_the_same_way(self, monkeypatch):
        monkeypatch.setenv("JOB_BACKEND", "thread")

        class DeadPool:
            def submit(self, *args, **kwargs):
                raise RuntimeError("cannot schedule new futures after shutdown")

        monkeypatch.setattr(jobs, "_get_executor", lambda: DeadPool())

        with pytest.raises(jobs.JobDispatchError) as caught:
            jobs.submit("multi_segment_clipping", {})
        assert job_store.get_job(caught.value.job_id)["status"] == job_store.FAILED
