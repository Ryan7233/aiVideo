"""What the review UI needs from the API: playback of the source, progress
that names a phase, and a history list a person can read."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core import job_store
from core.runtime import INPUT_DIR, ensure_runtime_directories

ensure_runtime_directories()


@pytest.fixture
def source_file():
    path = INPUT_DIR / "uploads" / "workbench_source.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 16)
    yield path
    path.unlink(missing_ok=True)


class TestSourcePlayback:
    def test_the_source_is_served_under_its_reported_path(self, source_file):
        client = TestClient(app)
        assert client.get("/input_data/uploads/workbench_source.mp4").status_code == 200

    def test_but_only_with_the_key(self, source_file, monkeypatch):
        monkeypatch.setenv("AIVIDEO_API_KEY", "secret123")
        client = TestClient(app)
        assert client.get("/input_data/uploads/workbench_source.mp4").status_code == 401
        ok = client.get("/input_data/uploads/workbench_source.mp4", headers={"X-API-Key": "secret123"})
        assert ok.status_code == 200

    def test_the_result_names_the_source_relative_to_the_data_root(self, source_file):
        from core.video_workflow import _data_relative

        assert _data_relative(source_file) == "input_data/uploads/workbench_source.mp4"
        assert _data_relative(Path("/somewhere/else.mp4")) is None


class TestProgressPhases:
    def test_the_workflow_reports_each_phase(self, monkeypatch, tmp_path):
        """Poll the job and you see 'transcribing', not just 'running'."""
        import core.video_workflow as workflow

        source = INPUT_DIR / "uploads" / "phases.mp4"
        source.write_bytes(b"\x00")
        monkeypatch.setattr(workflow, "_probe_duration", lambda p: 60.0)
        monkeypatch.setattr(workflow, "_probe_fps", lambda p: 25.0)
        monkeypatch.setattr(workflow, "_load_transcript", lambda *a, **k: (
            [{"start": 0, "end": 10, "text": "第一句完整的话。"},
             {"start": 20, "end": 30, "text": "第二句完整的话。"}], {"source": "provided"}))

        class Engine:
            def analyze_video_content(self, *a, **k):
                return {"scene_changes": [], "audio_energy": [], "motion_activity": [],
                        "measured_intervals": [{"start": 0, "end": 60}], "measured_seconds": 60}

        monkeypatch.setattr(workflow, "SmartClippingEngine", Engine)
        monkeypatch.setenv("SEMANTIC_SCORING_MODE", "rules")
        monkeypatch.setattr(workflow, "OUTPUT_DIR", tmp_path)

        phases = []
        try:
            result = workflow.process_multi_segment_video(
                {"video_path": str(source), "topic": "话", "target_segments": 2, "total_duration": 30},
                report=phases.append,
            )
        finally:
            source.unlink(missing_ok=True)
        assert phases == ["materializing", "transcribing", "measuring", "scoring", "exporting"]
        assert result["source_video"] == "input_data/uploads/phases.mp4"

    def test_a_failing_callback_does_not_fail_the_job(self, monkeypatch):
        import core.video_workflow as workflow

        def explode(_):
            raise RuntimeError("progress store down")

        monkeypatch.setattr(workflow, "materialize_video_source",
                            lambda *a, **k: (_ for _ in ()).throw(ValueError("stop here")))
        with pytest.raises(ValueError, match="stop here"):
            workflow.process_multi_segment_video({"video_path": "x.mp4"}, report=explode)

    def test_the_job_handler_writes_phases_to_the_store(self, monkeypatch):
        import core.jobs as jobs

        seen = []
        monkeypatch.setattr(jobs.job_store, "set_progress", lambda job_id, progress: seen.append(progress["step"]))

        def fake_workflow(params, report=None):
            report("transcribing")
            return {"ok": True}

        monkeypatch.setattr("core.video_workflow.process_multi_segment_video", fake_workflow)
        assert jobs._multi_segment_handler({}, "job-1") == {"ok": True}
        assert seen == ["queued", "transcribing"]


class TestHistoryListing:
    def test_jobs_carry_a_human_label(self):
        client = TestClient(app)
        by_topic = job_store.create_job("multi_segment_clipping",
                                        {"topic": "小红书运营", "video_path": "input_data/uploads/a.mp4"})
        by_file = job_store.create_job("multi_segment_clipping",
                                       {"topic": "", "video_path": "input_data/uploads/talk.mp4"})
        try:
            listed = {job["id"]: job for job in client.get("/jobs?limit=200").json()["jobs"]}
            assert listed[by_topic]["label"] == "小红书运营"
            assert listed[by_file]["label"] == "talk.mp4"
        finally:
            with job_store._connect() as connection:
                connection.execute("DELETE FROM jobs WHERE id IN (?, ?)", (by_topic, by_file))


class TestTheFrontendUsesAllOfIt:
    @pytest.fixture(scope="class")
    def source(self):
        return Path("frontend/app.js").read_text(encoding="utf-8")

    def test_it_plays_the_source_and_names_phases(self, source):
        assert "source_video" in source
        for phase in ("transcribing", "measuring", "scoring", "exporting"):
            assert phase in source

    def test_it_exports_labels_the_evaluator_accepts(self, source):
        assert '"accepted"' in source or "accepted," in source
        assert "start_time" in source and "end_time" in source

    def test_it_lists_and_cancels_jobs(self, source):
        assert "/jobs?kind=" in source
        assert "method: 'DELETE'" in source


def test_the_page_has_no_duplicate_element_ids():
    """The segment list was once appended into the '片段数' input because both
    carried id="segments"; getElementById returns the first one, silently."""
    import collections
    import re

    ids = re.findall(r'\bid="([^"]+)"', Path("frontend/index.html").read_text(encoding="utf-8"))
    duplicates = [name for name, count in collections.Counter(ids).items() if count > 1]
    assert not duplicates, duplicates


class TestLanguageIsSelectable:
    """Detection listens to the opening 30 s only. On a Mandarin video that
    starts in English it returned "en", and the whole transcript came back as
    romanised nonsense -- every candidate then scored within 0.02 of the next.
    The API always took asr_language; the page did not offer it."""

    def test_the_page_offers_it_and_sends_it(self):
        page = Path("frontend/index.html").read_text(encoding="utf-8")
        assert 'id="asr-language"' in page
        assert 'value="zh"' in page
        assert 'value="" selected' in page, "auto-detect stays the default"
        assert "asr_language" in Path("frontend/app.js").read_text(encoding="utf-8")

    def test_the_request_model_passes_it_through(self, source_file):
        from api.main import MultiSegmentClippingReq

        assert MultiSegmentClippingReq.model_fields["asr_language"].default is None
        req = MultiSegmentClippingReq(video_path=str(source_file), topic="t", asr_language="zh")
        assert req.asr_language == "zh"


class TestEvaluateFromThePage:
    """The page exported a labels file and told the reviewer to run a CLI with
    it -- but never said where the file went, and nothing in the UI consumed
    one. The loop was open at the handoff; this endpoint closes it."""

    @pytest.fixture
    def finished_job(self):
        job_id = job_store.create_job("multi_segment_clipping", {"topic": "校准"})
        job_store.mark_running(job_id)
        job_store.mark_succeeded(job_id, {"selected_segments": [
            {"start_time": 10, "end_time": 20, "text": "第一段", "cues": []},
            {"start_time": 40, "end_time": 50, "text": "第二段", "cues": []},
        ]})
        yield job_id
        with job_store._connect() as connection:
            connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def test_a_disagreement_produces_real_numbers(self, finished_job):
        client = TestClient(app)
        # Keeps the first suggestion, rejects the second, adds a missed span.
        body = client.post("/evaluate/selection", json={
            "job_id": finished_job,
            "accepted": [{"start_time": 10, "end_time": 20}, {"start_time": 70, "end_time": 80}],
        }).json()
        human = body["evaluation"]["human_metrics"]
        assert human["matched_count"] == 1
        assert human["precision_at_returned_k"] == 0.5
        assert human["approved_moment_recall"] == 0.5
        assert body["trivial"] is False

    def test_accepting_everything_is_flagged_as_meaningless(self, finished_job):
        """Labels identical to the output score 1.0 by construction."""
        client = TestClient(app)
        body = client.post("/evaluate/selection", json={
            "job_id": finished_job,
            "accepted": [{"start_time": 10, "end_time": 20}, {"start_time": 40, "end_time": 50}],
        }).json()
        assert body["evaluation"]["human_metrics"]["precision_at_returned_k"] == 1.0
        assert body["trivial"] is True

    def test_bad_spans_are_rejected(self, finished_job):
        client = TestClient(app)
        assert client.post("/evaluate/selection", json={
            "job_id": finished_job, "accepted": [{"start_time": 30, "end_time": 20}],
        }).status_code == 422

    def test_an_unfinished_or_missing_job_is_refused(self):
        client = TestClient(app)
        assert client.post("/evaluate/selection", json={
            "job_id": "nope", "accepted": []}).status_code == 404
        pending = job_store.create_job("multi_segment_clipping", {})
        try:
            assert client.post("/evaluate/selection", json={
                "job_id": pending, "accepted": []}).status_code == 409
        finally:
            with job_store._connect() as connection:
                connection.execute("DELETE FROM jobs WHERE id = ?", (pending,))

    def test_the_page_calls_it_and_shows_the_caveat(self):
        page = Path("frontend/index.html").read_text(encoding="utf-8")
        source = Path("frontend/app.js").read_text(encoding="utf-8")
        assert 'id="evaluate"' in page and 'id="scorecard"' in page
        assert "/evaluate/selection" in source
        assert "trivial" in source, "a 100% score from unchanged labels must be labelled as such"
