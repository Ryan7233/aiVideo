"""Regressions for the decisions a user actually receives, not model mocks."""

from pathlib import Path
import subprocess

import pytest

from core.candidates import build_windows
from core.edl import to_srt
from core.smart_clipping import SmartClippingEngine
from core.video_workflow import _build_candidates, _select_segments


def test_complete_sentence_never_leaks_into_a_partial_window():
    transcript = [{"start": 8, "end": 13, "text": "从第八秒说到第十三秒。"}]
    windows = build_windows(30, 10, transcript)
    assert [(w["start"], w["end"]) for w in windows] == [(8, 13)]
    assert windows[0]["text"] == transcript[0]["text"]


def test_long_sentence_is_not_cut_to_fill_a_short_budget():
    transcript = [{"start": 0, "end": 20, "text": "一段没有内部时间戳的长句"}]
    assert build_windows(40, 10, transcript, maximum=10) == []
    assert build_windows(40, 10, transcript, maximum=25)[0]["end"] == 20


def test_overlapping_cues_cannot_be_cut_apart():
    cues = [{"start": 0, "end": 8, "text": "第一人"},
            {"start": 6, "end": 12, "text": "第二人"}]
    assert build_windows(20, 10, cues)[0]["end"] == 12


def test_silence_is_not_filled_with_unrelated_sentences():
    cues = [{"start": 0, "end": 3, "text": "甲"},
            {"start": 20, "end": 23, "text": "乙"}]
    assert build_windows(30, 10, cues) == []


@pytest.mark.parametrize("count", [1, 2, 3])
def test_weak_bookends_never_displace_the_best_moment(count):
    candidates = [{"start_time": s, "end_time": s + 10, "score": score}
                  for s, score in [(0, .01), (30, .95), (50, .99), (90, .02)]]
    for summary in (False, True):
        selected = _select_segments(candidates, 100, count, summary, True, summary)
        assert any(s["score"] == .99 for s in selected)
        if count <= 2:
            assert all(s["score"] >= .95 for s in selected)


def test_summary_picks_the_best_opening_not_the_first_one():
    candidates = [{"start_time": s, "end_time": s + 5, "score": score}
                  for s, score in [(0, .4), (10, .9), (50, 1.0), (90, .8)]]
    selected = _select_segments(candidates, 100, 3, True, True, True)
    assert selected[0]["start_time"] == 10
    assert selected[-1]["type"] == "conclusion"


def test_variable_sentence_lengths_honor_total_budget():
    candidates = [{"start_time": s, "end_time": s + width, "score": score}
                  for s, width, score in [(0, 18, .9), (20, 15, .8), (40, 10, .7)]]
    result = _select_segments(candidates, 60, 3, False, True, False, total_budget=30)
    assert sum(s["end_time"] - s["start_time"] for s in result) == 28


def test_missing_measurements_do_not_lower_a_semantic_score(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCORING_MODE", "rules")
    text = "方法是先按完整句子切分，再核实每段的内容是否与主题相关。"
    cues = [{"start": s, "end": s + 10, "text": text} for s in (0, 1000)]
    analysis = {"measured_intervals": [{"start": 0, "end": 300}],
                "motion_activity": [{"timestamp": 2, "activity": 1}],
                "audio_energy": [{"timestamp": 2, "energy": 1}]}
    candidates = _build_candidates(1800, 10, "方法", cues, analysis,
                                   {"semantic": .4, "visual": .3, "audio": .3})
    late = next(c for c in candidates if c["start_time"] == 1000)
    assert late["score"] == late["semantic_score"]
    assert late["effective_weights"] == {"semantic": 1., "visual": 0., "audio": 0.}


def test_full_video_measures_after_fifteen_minutes(monkeypatch):
    engine = SmartClippingEngine()
    monkeypatch.setattr(engine, "_get_video_duration", lambda p: 1805)
    starts = []

    def measured(path, duration, start=0):
        starts.append(start)
        return {"scene_changes": [], "audio_energy": [],
                "motion_activity": [{"timestamp": 1, "activity": .7}]}

    monkeypatch.setattr(engine, "_measure_streams", measured)
    result = engine.analyze_video_content("placeholder", max_duration=None)
    assert starts == [0, 300, 600, 900, 1200, 1500, 1800]
    assert result["measured_seconds"] == 1805
    assert result["motion_activity"][-1]["timestamp"] == 1801


def test_failed_chunk_is_reported_without_inventing_measurements(monkeypatch):
    engine = SmartClippingEngine()
    monkeypatch.setattr(engine, "_get_video_duration", lambda p: 700)

    def measured(path, duration, start=0):
        if start == 300:
            raise RuntimeError("decode error")
        return {"scene_changes": [], "audio_energy": [], "motion_activity": []}

    monkeypatch.setattr(engine, "_measure_streams", measured)
    result = engine.analyze_video_content("placeholder", max_duration=None)
    assert result["degraded"]
    assert result["measured_seconds"] == 400
    assert result["measurement_errors"][0]["start"] == 300


def test_subtitle_cues_keep_timing_and_complete_text():
    text = "完整字幕" * 100
    segments = [{"start_time": 0, "end_time": 5, "duration": 5, "cues": []},
                {"start_time": 10, "end_time": 20, "duration": 10,
                 "preview_text": "截断预览", "cues": [
                     {"start": 11, "end": 13, "text": text},
                     {"start": 15, "end": 19, "text": "第二句"}]}]
    srt = to_srt(segments)
    assert "00:00:06,000 --> 00:00:08,000" in srt
    assert "00:00:10,000 --> 00:00:14,000" in srt
    assert text in srt and "截断预览" not in srt


@pytest.mark.integration
@pytest.mark.parametrize("duration,chunk_seconds", [(18, 6), (910, 300)])
def test_real_ffmpeg_chunks_have_absolute_timestamps(tmp_path, duration, chunk_seconds):
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg required")
    source = tmp_path / "chunks.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc2=size=96x64:rate=5:duration={duration}", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={duration}", "-c:v", "libx264", "-c:a", "aac",
                    "-shortest", str(source)], check=True, capture_output=True)
    result = SmartClippingEngine().analyze_video_content(str(source), max_duration=None, chunk_seconds=chunk_seconds)
    assert not result["degraded"], result["measurement_errors"]
    assert result["measured_seconds"] >= duration
    for key in ("motion_activity", "audio_energy"):
        times = [p["timestamp"] for p in result[key]]
        assert times == sorted(times)
        assert any(t >= duration - 6 for t in times)
        assert all(0 <= t < duration + .1 for t in times)


def test_default_mode_matches_ui_and_batch():
    from api.main import app
    props = app.openapi()["components"]["schemas"]["MultiSegmentClippingReq"]["properties"]
    assert props["selection_mode"]["default"] == "highlights"
    from api.main import MultiSegmentClippingReq
    assert MultiSegmentClippingReq.model_fields["include_intro"].default is None
    assert 'value="highlights" selected' in Path("frontend/index.html").read_text()


def test_human_metrics_require_annotations_and_do_not_double_count():
    from core.evaluation import evaluate_selection
    selected = [{"start_time": 10, "end_time": 20}, {"start_time": 11, "end_time": 21}]
    assert evaluate_selection(selected)["human_metrics"] is None
    result = evaluate_selection(selected, [{"start_time": 10, "end_time": 20}])
    assert result["overlapping_pairs"] == 1
    assert result["human_metrics"]["precision_at_returned_k"] == .5
    assert result["human_metrics"]["approved_moment_recall"] == 1.


def test_srt_milliseconds_carry_at_minute_boundary():
    from core.edl import _clock
    assert _clock(59.9999) == "00:01:00,000"
