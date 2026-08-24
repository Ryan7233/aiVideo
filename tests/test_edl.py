"""The decision list -- the output that survives without a renderer."""

import re

import pytest

from core.edl import to_edl, to_markdown, to_srt

SEGMENTS = [
    {"start_time": 12.5, "end_time": 27.5, "duration": 15.0, "score": 0.88,
     "semantic_score": 0.91, "visual_score": 0.74, "audio_score": 0.69,
     "type": "highlight", "preview_text": "打开率能比中午高三成",
     "semantic_details": {"reason": "给出了具体数据"}},
    {"start_time": 63.0, "end_time": 75.0, "duration": 12.0, "score": 0.81,
     "semantic_score": 0.84, "visual_score": 0.70, "audio_score": 0.66,
     "type": "conclusion", "preview_text": "记得点赞收藏",
     "semantic_details": {"reason": "明确的行动号召"}},
]


class TestMarkdown:
    def test_every_segment_gets_a_row(self):
        body = to_markdown(SEGMENTS, source="a.mp4", topic="运营")
        assert body.count("\n|") == len(SEGMENTS) + 2  # header + separator

    def test_the_reason_is_carried_through(self):
        """The reason is the point -- a bare timestamp is not a decision."""
        body = to_markdown(SEGMENTS)
        assert "给出了具体数据" in body
        assert "明确的行动号召" in body

    def test_the_total_is_reported(self):
        assert "27.0 秒" in to_markdown(SEGMENTS)

    def test_no_segments_still_renders(self):
        assert "剪辑清单" in to_markdown([])


class TestEdl:
    def test_it_looks_like_cmx3600(self):
        body = to_edl(SEGMENTS, title="demo")
        assert body.startswith("TITLE: demo")
        assert "FCM: NON-DROP FRAME" in body

    def test_one_event_per_segment(self):
        events = [l for l in to_edl(SEGMENTS).splitlines() if re.match(r"^\d{3}\s+AX", l)]
        assert len(events) == len(SEGMENTS)

    def test_timecodes_are_well_formed(self):
        for line in to_edl(SEGMENTS).splitlines():
            if re.match(r"^\d{3}\s+AX", line):
                codes = re.findall(r"\d{2}:\d{2}:\d{2}:\d{2}", line)
                assert len(codes) == 4, line

    def test_record_timecode_is_continuous(self):
        """Segments must land back to back on the timeline."""
        events = [l for l in to_edl(SEGMENTS).splitlines() if re.match(r"^\d{3}\s+AX", l)]
        first_out = re.findall(r"\d{2}:\d{2}:\d{2}:\d{2}", events[0])[3]
        second_in = re.findall(r"\d{2}:\d{2}:\d{2}:\d{2}", events[1])[2]
        assert first_out == second_in

    def test_source_timecode_matches_the_segment(self):
        event = [l for l in to_edl(SEGMENTS).splitlines() if l.startswith("001")][0]
        assert re.findall(r"\d{2}:\d{2}:\d{2}:\d{2}", event)[0] == "00:00:12:12"

    def test_frames_never_overflow(self):
        body = to_edl([{"start_time": 1.999, "end_time": 2.999, "duration": 1.0}], fps=25)
        for code in re.findall(r"\d{2}:\d{2}:\d{2}:(\d{2})", body):
            assert int(code) < 25


class TestSrt:
    def test_relative_timing_starts_at_zero(self):
        assert to_srt(SEGMENTS).startswith("1\n00:00:00,000 -->")

    def test_source_timing_is_available(self):
        assert "00:00:12,500" in to_srt(SEGMENTS, relative=False)

    def test_segments_without_text_are_skipped(self):
        assert to_srt([{"start_time": 0, "end_time": 5, "duration": 5}]) == ""

    def test_blocks_are_well_formed(self):
        for block in [b for b in to_srt(SEGMENTS).split("\n\n") if b.strip()]:
            lines = block.strip().splitlines()
            assert lines[0].isdigit()
            assert "-->" in lines[1]
            assert lines[2].strip()


@pytest.mark.integration
def test_the_pipeline_writes_a_list_without_rendering(tmp_path):
    """The list is the product; the render is optional."""
    import shutil
    import subprocess
    from pathlib import Path

    from core.runtime import INPUT_DIR, OUTPUT_DIR
    from core.video_workflow import process_multi_segment_video

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required")
    source = INPUT_DIR / "edl_probe.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15:duration=30",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=30",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", str(source)], check=True, capture_output=True)
    written = []
    try:
        result = process_multi_segment_video({
            "video_path": str(source), "topic": "测试", "target_segments": 2,
            "total_duration": 16, "enable_content_analysis": False, "render": False,
        })
        assert result["rendered"] is False
        assert result["output_video"] is None
        assert "md" in result["decision_list"]
        for relative in result["decision_list"].values():
            path = OUTPUT_DIR / Path(relative).name
            written.append(path)
            assert path.is_file() and path.stat().st_size > 0
    finally:
        source.unlink(missing_ok=True)
        for path in written:
            path.unlink(missing_ok=True)
