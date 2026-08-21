import shutil
import subprocess
from pathlib import Path

import pytest

from core.runtime import INPUT_DIR, OUTPUT_DIR
from core.video_workflow import _build_candidates, _select_segments, process_multi_segment_video


def test_selection_honors_requested_count_and_sections():
    analysis = {
        "scene_changes": [
            {"timestamp": 2, "score": 0.8},
            {"timestamp": 12, "score": 0.8},
            {"timestamp": 22, "score": 0.8},
        ],
        "audio_energy": [
            {"timestamp": 2, "energy": 0.4},
            {"timestamp": 12, "energy": 0.8},
            {"timestamp": 22, "energy": 0.6},
        ],
        "motion_activity": [{"timestamp": 2}, {"timestamp": 12}, {"timestamp": 22}],
    }
    transcript = [
        {"start": 0, "end": 10, "text": "主题介绍和背景"},
        {"start": 10, "end": 20, "text": "核心主题重点内容"},
        {"start": 20, "end": 30, "text": "主题总结"},
    ]
    candidates = _build_candidates(
        30, 10, "主题", transcript, analysis, {"semantic": 0.4, "visual": 0.3, "audio": 0.3}
    )
    selected = _select_segments(candidates, 30, 3, True, True, True)
    assert len(selected) == 3
    assert selected[0]["type"] == "intro"
    assert selected[-1]["type"] == "conclusion"


@pytest.mark.integration
def test_synthetic_video_end_to_end():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required")

    source = INPUT_DIR / "workflow_synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=15:duration=12",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=12",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    output = None
    try:
        result = process_multi_segment_video(
            {
                "video_path": str(source),
                "topic": "测试",
                "target_segments": 2,
                "total_duration": 10,
                "semantic_weight": 0.4,
                "visual_weight": 0.3,
                "audio_weight": 0.3,
                "include_intro": True,
                "include_highlights": True,
                "include_conclusion": True,
                "enable_content_analysis": False,
            }
        )
        output = OUTPUT_DIR / Path(result["output_video"]).name
        assert output.exists()
        assert result["analysis"]["selected_segments"] == 2
        assert result["analysis"]["selection_strategy"]["target_segments"] == 2
        assert result["analysis"]["total_output_duration"] <= 10.1
    finally:
        source.unlink(missing_ok=True)
        if output:
            output.unlink(missing_ok=True)
