"""Integration tests for the FFmpeg measurement pass.

The previous implementation scraped scene and motion numbers out of FFmpeg's
stderr with regexes. On FFmpeg 7.x none of them matched, so the engine returned
empty scene/motion lists and the visual score silently collapsed to zero for
every candidate. Nothing failed and nothing logged an error. These tests assert
the measurements are actually non-empty and actually vary.
"""

import shutil
import subprocess

import pytest

from core.runtime import INPUT_DIR
from core.smart_clipping import SmartClippingEngine


pytestmark = pytest.mark.integration


def _require_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required")


def _synthesize(name: str, seconds: int, with_audio: bool):
    path = INPUT_DIR / name
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=15:duration={seconds}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=16000:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd.append(str(path))
    subprocess.run(cmd, check=True, capture_output=True)
    return path


@pytest.fixture(scope="module")
def analysis():
    _require_ffmpeg()
    source = _synthesize("measurement_probe.mp4", 20, with_audio=True)
    try:
        yield SmartClippingEngine().analyze_video_content(str(source))
    finally:
        source.unlink(missing_ok=True)


def test_analysis_is_not_a_fallback(analysis):
    assert not analysis.get("fallback"), "measurement pass fell back to empty results"
    assert analysis["duration"] == pytest.approx(20, abs=1)


def test_scene_changes_are_detected(analysis):
    assert analysis["scene_changes"], "scene detection returned nothing"
    for scene in analysis["scene_changes"]:
        assert 0 <= scene["timestamp"] <= analysis["duration"] + 1
        assert scene["score"] > 0


def test_motion_is_measured_not_counted(analysis):
    motion = analysis["motion_activity"]
    assert motion, "motion analysis returned nothing"
    activities = {round(point["activity"], 4) for point in motion}
    # The old implementation hardcoded activity = 1.0 for every sample.
    assert len(activities) > 1, "motion activity is constant; it is not a measurement"
    assert all(0.0 <= point["activity"] <= 1.0 for point in motion)


def test_audio_energy_is_measured(analysis):
    energy = analysis["audio_energy"]
    assert energy
    assert all(0.0 <= point["energy"] <= 1.0 for point in energy)


def test_video_without_audio_track_still_analyses():
    _require_ffmpeg()
    source = _synthesize("measurement_silent.mp4", 8, with_audio=False)
    try:
        result = SmartClippingEngine().analyze_video_content(str(source))
        assert not result.get("fallback")
        assert result["scene_changes"]
        assert result["motion_activity"]
        assert result["audio_energy"] == []
    finally:
        source.unlink(missing_ok=True)


def test_analysis_duration_is_capped():
    _require_ffmpeg()
    source = _synthesize("measurement_capped.mp4", 12, with_audio=False)
    try:
        result = SmartClippingEngine().analyze_video_content(str(source), max_duration=5)
        assert result["analysis_duration"] == 5
        assert max(p["timestamp"] for p in result["motion_activity"]) <= 6
    finally:
        source.unlink(missing_ok=True)
