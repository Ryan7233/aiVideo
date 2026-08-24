"""Real Whisper on real speech.

Everything else stubs the model. This runs it, because two defects only showed
up against real audio: Mandarin came back in Traditional characters, and the
fix for that (an initial_prompt) silently collapsed segmentation from 12
segments to 2 spanning 24 seconds each.

Speech is synthesised with macOS `say`, so no audio fixture is committed. The
test skips where `say`, FFmpeg, or a cached model is unavailable; set
ASR_INTEGRATION_TESTS=1 to allow downloading the tiny model.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from core.runtime import INPUT_DIR, MODEL_DIR

pytestmark = pytest.mark.integration

SCRIPT = (
    "大家好，今天来聊聊小红书的内容运营。"
    "第一个重点是发布时间。笔记放在晚上七点到九点，打开率能比中午高三成。"
    "第二个重点是封面。封面上的字不要超过十二个。"
)

# Characters that exist only in Traditional Chinese. If these show up, Whisper
# transcribed Mandarin into the wrong script.
TRADITIONAL_ONLY = set("這個內容運營發時間點鐘臺灣體讚樹術書總結學習實現準備")

# Below this the clip is too short to say anything about segmentation.
MIN_SPEECH_SECONDS = 5.0


def _requirements_met():
    if not shutil.which("say") or not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return False, "needs macOS `say` and FFmpeg to synthesise speech"
    model_cached = (MODEL_DIR / "whisper").is_dir() and any((MODEL_DIR / "whisper").iterdir())
    if not model_cached and os.getenv("ASR_INTEGRATION_TESTS", "") not in {"1", "true"}:
        return False, "no cached Whisper model; set ASR_INTEGRATION_TESTS=1 to download"
    return True, ""


def _media_duration(path) -> float:
    """Duration in seconds, or 0.0 if the file is unreadable or empty.

    `say` exits 0 even for a voice that does not exist, writing a header-only
    AIFF; ffprobe then reports N/A. Passing that on to FFmpeg made the whole
    fixture explode instead of skipping, so every step is measured.
    """
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if completed.returncode != 0:
            return 0.0
        return float(completed.stdout.strip())
    except (ValueError, OSError, subprocess.SubprocessError):
        return 0.0


def _synthesise(voice: str, script_path, out_path) -> float:
    """Speak the script to a file and report how long the audio actually is."""
    result = subprocess.run(
        ["say", "-v", voice, "-f", str(script_path), "-o", str(out_path)],
        capture_output=True,
    )
    if result.returncode != 0 or not Path(out_path).is_file():
        return 0.0
    return _media_duration(out_path)


@pytest.fixture(scope="module")
def spoken_video(tmp_path_factory):
    ok, reason = _requirements_met()
    if not ok:
        pytest.skip(reason)

    tmp = tmp_path_factory.mktemp("asr")
    script = tmp / "script.txt"
    script.write_text(SCRIPT, encoding="utf-8")
    aiff, wav = tmp / "speech.aiff", tmp / "speech.wav"

    # Pick a voice by whether it produces real audio, not by exit code.
    spoken = 0.0
    for candidate in ("Tingting", "Meijia", "Sinji", "Eddy", "Flo", "Li-mu"):
        spoken = _synthesise(candidate, script, aiff)
        if spoken >= MIN_SPEECH_SECONDS:
            break
    if spoken < MIN_SPEECH_SECONDS:
        pytest.skip(
            "no Chinese TTS voice produced usable audio "
            f"(best was {spoken:.2f}s, need {MIN_SPEECH_SECONDS}s)"
        )

    converted = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
         "-ar", "16000", "-ac", "1", str(wav)],
        capture_output=True,
    )
    duration = _media_duration(wav) if converted.returncode == 0 else 0.0
    if duration < MIN_SPEECH_SECONDS:
        pytest.skip(f"speech conversion produced {duration:.2f}s of audio")

    video = INPUT_DIR / "asr_integration.mp4"
    built = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=15:duration={duration:.3f}",
         "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(video)],
        capture_output=True,
    )
    if built.returncode != 0 or _media_duration(video) < MIN_SPEECH_SECONDS:
        video.unlink(missing_ok=True)
        pytest.skip("could not build a speech video fixture")

    try:
        yield video, duration
    finally:
        video.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def transcript(spoken_video):
    from core.whisper_asr import get_asr_service

    video, _ = spoken_video
    return get_asr_service(model_size="tiny").transcribe_video(str(video), cleanup_audio=True)


def test_language_is_detected_as_chinese(transcript):
    assert transcript["language"] == "zh"


def test_output_is_simplified_not_traditional(transcript):
    text = "".join(segment["text"] for segment in transcript["segments"])
    offenders = sorted({char for char in text if char in TRADITIONAL_ONLY})
    # A couple of stray characters are within tiny-model noise; a script-wide
    # flip produces many.
    assert len(offenders) <= 3, f"transcribed into Traditional Chinese: {offenders}"


def test_segmentation_is_not_collapsed(transcript):
    """One segment covering the clip makes every scoring window identical.

    Boundaries now come from word timestamps rather than Whisper's decoding
    windows, so the exact count varies with the speech; what must hold is that
    the transcript is split at all and no piece swallows the whole clip.
    """
    segments = transcript["segments"]
    assert len(segments) >= 2, f"only {len(segments)} segment(s); segmentation collapsed"
    longest = max(segment["end"] - segment["start"] for segment in segments)
    assert longest <= 15, f"longest segment is {longest:.1f}s; segmentation collapsed"


def test_content_words_survive(transcript):
    text = "".join(segment["text"] for segment in transcript["segments"])
    assert "内容" in text or "运营" in text


def test_windows_get_distinct_text(spoken_video, transcript):
    """Coarse segments make every scoring window see the same words."""
    from core.video_workflow import _build_candidates

    _, duration = spoken_video
    candidates = _build_candidates(
        duration, 8.0, "小红书运营",
        [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in transcript["segments"]],
        {"scene_changes": [], "audio_energy": [], "motion_activity": []},
        {"semantic": 0.5, "visual": 0.2, "audio": 0.3},
    )
    texts = [c["text"] for c in candidates if c["text"]]
    assert len(set(texts)) > 1, "every window received identical transcript text"
