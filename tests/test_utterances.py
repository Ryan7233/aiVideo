"""Rebuilding utterances from word timings.

Whisper's segment boundaries follow its decoding windows rather than speech.
On continuous narration it returned the whole 28-second clip as one segment,
so every candidate window in the clipper saw identical text and scored
identically -- which defeats the only thing the tool does.
"""

import pytest

from core.utterances import MAX_SECONDS, group_words, resegment


def words(*spans):
    return [{"word": text, "start": start, "end": end} for start, end, text in spans]


class TestGrouping:
    def test_a_pause_starts_a_new_utterance(self):
        grouped = group_words(words(
            (0.0, 0.5, "大家好"), (0.5, 1.0, "今天"),
            (2.0, 2.5, "第一个"), (2.5, 3.0, "重点"),
        ))
        assert len(grouped) == 2
        assert grouped[0]["text"].startswith("大家好")
        assert grouped[1]["start"] == 2.0

    def test_sentence_punctuation_splits(self):
        grouped = group_words(words(
            (0.0, 0.5, "这样。"), (0.55, 1.0, "然后"), (1.0, 1.5, "继续"),
            (1.5, 2.0, "讲"), (2.0, 2.6, "完了"),
        ))
        assert len(grouped) >= 2

    def test_a_long_run_is_capped(self):
        """A monologue without punctuation still has to yield usable pieces."""
        spans = [(i * 0.5, i * 0.5 + 0.5, "字") for i in range(80)]  # 40s, no gaps
        grouped = group_words(words(*spans))
        assert len(grouped) > 1
        assert max(g["end"] - g["start"] for g in grouped) <= MAX_SECONDS + 1

    def test_an_adjacent_sliver_is_folded_in(self):
        grouped = group_words(words(
            (0.0, 2.0, "完整的一句话"), (2.1, 2.2, "嗯"), (2.9, 5.0, "接着说下去"),
        ))
        # The filler abuts the first utterance, so it joins it rather than
        # becoming a "moment" of its own.
        assert len(grouped) == 2, grouped
        assert grouped[0]["text"].endswith("嗯")

    def test_a_sliver_across_a_pause_is_left_alone(self):
        """Merging across silence would join two unrelated fragments."""
        grouped = group_words(words(
            (0.0, 2.0, "完整的一句话"), (3.0, 3.1, "嗯"), (4.5, 6.5, "接着说下去"),
        ))
        assert len(grouped) == 3, grouped

    def test_timings_stay_ordered_and_within_the_source(self):
        grouped = group_words(words(
            (0.0, 0.5, "a"), (1.2, 1.8, "b"), (3.0, 3.9, "c"), (3.9, 4.4, "d"),
        ))
        for item in grouped:
            assert item["end"] > item["start"]
        for earlier, later in zip(grouped, grouped[1:]):
            assert later["start"] >= earlier["start"]


class TestResegment:
    def test_one_giant_segment_becomes_several(self):
        """The exact failure this exists for."""
        original = [{
            "start": 0.0, "end": 28.0, "text": "全部内容",
            "words": [{"word": f"字{i}", "start": i * 1.0, "end": i * 1.0 + 0.6}
                      for i in range(28)],
        }]
        assert len(resegment(original)) > 1

    def test_without_word_timings_the_input_is_returned(self):
        original = [{"start": 0.0, "end": 5.0, "text": "没有词级时间戳"}]
        assert resegment(original) == original

    def test_empty_input_is_safe(self):
        assert resegment([]) == []

    def test_the_transcript_text_survives(self):
        original = [{
            "start": 0.0, "end": 4.0, "text": "一二三四",
            "words": words((0.0, 1.0, "一"), (1.0, 2.0, "二"),
                           (2.8, 3.0, "三"), (3.0, 4.0, "四")),
        }]
        joined = "".join(item["text"] for item in resegment(original)).replace(" ", "")
        assert joined == "一二三四"


@pytest.mark.integration
def test_real_speech_is_split_into_several_utterances(tmp_path):
    """Against real audio, not synthetic timings."""
    import shutil
    import subprocess

    from core.runtime import INPUT_DIR, MODEL_DIR

    if not shutil.which("say") or not shutil.which("ffmpeg"):
        pytest.skip("needs macOS `say` and FFmpeg")
    if not ((MODEL_DIR / "whisper").is_dir() and any((MODEL_DIR / "whisper").iterdir())):
        pytest.skip("no cached Whisper model")

    script = tmp_path / "s.txt"
    script.write_text("大家好，今天聊聊长视频怎么剪。第一个重点是选段。第二个重点是时间码。",
                      encoding="utf-8")
    aiff, wav = tmp_path / "s.aiff", tmp_path / "s.wav"
    if subprocess.run(["say", "-v", "Tingting", "-f", str(script), "-o", str(aiff)],
                      capture_output=True).returncode != 0:
        pytest.skip("no Chinese TTS voice")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
                    "-ar", "16000", "-ac", "1", str(wav)], check=True, capture_output=True)

    video = INPUT_DIR / "utterance_probe.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15:duration=14",
                    "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(video)], check=True, capture_output=True)
    try:
        from core.whisper_asr import get_asr_service

        result = get_asr_service(model_size="tiny").transcribe_video(str(video))
        assert result["segment_count"] >= 2, result["segments"]
        assert result["segment_count"] >= result.get("raw_segment_count", 0)
    finally:
        video.unlink(missing_ok=True)
