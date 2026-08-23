"""Device selection must not require torch.

torch and torchaudio were hard requirements (~3 GB) but nothing imported them
unconditionally. The only uses were this CUDA check and a CLIP path whose
`clip` package was never in requirements.txt, so it could never activate.
CTranslate2 -- which faster-whisper already runs inference on -- answers the
same question.
"""

import builtins

import pytest

from core.whisper_asr import WhisperASRService


@pytest.fixture
def no_torch(monkeypatch):
    """Simulate an install without torch."""
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_detects_a_device_without_torch(no_torch):
    assert WhisperASRService._detect_device() in {"cpu", "cuda"}


def test_falls_back_to_cpu_when_nothing_is_available(monkeypatch, no_torch):
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"torch", "ctranslate2"}:
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert WhisperASRService._detect_device() == "cpu"


def test_explicit_device_is_respected():
    """A GPU user can still ask for cuda directly."""
    service = WhisperASRService(model_size="tiny", device="cuda")
    assert service.device == "cuda"


def test_auto_is_resolved_lazily_not_at_construction():
    """Constructing the service must not probe hardware or load a model."""
    service = WhisperASRService(model_size="tiny", device="auto")
    assert service.device == "auto"
    assert not service.model_loaded


class TestSimplifiedChinesePrompt:
    """Whisper transcribes Mandarin into Traditional characters by default.

    That is wrong for a Xiaohongshu tool twice over: the subtitles get burnt
    into the video, and every sentiment and topic dictionary in
    core/semantic_analysis.py is Simplified, so matching degrades.

    The fix is a two-character initial_prompt. It has to stay short: the
    prompt carries forward through condition_on_previous_text, and a longer
    one collapses segmentation (measured: "简体" -> 7 segments / 6.2s longest,
    "以下是普通话的句子。" -> 2 segments / 29.6s). Coarse segments make every
    scoring window see the same text.
    """

    def test_prompt_is_short(self):
        from core.whisper_asr import DEFAULT_SIMPLIFIED_PROMPT

        assert len(DEFAULT_SIMPLIFIED_PROMPT) <= 4, (
            "a longer prompt collapses Whisper's segmentation; see the table "
            "in core/whisper_asr.py"
        )

    def test_prompt_is_overridable(self, monkeypatch):
        from core.whisper_asr import simplified_chinese_prompt

        monkeypatch.setenv("ASR_INITIAL_PROMPT", "自定义")
        assert simplified_chinese_prompt() == "自定义"

    def _capture_params(self, monkeypatch, service, language):
        captured = {}

        class FakeModel:
            def transcribe(self, audio_path, language=None, task=None, **params):
                captured.update(params)
                captured["language"] = language

                class Info:
                    language = "zh"
                    language_probability = 1.0
                    duration = 1.0

                return iter(()), Info()

        service.model = FakeModel()
        service.model_loaded = True
        monkeypatch.setattr(service, "_detect_language", lambda path: "zh")
        service.transcribe_audio("dummy.wav", language=language)
        return captured

    def test_chinese_audio_gets_the_prompt(self, monkeypatch):
        from core.whisper_asr import DEFAULT_SIMPLIFIED_PROMPT, WhisperASRService

        service = WhisperASRService(model_size="tiny")
        params = self._capture_params(monkeypatch, service, "zh")
        assert params["initial_prompt"] == DEFAULT_SIMPLIFIED_PROMPT

    def test_english_audio_does_not_get_it(self, monkeypatch):
        from core.whisper_asr import WhisperASRService

        service = WhisperASRService(model_size="tiny")
        captured = {}

        class FakeModel:
            def transcribe(self, audio_path, language=None, task=None, **params):
                captured.update(params)

                class Info:
                    language = "en"
                    language_probability = 1.0
                    duration = 1.0

                return iter(()), Info()

        service.model = FakeModel()
        service.model_loaded = True
        monkeypatch.setattr(service, "_detect_language", lambda path: "en")
        service.transcribe_audio("dummy.wav")
        assert captured["initial_prompt"] is None

    def test_caller_supplied_prompt_wins(self, monkeypatch):
        from core.whisper_asr import WhisperASRService

        service = WhisperASRService(model_size="tiny")
        captured = {}

        class FakeModel:
            def transcribe(self, audio_path, language=None, task=None, **params):
                captured.update(params)

                class Info:
                    language = "zh"
                    language_probability = 1.0
                    duration = 1.0

                return iter(()), Info()

        service.model = FakeModel()
        service.model_loaded = True
        service.transcribe_audio("dummy.wav", language="zh", initial_prompt="我的提示")
        assert captured["initial_prompt"] == "我的提示"

    def test_detection_can_be_disabled(self, monkeypatch):
        from core.whisper_asr import WhisperASRService

        monkeypatch.setenv("ASR_LANGUAGE_DETECTION", "false")
        assert WhisperASRService(model_size="tiny")._detect_language("x.wav") is None
