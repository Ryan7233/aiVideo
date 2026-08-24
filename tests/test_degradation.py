"""A fallback result must say it is one.

Seven separate incidents on this branch shared one shape: something threw, a
broad `except` returned a substitute of the same shape, and the caller
reported success. Three NameErrors, a cover that never wrote a file, audio
analysis with every feature missing, a collage that raised on every layout,
and a frontend that quietly drew its own canvas while showing "生成完成".

Carrying on with less is often right. Doing it invisibly is not.
"""

import ast
import pathlib
import re

from core.degradation import DEGRADED, DEGRADED_REASON, is_degraded, mark_degraded


class TestHelper:
    def test_it_marks_and_returns_the_payload(self):
        payload = mark_degraded({"a": 1}, ValueError("boom"))
        assert payload["a"] == 1
        assert payload[DEGRADED] is True
        assert "ValueError: boom" == payload[DEGRADED_REASON]

    def test_a_plain_result_is_not_degraded(self):
        assert not is_degraded({"a": 1})
        assert not is_degraded(None)
        assert not is_degraded("text")

    def test_it_never_raises_inside_an_error_handler(self):
        assert mark_degraded(None, ValueError("x")) is None
        assert mark_degraded([1, 2], ValueError("x")) == [1, 2]

    def test_the_reason_is_truncated(self):
        payload = mark_degraded({}, "x" * 5000)
        assert len(payload[DEGRADED_REASON]) <= 500


class TestRealFallbacks:
    """Force each path to fail and check the result admits it."""

    def test_video_analysis(self, monkeypatch):
        from core.smart_clipping import SmartClippingEngine

        engine = SmartClippingEngine()
        monkeypatch.setattr(engine, "_get_video_duration", lambda path: 0)
        assert is_degraded(engine.analyze_video_content("nope.mp4"))





class TestConvention:
    """New fallbacks should use the shared marker, not invent another key."""

    def test_no_new_ad_hoc_fallback_keys(self):
        allowed = {"core/smart_clipping.py", "core/llm_service.py", "core/personalized_writing.py"}
        offenders = []
        for path in sorted(pathlib.Path("core").glob("*.py")):
            if str(path) in allowed or path.name == "degradation.py":
                continue
            text = path.read_text(encoding="utf-8")
            for key in ("'fallback': True", '"fallback": True', "'is_fallback'", '"is_fallback"'):
                if key in text:
                    offenders.append(f"{path}: {key}")
        assert not offenders, f"use core.degradation.mark_degraded instead: {offenders}"

    def test_payload_fallbacks_are_marked(self):
        """Every except that returns a full payload must mark it.

        A three-key dict or a *_fallback()/_default()/_basic() call returned
        from a broad except is a substitute result, not a sentinel.
        """
        fallbackish = re.compile(r"fallback|default|basic", re.I)
        unmarked = []
        for path in sorted(pathlib.Path("core").glob("*.py")):
            if path.name == "degradation.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                kind = node.type
                broad = kind is None or (
                    isinstance(kind, ast.Name) and kind.id in {"Exception", "BaseException"}
                )
                if not broad:
                    continue
                for ret in ast.walk(node):
                    if not isinstance(ret, ast.Return) or ret.value is None:
                        continue
                    rendered = ast.unparse(ret.value)
                    if "mark_degraded" in rendered or DEGRADED in rendered:
                        continue
                    payload = (
                        isinstance(ret.value, ast.Dict) and len(ret.value.keys) >= 3
                    ) or (
                        isinstance(ret.value, ast.Call)
                        and fallbackish.search(ast.unparse(ret.value.func))
                    )
                    if payload:
                        unmarked.append(f"{path}:{ret.lineno}")
        assert not unmarked, (
            "these return a substitute payload without marking it; wrap with "
            f"core.degradation.mark_degraded: {unmarked}"
        )
