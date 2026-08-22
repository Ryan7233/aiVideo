"""LLM-first window scoring, with the dictionary method as the fallback.

The rule-based scorer cannot tell that a fluent, on-topic sentence says
nothing, so an LLM takes the primary path when one is configured. These tests
pin down the contract that matters: the LLM never becomes a single point of
failure for a clipping run.
"""

import asyncio
import json

import pytest

from core import semantic_scoring
from core.semantic_scoring import SemanticScorer, WindowScore, rule_score, topic_terms


WINDOWS = [
    {"text": "今天我们来聊聊小红书的内容运营方法，这个方法非常有效。", "duration": 10},
    {"text": "嗯 那个 就是 呃 我 我 不 不知道 啊", "duration": 10},
    {"text": "这家咖啡店的拿铁真的太好喝了，强烈推荐给大家！", "duration": 10},
]


class FakeLLM:
    """Stands in for a configured provider."""

    def __init__(self, responses, configured=True):
        self._responses = list(responses)
        self._configured = configured
        self.prompts = []

    def is_configured(self):
        return self._configured

    async def complete(self, prompt):
        self.prompts.append(prompt)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("SEMANTIC_SCORING_MODE", raising=False)
    monkeypatch.delenv("SEMANTIC_SCORING_BATCH", raising=False)
    semantic_scoring.reset_scorer()
    yield
    semantic_scoring.reset_scorer()


def use_llm(monkeypatch, fake):
    monkeypatch.setattr("core.llm_service.get_llm_service", lambda: fake)


class TestTopicTerms:
    def test_chinese_topic_is_segmented(self):
        assert "咖啡店" in topic_terms("咖啡店探店")

    def test_empty_topic(self):
        assert topic_terms("") == []


class TestRuleScore:
    def test_filler_scores_below_real_content(self):
        real = rule_score(WINDOWS[0]["text"], "小红书运营", 10).score
        filler = rule_score(WINDOWS[1]["text"], "小红书运营", 10).score
        assert real > filler

    def test_empty_text_is_zero(self):
        assert rule_score("", "topic", 10).score == 0.0

    def test_source_is_labelled(self):
        assert rule_score("有内容的一句话测试", "", 10).source == "rules"


class TestLlmPath:
    def test_scores_come_from_the_model(self, monkeypatch):
        fake = FakeLLM([json.dumps([
            {"index": 0, "score": 0.9, "reason": "有具体方法"},
            {"index": 1, "score": 0.05, "reason": "纯语气词"},
            {"index": 2, "score": 0.75, "reason": "有情绪有推荐"},
        ])])
        use_llm(monkeypatch, fake)

        results = SemanticScorer().score_windows(WINDOWS, "小红书运营")

        assert [r.source for r in results] == ["llm"] * 3
        assert results[0].score == 0.9
        assert results[1].score == 0.05
        assert results[1].reason == "纯语气词"

    def test_fenced_json_is_parsed(self, monkeypatch):
        fake = FakeLLM(['```json\n[{"index": 0, "score": 0.6}]\n```'])
        use_llm(monkeypatch, fake)

        result = SemanticScorer().score_windows(WINDOWS[:1], "主题")[0]

        assert result.source == "llm"
        assert result.score == 0.6

    def test_prose_around_the_array_is_tolerated(self, monkeypatch):
        fake = FakeLLM(['这是结果：[{"index": 0, "score": 0.4}] 希望有帮助'])
        use_llm(monkeypatch, fake)

        assert SemanticScorer().score_windows(WINDOWS[:1], "主题")[0].score == 0.4

    def test_out_of_range_scores_are_clamped(self, monkeypatch):
        fake = FakeLLM([json.dumps([
            {"index": 0, "score": 7.5}, {"index": 1, "score": -3},
        ])])
        use_llm(monkeypatch, fake)

        results = SemanticScorer().score_windows(WINDOWS[:2], "主题")

        assert results[0].score == 1.0
        assert results[1].score == 0.0

    def test_windows_are_batched(self, monkeypatch):
        many = [{"text": f"第{i}段有意义的内容测试", "duration": 5} for i in range(25)]
        batches = [
            json.dumps([{"index": i, "score": 0.5} for i in range(10)]),
            json.dumps([{"index": i, "score": 0.5} for i in range(10, 20)]),
            json.dumps([{"index": i, "score": 0.5} for i in range(20, 25)]),
        ]
        fake = FakeLLM(batches)
        use_llm(monkeypatch, fake)

        results = SemanticScorer(batch_size=10).score_windows(many, "主题")

        assert len(fake.prompts) == 3
        assert all(r.source == "llm" for r in results)

    def test_empty_windows_are_not_sent_to_the_model(self, monkeypatch):
        fake = FakeLLM([json.dumps([{"index": 1, "score": 0.8}])])
        use_llm(monkeypatch, fake)

        results = SemanticScorer().score_windows(
            [{"text": "", "duration": 5}, {"text": "一段真实的内容", "duration": 5}], "主题"
        )

        prompt = fake.prompts[0]
        assert "1. 「一段真实的内容」" in prompt
        assert "0. 「" not in prompt, "an empty window was sent to the model"
        assert results[0].score == 0.0 and results[0].source == "rules"
        assert results[1].source == "llm"

    def test_topic_appears_in_the_prompt(self, monkeypatch):
        fake = FakeLLM([json.dumps([{"index": 0, "score": 0.5}])])
        use_llm(monkeypatch, fake)

        SemanticScorer().score_windows(WINDOWS[:1], "咖啡店探店")

        assert "咖啡店探店" in fake.prompts[0]


class TestFallback:
    def test_missing_indices_fall_back_individually(self, monkeypatch):
        """A partial answer must not lose the windows the model skipped."""
        fake = FakeLLM([json.dumps([{"index": 0, "score": 0.9}])])
        use_llm(monkeypatch, fake)

        results = SemanticScorer().score_windows(WINDOWS, "主题")

        assert results[0].source == "llm"
        assert [r.source for r in results[1:]] == ["rules", "rules"]
        assert all(r.score >= 0 for r in results)

    def test_unparseable_response_falls_back_entirely(self, monkeypatch):
        use_llm(monkeypatch, FakeLLM(["抱歉，我无法完成这个请求。"]))

        results = SemanticScorer().score_windows(WINDOWS, "主题")

        assert [r.source for r in results] == ["rules"] * 3

    def test_provider_error_falls_back(self, monkeypatch):
        use_llm(monkeypatch, FakeLLM([RuntimeError("429 rate limited")]))

        results = SemanticScorer().score_windows(WINDOWS, "主题")

        assert [r.source for r in results] == ["rules"] * 3

    def test_unconfigured_provider_uses_rules(self, monkeypatch):
        use_llm(monkeypatch, FakeLLM([], configured=False))

        results = SemanticScorer().score_windows(WINDOWS, "主题")

        assert [r.source for r in results] == ["rules"] * 3

    def test_rules_mode_never_calls_the_model(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_SCORING_MODE", "rules")
        fake = FakeLLM([json.dumps([{"index": 0, "score": 0.9}])])
        use_llm(monkeypatch, fake)

        results = SemanticScorer().score_windows(WINDOWS, "主题")

        assert fake.prompts == []
        assert [r.source for r in results] == ["rules"] * 3

    def test_llm_mode_surfaces_the_error(self, monkeypatch):
        """Opt in to strict mode and a provider outage fails loudly."""
        monkeypatch.setenv("SEMANTIC_SCORING_MODE", "llm")
        use_llm(monkeypatch, FakeLLM([RuntimeError("provider down")]))

        with pytest.raises(RuntimeError, match="provider down"):
            SemanticScorer().score_windows(WINDOWS, "主题")


class TestAsyncBridge:
    def test_works_with_no_running_loop(self, monkeypatch):
        use_llm(monkeypatch, FakeLLM([json.dumps([{"index": 0, "score": 0.5}])]))
        assert SemanticScorer().score_windows(WINDOWS[:1], "主题")[0].source == "llm"

    def test_works_from_inside_a_running_loop(self, monkeypatch):
        """The API scores inside a worker thread, but a loop may already exist."""
        use_llm(monkeypatch, FakeLLM([json.dumps([{"index": 0, "score": 0.5}])]))

        async def driver():
            return await asyncio.to_thread(
                SemanticScorer().score_windows, WINDOWS[:1], "主题"
            )

        assert asyncio.run(driver())[0].source == "llm"


def test_scores_reach_the_candidate_list(monkeypatch):
    """The workflow must actually surface the model's score and reason."""
    from core.video_workflow import _build_candidates

    fake = FakeLLM([json.dumps([
        {"index": i, "score": 0.9, "reason": "有信息量"} for i in range(10)
    ])])
    use_llm(monkeypatch, fake)

    candidates = _build_candidates(
        30, 10, "主题",
        [{"start": 0, "end": 30, "text": "一段有内容的转写文本用于测试"}],
        {"scene_changes": [], "audio_energy": [], "motion_activity": []},
        {"semantic": 0.4, "visual": 0.3, "audio": 0.3},
    )

    assert candidates
    assert candidates[0]["semantic_score"] == 0.9
    assert candidates[0]["semantic_details"]["source"] == "llm"
    assert candidates[0]["semantic_details"]["reason"] == "有信息量"


class TestWhyTheLlmPathExists:
    """The specific discrimination the dictionary method cannot make."""

    SUBSTANTIVE = "小红书的笔记发布时间选在晚上七点到九点，打开率能比中午高三成左右。"
    FLUENT_BUT_EMPTY = "小红书这个平台真的非常好，内容特别棒，大家一定要好好做，做得好就会很好。"

    def test_rules_cannot_tell_substance_from_fluency(self):
        """Documents the known limit: sentiment and keyword hits win.

        The empty sentence is longer, hits more topic keywords and carries more
        sentiment words, so the dictionary method ranks it higher. This is not
        fixable by tuning the weights -- it needs to read the meaning.
        """
        substantive = rule_score(self.SUBSTANTIVE, "小红书运营", 10).score
        empty = rule_score(self.FLUENT_BUT_EMPTY, "小红书运营", 10).score

        assert empty > substantive, (
            "if this ever fails the dictionary method improved; "
            "re-check whether the LLM path is still needed for this case"
        )

    def test_llm_path_ranks_substance_higher(self, monkeypatch):
        fake = FakeLLM([json.dumps([
            {"index": 0, "score": 0.85, "reason": "给出具体数据"},
            {"index": 1, "score": 0.2, "reason": "没有实际信息"},
        ])])
        use_llm(monkeypatch, fake)

        results = SemanticScorer().score_windows(
            [{"text": self.SUBSTANTIVE, "duration": 10},
             {"text": self.FLUENT_BUT_EMPTY, "duration": 10}],
            "小红书运营",
        )

        assert results[0].score > results[1].score
        assert all(r.source == "llm" for r in results)
