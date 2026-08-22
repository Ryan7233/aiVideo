r"""Regression tests for Chinese text scoring.

These exist because the scoring used to tokenize with ``\b\w+\b`` and count
words with ``str.split()``. A whole Chinese sentence came back as one token, so
filler speech outscored real content and every dictionary lookup returned zero.
"""

import pytest

from core.semantic_analysis import SemanticAnalyzer
from core.text_tokenizer import scan_lexicon, split_sentences, tokenize


REAL_CONTENT = "今天我们来聊聊小红书的内容运营方法，这个方法非常有效，能帮你快速涨粉。"
FILLER = "嗯 那个 就是 呃 我 我 不 不知道 啊"
PRAISE = "这家咖啡店的拿铁真的太好喝了，环境也很棒，强烈推荐给大家！"
COMPLAINT = "这次体验很糟糕，服务态度差，非常失望，不推荐。"


@pytest.fixture(scope="module")
def analyzer():
    return SemanticAnalyzer()


class TestTokenizer:
    def test_chinese_splits_into_multiple_tokens(self):
        tokens = tokenize("今天我们来聊聊小红书的内容运营方法")
        assert len(tokens) > 5
        assert "".join(tokens) == "今天我们来聊聊小红书的内容运营方法"

    def test_mixed_script_keeps_latin_words_whole(self):
        assert "machine" in tokenize("用 machine learning 做视频剪辑")

    def test_empty_input(self):
        assert tokenize("") == []
        assert tokenize("   ") == []

    def test_longest_lexicon_match_wins(self):
        # 不好 must consume the 好 inside it rather than both firing.
        assert scan_lexicon("这个不好", {"好": 0.6, "不好": -0.6}) == [("不好", -0.6)]

    def test_latin_lexicon_respects_word_boundaries(self):
        assert scan_lexicon("I do not like it", {"no": -0.7}) == []

    def test_sentences_drop_empty_pieces(self):
        assert split_sentences("第一句。第二句！") == ["第一句", "第二句"]


class TestKeywords:
    def test_chinese_keywords_score_above_zero(self, analyzer):
        keywords = analyzer.extract_keywords(REAL_CONTENT, top_k=5)
        assert keywords
        assert all(item["score"] > 0 for item in keywords)

    def test_repeated_term_outranks_incidental_ones(self, analyzer):
        keywords = analyzer.extract_keywords(REAL_CONTENT, top_k=5)
        assert keywords[0]["word"] == "方法"

    def test_stop_words_and_filler_are_dropped(self, analyzer):
        words = {item["word"] for item in analyzer.extract_keywords(REAL_CONTENT, top_k=20)}
        assert not words & {"的", "这个", "我们"}


class TestSentiment:
    def test_chinese_praise_reads_positive(self, analyzer):
        result = analyzer.analyze_sentiment(PRAISE)
        assert result["positive"] > 0
        assert result["dominant"] == "positive"

    def test_chinese_complaint_reads_negative(self, analyzer):
        result = analyzer.analyze_sentiment(COMPLAINT)
        assert result["negative"] > 0
        assert result["dominant"] == "negative"

    def test_negation_is_not_scored_as_praise(self, analyzer):
        result = analyzer.analyze_sentiment("这个不好")
        assert result["positive"] == 0
        assert result["negative"] > 0

    def test_neutral_text_reports_neutral(self, analyzer):
        assert analyzer.analyze_sentiment("今天天气还行")["dominant"] == "neutral"


class TestTopicRelevance:
    def test_on_topic_chinese_text_scores_high(self, analyzer):
        scores = analyzer.analyze_topic_relevance("深度学习算法和人工智能技术")
        assert scores["technology"] >= 0.9

    def test_food_content_is_recognised(self, analyzer):
        assert analyzer.analyze_topic_relevance(PRAISE)["food"] > 0

    def test_off_topic_text_scores_zero(self, analyzer):
        assert max(analyzer.analyze_topic_relevance("嗯 那个 就是").values()) == 0


class TestQualityScore:
    def test_real_content_outscores_filler(self, analyzer):
        """The invariant the old implementation inverted."""
        real = analyzer.calculate_content_quality_score(REAL_CONTENT, 10)["overall_score"]
        filler = analyzer.calculate_content_quality_score(FILLER, 10)["overall_score"]
        assert real > filler

    def test_praise_outscores_filler(self, analyzer):
        praise = analyzer.calculate_content_quality_score(PRAISE, 10)["overall_score"]
        filler = analyzer.calculate_content_quality_score(FILLER, 10)["overall_score"]
        assert praise > filler

    def test_vocabulary_diversity_is_not_constant(self, analyzer):
        rich = analyzer.calculate_content_quality_score(REAL_CONTENT, 10)["vocabulary_diversity"]
        poor = analyzer.calculate_content_quality_score(FILLER, 10)["vocabulary_diversity"]
        assert rich > poor
        assert poor < 1.0

    def test_chinese_word_count_is_not_one(self, analyzer):
        assert analyzer.calculate_content_quality_score(REAL_CONTENT, 10)["word_count"] > 5

    def test_emotional_intensity_fires_on_chinese(self, analyzer):
        assert analyzer.calculate_content_quality_score(PRAISE, 10)["emotional_intensity"] > 0

    def test_scores_stay_in_range(self, analyzer):
        for text in (REAL_CONTENT, FILLER, PRAISE, COMPLAINT, "", "a"):
            result = analyzer.calculate_content_quality_score(text, 10)
            for key, value in result.items():
                if isinstance(value, float):
                    assert 0.0 <= value <= 1.0, f"{key}={value} for {text!r}"

    def test_empty_text_is_safe(self, analyzer):
        assert analyzer.calculate_content_quality_score("", 10)["overall_score"] == 0.0
