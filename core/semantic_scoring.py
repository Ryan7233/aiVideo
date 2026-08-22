"""Scoring how much a candidate window is worth clipping.

The rule-based path counts words, matches sentiment and topic dictionaries and
measures information density. It is cheap and offline, but it cannot tell that
a fluent, on-topic sentence says nothing. An LLM can, so when one is configured
it scores the windows and the dictionary method becomes the fallback.

Windows are scored in batches -- a single request covers many candidates -- and
any window the model does not return a usable score for falls back individually
rather than failing the run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core.semantic_analysis import get_semantic_analyzer
from core.text_tokenizer import tokenize

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_CHARS = 240


@dataclass
class WindowScore:
    """One window's score, and where the number came from."""

    index: int
    score: float
    reason: str = ""
    source: str = "rules"
    details: Dict[str, Any] = field(default_factory=dict)


def scoring_mode() -> str:
    """``auto`` (LLM when configured), ``llm`` (require it) or ``rules``."""
    mode = os.getenv("SEMANTIC_SCORING_MODE", "auto").strip().lower()
    return mode if mode in {"auto", "llm", "rules"} else "auto"


def _batch_size() -> int:
    try:
        return max(1, min(int(os.getenv("SEMANTIC_SCORING_BATCH", DEFAULT_BATCH_SIZE)), 50))
    except ValueError:
        return DEFAULT_BATCH_SIZE


def topic_terms(topic: str) -> List[str]:
    """Segment a topic so a multi-word Chinese topic can match partially."""
    terms = {term for term in tokenize(topic) if len(term) >= 2}
    for part in re.split(r"[\s,，、/|]+", topic or ""):
        part = part.strip().lower()
        if len(part) >= 2:
            terms.add(part)
    return sorted(terms) or ([topic.strip().lower()] if topic and topic.strip() else [])


def rule_score(text: str, topic: str, duration: float) -> WindowScore:
    """Dictionary and density based score. Always available, never raises."""
    if not text or not text.strip():
        return WindowScore(index=-1, score=0.0, source="rules",
                           details={"quality": 0.0, "topic_match": 0.0})

    analyzer = get_semantic_analyzer()
    quality = analyzer.calculate_content_quality_score(text, duration)
    terms = topic_terms(topic)
    lowered = text.lower()
    matched = [term for term in terms if term in lowered]
    topic_match = len(matched) / len(terms) if terms else quality.get("topic_relevance", 0.0)
    score = min(1.0, quality.get("overall_score", 0.0) * 0.7 + topic_match * 0.3)
    return WindowScore(
        index=-1,
        score=score,
        source="rules",
        details={
            "quality": quality.get("overall_score", 0.0),
            "topic_match": topic_match,
            "matched_terms": matched,
        },
    )


def _run_async(coro):
    """Await ``coro`` from sync code, whether or not a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Called from inside a loop: hand the coroutine to a thread with its own.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _extract_json_array(response: str) -> Optional[list]:
    """Pull a JSON array out of a model response, fenced or not."""
    if not response:
        return None
    text = response.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("["):
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


class SemanticScorer:
    """Scores candidate windows, preferring an LLM and falling back to rules."""

    def __init__(self, batch_size: Optional[int] = None, max_chars: int = DEFAULT_MAX_CHARS):
        self.batch_size = batch_size or _batch_size()
        self.max_chars = max_chars

    # -- prompt ---------------------------------------------------------

    def _build_prompt(self, topic: str, windows: Sequence[Dict[str, Any]]) -> str:
        listing = "\n".join(
            f'{item["index"]}. 「{item["text"][: self.max_chars]}」'
            for item in windows
        )
        topic_line = f"视频主题：{topic}\n" if topic and topic.strip() else ""
        return f"""你是短视频剪辑师。下面是同一个视频里若干候选片段的语音转写文本。
{topic_line}
请为每个片段打分，判断它**是否值得剪进一条短视频**。评分标准：

- 0.8-1.0：信息量大、有明确观点或结论、有情绪张力，单独拿出来也能看懂
- 0.5-0.7：有内容但比较平淡，或依赖上下文
- 0.2-0.4：过渡句、重复内容、信息量低
- 0.0-0.1：纯口水话、语气词、无意义内容

{"与主题相关的片段应当加分。" if topic_line else ""}

候选片段：
{listing}

只输出 JSON 数组，不要输出任何其他文字。每项包含 index、score（0 到 1 的小数）、
reason（不超过 20 字的中文理由）：
[{{"index": 0, "score": 0.8, "reason": "给出了具体方法"}}]"""

    # -- llm path -------------------------------------------------------

    def _score_batch_with_llm(
        self, topic: str, windows: Sequence[Dict[str, Any]]
    ) -> Dict[int, WindowScore]:
        from core.llm_service import get_llm_service

        service = get_llm_service()
        response = _run_async(service.complete(self._build_prompt(topic, windows)))
        parsed = _extract_json_array(response)
        if parsed is None:
            raise ValueError(f"模型未返回可解析的 JSON 数组: {str(response)[:200]}")

        valid_indices = {item["index"] for item in windows}
        scored: Dict[int, WindowScore] = {}
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            try:
                index = int(entry["index"])
                score = float(entry["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if index not in valid_indices:
                continue
            scored[index] = WindowScore(
                index=index,
                score=max(0.0, min(1.0, score)),
                reason=str(entry.get("reason", ""))[:60],
                source="llm",
            )
        return scored

    # -- public ---------------------------------------------------------

    def score_windows(self, windows: Sequence[Dict[str, Any]], topic: str) -> List[WindowScore]:
        """Score every window. Returns one entry per input, in input order.

        ``windows`` items need ``text`` and ``duration``; the index is the
        position in the sequence.
        """
        indexed = [
            {"index": position, "text": (window.get("text") or "").strip(),
             "duration": float(window.get("duration") or 0.0)}
            for position, window in enumerate(windows)
        ]

        results: Dict[int, WindowScore] = {}
        mode = scoring_mode()
        # Empty windows never reach the model; there is nothing to judge.
        candidates = [item for item in indexed if item["text"]]

        if candidates and mode != "rules":
            try:
                results = self._score_with_llm(topic, candidates)
            except Exception as exc:
                if mode == "llm":
                    raise
                logger.warning("LLM 语义评分不可用，回退到规则评分: %s", exc)
                results = {}

        for item in indexed:
            if item["index"] in results:
                continue
            fallback = rule_score(item["text"], topic, item["duration"])
            fallback.index = item["index"]
            results[item["index"]] = fallback

        return [results[position] for position in range(len(indexed))]

    def _score_with_llm(
        self, topic: str, candidates: Sequence[Dict[str, Any]]
    ) -> Dict[int, WindowScore]:
        from core.llm_service import get_llm_service

        if not get_llm_service().is_configured():
            raise RuntimeError("未配置 LLM API")

        scored: Dict[int, WindowScore] = {}
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            scored.update(self._score_batch_with_llm(topic, batch))
        if not scored:
            raise ValueError("模型没有返回任何有效评分")
        logger.info("LLM 语义评分覆盖 %d/%d 个候选窗口", len(scored), len(candidates))
        return scored


_scorer: Optional[SemanticScorer] = None


def get_semantic_scorer() -> SemanticScorer:
    global _scorer
    if _scorer is None:
        _scorer = SemanticScorer()
    return _scorer


def reset_scorer() -> None:
    """Drop the cached scorer so new environment settings take effect."""
    global _scorer
    _scorer = None
