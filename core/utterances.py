"""Turning word timings into utterances.

Whisper's own segment boundaries follow its decoding windows, not speech. On
continuous narration it happily returns one segment covering half a minute,
and then every candidate window in the clipper sees the same transcript text
and scores identically -- which defeats the one thing this tool exists to do.

Word timestamps are reliable where segment boundaries are not, so utterances
are rebuilt from them: split on sentence-ending punctuation, on a pause long
enough to be a breath, and on a hard length cap so a monologue without
punctuation still yields usable pieces.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)

# A pause longer than this reads as a boundary rather than a hesitation.
PAUSE_SECONDS = 0.45
# Never let an utterance run longer than this, punctuation or not.
MAX_SECONDS = 12.0
# Below this, a fragment is glued onto its neighbour instead of standing alone.
MIN_SECONDS = 1.2

_SENTENCE_END = re.compile(r"[。！？!?；;…]$|[.](\s|$)")


def _flush(words: Sequence[Dict[str, Any]], *, closed: bool = False) -> Dict[str, Any]:
    """``closed`` marks an utterance that ended on sentence punctuation."""
    text = "".join(word["word"] for word in words).strip()
    return {"start": float(words[0]["start"]), "end": float(words[-1]["end"]),
            "text": text, "closed": closed}


def group_words(words: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group timed words into utterances."""
    grouped: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []

    for index, word in enumerate(words):
        current.append(word)
        text = str(word.get("word", ""))
        span = float(word["end"]) - float(current[0]["start"])
        following = words[index + 1] if index + 1 < len(words) else None
        gap = (float(following["start"]) - float(word["end"])) if following else 0.0

        if not following:
            break
        ended_sentence = bool(_SENTENCE_END.search(text.strip()))
        if ended_sentence or gap >= PAUSE_SECONDS or span >= MAX_SECONDS:
            grouped.append(_flush(current, closed=ended_sentence))
            current = []

    if current:
        grouped.append(_flush(current))

    # Fold away slivers so a stray word does not become its own "moment" --
    # but only into a neighbour it actually abuts. Merging across a real pause
    # would produce an utterance that is mostly silence joining two unrelated
    # fragments.
    merged: List[Dict[str, Any]] = []
    for item in grouped:
        if not item["text"]:
            continue
        if not merged:
            merged.append(item)
            continue
        previous = merged[-1]
        # A finished sentence stays finished; only an unterminated fragment
        # may absorb what follows it.
        adjacent = (item["start"] - previous["end"] < PAUSE_SECONDS
                    and not previous.get("closed"))
        short = (item["end"] - item["start"] < MIN_SECONDS
                 or previous["end"] - previous["start"] < MIN_SECONDS)
        if short and adjacent and (item["end"] - previous["start"]) <= MAX_SECONDS:
            previous["end"] = item["end"]
            previous["text"] = f"{previous['text']} {item['text']}".strip()
        else:
            merged.append(item)
    return merged


def resegment(segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rebuild utterances from ASR output, using word timings when present.

    Falls back to the original segments when the model returned no word
    timings, so this can never make the transcript worse than it was.
    """
    words = [
        word for segment in segments
        for word in (segment.get("words") or [])
        if word.get("start") is not None and word.get("end") is not None
    ]
    if not words:
        return [dict(segment) for segment in segments]

    grouped = group_words(words)
    if not grouped:
        return [dict(segment) for segment in segments]

    logger.info("按词级时间戳重新分句: %d 段 -> %d 段", len(segments), len(grouped))
    for item in grouped:
        item.pop("closed", None)
    return grouped
