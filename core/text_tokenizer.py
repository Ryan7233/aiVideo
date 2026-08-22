r"""Chinese-aware tokenization and lexicon matching.

The scoring modules used to tokenize with ``re.findall(r'\b\w+\b', text)`` and
count words with ``text.split()``.  Neither works on Chinese: a whole sentence
comes back as a single token, so stop-word filtering, TF-IDF and every
dictionary lookup silently degrade to noise.  Everything that needs to look
inside text goes through this module instead.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

CJK = r"㐀-䶿一-鿿豈-﫿"

_CJK_RUN = re.compile(f"[{CJK}]+")
_LATIN_WORD = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?", re.IGNORECASE)
_TOKEN_SPLIT = re.compile(f"([{CJK}]+)")
_SENTENCE_SPLIT = re.compile(r"[.!?;。！？；\n]+")


@lru_cache(maxsize=1)
def _jieba_cut():
    """Return jieba's cut function, or None when jieba is not installed."""
    try:
        import jieba

        jieba.setLogLevel(60)  # keep dictionary loading out of the app logs
        return jieba.lcut
    except Exception:  # pragma: no cover - exercised only on stripped installs
        return None


def _split_cjk_run(run: str) -> List[str]:
    """Segment one run of Chinese characters."""
    cut = _jieba_cut()
    if cut is not None:
        return [token for token in cut(run) if token.strip()]
    # Fallback: non-overlapping bigrams. Boundaries are wrong more often than
    # jieba's, but the token count stays in the same range, which is what the
    # density and diversity scores actually depend on.
    return [run[i : i + 2] for i in range(0, len(run), 2)]


def tokenize(text: str) -> List[str]:
    """Split mixed Chinese/Latin text into comparable lowercase tokens."""
    if not text or not text.strip():
        return []
    tokens: List[str] = []
    for chunk in _TOKEN_SPLIT.split(text):
        if not chunk:
            continue
        if _CJK_RUN.fullmatch(chunk):
            tokens.extend(_split_cjk_run(chunk))
        else:
            tokens.extend(match.group(0).lower() for match in _LATIN_WORD.finditer(chunk))
    return tokens


def content_tokens(tokens: Sequence[str], stop_words: Iterable[str]) -> List[str]:
    """Drop stop words and single characters, which carry no topical signal."""
    stops = stop_words if isinstance(stop_words, (set, frozenset)) else set(stop_words)
    return [token for token in tokens if len(token) >= 2 and token not in stops]


def split_sentences(text: str) -> List[str]:
    """Split into sentences, dropping the empty pieces the delimiters leave."""
    return [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part.strip()]


def _is_latin_term(term: str) -> bool:
    return not _CJK_RUN.search(term)


def _has_word_boundary(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return not (before.isalnum() or after.isalnum())


def scan_lexicon(text: str, lexicon: Dict[str, float]) -> List[Tuple[str, float]]:
    """Find lexicon entries in ``text``, longest match first, without overlap.

    Chinese entries are matched as substrings because there is no reliable
    whitespace boundary; Latin entries still require one.  Matching longest
    first and consuming the span is what keeps ``不好`` from also scoring as
    ``好``.
    """
    if not text or not lexicon:
        return []
    lowered = text.lower()
    consumed = bytearray(len(lowered))
    hits: List[Tuple[str, float]] = []

    for term in sorted(lexicon, key=len, reverse=True):
        if not term:
            continue
        needle = term.lower()
        latin = _is_latin_term(needle)
        start = 0
        while True:
            index = lowered.find(needle, start)
            if index < 0:
                break
            end = index + len(needle)
            if any(consumed[index:end]) or (latin and not _has_word_boundary(lowered, index, end)):
                start = index + 1
                continue
            consumed[index:end] = b"\x01" * (end - index)
            hits.append((term, lexicon[term]))
            start = end
    return hits
