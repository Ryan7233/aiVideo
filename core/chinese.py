"""Traditional -> Simplified normalisation for transcripts.

The ``简体`` initial prompt biases Whisper toward Simplified output, but the
prompt only conditions the first decoding window. On a longer recording the
model can drift into Traditional part-way through and stay there, so one
speaker ends up with 选断 at 7 s and 選斷 at 95 s. That breaks two things:
the exported SRT mixes scripts, and the duplicate check in the selector sees
two different strings for the same words. Converting after decoding is the
guarantee the prompt cannot give.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _converter() -> Optional[Any]:
    try:
        from opencc import OpenCC
    except ImportError:  # pragma: no cover - exercised only in partial installs
        logger.warning("opencc 未安装；繁体转写不会转换为简体")
        return None
    return OpenCC("t2s")


def to_simplified(text: Any) -> str:
    """Return ``text`` in Simplified Chinese; non-Chinese text passes through."""
    text = str(text or "")
    if not text:
        return text
    converter = _converter()
    return converter.convert(text) if converter is not None else text
