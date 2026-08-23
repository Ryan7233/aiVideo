"""Marking a result that came from a fallback path.

The recurring failure in this codebase is not the broad `except` itself --
carrying on with less is often the right call for one bad photo or a missing
optional model. The problem is that the substitute is shaped like the real
thing, so the caller reports success and nobody finds out for weeks. It has
happened seven times: three NameErrors, a cover that never wrote a file, audio
analysis with every feature missing, a collage that always raised, and a
frontend that fell back to a local canvas while showing "生成完成".

So fallbacks say so. Callers, tests and the UI can all check one key instead
of guessing from the shape, and four different ad-hoc spellings
(`fallback`, `is_fallback`, `status: "fallback"`, `degraded`) collapse into
one.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

DEGRADED = "degraded"
DEGRADED_REASON = "degraded_reason"


def mark_degraded(
    payload: Dict[str, Any],
    reason: Any,
    *,
    logger: Optional[logging.Logger] = None,
    context: str = "",
) -> Dict[str, Any]:
    """Tag ``payload`` as a fallback and return it.

    ``reason`` is usually the caught exception. Safe to call on anything that
    is not a dict -- it is returned untouched rather than raising inside an
    error handler.
    """
    if not isinstance(payload, dict):
        return payload

    text = f"{type(reason).__name__}: {reason}" if isinstance(reason, BaseException) else str(reason)
    payload[DEGRADED] = True
    payload[DEGRADED_REASON] = text[:500]
    if logger is not None:
        logger.warning("%s降级返回: %s", f"{context} " if context else "", text[:300])
    return payload


def is_degraded(payload: Any) -> bool:
    """True when ``payload`` came from a fallback path."""
    return bool(isinstance(payload, dict) and payload.get(DEGRADED))
