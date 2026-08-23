"""Environment reads where an empty value means "not configured".

`env.example` ships keys with blank values as a way of documenting them:

    MAX_CONCURRENT_MEDIA_JOBS=
    AIVIDEO_DB_PATH=

`os.getenv(name, default)` returns "" for those, not the default, so following
the README -- copy env.example to .env, start the app -- crashed on
`int("")`, pointed the job database at the current directory, and passed an
empty Whisper prompt where a default was intended.

Everything here treats blank as absent, and falls back with a warning rather
than raising when a value cannot be parsed: a malformed number in a config
file should not take the process down.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TRUTHY = {"1", "true", "yes", "on"}
FALSY = {"0", "false", "no", "off"}


def env_str(name: str, default: str = "") -> str:
    """Value with surrounding whitespace stripped, or ``default`` if blank."""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def env_int(name: str, default: int) -> int:
    raw = env_str(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %s", name, raw, default)
        return default


def env_float(name: str, default: float) -> float:
    raw = env_str(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = env_str(name).lower()
    if not raw:
        return default
    if raw in TRUTHY:
        return True
    if raw in FALSY:
        return False
    logger.warning("%s=%r is not a boolean; using %s", name, raw, default)
    return default


def env_path(name: str, default: Path) -> Path:
    raw = env_str(name)
    return Path(raw).expanduser() if raw else Path(default)


def env_list(name: str, default: str, separator: str = ",") -> list[str]:
    """Split on ``separator``, dropping blank entries."""
    return [item.strip() for item in env_str(name, default).split(separator) if item.strip()]


def env_optional(name: str) -> Optional[str]:
    """The value, or None when unset or blank."""
    return env_str(name) or None
