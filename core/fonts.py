"""Finding a font that can actually render the text on a cover.

The cover code hardcoded one font path per platform and fell back to
``ImageFont.load_default()`` on any failure. Two ways that went wrong:

- ``/System/Library/Fonts/PingFang.ttc`` does not exist on current macOS, so
  every cover fell back to the default bitmap font, which has no Chinese
  glyphs at all. Titles came out as a few unreadable pixels.
- On Linux the fallback was DejaVuSans, which opens fine and then renders
  Chinese as tofu boxes. A file-exists check cannot catch that.

So candidates are probed by *rendering* a Chinese glyph and comparing it with
a private-use codepoint no font defines. If they match, the glyph is missing.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# No font defines this private-use codepoint, so it always renders .notdef.
_SENTINEL = "\ue000"
_PROBE = "中"

# Ordered by preference: a bold/medium weight suited to cover titles first.
CJK_FONT_CANDIDATES: List[str] = [
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    # Linux (Noto CJK is what most distros and container images ship)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    # Windows
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]

LATIN_FONT_CANDIDATES: List[str] = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

INSTALL_HINT = (
    "未找到可渲染中文的字体，封面标题将无法正常显示。"
    "Debian/Ubuntu 可安装：apt-get install -y fonts-noto-cjk；"
    "也可以用 AIVIDEO_CJK_FONT 指定字体文件路径。"
)


def _can_render_cjk(path: str) -> bool:
    """True when the font has a real glyph for a Chinese character."""
    try:
        from PIL import ImageFont

        font = ImageFont.truetype(path, 40)
        return bytes(font.getmask(_PROBE, mode="L")) != bytes(font.getmask(_SENTINEL, mode="L"))
    except Exception:
        return False


@lru_cache(maxsize=1)
def find_cjk_font() -> Optional[str]:
    """Path to a Chinese-capable font, or None. Result is cached."""
    override = os.getenv("AIVIDEO_CJK_FONT", "").strip()
    if override:
        if Path(override).is_file() and _can_render_cjk(override):
            logger.info("Using AIVIDEO_CJK_FONT: %s", override)
            return override
        logger.warning("AIVIDEO_CJK_FONT is set but unusable: %s", override)

    for candidate in CJK_FONT_CANDIDATES:
        if Path(candidate).is_file() and _can_render_cjk(candidate):
            logger.info("Cover font: %s", candidate)
            return candidate

    logger.warning(INSTALL_HINT)
    return None


@lru_cache(maxsize=1)
def find_latin_font() -> Optional[str]:
    """A Latin font, used only when no CJK font is available."""
    for candidate in LATIN_FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def load_font(size: int, *, prefer_cjk: bool = True):
    """Load a font at ``size``, preferring one that can render Chinese.

    Always returns something usable: a CJK font, else a Latin font, else
    PIL's built-in default (which cannot render Chinese, hence the warning
    from :func:`find_cjk_font`).
    """
    from PIL import ImageFont

    paths = []
    if prefer_cjk:
        paths.append(find_cjk_font())
    paths.append(find_latin_font())

    for path in paths:
        if not path:
            continue
        try:
            return ImageFont.truetype(path, size)
        except Exception as exc:
            logger.debug("Could not load %s at size %s: %s", path, size, exc)

    return ImageFont.load_default()


def reset_cache() -> None:
    """Forget the resolved fonts (used by tests)."""
    find_cjk_font.cache_clear()
    find_latin_font.cache_clear()
