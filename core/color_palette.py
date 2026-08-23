"""Deriving a colour palette from the images a cover is actually made of.

The cover renderer only offered five hardcoded themes, so every generated
cover looked the same whatever the photos were. The cover *designer* already
knew how to pull dominant colours out of a frame, but that logic was private
to it and produced RGB tuples the renderer could not consume.

This is that logic, shared, with the conversion both sides need.
"""

from __future__ import annotations

import colorsys
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

RGB = Tuple[int, int, int]

# Used when there is nothing to sample or the sampling fails.
NEUTRAL_PALETTE: Dict[str, str] = {
    "primary": "#2C3E50",
    "secondary": "#7F8C8D",
    "accent": "#E74C3C",
    "background": "#F8F9FA",
}


def rgb_to_hex(color: Sequence[int]) -> str:
    r, g, b = (max(0, min(255, int(channel))) for channel in color[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(value: str) -> RGB:
    text = value.lstrip("#")
    if len(text) == 3:
        text = "".join(char * 2 for char in text)
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def adjust_for_text(color: RGB) -> RGB:
    """Push a colour toward the saturation/brightness band text reads on."""
    try:
        h, s, v = colorsys.rgb_to_hsv(*(channel / 255 for channel in color))
        s = min(1.0, s * 1.2)
        v = max(0.4, min(0.8, v))
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))
    except Exception:
        return color


def complement(color: RGB) -> RGB:
    """Opposite hue, damped so it works as a supporting colour."""
    try:
        h, s, v = colorsys.rgb_to_hsv(*(channel / 255 for channel in color))
        r, g, b = colorsys.hsv_to_rgb((h + 0.5) % 1.0, max(0.3, s * 0.8), max(0.3, v * 0.9))
        return (int(r * 255), int(g * 255), int(b * 255))
    except Exception:
        return (128, 128, 128)


def shade(color: RGB, factor: float = 0.65) -> RGB:
    """A darker version of a colour at the same hue.

    The built-in themes all pair primary with a darker sibling rather than a
    complement, because the cover background is a primary->secondary gradient
    and two opposing hues there read as a mistake.
    """
    try:
        h, s, v = colorsys.rgb_to_hsv(*(channel / 255 for channel in color))
        r, g, b = colorsys.hsv_to_rgb(h, min(1.0, s * 1.05), max(0.15, v * factor))
        return (int(r * 255), int(g * 255), int(b * 255))
    except Exception:
        return color


def tint(color: RGB, strength: float = 0.92) -> RGB:
    """A near-white version of a colour, for page backgrounds."""
    return tuple(int(channel + (255 - channel) * strength) for channel in color)  # type: ignore[return-value]


def dominant_colors(image, num_colors: int = 5) -> List[RGB]:
    """Cluster an image's pixels and return the centres, most common first."""
    try:
        import numpy as np
        from sklearn.cluster import KMeans

        small = image.convert("RGB").resize((150, 150))
        pixels = np.array(small).reshape(-1, 3)

        clusters = max(1, min(num_colors, len(np.unique(pixels, axis=0))))
        kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)

        centres = kmeans.cluster_centers_.astype(int)
        counts = Counter(kmeans.labels_)
        return [tuple(int(channel) for channel in centres[label]) for label, _ in counts.most_common()]
    except Exception as exc:
        logger.warning("Dominant colour extraction failed, using neutrals: %s", exc)
        return [(128, 128, 128), (64, 64, 64), (192, 192, 192)]


def _open(path) -> Optional["object"]:
    try:
        from PIL import Image

        candidate = Path(path)
        if not candidate.is_file():
            return None
        return Image.open(candidate)
    except Exception as exc:
        logger.debug("Could not open %s for colour sampling: %s", path, exc)
        return None


def palette_from_images(paths: Sequence[str], sample_limit: int = 5) -> Dict[str, str]:
    """Build a renderer-ready hex palette from the images going into a cover.

    Samples the first few images rather than all of them: a cover's images are
    usually from one shoot, and clustering is the expensive part.
    """
    samples: List[RGB] = []
    for path in list(paths)[:sample_limit]:
        image = _open(path)
        if image is None:
            continue
        with image:
            samples.extend(dominant_colors(image, num_colors=3)[:2])

    if not samples:
        logger.info("No sampleable images; falling back to the neutral palette")
        return dict(NEUTRAL_PALETTE)

    # The most-represented colour across the set anchors the palette.
    counts = Counter(samples)
    main = counts.most_common(1)[0][0]
    primary = adjust_for_text(main)

    return {
        "primary": rgb_to_hex(primary),
        # Same hue, darker: the background is a primary->secondary gradient.
        "secondary": rgb_to_hex(shade(primary)),
        # The complement earns its contrast here, on small accent elements.
        "accent": rgb_to_hex(adjust_for_text(complement(primary))),
        "background": rgb_to_hex(tint(primary)),
    }
