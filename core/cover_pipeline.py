"""Choosing what goes on a cover, then rendering it.

Two modules existed and never spoke to each other. `smart_cover_design`
extracts key frames from clips, scores every frame and photo on sharpness,
aspect fit and how much room is left for a title, and picks a winner.
`smart_cover_generator` composites a set of images into a finished cover but
has no opinion about which images: whatever the caller passes, in order.

So the ranking was thrown away and the renderer was fed unsorted input. This
joins them: rank the material, take the best few, render them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from core.degradation import mark_degraded

logger = logging.getLogger(__name__)

# Beyond this a cover is a wall of thumbnails; the adaptive grid tops out here.
MAX_COVER_IMAGES = 9


def _dedupe(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the best-scoring entry per file.

    Several key frames can resolve to the same path, and a cover made of six
    copies of one shot is not a cover. Deduplicating the ranking rather than
    just the render also keeps the reported selection honest -- otherwise
    every duplicate row is marked "used".
    """
    seen, unique = set(), []
    for candidate in candidates:
        path = candidate.get("path")
        if path and path not in seen:
            seen.add(path)
            unique.append(candidate)
    return unique


def build_cover(
    clips: Optional[List[Dict[str, Any]]] = None,
    photos: Optional[List[Dict[str, Any]]] = None,
    *,
    title: str = "",
    subtitle: str = "",
    image_count: int = 4,
    layout: str = "auto",
    theme: str = "auto",
    platform: str = "xiaohongshu",
) -> Dict[str, Any]:
    """Rank the available material and render a cover from the best of it.

    ``clips`` are results from the clipping workflow (each needs an
    ``output_path``); ``photos`` are ranked photos (each needs a ``path``).
    Either may be empty. Returns the renderer's result with the ranking
    attached under ``selection``.
    """
    from core.smart_cover_design import get_smart_cover_designer
    from core.smart_cover_generator import get_smart_cover_generator

    wanted = max(1, min(int(image_count), MAX_COVER_IMAGES))

    ranked = get_smart_cover_designer().rank_cover_candidates(clips or [], photos or [])
    if not ranked:
        return mark_degraded(
            {"status": "error", "message": "没有可用的封面素材", "selection": []},
            "no usable cover candidates",
            logger=logger,
            context="build_cover",
        )

    ranked = _dedupe(ranked)
    chosen = [item["path"] for item in ranked[:wanted]]
    logger.info(
        "封面选材: 从 %d 个候选里选出 %d 张（最高分 %.2f）",
        len(ranked), len(chosen), ranked[0].get("cover_score", 0.0),
    )

    result = get_smart_cover_generator().generate_cover(
        images=chosen,
        title=title,
        subtitle=subtitle,
        layout=layout,
        theme=theme,
        platform=platform,
    )

    # Keep the reasoning visible: which frames won, and why they scored.
    if isinstance(result, dict):
        result["selection"] = [
            {
                "path": item.get("path"),
                "type": item.get("type"),
                "cover_score": round(float(item.get("cover_score", 0.0)), 4),
                "source": item.get("source"),
                "used": item.get("path") in chosen,
            }
            for item in ranked[: max(wanted * 2, 8)]
        ]
    return result
