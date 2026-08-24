"""The decision list: which moments to cut, and why.

The rendered 9:16 file is the least defensible half of this tool -- a
dedicated editor does that better. The judgement about *which fifteen seconds
matter* is the part worth keeping, so it becomes a first-class output rather
than an intermediate the renderer consumes and discards.

Three formats, because the answer has three audiences: a person reading it, a
timeline importing it, and a caption burner.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def _hms(seconds: float, *, frames: bool = False, fps: int = 25) -> str:
    total = max(0.0, float(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if frames:
        whole = int(secs)
        frame = int(round((secs - whole) * fps))
        if frame >= fps:  # rounding can push it past the last frame
            whole, frame = whole + 1, 0
        return f"{int(hours):02d}:{int(minutes):02d}:{whole:02d}:{frame:02d}"
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}".replace(".", ",")


def to_markdown(segments: Sequence[Dict[str, Any]], *, source: str = "", topic: str = "") -> str:
    """A table you can read and act on without opening anything."""
    header = ["# 剪辑清单", ""]
    if source:
        header.append(f"- 素材：`{source}`")
    if topic:
        header.append(f"- 主题：{topic}")
    header += [
        f"- 片段：{len(segments)} 个，"
        f"共 {sum(float(s.get('duration', 0)) for s in segments):.1f} 秒",
        "",
        "| # | 起 | 止 | 时长 | 总分 | 语义 | 画面 | 音频 | 理由 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    rows = []
    for index, segment in enumerate(segments, 1):
        details = segment.get("semantic_details") or {}
        reason = str(details.get("reason") or segment.get("type") or "").strip()
        rows.append(
            f"| {index} "
            f"| {_hms(segment.get('start_time', 0))[:-4]} "
            f"| {_hms(segment.get('end_time', 0))[:-4]} "
            f"| {float(segment.get('duration', 0)):.1f}s "
            f"| {float(segment.get('score', 0)):.2f} "
            f"| {float(segment.get('semantic_score', 0)):.2f} "
            f"| {float(segment.get('visual_score', 0)):.2f} "
            f"| {float(segment.get('audio_score', 0)):.2f} "
            f"| {reason} |"
        )
    return "\n".join(header + rows) + "\n"


def to_edl(segments: Sequence[Dict[str, Any]], *, title: str = "aiVideo", fps: int = 25) -> str:
    """CMX3600, which Premiere, Resolve and Final Cut all import.

    Record timecode runs continuously from zero so the segments land back to
    back on the timeline in the order they were chosen.
    """
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    record = 0.0
    for index, segment in enumerate(segments, 1):
        start = float(segment.get("start_time", 0))
        end = float(segment.get("end_time", 0))
        duration = max(0.0, end - start)
        lines.append(
            f"{index:03d}  AX       V     C        "
            f"{_hms(start, frames=True, fps=fps)} {_hms(end, frames=True, fps=fps)} "
            f"{_hms(record, frames=True, fps=fps)} {_hms(record + duration, frames=True, fps=fps)}"
        )
        reason = ((segment.get("semantic_details") or {}).get("reason")
                  or segment.get("type") or "")
        if reason:
            lines.append(f"* COMMENT: {reason}")
        record += duration
    return "\n".join(lines) + "\n"


def to_srt(segments: Sequence[Dict[str, Any]], *, relative: bool = True) -> str:
    """Captions for the selected segments.

    ``relative`` re-times them against the concatenated output, which is what
    matches the rendered clip; otherwise they keep source timing.
    """
    blocks: List[str] = []
    offset = 0.0
    for index, segment in enumerate(segments, 1):
        text = (segment.get("preview_text") or segment.get("text") or "").strip()
        if not text:
            continue
        start = float(segment.get("start_time", 0))
        end = float(segment.get("end_time", 0))
        if relative:
            start, end = offset, offset + max(0.0, end - start)
        blocks.append(f"{index}\n{_hms(start)} --> {_hms(end)}\n{text}\n")
        offset += max(0.0, float(segment.get("duration", 0)))
    return "\n".join(blocks)
