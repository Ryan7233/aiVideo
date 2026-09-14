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


def _clock(seconds: float) -> str:
    """SRT-style hh:mm:ss,mmm."""
    total = round(max(0.0, float(seconds)) * 1000)
    seconds_total, millis = divmod(total, 1000)
    minutes_total, secs = divmod(seconds_total, 60)
    hours, minutes = divmod(minutes_total, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _timecode(seconds: float, fps: float) -> str:
    """SMPTE hh:mm:ss:ff, carried through a total frame count.

    Rounding each field on its own let 59.999s at 25fps render as
    ``00:00:60:00`` -- a timecode no editor will accept. Converting to whole
    frames first and dividing back out makes the carry fall out naturally.
    """
    rate = max(1, int(round(fps)))
    frame_index = int(round(max(0.0, float(seconds)) * rate))
    seconds_total, frame = divmod(frame_index, rate)
    minutes_total, secs = divmod(seconds_total, 60)
    hours, minutes = divmod(minutes_total, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frame:02d}"


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
        def metric(name):
            if segment.get("available_dimensions", {}).get(name) is False:
                return "—"
            return f"{float(segment.get(name + '_score', 0)):.2f}"
        rows.append(
            f"| {index} "
            f"| {_clock(segment.get('start_time', 0))[:-4]} "
            f"| {_clock(segment.get('end_time', 0))[:-4]} "
            f"| {float(segment.get('duration', 0)):.1f}s "
            f"| {float(segment.get('score', 0)):.2f} "
            f"| {metric('semantic')} "
            f"| {metric('visual')} "
            f"| {metric('audio')} "
            f"| {reason} |"
        )
    return "\n".join(header + rows) + "\n"


def to_edl(
    segments: Sequence[Dict[str, Any]],
    *,
    title: str = "aiVideo",
    fps: float = 25.0,
) -> str:
    """CMX3600, which Premiere, Resolve and Final Cut all import.

    ``fps`` should be the source's own frame rate -- the workflow reads it
    with ffprobe. A 30fps clip written at 25 lands every cut on the wrong
    frame. Record timecode runs continuously from zero so the segments sit
    back to back on the timeline in the order they were chosen.
    """
    rate = max(1.0, float(fps))
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    record = 0.0
    for index, segment in enumerate(segments, 1):
        start = float(segment.get("start_time", 0))
        end = float(segment.get("end_time", 0))
        duration = max(0.0, end - start)
        lines.append(
            f"{index:03d}  AX       V     C        "
            f"{_timecode(start, rate)} {_timecode(end, rate)} "
            f"{_timecode(record, rate)} {_timecode(record + duration, rate)}"
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
    number = 0
    for segment in segments:
        span = max(0.0, float(segment.get("duration", 0)))
        cut_start = float(segment.get("start_time", 0))
        cut_end = float(segment.get("end_time", 0))
        cues = segment.get("cues")
        if cues is None:  # Compatibility for callers without timed transcripts.
            cues = [{"start": cut_start, "end": cut_end,
                     "text": segment.get("text") or segment.get("preview_text") or ""}]
        for cue in cues:
            text = cue.get("text", "").strip()
            start, end = float(cue["start"]), float(cue["end"])
            if not text or start < cut_start - 0.001 or end > cut_end + 0.001 or end <= start:
                continue
            if relative:
                start, end = offset + start - cut_start, offset + end - cut_start
            number += 1
            blocks.append(f"{number}\n{_clock(start)} --> {_clock(end)}\n{text}\n")
        # Advance regardless: skipping a silent segment without moving the
        # offset pushed every later caption back onto the wrong moment, and
        # numbering has to stay contiguous for a valid SRT.
        offset += span
    return "\n".join(blocks)
