"""Candidate intervals whose scoring text is actually inside the cut."""

import math
from typing import Any, Dict, List, Sequence


def build_windows(duration: float, target: float, transcript: Sequence[Dict[str, Any]],
                  maximum: float = 30.0) -> List[Dict[str, Any]]:
    maximum = min(30.0, maximum, duration)
    if not transcript:
        width = min(target, maximum)
        step = max(3.0, width / 2)
        last = max(0.0, duration - width)
        starts = [i * step for i in range(int(last / step) + 1)]
        if abs(starts[-1] - last) > 0.001:
            starts.append(last)
        return [dict(start=s, end=s + width, duration=width, text="", cues=[],
                     boundary_source="audiovisual_window") for s in starts]

    cues = []
    for item in transcript:
        start, end = float(item["start"]), float(item["end"])
        text = str(item.get("text", "")).strip()
        if text and math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + 0.001:
            cues.append(dict(start=start, end=min(end, duration), text=text))
    cues.sort(key=lambda c: (c["start"], c["end"]))

    # Merge overlapping subtitle cues: cutting one would otherwise bisect
    # another. Keep their individual timings for the eventual SRT export.
    units: List[List[Dict[str, Any]]] = []
    for cue in cues:
        if units and cue["start"] < max(c["end"] for c in units[-1]):
            units[-1].append(cue)
        else:
            units.append([cue])

    windows = []
    for i, unit in enumerate(units):
        start = unit[0]["start"]
        accumulated: List[Dict[str, Any]] = []
        choices = []
        previous_end = start
        for j in range(i, len(units)):
            following = units[j]
            end = max(c["end"] for c in following)
            # Do not join unrelated sentences across a long silence.
            if following[0]["start"] - previous_end > 2.0 or end - start > maximum + 0.001:
                break
            accumulated.extend(following)
            previous_end = end
            if end - start >= 5.0:
                choices.append(dict(start=start, end=end, duration=end - start,
                                    text=" ".join(c["text"] for c in accumulated),
                                    cues=list(accumulated), boundary_source="utterance"))
        if choices:
            # Keep a short alternative so a higher scoring long candidate does
            # not make the duration budget impossible to satisfy.
            nearest = min(choices, key=lambda c: (abs(c["duration"] - target), c["duration"]))
            windows.append(choices[0])
            if nearest is not choices[0]:
                windows.append(nearest)
    return windows
