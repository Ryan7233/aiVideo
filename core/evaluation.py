"""Separate export correctness from human-labelled selection quality."""

from typing import Optional


def evaluate_selection(segments: list[dict], accepted: Optional[list[dict]] = None) -> dict:
    def overlaps(a, b):
        return max(0.0, min(a["end_time"], b["end_time"]) - max(a["start_time"], b["start_time"]))

    def iou(a, b):
        shared = overlaps(a, b)
        union = a["end_time"] - a["start_time"] + b["end_time"] - b["start_time"] - shared
        return shared / union if union > 0 else 0.0

    cut_cues = sum(1 for s in segments for cue in s.get("cues", [])
                   if cue["start"] < s["start_time"] - .001 or cue["end"] > s["end_time"] + .001)
    overlapping = sum(1 for i, a in enumerate(segments) for b in segments[i + 1:] if overlaps(a, b) > .001)
    texts = ["".join(s.get("text", "").split()) for s in segments if s.get("text", "").strip()]
    report = {
        "selected_count": len(segments),
        "overlapping_pairs": overlapping,
        "cut_cue_count": cut_cues,
        "utterance_aligned_count": sum(s.get("boundary_source") == "utterance" for s in segments),
        "duplicate_text_count": len(texts) - len(set(texts)),
        "human_metrics": None,
    }
    if accepted is not None:
        for span in accepted:
            if not (0 <= span["start_time"] < span["end_time"]):
                raise ValueError("标注必须满足 0 <= start_time < end_time")
        # One-to-one matching stops multiple overlapping recommendations from
        # receiving credit for the same labelled moment. Empty labels mean no
        # approved moments, not missing annotation.
        pairs = sorted(((iou(s, a), i, j) for i, s in enumerate(segments)
                        for j, a in enumerate(accepted)), reverse=True)
        picked, matched = set(), set()
        for score, i, j in pairs:
            if score >= .5 and i not in picked and j not in matched:
                picked.add(i)
                matched.add(j)
        report["human_metrics"] = {
            "match_rule": "one-to-one interval IoU >= 0.5",
            "approved_count": len(accepted),
            "matched_count": len(picked),
            "precision_at_returned_k": len(picked) / len(segments) if segments else 0.0,
            "approved_moment_recall": len(matched) / len(accepted) if accepted else None,
        }
    return report
