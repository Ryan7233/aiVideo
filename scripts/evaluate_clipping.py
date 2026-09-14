#!/usr/bin/env python3
"""Run a real local source and report structure separately from human quality.

Optional annotation JSON: {"accepted": [{"start_time": 10, "end_time": 20}]}.
Without annotations, human_metrics stays null; ASR/LLM scores are not accuracy.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Support `python scripts/evaluate_clipping.py` from a project checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.evaluation import evaluate_selection
from core.video_workflow import process_multi_segment_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--topic", default="内容要点")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--asr-model", default="tiny")
    parser.add_argument("--selection-mode", choices=["highlights", "summary"], default="highlights")
    parser.add_argument("--scoring-mode", choices=["rules", "auto", "llm"], default="rules",
                        help="默认 rules 离线评分；auto/llm 会按配置调用外部模型")
    args = parser.parse_args()
    accepted = None
    if args.annotations:
        accepted = json.loads(args.annotations.read_text(encoding="utf-8"))["accepted"]
        evaluate_selection([], accepted)  # Reject bad labels before doing media work.
    os.environ["SEMANTIC_SCORING_MODE"] = args.scoring_mode
    options = {"video_path": args.video, "topic": args.topic,
               "target_segments": args.segments, "total_duration": args.duration,
               "asr_model_size": args.asr_model, "selection_mode": args.selection_mode,
               "render": False}
    started = time.perf_counter()
    result = process_multi_segment_video(options)
    report = {
        "source": args.video,
        "annotations": str(args.annotations) if args.annotations else None,
        "options": options,
        "scoring_mode": args.scoring_mode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "evaluation": evaluate_selection(result["selected_segments"], accepted),
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.output), "elapsed_seconds": report["elapsed_seconds"],
                      "evaluation": report["evaluation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
