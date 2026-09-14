"""Production-oriented multi-segment video workflow.

This module keeps blocking media work out of the FastAPI route layer and makes
every request parameter participate in selection.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.chinese import to_simplified
from core.concurrency import run_ffmpeg
from core.candidates import build_windows
from core.edl import to_edl, to_markdown, to_srt
from core.config import MAX_FILE_SIZE
from core.runtime import (
    OUTPUT_DIR,
    materialize_video_source,
    resolve_media_path,
)
from core.semantic_scoring import get_semantic_scorer
from core.smart_clipping import SmartClippingEngine
from core.whisper_asr import get_asr_service


logger = logging.getLogger(__name__)


def _probe_fps(video_path: Path, default: float = 25.0) -> float:
    """Source frame rate, for timecodes an editor will accept."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "json", str(video_path)],
            capture_output=True, text=True, check=True, timeout=30,
        )
        raw = json.loads(result.stdout)["streams"][0]["r_frame_rate"]
        numerator, _, denominator = raw.partition("/")
        rate = float(numerator) / float(denominator or 1)
        return rate if 1.0 <= rate <= 480.0 else default
    except Exception as exc:
        logger.debug("读取帧率失败，按 %s fps 处理: %s", default, exc)
        return default


def _probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def _time_to_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    parts = text.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (ValueError, IndexError):
        return 0.0


def _parse_srt(path: Path) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    content = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        time_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if time_index is None:
            continue
        start_text, end_text = [part.strip() for part in lines[time_index].split("-->", 1)]
        text = " ".join(lines[time_index + 1 :]).strip()
        if text:
            segments.append(
                {
                    "start": _time_to_seconds(start_text),
                    "end": _time_to_seconds(end_text.split()[0]),
                    "text": text,
                }
            )
    return segments


def _load_transcript(
    video_path: Path,
    subtitle_path: Optional[str],
    enabled: bool,
    model_size: str,
    language: Optional[str],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if subtitle_path:
        path = resolve_media_path(subtitle_path)
        return _parse_srt(path), {"source": "provided_subtitle", "path": str(path)}
    if not enabled:
        return [], {"source": "disabled"}

    try:
        service = get_asr_service(model_size=model_size)
        result = service.transcribe_video(str(video_path), language=language, cleanup_audio=True)
        segments = [
            {
                "start": float(item.get("start", 0)),
                "end": float(item.get("end", 0)),
                "text": item.get("text", "").strip(),
            }
            for item in result.get("segments", [])
            if item.get("text", "").strip()
        ]
        return segments, {
            "source": "whisper",
            "language": result.get("language"),
            "word_count": result.get("word_count", 0),
            "segment_count": len(segments),
        }
    except Exception as exc:
        logger.warning("ASR unavailable; continuing with measured audiovisual features: %s", exc)
        return [], {"source": "unavailable", "error": str(exc)}


def _points_in_window(points: Sequence[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    return [point for point in points if start <= float(point.get("timestamp", -1)) < end]


def _build_candidates(
    duration: float,
    window_duration: float,
    topic: str,
    transcript: Sequence[Dict[str, Any]],
    analysis: Dict[str, Any],
    weights: Dict[str, float],
    maximum_duration: float = 30.0,
) -> List[Dict[str, Any]]:
    scenes = analysis.get("scene_changes", [])
    audio = analysis.get("audio_energy", [])
    motion = analysis.get("motion_activity", [])
    windows = build_windows(duration, window_duration, transcript, maximum_duration)

    semantic_scores = get_semantic_scorer().score_windows(windows, topic)

    candidates: List[Dict[str, Any]] = []
    for position, window in enumerate(windows):
        start, end = window["start"], window["end"]
        window_scenes = _points_in_window(scenes, start, end)
        window_motion = _points_in_window(motion, start, end)
        window_audio = _points_in_window(audio, start, end)
        window_text = window["text"]

        ideal_scenes = max(1.0, (end - start) / 8.0)
        scene_density = min(1.0, len(window_scenes) / ideal_scenes)
        # Average the measured motion magnitude rather than counting samples:
        # the sample count only reflects the fixed sampling rate.
        motion_density = (
            sum(float(point.get("activity", 0.0)) for point in window_motion) / len(window_motion)
            if window_motion
            else 0.0
        )
        visual_score = scene_density * 0.65 + motion_density * 0.35
        audio_score = (
            sum(float(point.get("energy", 0.0)) for point in window_audio) / len(window_audio)
            if window_audio
            else 0.0
        )
        scored = semantic_scores[position]
        semantic_score = scored.score
        semantic_details = dict(scored.details)
        semantic_details["source"] = scored.source
        if scored.reason:
            semantic_details["reason"] = scored.reason
        intervals = analysis.get("measured_intervals")
        if intervals is None:
            covered = not analysis.get("degraded") and end <= analysis.get("analysis_duration", duration) + 0.001
        else:
            cursor = start
            for interval in intervals:
                if interval["start"] <= cursor + 0.001:
                    cursor = max(cursor, interval["end"])
            covered = cursor >= end - 0.001
        available = {
            "semantic": bool(window_text),
            "visual": covered and bool(window_motion or window_scenes),
            "audio": covered and bool(window_audio),
        }
        active_weights = {key: max(0.0, weights[key]) if available[key] else 0.0
                          for key in weights}
        total_weight = sum(active_weights.values())
        if total_weight <= 0:
            continue  # No requested dimension has evidence; do not invent a score.
        active_weights = {key: value / total_weight for key, value in active_weights.items()}
        total_score = (
            semantic_score * active_weights["semantic"]
            + visual_score * active_weights["visual"]
            + audio_score * active_weights["audio"]
        )
        candidates.append(
            {
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "duration": round(end - start, 3),
                "text": window_text,
                "cues": window["cues"],
                "boundary_source": window["boundary_source"],
                "available_dimensions": available,
                "effective_weights": active_weights,
                "semantic_score": semantic_score,
                "visual_score": visual_score,
                "audio_score": audio_score,
                "score": total_score,
                "semantic_details": semantic_details,
            }
        )
    return candidates


def _overlaps(candidate: Dict[str, Any], selected: Iterable[Dict[str, Any]]) -> bool:
    return any(
        candidate["start_time"] < item["end_time"] and candidate["end_time"] > item["start_time"]
        for item in selected
    )


# Share of the shorter text's character bigrams that must reappear in the
# other text before the two count as the same content. Engineering default,
# not calibrated on annotated material; see `section_min_score`.
DUPLICATE_TEXT_SIMILARITY = 0.8


def _normalized_text(text: Any) -> str:
    # Simplify here as well as in the ASR: a user-supplied SRT can arrive in
    # either script, and 选断 must match 選斷.
    return "".join(ch.lower() for ch in to_simplified(text) if ch.isalnum())


def _text_similarity(a: str, b: str) -> float:
    """Containment of character bigrams, so a repeated line is caught whether
    it fills a short window or is padded with more speech in a long one.
    Bigrams rather than words: Chinese carries no word separators."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    def bigrams(text: str) -> set:
        return {text[i:i + 2] for i in range(len(text) - 1)} or {text}

    ga, gb = bigrams(a), bigrams(b)
    return len(ga & gb) / min(len(ga), len(gb))


def _duplicates(candidate: Dict[str, Any], selected: Iterable[Dict[str, Any]]) -> bool:
    """Non-overlapping windows can still carry the same words: a repeated
    slogan or sponsor read, or Whisper looping one phrase over silence.
    Windows without text (audiovisual fallback) never count as duplicates."""
    text = _normalized_text(candidate.get("text"))
    if not text:
        return False
    return any(
        _text_similarity(text, _normalized_text(item.get("text"))) >= DUPLICATE_TEXT_SIMILARITY
        for item in selected
    )


def _pick_best(
    candidates: Iterable[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    kind: str,
) -> bool:
    available = [item for item in candidates if not _overlaps(item, selected)]
    if not available:
        return False
    chosen = max(available, key=lambda item: item["score"]).copy()
    chosen["type"] = kind
    selected.append(chosen)
    return True


def _select_segments(
    candidates: Sequence[Dict[str, Any]],
    video_duration: float,
    count: int,
    include_intro: bool,
    include_highlights: bool,
    include_conclusion: bool,
    total_budget: Optional[float] = None,
    section_min_score: float = 0.35,
    skipped_duplicates: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []

    def eligible(items):
        used = sum(item["end_time"] - item["start_time"] for item in selected)
        kept = []
        for item in items:
            if _overlaps(item, selected):
                continue
            if total_budget is not None and used + item["end_time"] - item["start_time"] > total_budget + 0.001:
                continue
            if _duplicates(item, selected):
                # Record the rejection so a wrong similarity call is visible
                # in the result rather than silently costing a segment.
                if skipped_duplicates is not None and not any(
                    item is skipped for skipped in skipped_duplicates
                ):
                    skipped_duplicates.append(item)
                continue
            kept.append(item)
        return kept

    if include_intro and len(selected) < count:
        intro_candidates = eligible([
            item for item in candidates if item["start_time"] <= video_duration * 0.2
            and item["score"] >= section_min_score
        ])
        if intro_candidates:
            chosen = max(intro_candidates, key=lambda item: item["score"]).copy()
            chosen["type"] = "intro"
            selected.append(chosen)
    if include_conclusion and len(selected) < count:
        conclusion_candidates = eligible([
            item
            for item in candidates
            if item["end_time"] >= video_duration * 0.8 and item["score"] >= section_min_score
        ])
        if conclusion_candidates:
            chosen = max(
                conclusion_candidates, key=lambda item: item["score"]
            ).copy()
            chosen["type"] = "conclusion"
            selected.append(chosen)
    if include_highlights:
        for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
            if len(selected) >= count:
                break
            if eligible([candidate]):
                chosen = candidate.copy()
                chosen["type"] = "highlight"
                selected.append(chosen)
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if len(selected) >= count:
            break
        if eligible([candidate]):
            chosen = candidate.copy()
            chosen["type"] = "best_available"
            selected.append(chosen)
    return sorted(selected, key=lambda item: item["start_time"])


def _run_ffmpeg(command: List[str], timeout: int = 600) -> None:
    result = run_ffmpeg(command, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:] or "FFmpeg failed")


def _combine_segments(video_path: Path, segments: Sequence[Dict[str, Any]], output_path: Path) -> None:
    temp_files: List[Path] = []
    concat_file = OUTPUT_DIR / f"concat_{uuid.uuid4().hex}.txt"
    try:
        for index, segment in enumerate(segments):
            temp_file = OUTPUT_DIR / f"segment_{uuid.uuid4().hex}_{index}.mp4"
            _run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(segment["start_time"]),
                    "-i",
                    str(video_path),
                    "-t",
                    str(segment["duration"]),
                    "-vf",
                    "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1:1",
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a?",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-movflags",
                    "+faststart",
                    str(temp_file),
                ]
            )
            temp_files.append(temp_file)
        concat_file.write_text(
            "".join(f"file '{path.as_posix()}'\n" for path in temp_files), encoding="utf-8"
        )
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
    finally:
        concat_file.unlink(missing_ok=True)
        for path in temp_files:
            path.unlink(missing_ok=True)


def _write_decision_list(
    stem: str,
    segments: Sequence[Dict[str, Any]],
    source: str,
    topic: str,
    fps: float,
) -> Tuple[Dict[str, str], List[str]]:
    """Write the selection as Markdown, EDL and SRT.

    Returns the paths written and any per-format failures.
    """
    written: Dict[str, str] = {}
    failures: List[str] = []
    for suffix, body in (
        ("md", to_markdown(segments, source=source, topic=topic)),
        ("edl", to_edl(segments, title=Path(source).stem or "aiVideo", fps=fps)),
        ("srt", to_srt(segments)),
    ):
        if not body.strip():
            continue
        path = OUTPUT_DIR / f"clips_{stem}.{suffix}"
        try:
            path.write_text(body, encoding="utf-8")
            written[suffix] = f"output_data/{path.name}"
        except OSError as exc:
            logger.error("写出 %s 清单失败: %s", suffix, exc)
            failures.append(f"{suffix}: {exc}")

    # The list is the product now. A run that wrote none of it did not
    # succeed, whatever the renderer managed; a partial write is reported so
    # the caller knows which formats are missing.
    if failures and not written:
        raise RuntimeError("剪辑清单写入失败：" + "；".join(failures))
    return written, failures


def process_multi_segment_video(options: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete measured/semantic multi-segment workflow."""
    mode = options.get("selection_mode", "highlights")
    if mode not in {"highlights", "summary"}:
        raise ValueError("selection_mode 必须是 highlights 或 summary")
    # Optional old flags remain explicit overrides for existing API clients.
    def flag(name, default):
        value = options.get(name)
        return default if value is None else bool(value)

    intro = flag("include_intro", mode == "summary")
    conclusion = flag("include_conclusion", mode == "summary")
    # Materialise here rather than in a route: a local path, a file:// URL, a
    # direct .mp4 and a platform link all have to work from every caller --
    # the sync route, the job handler and the batch script alike. Doing it in
    # one of them left the others rejecting URLs outright.
    video_path = materialize_video_source(
        str(options["video_path"]), "clip", MAX_FILE_SIZE
    )
    video_duration = _probe_duration(video_path)
    if video_duration < 8:
        raise ValueError("视频时长至少需要 8 秒")

    count = max(1, min(int(options.get("target_segments", 3)), 10))
    desired_total = max(5.0, min(float(options.get("total_duration", 60)), video_duration))
    if video_duration + 0.01 < count * 5:
        raise ValueError(f"视频时长不足以生成 {count} 个至少 5 秒且互不重叠的片段")
    requested_segment = options.get("segment_duration")
    window_duration = float(requested_segment) if requested_segment else desired_total / count
    window_duration = max(5.0, min(window_duration, 30.0, video_duration))
    if window_duration * count > desired_total:
        window_duration = max(5.0, desired_total / count)

    transcript, transcription_meta = _load_transcript(
        video_path,
        options.get("subtitle_path"),
        bool(options.get("enable_content_analysis", True)),
        options.get("asr_model_size", "base"),
        options.get("asr_language"),
    )
    analysis = SmartClippingEngine().analyze_video_content(
        str(video_path), max_duration=None
    )
    weights = {
        "semantic": float(options.get("semantic_weight", 0.4)),
        "visual": float(options.get("visual_weight", 0.3)),
        "audio": float(options.get("audio_weight", 0.3)),
    }
    candidates = _build_candidates(
        video_duration, window_duration, options.get("topic", ""), transcript, analysis, weights,
        maximum_duration=min(30.0, desired_total),
    )
    skipped_duplicates: List[Dict[str, Any]] = []
    selected = _select_segments(
        candidates,
        video_duration,
        count,
        intro,
        bool(options.get("include_highlights", True)),
        conclusion,
        total_budget=desired_total,
        skipped_duplicates=skipped_duplicates,
    )
    if not selected:
        raise ValueError("未找到符合时长与完整语句边界的可用片段，或所选评分维度没有有效测量；请调整时长或权重")

    warnings = []
    if len(selected) < count:
        warnings.append(f"在完整语句、不重叠和总时长约束下仅选出 {len(selected)}/{count} 段")
    if skipped_duplicates:
        spots = "、".join(f"{item['start_time']:.1f}s" for item in skipped_duplicates[:5])
        warnings.append(f"跳过 {len(skipped_duplicates)} 个与已选片段文本重复的候选（{spots}）")
    if analysis.get("degraded"):
        warnings.append("部分视听测量不可用；缺失维度已排除并重新归一化权重")
    if transcription_meta.get("source") == "unavailable":
        warnings.append("转写不可用，已使用视听窗口；切口不保证完整语句")

    render = bool(options.get("render", False))
    stem = uuid.uuid4().hex
    output_path = OUTPUT_DIR / f"multi_clip_{stem}.mp4"
    if render:
        _combine_segments(video_path, selected, output_path)

    for segment in selected:
        segment["preview_text"] = segment.get("text", "")[:300]
        segment["score"] = round(segment["score"], 4)
        segment["semantic_score"] = round(segment["semantic_score"], 4)
        segment["visual_score"] = round(segment["visual_score"], 4)
        segment["audio_score"] = round(segment["audio_score"], 4)

    # The decision list is an output in its own right, not a by-product: the
    # judgement about which moments matter is the part a dedicated editor
    # cannot do for you, and it is useful even when nothing is rendered.
    decisions, decision_errors = _write_decision_list(
        stem, selected, str(video_path), options.get("topic", ""), _probe_fps(video_path)
    )

    return {
        "status": "success",
        "output_video": f"output_data/{output_path.name}" if render else None,
        "rendered": render,
        "decision_list": decisions,
        "decision_list_errors": decision_errors,
        "warnings": warnings,
        "selected_segments": selected,
        "analysis": {
            "total_video_duration": round(video_duration, 3),
            "analyzed_segments": len(candidates),
            "selected_segments": len(selected),
            "total_output_duration": round(sum(item["duration"] for item in selected), 3),
            "transcription": transcription_meta,
            "measurement": {
                "measured_seconds": analysis.get("measured_seconds", 0.0),
                "intervals": analysis.get("measured_intervals", []),
                "errors": analysis.get("measurement_errors", []),
            },
            "selection_strategy": {
                "mode": mode,
                "weights": weights,
                "target_segments": count,
                "requested_total_duration": desired_total,
                "segment_duration": window_duration,
                "include_intro": intro,
                "include_highlights": bool(options.get("include_highlights", True)),
                "include_conclusion": conclusion,
            },
        },
        "quality_metrics": {
            "semantic_quality": round(sum(item["semantic_score"] for item in selected) / len(selected), 4),
            "visual_quality": round(sum(item["visual_score"] for item in selected) / len(selected), 4),
            "audio_quality": round(sum(item["audio_score"] for item in selected) / len(selected), 4),
            "overall_score": round(sum(item["score"] for item in selected) / len(selected), 4),
        },
        "message": f"成功生成 {len(selected)} 个片段组合的智能剪辑视频",
    }
