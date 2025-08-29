#!/usr/bin/env python3
"""
纯代码端到端批处理脚本（不依赖 Dify）。

流程：segment -> sanitize -> cut916 -> 可选 burnsub -> upload
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import requests
from loguru import logger

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")


def post_json(path: str, payload: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
    url = f"{API_BASE}{path}"
    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"POST {path} attempt {attempt+1} failed: {e}")
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))


def sanitize_clips(clips: List[Dict[str, Any]], min_sec: int, max_sec: int) -> List[Dict[str, Any]]:
    def to_seconds(ts: str) -> int:
        parts = list(map(int, ts.split(":")))
        if len(parts) == 2:
            mm, ss = parts
            return mm * 60 + ss
        hh, mm, ss = parts
        return hh * 3600 + mm * 60 + ss

    sanitized: List[Dict[str, Any]] = []
    for c in clips:
        s, e = to_seconds(c["start"]), to_seconds(c["end"])
        if e <= s:
            continue
        dur = e - s
        if dur < min_sec:
            continue
        if dur > max_sec:
            e = s + max_sec
        sanitized.append({"start": c["start"], "end": f"{e//60:02d}:{e%60:02d}", "reason": c.get("reason", "")})
    return sanitized


def parse_srt(path: str) -> List[Tuple[str, str, str]]:
    """Parse a simple SRT file -> list of (start, end, text)."""
    items: List[Tuple[str, str, str]] = []
    if not os.path.exists(path):
        return items
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    for b in blocks:
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        # locate time line
        time_line = next((l for l in lines if "-->" in l), "")
        if not time_line:
            # maybe index on first line
            if len(lines) >= 2 and "-->" in lines[1]:
                time_line = lines[1]
                text_lines = lines[2:]
            else:
                continue
        else:
            idx = lines.index(time_line)
            text_lines = lines[idx + 1 :]
        try:
            start, end = [t.strip().replace(",", ":")[0:8] for t in time_line.split("-->")]
        except Exception:
            continue
        text = " ".join(text_lines)
        items.append((start, end, text))
    return items


def _format_time_srt(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def try_python_asr_to_srt(src: str, out_srt: str) -> bool:
    """Try to run ASR via Python libs (openai-whisper or faster_whisper). Returns True on success."""
    # 1) openai-whisper
    try:
        import whisper  # type: ignore
        model = whisper.load_model("small")
        result = model.transcribe(src, language="zh", task="transcribe")
        segments = result.get("segments", [])
        with open(out_srt, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = _format_time_srt(float(seg.get("start", 0)))
                end = _format_time_srt(float(seg.get("end", 0)))
                text = str(seg.get("text", "")).strip()
                if not text:
                    continue
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        return True
    except Exception as e:
        logger.warning(f"openai-whisper ASR 失败: {e}")

    # 2) faster-whisper
    try:
        from faster_whisper import WhisperModel  # type: ignore
        model = WhisperModel("small", device="cpu")
        segments, _ = model.transcribe(src, language="zh")
        with open(out_srt, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = _format_time_srt(seg.start)
                end = _format_time_srt(seg.end)
                text = seg.text.strip()
                if not text:
                    continue
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        return True
    except Exception as e:
        logger.warning(f"faster-whisper ASR 失败: {e}")

    return False


def score_text(t: str) -> float:
    """Heuristic highlight score for a caption line."""
    score = 0.0
    # punctuation/emphasis
    if "！" in t or "!" in t:
        score += 2
    if "？" in t or "?" in t:
        score += 1.2
    # numbers / steps
    if any(ch.isdigit() for ch in t):
        score += 1.0
    # hooks/keywords
    hooks = ["秘诀", "爆点", "亮点", "关键", "最", "注意", "避坑", "做法", "案例", "总结", "教你", "步骤", "因此", "所以"]
    score += sum(0.8 for w in hooks if w in t)
    # length cap
    score += min(len(t) / 20.0, 1.5)
    return score


def select_highlight_windows(
    srt_items: List[Tuple[str, str, str]], min_sec: int, max_sec: int, top_k: int = 3
) -> List[Dict[str, str]]:
    def to_s(ts: str) -> int:
        parts = list(map(int, ts.split(":")))
        if len(parts) == 2:
            mm, ss = parts
            return mm * 60 + ss
        hh, mm, ss = parts
        return hh * 3600 + mm * 60 + ss

    # sliding window by captions, accumulate score
    n = len(srt_items)
    windows: List[Tuple[float, int, int]] = []  # (score, i_start, i_end)
    for i in range(n):
        start_s = to_s(srt_items[i][0])
        acc = 0.0
        j = i
        while j < n:
            acc += score_text(srt_items[j][2])
            end_s = to_s(srt_items[j][1])
            dur = end_s - start_s
            if dur >= min_sec:
                windows.append((acc, i, j))
            if dur >= max_sec:
                break
            j += 1

    # sort by score descending, apply non-overlap suppression
    windows.sort(key=lambda x: x[0], reverse=True)
    chosen: List[Tuple[int, int]] = []
    for _, i, j in windows:
        if len(chosen) >= top_k:
            break
        # suppress overlap: accept if start or end is outside existing ranges
        ok = True
        for ci, cj in chosen:
            if not (j < ci or i > cj):
                ok = False
                break
        if ok:
            chosen.append((i, j))

    clips: List[Dict[str, str]] = []
    for i, j in chosen:
        s = srt_items[i][0]
        e = srt_items[j][1]
        clips.append({"start": s, "end": e, "reason": "highlight"})
    return clips


def run_pipeline(
    src: str,
    transcript: str,
    out_dir: str = "output_data",
    min_sec: int = 25,
    max_sec: int = 60,
    burn_sub: bool = False,
    srt_path: str | None = None,
    auto_asr: bool = False,
) -> Dict[str, Any]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 0) 若开启自动 ASR 且没有 SRT，则尝试调用本地 whisper 生成 SRT
    generated_srt: str | None = None
    if auto_asr and (not (os.path.exists(transcript) and transcript.lower().endswith(".srt"))):
        tmp_srt = str(Path(out_dir) / "asr_tmp.srt")
        try:
            # 优先使用 faster-whisper 命令（若已安装），否则尝试 openai/whisper CLI
            # 这两个命令均为可选依赖；未安装时会抛错，脚本不退出，仅打印提示
            cmd = f"whisper '{src}' --model small --language zh --task transcribe --output_format srt --output_dir '{out_dir}'"
            rc = os.system(cmd)
            if rc == 0:
                guess = Path(out_dir) / (Path(src).stem + ".srt")
                if guess.exists():
                    transcript = str(guess)
                    generated_srt = transcript
                elif Path(tmp_srt).exists():
                    transcript = tmp_srt
                    generated_srt = transcript
            else:
                # 尝试 Python 版 ASR 回退
                ok = try_python_asr_to_srt(src, tmp_srt)
                if ok:
                    transcript = tmp_srt
                    generated_srt = transcript
                else:
                    logger.warning("whisper CLI 未找到或执行失败，且 Python ASR 回退也失败。可安装: pip install openai-whisper 或 faster-whisper")
        except Exception as e:
            logger.warning(f"自动 ASR 失败: {e}")

    # 1) 生成候选片段：优先解析 SRT 做亮点抓取；否则走 /segment（大模型分析）
    clips: List[Dict[str, Any]]
    if os.path.exists(transcript) and transcript.lower().endswith(".srt"):
        srt_items = parse_srt(transcript)
        clips = select_highlight_windows(srt_items, min_sec, max_sec, top_k=3)
        logger.info(f"Highlight-selected {len(clips)} clips from SRT")
    else:
        seg_payload = {"transcript": transcript, "min_sec": min_sec, "max_sec": max_sec}
        seg = post_json("/segment", seg_payload)
        clips = seg.get("clips", [])
        logger.info(f"AI suggested {len(clips)} clips")

    # 2) sanitize
    clips = sanitize_clips(clips, min_sec, max_sec)
    logger.info(f"Sanitized to {len(clips)} clips")

    # 决定要不要烧字幕：如果提供了 srt_path 或者自动 ASR 生成了 SRT，则默认烧
    effective_srt = srt_path or generated_srt
    do_burn = burn_sub or bool(effective_srt)

    results: List[str] = []
    for i, c in enumerate(clips, 1):
        out_path = str(Path(out_dir) / f"clip_{i:02d}_916.mp4")
        cut_payload = {"src": src, "start": c["start"], "end": c["end"], "out": out_path}
        cut_res = post_json("/cut916", cut_payload)
        logger.info(f"cut916 ok -> {cut_res.get('out', out_path)}")

        if do_burn and effective_srt:
            burn_out = str(Path(out_dir) / f"clip_{i:02d}_916_sub.mp4")
            burn_payload = {"src": out_path, "srt": effective_srt, "out": burn_out}
            post_json("/burnsub", burn_payload)
            out_path = burn_out

        # 3) upload (mock)
        up = post_json("/upload", {"path": out_path, "bucket": os.getenv("UPLOAD_BUCKET", "clips")})
        results.append(up.get("url", out_path))

    return {"count": len(results), "results": results}


def main():
    import argparse

    p = argparse.ArgumentParser(description="Run batch pipeline without Dify")
    p.add_argument("--src", required=True, help="source video path")
    p.add_argument("--transcript", required=True, help="transcript text or path to .txt")
    p.add_argument("--out_dir", default="output_data")
    p.add_argument("--min_sec", type=int, default=25)
    p.add_argument("--max_sec", type=int, default=60)
    p.add_argument("--burn_sub", action="store_true")
    p.add_argument("--srt", default=None)
    p.add_argument("--auto_asr", action="store_true", help="auto-generate SRT with whisper if missing")
    args = p.parse_args()

    transcript = args.transcript
    if os.path.exists(transcript) and transcript.lower().endswith(".txt"):
        transcript = Path(transcript).read_text(encoding="utf-8")

    logger.add("logs/batch.log", rotation="10 MB", level="INFO")
    logger.info(f"API_BASE={API_BASE}")

    out = run_pipeline(
        src=args.src,
        transcript=transcript,
        out_dir=args.out_dir,
        min_sec=args.min_sec,
        max_sec=args.max_sec,
        burn_sub=args.burn_sub,
        srt_path=args.srt,
        auto_asr=args.auto_asr,
    )

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())


