#!/usr/bin/env python3
"""Run a folder of long videos through the clipper and collect the cut lists.

The volume case: a shoot leaves you with a dozen files and you want to know
which moments in each are worth cutting, without sitting through them.

    python scripts/run_batch.py input_data/raw --topic 露营 --segments 3
    python scripts/run_batch.py input_data/raw --render      # also cut the mp4s

Writes each list next to the others under output_data/ and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def call(base: str, path: str, payload=None, api_key: str = "", timeout: int = 60):
    url = f"{base.rstrip('/')}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("X-API-Key", api_key)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        detail = error.read().decode()[:300]
        raise SystemExit(f"{path} 失败 ({error.code}): {detail}") from None
    except urllib.error.URLError as error:
        raise SystemExit(f"连不上 {url}：{error.reason}") from None


def wait(base: str, job_id: str, api_key: str, poll: float, timeout: float):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = call(base, f"/jobs/{job_id}", api_key=api_key)
        if job["done"]:
            return job
        time.sleep(poll)
    raise SystemExit(f"任务 {job_id} 超时")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="放长视频的目录")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="", help="服务端设置了 AIVIDEO_API_KEY 时需要")
    parser.add_argument("--topic", default="", help="用于判断片段是否切题")
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--duration", type=float, default=60.0, help="总时长（秒）")
    parser.add_argument("--asr-model", default="base", choices=["tiny", "base", "small"])
    parser.add_argument("--no-asr", action="store_true", help="跳过语音识别")
    parser.add_argument("--render", action="store_true", help="同时渲染 9:16 成片")
    parser.add_argument("--poll", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"目录不存在：{args.folder}")
    videos = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise SystemExit(f"{args.folder} 里没有视频文件")

    print(f"{len(videos)} 个文件 → {args.api}\n")
    failures = 0
    for index, video in enumerate(videos, 1):
        print(f"[{index}/{len(videos)}] {video.name}", flush=True)
        started = time.time()
        try:
            submitted = call(args.api, "/jobs", {
                "kind": "multi_segment_clipping",
                "params": {
                    "video_path": str(video),
                    "topic": args.topic,
                    "target_segments": args.segments,
                    "total_duration": args.duration,
                    "enable_content_analysis": not args.no_asr,
                    "asr_model_size": args.asr_model,
                    "render": args.render,
                },
            }, api_key=args.api_key)
            job = wait(args.api, submitted["job_id"], args.api_key, args.poll, args.timeout)
        except SystemExit as error:
            print(f"     ✗ {error}\n")
            failures += 1
            continue

        if job["status"] != "succeeded":
            print(f"     ✗ {job.get('error', job['status'])}\n")
            failures += 1
            continue

        result = job["result"]
        elapsed = time.time() - started
        segments = result.get("selected_segments", [])
        print(f"     ✓ {len(segments)} 段 / "
              f"{result['analysis']['total_output_duration']:.1f}s  ({elapsed:.0f}s)")
        for path in result.get("decision_list", {}).values():
            print(f"       {path}")
        if result.get("output_video"):
            print(f"       {result['output_video']}")
        print()

    print(f"完成 {len(videos) - failures}/{len(videos)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
