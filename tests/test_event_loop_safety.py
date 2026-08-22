"""Guard against long-running work being called straight from a coroutine.

ASR and FFmpeg run for minutes. When they were invoked directly inside an
``async def`` route they blocked the single event loop, so every other request
-- including /health -- queued behind one clipping job.
"""

import ast
from pathlib import Path

import pytest

API_MAIN = Path(__file__).resolve().parent.parent / "api" / "main.py"

# Callables that must never be awaited-through directly from a coroutine.
BLOCKING_CALLS = {
    "transcribe_video",
    "safe_run_ffmpeg",
    "select_best_segments_with_asr",
    "detect_highlights",
    "generate_smart_cover",
    "analyze_video_intelligence",
    "get_smart_segments",
    "process_multi_segment_video",
}


def _called_names(node: ast.AST):
    """Names of every call in ``node`` that is not already inside to_thread."""
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Attribute) and func.attr == "to_thread":
            continue  # asyncio.to_thread(fn, ...) passes fn, it does not call it
        if isinstance(func, ast.Attribute):
            yield func.attr
        elif isinstance(func, ast.Name):
            yield func.id


@pytest.fixture(scope="module")
def coroutine_routes():
    tree = ast.parse(API_MAIN.read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)]


def test_no_blocking_media_work_inside_coroutines(coroutine_routes):
    offenders = {}
    for route in coroutine_routes:
        hits = sorted(set(_called_names(route)) & BLOCKING_CALLS)
        if hits:
            offenders[route.name] = hits
    assert not offenders, (
        "these coroutines call blocking media work directly; wrap them in "
        f"asyncio.to_thread(...): {offenders}"
    )


def test_pipeline_bodies_stay_synchronous(coroutine_routes):
    """The extracted helpers must remain plain functions a thread can run."""
    tree = ast.parse(API_MAIN.read_text(encoding="utf-8"))
    helpers = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_run_xiaohongshu")
    }
    assert helpers == {"_run_xiaohongshu_pipeline", "_run_xiaohongshu_pipeline_pro"}

    names = {route.name for route in coroutine_routes}
    assert {"xiaohongshu_pipeline", "xiaohongshu_pipeline_pro"} <= names
