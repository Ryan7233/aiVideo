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


def test_no_fake_async_media_helpers(coroutine_routes):
    """`async def` with no await inside runs entirely on the caller's loop.

    The cover and collage generators were declared async and contained only
    Pillow, clustering and PNG encoding. Awaiting one from a route blocked
    /health for the whole render. They are plain functions now, reached
    through asyncio.to_thread; this keeps them that way.
    """
    import pathlib

    offenders = []
    for path in sorted(pathlib.Path("core").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            if any(isinstance(child, (ast.Await, ast.AsyncFor, ast.AsyncWith))
                   for child in ast.walk(node)):
                continue
            body_lines = node.end_lineno - node.lineno
            # Small stubs are harmless; the problem is real work on the loop.
            if body_lines >= 20:
                offenders.append(f"{path.name}::{node.name} ({body_lines} lines)")

    assert not offenders, (
        "these are declared async but never await, so they run synchronously "
        f"on the event loop: {offenders}"
    )


def test_no_await_on_a_synchronous_function():
    """`await f()` where f is a plain function raises at runtime.

    Converting the media helpers from fake-async left two of these behind. The
    first check only compared names within a single file, so it missed
    api/main.py awaiting a function defined in core/ -- which is exactly how
    /xiaohongshu/render_page?mode=single kept returning 500. This one resolves
    across modules.
    """
    import pathlib

    files = (
        list(pathlib.Path("core").glob("*.py"))
        + list(pathlib.Path("routers").glob("*.py"))
        + list(pathlib.Path("worker").glob("*.py"))
        + [pathlib.Path("api/main.py")]
    )

    sync_defs, async_defs = {}, set()
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                sync_defs.setdefault(node.name, set()).add(path.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                async_defs.add(node.name)

    offenders = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Await) or not isinstance(node.value, ast.Call):
                continue
            func = node.value.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name and name in sync_defs and name not in async_defs:
                where = ", ".join(sorted(sync_defs[name]))
                offenders.append(f"{path}:{node.lineno} awaits {name}() (sync def in {where})")

    assert not offenders, offenders


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
