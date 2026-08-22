"""Pins the HTTP surface so it converges instead of growing by accident.

The app reached 50 routes while the built-in UI used 14. Rather than delete
everything uncalled, the rule applied was: remove surface that is fake, a
no-op, or can never become real; keep and document surface that does real work
even if nothing in this repo calls it yet.
"""

import pytest

from api.main import app


# Removed deliberately. Listed so a reintroduction is a conscious decision:
#   /xiaohongshu/authorize, /profile, /note/{id}/stats returned fabricated
#     OAuth sessions, follower counts and note statistics. Xiaohongshu has no
#     open third-party API, so these could never become real.
#   /xiaohongshu/edit_text returned "文案编辑成功" without editing anything.
REMOVED = {
    "/xiaohongshu/authorize",
    "/xiaohongshu/profile",
    "/xiaohongshu/note/{note_id}/stats",
    "/xiaohongshu/edit_text",
}

# Real capabilities that simply had no in-repo caller. Kept and documented.
UNCALLED_BUT_REAL = {
    "/asr/info",
    "/asr/transcribe",
    "/asr/extract_audio",
    "/auto_intro",
    "/burnsub",
    "/collage/layouts",
    "/cover/templates",
    "/xiaohongshu/layouts",
    "/image/decorate",
    "/image/decorations/smart",
}

# Everything the built-in UI or the documented workflow depends on.
LOAD_BEARING = {
    "/",
    "/health",
    "/info",
    "/jobs",
    "/jobs/{job_id}",
    "/jobs/kinds",
    "/upload/video",
    "/upload/photos",
    "/video/multi_segment_clipping",
    "/xiaohongshu/pipeline",
    "/xiaohongshu/pipeline_pro",
    "/xiaohongshu/generate_collage",
    "/collage/generate_advanced",
    "/cover/generate",
    "/analyze_video",
    "/semantic/analyze",
}


@pytest.fixture(scope="module")
def schema():
    """The OpenAPI document is authoritative: included routers stay nested in
    ``app.routes`` on this FastAPI version, so walking that misses /jobs."""
    return app.openapi()


@pytest.fixture(scope="module")
def paths(schema):
    return set(schema["paths"])


def _operations(schema, path):
    return [op for method, op in schema["paths"][path].items()
            if method in {"get", "post", "put", "delete", "patch"}]


def test_load_bearing_routes_all_exist(paths):
    missing = LOAD_BEARING - paths
    assert not missing, f"routes the UI or docs depend on are gone: {sorted(missing)}"


def test_removed_routes_stay_removed(paths):
    back = sorted(REMOVED & paths)
    assert not back, (
        f"routes removed as fake or no-op are routed again: {back}. "
        "If one is now backed by a real implementation, drop it from REMOVED."
    )


def test_kept_routes_still_exist(paths):
    """These were audited and deliberately kept; losing them silently is a bug."""
    missing = UNCALLED_BUT_REAL - paths
    assert not missing, f"deliberately-kept routes disappeared: {sorted(missing)}"


def test_nothing_is_left_marked_deprecated(schema, paths):
    """Deprecation was resolved either way -- deleted, or kept and documented."""
    stragglers = sorted(
        path for path in paths
        if any(op.get("deprecated") for op in _operations(schema, path))
    )
    assert not stragglers, f"still marked deprecated with no decision: {stragglers}"


def test_every_route_is_tagged(schema, paths):
    """Untagged routes all land in one bucket in /docs, which is unreadable."""
    untagged = sorted(
        path for path in paths
        if any(not op.get("tags") for op in _operations(schema, path))
    )
    assert not untagged, f"routes with no OpenAPI tag: {untagged}"


def test_surface_has_not_grown(paths):
    """A ratchet: the route count may shrink, never grow, without a decision."""
    assert len(paths) <= 56, (
        f"the API grew to {len(paths)} paths. If the addition is intended, "
        "raise this bound deliberately."
    )


def test_post_bodies_are_not_query_parameters(schema, paths):
    """A POST that takes its payload in the query string is an API bug."""
    offenders = []
    for path in paths:
        post = schema["paths"][path].get("post")
        if not post:
            continue
        query_params = [
            param["name"] for param in post.get("parameters", [])
            if param.get("in") == "query" and param.get("required")
        ]
        # Path parameters are fine; required query params on a POST are not.
        if query_params and not post.get("requestBody"):
            offenders.append(f"{path}: {query_params}")
    assert not offenders, f"POST routes taking a required body via query string: {offenders}"


def test_openapi_schema_builds(schema):
    assert schema["info"]["title"] == "AI Video Clipper API"
    assert schema["paths"]
