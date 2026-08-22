from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from core.runtime import INPUT_DIR, OUTPUT_DIR, ensure_runtime_directories


ensure_runtime_directories()
client = TestClient(app)


def test_root_serves_frontend():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "AI Video" in response.text


def test_info_endpoint():
    response = client.get("/info")
    assert response.status_code == 200
    assert response.json()["version"] == "1.0.0"


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert "timestamp" in data


def test_segment_validation_and_simulation_label():
    invalid = client.post("/segment", json={"transcript": "", "min_sec": 25, "max_sec": 60})
    assert invalid.status_code == 422

    response = client.post(
        "/segment",
        json={"transcript": "00:10 测试字幕内容", "min_sec": 25, "max_sec": 60},
    )
    assert response.status_code == 200
    assert response.json()["mode"] == "simulation"


def test_semantic_analysis():
    response = client.post("/semantic/analyze", json={"text": "AI 技术很好，值得推荐。"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "quality_score" in data["analysis"]


def test_cut916_restricts_source_and_output():
    source = INPUT_DIR / "test_cut_source.mp4"
    source.write_bytes(b"synthetic-test-placeholder")
    try:
        with patch("api.main.safe_run_ffmpeg", return_value={"code": 0, "duration": 0.1}):
            response = client.post(
                "/cut916",
                json={
                    "src": str(source),
                    "start": "00:10",
                    "end": "00:20",
                    "out": "test_cut_output.mp4",
                },
            )
        assert response.status_code == 200
        assert Path(response.json()["out"]).parent == OUTPUT_DIR
    finally:
        source.unlink(missing_ok=True)
        (OUTPUT_DIR / "test_cut_output.mp4").unlink(missing_ok=True)


def test_upload_metadata_endpoint_is_explicitly_simulated():
    source = INPUT_DIR / "test_upload_source.mp4"
    source.write_bytes(b"placeholder")
    try:
        response = client.post("/upload", json={"path": str(source), "bucket": "test"})
        assert response.status_code == 200
        assert response.json()["status"] == "simulation"
        assert response.json()["uploaded"] is False
    finally:
        source.unlink(missing_ok=True)


def test_optional_api_key(monkeypatch):
    monkeypatch.setenv("AIVIDEO_API_KEY", "test-secret")
    denied = client.post("/semantic/analyze", json={"text": "测试文本"})
    assert denied.status_code == 401
    allowed = client.post(
        "/semantic/analyze",
        json={"text": "测试文本"},
        headers={"X-API-Key": "test-secret"},
    )
    assert allowed.status_code == 200
    assert client.get("/output/not-present.mp4").status_code == 401


def test_xiaohongshu_publish_is_explicitly_simulated():
    image = OUTPUT_DIR / "test_publish_image.jpg"
    image.write_bytes(b"placeholder")
    try:
        response = client.post(
            "/xiaohongshu/publish",
            json={
                "title": "测试",
                "content": "测试内容",
                "images": [str(image)],
                "tags": ["测试"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "simulation"
        assert data["published"] is False
        assert data["data"]["note_id"] is None
    finally:
        image.unlink(missing_ok=True)


def test_multi_segment_duration_budget_validation():
    source = INPUT_DIR / "duration_budget.mp4"
    source.write_bytes(b"placeholder")
    try:
        response = client.post(
            "/video/multi_segment_clipping",
            json={
                "video_path": str(source),
                "topic": "测试",
                "target_segments": 3,
                "total_duration": 10,
            },
        )
        assert response.status_code == 422
    finally:
        source.unlink(missing_ok=True)


def test_remote_source_download_has_its_size_limit_wired(monkeypatch):
    """MAX_FILE_SIZE was used but never imported, so every remote URL 500'd."""
    import asyncio

    from api import main
    from core import runtime

    captured = {}

    def fake_download(url, destination, max_bytes):
        captured["max_bytes"] = max_bytes
        destination.write_bytes(b"placeholder")
        return destination

    monkeypatch.setattr(runtime, "validate_remote_url", lambda url: url)
    monkeypatch.setattr(runtime, "download_public_file", fake_download)

    path = asyncio.run(main.materialize_video_source("https://example.com/a.mp4", "sizecheck"))
    try:
        assert captured["max_bytes"] == main.MAX_FILE_SIZE > 0
    finally:
        Path(path).unlink(missing_ok=True)


def test_video_extension_allowlist_is_importable():
    """Same latent NameError guarded /upload/video's extension check."""
    from api.main import ALLOWED_VIDEO_EXTENSIONS

    assert ".mp4" in ALLOWED_VIDEO_EXTENSIONS
