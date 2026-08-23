
import pytest

from core.runtime import INPUT_DIR, resolve_media_path, resolve_output_path, validate_remote_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/video.mp4",
        "http://localhost/video.mp4",
        "file:///etc/passwd",
        "ftp://example.com/video.mp4",
    ],
)
def test_remote_url_rejects_unsafe_targets(url):
    with pytest.raises(ValueError):
        validate_remote_url(url)


def test_media_path_is_confined_to_runtime_roots(tmp_path):
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError):
        resolve_media_path(str(outside))


def test_managed_media_path_is_allowed():
    path = INPUT_DIR / "security_test.mp4"
    path.write_bytes(b"x")
    try:
        assert resolve_media_path(str(path)) == path.resolve()
    finally:
        path.unlink(missing_ok=True)


def test_output_path_uses_basename_only():
    output = resolve_output_path("../../escape.mp4", "default.mp4")
    assert output.name == "escape.mp4"
    assert output.parent.name == "output_data"
