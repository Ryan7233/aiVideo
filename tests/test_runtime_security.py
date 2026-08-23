
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


class TestPeerVerification:
    """DNS can change between validating a hostname and connecting to it.

    validate_remote_url resolves the name and checks the addresses; requests
    then resolves it again independently. A name that answers with a public
    address for the first lookup and an internal one for the second gets
    through. Checking the peer closes that, because it inspects the connection
    that was actually made.
    """

    @pytest.fixture
    def local_server(self):
        import http.server
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "7")
                self.end_headers()
                self.wfile.write(b"SECRET!")

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()

    def test_a_connection_to_loopback_is_rejected(self, monkeypatch, tmp_path, local_server):
        from core import runtime

        # Stand in for DNS that answered with a public address at check time.
        monkeypatch.setattr(runtime, "validate_remote_url", lambda url: url)

        with pytest.raises(ValueError, match="非公网地址"):
            runtime.download_public_file(f"{local_server}/secret", tmp_path / "out.bin", 10_000)

        assert not (tmp_path / "out.bin").exists()

    def test_the_peer_accessor_still_works(self, local_server):
        """urllib3 exposes no public API for this; fail loudly if it moves.

        download_public_file refuses to proceed when the peer cannot be
        determined, so a urllib3 upgrade that breaks the accessor turns into a
        hard failure. This makes it a CI failure instead of a production one.
        """
        import requests

        from core.runtime import _peer_address

        response = requests.get(f"{local_server}/x", stream=True, timeout=10)
        try:
            assert _peer_address(response) == "127.0.0.1", (
                "urllib3 internals changed; update _peer_address in core/runtime.py"
            )
        finally:
            response.close()

    def test_it_fails_closed_when_the_peer_is_unknown(self):
        from core.runtime import assert_public_peer

        class Opaque:
            raw = object()

        with pytest.raises(ValueError, match="无法确认"):
            assert_public_peer(Opaque())
