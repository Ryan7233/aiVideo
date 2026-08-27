"""Artefacts must be fetchable at the path the API reports.

The payload names files as "output_data/<name>", but the directory was only
mounted at /output, so every download link in the UI 404'd. And with
AIVIDEO_API_KEY set the whole UI was unusable: the page loaded and then upload,
submit, poll and download all returned 401.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.runtime import OUTPUT_DIR

FRONTEND = "frontend/app.js"


@pytest.fixture
def artefact():
    path = OUTPUT_DIR / "access_probe.md"
    path.write_text("# 剪辑清单\n", encoding="utf-8")
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


class TestReportedPathsResolve:
    def test_the_reported_prefix_is_served(self, artefact):
        """`output_data/x` from the payload has to work as a URL."""
        client = TestClient(app)
        assert client.get(f"/output_data/{artefact.name}").status_code == 200

    def test_the_older_prefix_still_works(self, artefact):
        client = TestClient(app)
        assert client.get(f"/output/{artefact.name}").status_code == 200


class TestApiKeyMode:
    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv("AIVIDEO_API_KEY", "secret123")
        return TestClient(app, raise_server_exceptions=False)

    def test_the_page_itself_stays_reachable(self, client):
        """Otherwise there is nowhere to type the key."""
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200

    def test_everything_else_needs_the_key(self, client):
        assert client.post("/jobs", json={}).status_code == 401
        assert client.get("/jobs/kinds").status_code == 401
        assert client.post("/upload/video",
                           files={"file": ("a.mp4", b"x", "video/mp4")}).status_code == 401

    def test_artefacts_need_the_key(self, client, artefact):
        assert client.get(f"/output_data/{artefact.name}").status_code == 401

    def test_the_key_gets_through(self, client, artefact):
        headers = {"X-API-Key": "secret123"}
        assert client.get("/jobs/kinds", headers=headers).status_code == 200
        assert client.get(f"/output_data/{artefact.name}", headers=headers).status_code == 200


class TestFrontendSendsTheKey:
    """The UI has to carry the key, or it is a page that can only 401."""

    @pytest.fixture(scope="class")
    def source(self):
        from pathlib import Path

        return Path(FRONTEND).read_text(encoding="utf-8")

    def test_it_sets_the_header(self, source):
        assert "X-API-Key" in source

    def test_every_fetch_carries_headers(self, source):
        """Only the helper and the auth probe may call fetch, and both of
        them attach the key; anything else would silently 401."""
        sites = [line.strip() for line in source.splitlines() if "fetch(" in line]
        assert len(sites) == 2, f"unexpected fetch call sites: {sites}"
        assert all("headers" in site for site in sites), sites

    def test_it_reacts_to_a_401(self, source):
        assert "401" in source and "revealAuth" in source


def test_the_batch_script_can_send_a_key():
    from pathlib import Path

    source = Path("scripts/run_batch.py").read_text(encoding="utf-8")
    assert "--api-key" in source
    assert "X-API-Key" in source
