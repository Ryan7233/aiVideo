"""Configuration must not be able to break startup or delete data.

Following the README -- copy env.example to .env, start the app -- used to
crash on import: env.example ships keys with blank values to document them,
and os.getenv returns "" for those rather than the default, so int("") raised,
the job database resolved to ".", and the Whisper prompt became empty.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from core import job_store, retention
from core.env import env_bool, env_float, env_int, env_list, env_path, env_str

ROOT = Path(__file__).resolve().parent.parent


class TestEnvHelpers:
    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_falls_back_to_the_default(self, monkeypatch, value):
        monkeypatch.setenv("AIVIDEO_TEST_VALUE", value)
        assert env_int("AIVIDEO_TEST_VALUE", 7) == 7
        assert env_float("AIVIDEO_TEST_VALUE", 1.5) == 1.5
        assert env_str("AIVIDEO_TEST_VALUE", "fallback") == "fallback"
        assert env_bool("AIVIDEO_TEST_VALUE", True) is True
        assert env_path("AIVIDEO_TEST_VALUE", Path("/tmp/x")) == Path("/tmp/x")

    def test_unparseable_warns_rather_than_raising(self, monkeypatch):
        monkeypatch.setenv("AIVIDEO_TEST_VALUE", "not-a-number")
        assert env_int("AIVIDEO_TEST_VALUE", 3) == 3
        assert env_float("AIVIDEO_TEST_VALUE", 3.5) == 3.5

    def test_real_values_are_honoured(self, monkeypatch):
        monkeypatch.setenv("AIVIDEO_TEST_VALUE", " 42 ")
        assert env_int("AIVIDEO_TEST_VALUE", 1) == 42
        assert env_str("AIVIDEO_TEST_VALUE", "x") == "42"

    def test_bool_accepts_the_usual_spellings(self, monkeypatch):
        for raw, expected in [("yes", True), ("on", True), ("0", False), ("off", False)]:
            monkeypatch.setenv("AIVIDEO_TEST_VALUE", raw)
            assert env_bool("AIVIDEO_TEST_VALUE", not expected) is expected

    def test_list_drops_blank_entries(self, monkeypatch):
        monkeypatch.setenv("AIVIDEO_TEST_VALUE", ".mp4, ,.mov,")
        assert env_list("AIVIDEO_TEST_VALUE", ".x") == [".mp4", ".mov"]


class TestEnvExample:
    def test_it_has_no_blank_values(self):
        """A blank value is read as "" and never reaches the default."""
        blanks = [
            line for line in (ROOT / "env.example").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#") and line.rstrip().endswith("=")
        ]
        assert not blanks, f"comment these out instead: {blanks}"

    def test_the_app_starts_with_it(self, tmp_path):
        """The exact README quick-start: copy env.example, import the app."""
        (tmp_path / ".env").write_text((ROOT / "env.example").read_text(encoding="utf-8"))
        script = (
            "from dotenv import load_dotenv; load_dotenv('.env')\n"
            f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
            "import api.main\n"
            "from core.config import API_HOST\n"
            "from core.concurrency import MAX_CONCURRENT_MEDIA_JOBS\n"
            "from core.whisper_asr import simplified_chinese_prompt\n"
            "assert MAX_CONCURRENT_MEDIA_JOBS >= 1, MAX_CONCURRENT_MEDIA_JOBS\n"
            "assert simplified_chinese_prompt(), 'empty ASR prompt'\n"
            "print('OK', API_HOST)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=180
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "OK" in result.stdout

    def test_it_does_not_bind_all_interfaces_by_default(self):
        """Unauthenticated uploads and transcoding should not be LAN-facing."""
        text = (ROOT / "env.example").read_text(encoding="utf-8")
        active = [
            line.strip() for line in text.splitlines()
            if line.strip().startswith("API_HOST=")
        ]
        assert active == ["API_HOST=127.0.0.1"], active

    def test_the_default_host_is_loopback(self):
        from core.config import DEFAULT_API_HOST

        assert DEFAULT_API_HOST == "127.0.0.1"


class TestRetentionZeroMeansDisabled:
    """0 disables a rule. It used to mean "cutoff = now", i.e. delete all."""

    @pytest.fixture(autouse=True)
    def store(self, tmp_path):
        job_store.reset_for_tests(tmp_path / "jobs.db")

    def _finished_job(self):
        job_id = job_store.create_job("multi_segment_clipping", {})
        job_store.mark_succeeded(job_id, {})
        return job_id

    @pytest.mark.parametrize("setting", ["0", "0.5", "30"])
    def test_a_fresh_job_is_never_swept(self, monkeypatch, setting):
        monkeypatch.setenv("JOB_RETENTION_DAYS", setting)
        monkeypatch.setenv("OUTPUT_RETENTION_DAYS", "0")
        monkeypatch.setenv("UPLOAD_RETENTION_DAYS", "0")
        job_id = self._finished_job()

        assert retention.run_sweep()["removed_jobs"] == 0
        assert job_store.get_job(job_id) is not None

    def test_fractional_days_are_not_truncated_to_zero(self):
        """int(0.5) == 0 made the cutoff "now" and wiped everything."""
        self._finished_job()
        assert job_store.delete_jobs_older_than(0.5) == 0

    def test_a_non_positive_window_never_deletes(self):
        self._finished_job()
        assert job_store.delete_jobs_older_than(0) == 0
        assert job_store.delete_jobs_older_than(-5) == 0

    def test_an_old_job_is_still_swept_when_enabled(self, monkeypatch):
        import sqlite3

        job_id = self._finished_job()
        with sqlite3.connect(str(job_store.DB_PATH)) as connection:
            connection.execute(
                "UPDATE jobs SET finished_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (job_id,)
            )
        assert job_store.delete_jobs_older_than(1) == 1
