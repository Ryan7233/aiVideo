"""Deployment config the test suite could not previously see.

The production profile could not have worked: nginx aliased /static/ at
output_data (the frontend's assets live inside the app), proxied five path
prefixes and left /jobs, /upload, /cover and the rest unrouted, redirected /
to /api/, referenced a certificate and an api-locations.conf that are not in
the repo, and depended on a service from a different profile. Meanwhile the
API published port 8000 on every interface with no authentication, and Redis
published 6379 with no password.
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILES = ["Dockerfile", "Dockerfile.api", "Dockerfile.worker"]


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nginx_conf():
    return (ROOT / "nginx" / "nginx.conf").read_text(encoding="utf-8")


class TestExposure:
    def test_only_the_reverse_proxy_faces_the_network(self, compose):
        """Everything else binds loopback; nginx is the deliberate entry point."""
        public = {}
        for name, service in compose["services"].items():
            for mapping in service.get("ports", []):
                if not str(mapping).startswith("127.0.0.1:"):
                    public.setdefault(name, []).append(mapping)
        assert set(public) <= {"nginx"}, f"published on all interfaces: {public}"

    def test_redis_is_not_reachable_from_the_network(self, compose):
        for mapping in compose["services"]["redis"].get("ports", []):
            assert str(mapping).startswith("127.0.0.1:"), (
                f"Redis has no password configured; {mapping} exposes the broker"
            )

    def test_the_api_key_is_passed_to_api_and_worker(self, compose):
        for name in ("api", "worker"):
            env = compose["services"][name].get("environment", [])
            assert any(item.startswith("AIVIDEO_API_KEY=") for item in env), (
                f"{name} cannot enforce auth without the key being passed through"
            )


class TestImages:
    @pytest.mark.parametrize("name", DOCKERFILES)
    def test_images_bind_all_interfaces(self, name):
        """The app defaults to loopback; a container must opt out of that."""
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "ENV API_HOST=0.0.0.0" in text, (
            f"{name} would bind loopback inside the container and be unreachable"
        )

    @pytest.mark.parametrize("name", DOCKERFILES)
    def test_images_install_a_cjk_font(self, name):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "fonts-noto-cjk" in text, f"{name}: cover titles would render as tofu"


class TestNginx:
    def test_it_proxies_everything(self, nginx_conf):
        """Path-by-path routing left most of the API unreachable."""
        active = [
            line.strip() for line in nginx_conf.splitlines()
            if "location" in line and not line.strip().startswith("#")
        ]
        assert any(line.startswith("location /") and "{" in line for line in active), active

    def test_it_does_not_alias_static_at_the_output_directory(self, nginx_conf):
        active = "\n".join(
            line for line in nginx_conf.splitlines() if not line.strip().startswith("#")
        )
        assert "/var/www/static" not in active, (
            "the frontend's assets are served by the app, not from output_data"
        )

    def test_the_root_is_not_redirected_away(self, nginx_conf):
        active = "\n".join(
            line for line in nginx_conf.splitlines() if not line.strip().startswith("#")
        )
        assert "return 302 /api/" not in active

    def test_no_active_reference_to_files_absent_from_the_repo(self, nginx_conf):
        active = "\n".join(
            line for line in nginx_conf.splitlines() if not line.strip().startswith("#")
        )
        for missing in ("api-locations.conf", "ssl_certificate"):
            assert missing not in active, (
                f"{missing} is referenced but not shipped; nginx -t would fail"
            )

    def test_upload_and_clipping_timeouts_are_generous(self, nginx_conf):
        assert "client_max_body_size 1G" in nginx_conf
        assert "proxy_read_timeout" in nginx_conf


class TestProfiles:
    def test_no_service_depends_on_another_profile(self, compose):
        services = compose["services"]
        broken = []
        for name, service in services.items():
            own = set(service.get("profiles") or [])
            for dependency in service.get("depends_on") or []:
                needed = set(services.get(dependency, {}).get("profiles") or [])
                # A dependency in a profile the dependent is not part of will
                # not be started with it.
                if needed and not (needed & own):
                    broken.append(f"{name} ({sorted(own) or 'default'}) -> {dependency} ({sorted(needed)})")
        assert not broken, broken
