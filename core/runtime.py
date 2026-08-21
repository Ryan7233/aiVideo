"""Runtime paths and security boundaries shared by the API and workers."""

from __future__ import annotations

import ipaddress
import os
import socket
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import unquote, urlparse

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.getenv("AIVIDEO_DATA_DIR", PROJECT_ROOT)).expanduser().resolve()

INPUT_DIR = DATA_ROOT / "input_data"
DOWNLOAD_DIR = INPUT_DIR / "downloads"
VIDEO_UPLOAD_DIR = INPUT_DIR / "uploads"
OUTPUT_DIR = DATA_ROOT / "output_data"
PHOTO_UPLOAD_DIR = OUTPUT_DIR / "uploads"
LOG_DIR = DATA_ROOT / "logs"
MODEL_DIR = DATA_ROOT / "models"


def ensure_runtime_directories() -> None:
    """Create every directory required while importing and serving the app."""
    for directory in (
        INPUT_DIR,
        DOWNLOAD_DIR,
        VIDEO_UPLOAD_DIR,
        OUTPUT_DIR,
        PHOTO_UPLOAD_DIR,
        LOG_DIR,
        MODEL_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _is_within(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.expanduser().resolve()
    return any(resolved == root.resolve() or root.resolve() in resolved.parents for root in roots)


def resolve_media_path(
    value: str,
    *,
    must_exist: bool = True,
    allowed_roots: Optional[Iterable[Path]] = None,
) -> Path:
    """Resolve a media path and keep it inside managed runtime directories."""
    raw = unquote(value[7:] if value.startswith("file://") else value)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = DATA_ROOT / candidate
    candidate = candidate.resolve()

    roots = tuple(allowed_roots or (INPUT_DIR, OUTPUT_DIR))
    allow_unsafe = os.getenv("ALLOW_UNSAFE_LOCAL_PATHS", "false").lower() in {"1", "true", "yes"}
    if not allow_unsafe and not _is_within(candidate, roots):
        raise ValueError("文件路径必须位于 input_data 或 output_data 目录内")
    if must_exist and not candidate.is_file():
        raise FileNotFoundError(f"文件不存在: {candidate}")
    return candidate


def resolve_output_path(value: Optional[str], default_name: str, suffix: str = ".mp4") -> Path:
    """Resolve a user-selected output filename inside output_data."""
    name = Path(value).name if value else default_name
    if suffix and Path(name).suffix.lower() != suffix.lower():
        name = f"{Path(name).stem}{suffix}"
    candidate = (OUTPUT_DIR / name).resolve()
    if not _is_within(candidate, (OUTPUT_DIR,)):
        raise ValueError("输出文件必须位于 output_data 目录内")
    return candidate


def validate_remote_url(url: str) -> str:
    """Allow public HTTP(S) targets and reject loopback/private/link-local hosts."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("仅支持有效的 http/https URL")
    if parsed.username or parsed.password:
        raise ValueError("URL 不允许包含用户名或密码")

    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("不允许访问本机地址")

    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or 443)}
    except socket.gaierror as exc:
        raise ValueError(f"无法解析远程主机: {host}") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("不允许访问内网、回环或保留地址")
    return url.strip()


def download_public_file(url: str, destination: Path, max_bytes: int) -> Path:
    """Stream a public URL to disk while validating every redirect target."""
    current = validate_remote_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(6):
        response = requests.get(current, stream=True, timeout=(10, 120), allow_redirects=False)
        if response.is_redirect or response.is_permanent_redirect:
            next_url = response.headers.get("location")
            response.close()
            if not next_url:
                raise ValueError("远程服务器返回了无效重定向")
            from urllib.parse import urljoin

            current = validate_remote_url(urljoin(current, next_url))
            continue

        response.raise_for_status()
        declared_size = int(response.headers.get("content-length", "0") or 0)
        if declared_size > max_bytes:
            response.close()
            raise ValueError("远程文件超过大小限制")

        written = 0
        try:
            with destination.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError("远程文件超过大小限制")
                    output.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            response.close()
        return destination

    raise ValueError("远程地址重定向次数过多")


ensure_runtime_directories()
