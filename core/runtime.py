"""Runtime paths and security boundaries shared by the API and workers."""

from __future__ import annotations

import ipaddress
import socket
import uuid
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import unquote, urlparse

import requests

from core.env import env_bool, env_path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = env_path("AIVIDEO_DATA_DIR", PROJECT_ROOT).resolve()

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
    allow_unsafe = env_bool("ALLOW_UNSAFE_LOCAL_PATHS", False)
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


def _peer_address(response) -> Optional[str]:
    """The IP actually connected to, or None if it cannot be determined.

    urllib3 does not expose this publicly. The accessors below are private and
    may move between versions, which is why the caller fails closed and a test
    asserts at least one of them still resolves.
    """
    candidates = (
        lambda: response.raw._fp.fp.raw._sock,
        lambda: response.raw._original_response.fp.raw._sock,
        lambda: response.raw._connection.sock,
    )
    for accessor in candidates:
        try:
            sock = accessor()
        except Exception:
            continue
        if sock is None:
            continue
        try:
            return sock.getpeername()[0]
        except Exception:
            continue
    return None


def assert_public_peer(response) -> None:
    """Reject a connection that actually landed on a private address.

    validate_remote_url resolves the hostname, then requests resolves it again
    when it connects; between the two, DNS can hand back an internal address.
    Checking the peer closes that window, because it inspects the connection
    that was really made rather than a name lookup.
    """
    peer = _peer_address(response)
    if peer is None:
        raise ValueError(
            "无法确认远程连接的实际地址，出于安全考虑中止下载"
            "（urllib3 内部结构可能已变化）"
        )
    if not ipaddress.ip_address(peer).is_global:
        raise ValueError(f"远程主机解析到非公网地址: {peer}")


def download_public_file(url: str, destination: Path, max_bytes: int) -> Path:
    """Stream a public URL to disk while validating every redirect target."""
    current = validate_remote_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(6):
        response = requests.get(current, stream=True, timeout=(10, 120), allow_redirects=False)
        try:
            assert_public_peer(response)
        except Exception:
            response.close()
            raise
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


# Platforms serve video behind manifests and signed URLs, so a plain HTTP GET
# cannot fetch them. yt-dlp resolves that. The URL is validated first, exactly
# as a direct download would be; yt-dlp follows its own redirects from there.
class DownloadTooLarge(ValueError):
    """Raised when a download passes the size cap."""


def _cleanup(stem: str) -> None:
    """Remove whatever yt-dlp left behind, including .part and merge temps."""
    for leftover in DOWNLOAD_DIR.glob(f"{stem}*"):
        try:
            leftover.unlink()
        except OSError:
            pass


def download_with_ytdlp(url: str, prefix: str, max_bytes: int) -> Path:
    """Fetch a platform video (YouTube, Bilibili, ...) into the download dir.

    yt-dlp does its own connection handling, so neither the size cap nor the
    per-hop peer check that guards the direct downloader applies to it. The
    cap is reinstated here with a progress hook that aborts mid-download --
    checking afterwards would mean the disk is already full.
    """
    validated = validate_remote_url(url)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_{uuid.uuid4().hex}"

    def guard(status: dict) -> None:
        seen = status.get("downloaded_bytes") or 0
        declared = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
        if seen > max_bytes or declared > max_bytes:
            raise DownloadTooLarge(
                f"远程视频超过大小限制（{max_bytes // 1048576} MB）"
            )

    options = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(DOWNLOAD_DIR / f"{stem}.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,
        "progress_hooks": [guard],
        "max_filesize": max_bytes,
    }
    try:
        import yt_dlp
    except ImportError as exc:
        raise ValueError("未安装 yt-dlp，无法解析平台链接") from exc

    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(validated, download=True)
    except DownloadTooLarge:
        _cleanup(stem)
        raise
    except Exception as exc:
        _cleanup(stem)
        raise ValueError(f"平台链接下载失败: {exc}") from exc

    requested = (info or {}).get("requested_downloads") or []
    candidate = Path(requested[0]["filepath"]) if requested and requested[0].get("filepath") else None
    if candidate is None:
        for found in sorted(DOWNLOAD_DIR.glob(f"{stem}.*")):
            if found.is_file() and found.suffix != ".part":
                candidate = found
                break
    if candidate is None or not candidate.is_file():
        _cleanup(stem)
        raise ValueError("yt-dlp 没有产出可用的视频文件")

    # max_filesize is advisory for some extractors; enforce it on the result.
    if candidate.stat().st_size > max_bytes:
        _cleanup(stem)
        raise DownloadTooLarge(f"远程视频超过大小限制（{max_bytes // 1048576} MB）")
    return candidate


def is_direct_media_url(url: str) -> bool:
    """True when the URL looks like a file a plain GET can fetch."""
    path = urlparse(url).path.lower()
    return path.endswith((".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"))


def materialize_video_source(url: str, prefix: str, max_bytes: int) -> Path:
    """Resolve a managed local video, or securely download a public remote one.

    Synchronous on purpose: background jobs call it directly, and the API wraps
    it in a worker thread.
    """
    if url.startswith("file://"):
        return resolve_media_path(url)
    if not url.lower().startswith(("http://", "https://")):
        # A managed local path, e.g. the one /upload/video hands back.
        return resolve_media_path(url)

    # A direct file URL streams through the size-capped downloader; anything
    # else is a platform page, which needs yt-dlp to resolve.
    if is_direct_media_url(url):
        validated_url = validate_remote_url(url)
        destination = DOWNLOAD_DIR / f"{prefix}_{uuid.uuid4().hex}.mp4"
        download_public_file(validated_url, destination, max_bytes)
        return destination
    return download_with_ytdlp(url, prefix, max_bytes)


ensure_runtime_directories()
