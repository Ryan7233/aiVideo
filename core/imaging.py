"""The handful of image operations this project used OpenCV for.

opencv-python bundles its own FFmpeg shared libraries, and faster-whisper
pulls in PyAV, which bundles a different major version. Loading both puts two
copies of libavdevice in one process; macOS reports duplicate Objective-C
classes and warns about "spurious casting failures and mysterious crashes".

Only ten OpenCV calls were used, all of them expressible with numpy, SciPy and
Pillow, which are already dependencies. Kernels and coefficients match
OpenCV's so the scores these feed stay comparable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# OpenCV's BGR->grey coefficients.
_B, _G, _R = 0.114, 0.587, 0.299

# cv2.Sobel(ksize=3) kernels.
_SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])

# cv2.Laplacian(ksize=1).
_LAPLACIAN = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def imread_bgr(path: str) -> Optional[np.ndarray]:
    """Read an image as a BGR uint8 array, or None. Mirrors cv2.imread."""
    try:
        from PIL import Image

        if not Path(path).is_file():
            return None
        with Image.open(path) as handle:
            rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        return rgb[:, :, ::-1].copy()
    except Exception as exc:
        logger.debug("Could not read %s: %s", path, exc)
        return None


def to_gray(image: np.ndarray, source: str = "BGR") -> np.ndarray:
    """Luma as uint8, using OpenCV's coefficients and rounding."""
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        return array.astype(np.uint8)
    first, second, third = (_B, _G, _R) if source.upper() == "BGR" else (_R, _G, _B)
    grey = array[:, :, 0] * first + array[:, :, 1] * second + array[:, :, 2] * third
    return np.clip(np.round(grey), 0, 255).astype(np.uint8)


def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize to (width, height), like cv2.resize's default."""
    from PIL import Image

    array = np.asarray(image)
    if array.ndim == 3:
        pil = Image.fromarray(np.ascontiguousarray(array[:, :, ::-1], dtype=np.uint8))
        resized = pil.resize(size, Image.BILINEAR)
        return np.asarray(resized, dtype=np.uint8)[:, :, ::-1].copy()
    pil = Image.fromarray(array.astype(np.uint8))
    return np.asarray(pil.resize(size, Image.BILINEAR), dtype=np.uint8)


def _convolve(grey: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Correlate with edge replication, which is OpenCV's default border."""
    from scipy.ndimage import correlate

    return correlate(grey.astype(np.float64), kernel, mode="nearest")


def sobel(grey: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx and not dy:
        return _convolve(grey, _SOBEL_X)
    if dy and not dx:
        return _convolve(grey, _SOBEL_Y)
    raise ValueError("sobel takes exactly one of dx/dy")


def laplacian_variance(grey: np.ndarray) -> float:
    """Variance of the Laplacian -- the usual blur/sharpness measure."""
    return float(_convolve(grey, _LAPLACIAN).var())


def _non_maximum_suppression(magnitude: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Thin gradient ridges to single-pixel edges.

    Without this an edge counts as every pixel across the slope, which reads
    far higher than cv2.Canny on textured images.
    """
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180
    thinned = np.zeros_like(magnitude)
    padded = np.pad(magnitude, 1, mode="constant")

    # Four gradient directions; compare each pixel with its two neighbours
    # along the gradient.
    sectors = [
        ((angle < 22.5) | (angle >= 157.5), (0, 1)),    # horizontal gradient
        ((angle >= 22.5) & (angle < 67.5), (1, 1)),     # diagonal
        ((angle >= 67.5) & (angle < 112.5), (1, 0)),    # vertical gradient
        ((angle >= 112.5) & (angle < 157.5), (1, -1)),  # anti-diagonal
    ]
    rows, cols = magnitude.shape
    for mask, (dy, dx) in sectors:
        ahead = padded[1 + dy : 1 + dy + rows, 1 + dx : 1 + dx + cols]
        behind = padded[1 - dy : 1 - dy + rows, 1 - dx : 1 - dx + cols]
        keep = mask & (magnitude >= ahead) & (magnitude > behind)
        thinned[keep] = magnitude[keep]
    return thinned


def canny(grey: np.ndarray, low: float = 50, high: float = 150) -> np.ndarray:
    """Canny edge map as uint8 0/255, matching cv2.Canny's output shape.

    Sobel gradients with the L1 norm (cv2's default), non-maximum suppression,
    then hysteresis: weak pixels survive only if connected to a strong one.

    Measured against cv2.Canny on FFmpeg's test patterns, the edge-pixel
    fraction lands within 0.63x-1.17x, and the callers use it comparatively --
    ranking photos against each other -- rather than as an absolute.
    """
    gx = sobel(grey, 1, 0)
    gy = sobel(grey, 0, 1)
    # cv2.Canny defaults to L2gradient=False, i.e. the L1 norm.
    magnitude = _non_maximum_suppression(np.abs(gx) + np.abs(gy), gx, gy)

    strong = magnitude >= high
    weak = magnitude >= low
    if not weak.any():
        return np.zeros_like(grey, dtype=np.uint8)

    from scipy.ndimage import label

    labels, count = label(weak)
    if count == 0:
        return np.zeros_like(grey, dtype=np.uint8)
    keep = np.zeros(count + 1, dtype=bool)
    keep[np.unique(labels[strong])] = True
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def edge_density(grey: np.ndarray, low: float = 50, high: float = 150) -> float:
    """Fraction of pixels lying on an edge."""
    edges = canny(grey, low, high)
    return float((edges > 0).mean()) if edges.size else 0.0
