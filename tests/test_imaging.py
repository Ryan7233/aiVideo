"""The numpy/Pillow replacements for the OpenCV calls this project used.

OpenCV bundles its own FFmpeg libraries and faster-whisper pulls in PyAV,
which bundles a different major version. With both loaded, macOS reported
duplicate Objective-C classes and warned about "spurious casting failures and
mysterious crashes".

These tests pin the properties the callers depend on. Where OpenCV is still
installed they also compare against it directly; where it is not, they check
the invariants on their own.
"""

import numpy as np
import pytest
from PIL import Image

from core import imaging

try:  # only used for the comparison tests
    import cv2
except ImportError:
    cv2 = None


requires_cv2 = pytest.mark.skipif(cv2 is None, reason="OpenCV not installed (it is no longer a dependency)")


@pytest.fixture
def images(tmp_path):
    rng = np.random.default_rng(7)
    shapes = np.zeros((240, 320, 3), np.uint8)
    shapes[60:180, 80:240] = (220, 180, 60)
    shapes[100:140, 140:200] = (20, 40, 90)
    made = {
        "flat": np.full((240, 320, 3), (40, 120, 200), np.uint8),
        "gradient": np.tile(np.linspace(0, 255, 320, dtype=np.uint8), (240, 1))[:, :, None].repeat(3, 2),
        "noise": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
        "shapes": shapes,
    }
    paths = {}
    for name, array in made.items():
        path = tmp_path / f"{name}.png"
        Image.fromarray(array).save(path)
        paths[name] = str(path)
    return paths


class TestReading:
    def test_returns_bgr(self, tmp_path):
        path = tmp_path / "rgb.png"
        Image.fromarray(np.full((4, 4, 3), (10, 20, 30), np.uint8)).save(path)
        pixel = imaging.imread_bgr(str(path))[0, 0]
        assert tuple(pixel) == (30, 20, 10), "channels should be BGR like cv2.imread"

    def test_missing_file_returns_none(self):
        assert imaging.imread_bgr("/nope/missing.png") is None

    def test_unreadable_file_returns_none(self, tmp_path):
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"not an image")
        assert imaging.imread_bgr(str(broken)) is None


class TestGrayscale:
    def test_uses_luma_weights(self):
        # Pure blue in BGR -> 0.114 * 255
        assert imaging.to_gray(np.full((2, 2, 3), (255, 0, 0), np.uint8))[0, 0] == 29

    def test_rgb_source_flips_the_weights(self):
        assert imaging.to_gray(np.full((2, 2, 3), (255, 0, 0), np.uint8), source="RGB")[0, 0] == 76

    def test_already_grey_passes_through(self):
        grey = np.full((3, 3), 120, np.uint8)
        assert (imaging.to_gray(grey) == grey).all()

    def test_output_is_uint8(self):
        assert imaging.to_gray(np.zeros((4, 4, 3), np.uint8)).dtype == np.uint8


class TestGradients:
    def test_sobel_detects_a_vertical_edge(self):
        grey = np.zeros((20, 20), np.uint8)
        grey[:, 10:] = 255
        assert np.abs(imaging.sobel(grey, 1, 0)).max() > 500
        assert np.abs(imaging.sobel(grey, 0, 1)).max() < 1e-6

    def test_sobel_rejects_ambiguous_arguments(self):
        grey = np.zeros((5, 5), np.uint8)
        with pytest.raises(ValueError):
            imaging.sobel(grey, 1, 1)

    def test_laplacian_variance_separates_sharp_from_flat(self):
        flat = np.full((60, 60), 128, np.uint8)
        sharp = np.zeros((60, 60), np.uint8)
        sharp[:, ::2] = 255
        assert imaging.laplacian_variance(flat) == pytest.approx(0.0, abs=1e-9)
        assert imaging.laplacian_variance(sharp) > 1000


class TestEdges:
    def test_flat_image_has_no_edges(self):
        assert imaging.edge_density(np.full((50, 50), 90, np.uint8)) == 0.0

    def test_edges_are_found_on_a_step(self):
        grey = np.zeros((50, 50), np.uint8)
        grey[:, 25:] = 255
        assert 0.0 < imaging.edge_density(grey) < 0.15

    def test_canny_output_matches_cv2_conventions(self):
        grey = np.zeros((30, 30), np.uint8)
        grey[:, 15:] = 255
        edges = imaging.canny(grey)
        assert edges.dtype == np.uint8
        assert set(np.unique(edges)) <= {0, 255}
        assert edges.shape == grey.shape

    def test_non_maximum_suppression_thins_the_ridge(self):
        """Without it an edge counts as every pixel across the slope."""
        grey = np.zeros((40, 40), np.uint8)
        grey[:, 20:] = 255
        edges = imaging.canny(grey)
        # A single step should light up roughly one column, not several.
        columns_with_edges = (edges > 0).any(axis=0).sum()
        assert columns_with_edges <= 3, f"{columns_with_edges} columns; the ridge was not thinned"


@requires_cv2
class TestMatchesOpenCV:
    """Direct comparison, for as long as OpenCV happens to be installed."""

    def test_imread_is_identical(self, images):
        for path in images.values():
            assert (imaging.imread_bgr(path) == cv2.imread(path)).all()

    def test_grayscale_within_one_level(self, images):
        for path in images.values():
            mine = imaging.to_gray(imaging.imread_bgr(path))
            theirs = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
            assert np.abs(mine.astype(int) - theirs.astype(int)).max() <= 1

    def test_laplacian_variance_within_a_few_percent(self, images):
        for name, path in images.items():
            grey = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
            theirs = cv2.Laplacian(grey, cv2.CV_64F).var()
            mine = imaging.laplacian_variance(imaging.to_gray(imaging.imread_bgr(path)))
            if theirs < 1e-9:
                assert mine < 1e-6
            else:
                assert mine / theirs == pytest.approx(1.0, rel=0.05), name

    def test_sobel_within_a_few_percent(self, images):
        for name, path in images.items():
            grey = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
            theirs = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
            mine = imaging.sobel(imaging.to_gray(imaging.imread_bgr(path)), 1, 0)
            scale = np.abs(theirs).mean() or 1.0
            assert np.abs(theirs - mine).mean() / scale < 0.05, name

    def test_edge_density_stays_in_the_same_range(self, images):
        """Measured 0.63x-1.17x of cv2.Canny across FFmpeg's test patterns."""
        for name, path in images.items():
            grey = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
            theirs = float((cv2.Canny(grey, 50, 150) > 0).mean())
            mine = imaging.edge_density(imaging.to_gray(imaging.imread_bgr(path)))
            if theirs < 1e-6:
                assert mine < 1e-3, name
            else:
                assert 0.5 <= mine / theirs <= 1.5, f"{name}: {mine:.4f} vs {theirs:.4f}"
