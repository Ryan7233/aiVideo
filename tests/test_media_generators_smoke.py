"""Execute every media generator once and check it produces a real image.

These modules sit at 7-13% coverage, and converting them from fake-async to
plain functions broke one without any test noticing: all seven layout methods
in the Xiaohongshu collage generator became sync while the caller still
awaited the one it dispatched, so /xiaohongshu/generate_collage -- a route the
UI uses -- returned "object Image can't be used in 'await' expression".

The point here is breadth, not depth: call each entry point and each layout
branch, and assert something valid comes back.
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from core.runtime import PHOTO_UPLOAD_DIR


@pytest.fixture(scope="module")
def photos():
    """Four images with structure, so layout code has something to work with."""
    rng = np.random.default_rng(3)
    paths = []
    for index in range(4):
        array = (
            np.tile(np.linspace(0, 255, 400), (400, 1))[:, :, None].repeat(3, 2)
            + rng.normal(0, 25, (400, 400, 3))
        ).clip(0, 255).astype(np.uint8)
        array[100:200, 100:300] = (200, 60, 40)
        path = PHOTO_UPLOAD_DIR / f"smoke_gen_{index}.jpg"
        Image.fromarray(array).save(path)
        paths.append(str(path))
    try:
        yield paths
    finally:
        for path in paths:
            try:
                __import__("pathlib").Path(path).unlink(missing_ok=True)
            except OSError:
                pass


def _assert_valid_png(data_uri_or_b64: str, label: str):
    raw = data_uri_or_b64.split(",", 1)[-1]
    decoded = base64.b64decode(raw)
    with Image.open(io.BytesIO(decoded)) as image:
        image.verify()
    assert len(decoded) > 2048, f"{label}: image is suspiciously small"


class TestAdvancedCollage:
    @pytest.mark.parametrize(
        "layout", ["dynamic", "grid", "magazine", "mosaic", "creative"]
    )
    def test_every_layout_produces_an_image(self, photos, layout):
        from core.advanced_collage_generator import get_advanced_collage_generator

        result = get_advanced_collage_generator().generate_advanced_collage(
            images=photos, title="冒烟测试", layout_type=layout
        )
        assert result["status"] == "success", result
        _assert_valid_png(result["collage_base64"], layout)


class TestXiaohongshuCollage:
    def test_every_layout_template_runs(self, photos):
        """The dispatch path that the async/sync conversion broke."""
        from core.xiaohongshu_collage_generator import get_xiaohongshu_collage_generator

        generator = get_xiaohongshu_collage_generator()
        failures = {}
        for name in generator.layout_templates:
            result = generator.generate_xiaohongshu_collage(photos, "冒烟测试", layout=name)
            if not result.get("success"):
                failures[name] = result.get("error")
        assert not failures, f"layouts failed: {failures}"

    def test_default_layout_returns_a_real_image(self, photos):
        from core.xiaohongshu_collage_generator import get_xiaohongshu_collage_generator

        result = get_xiaohongshu_collage_generator().generate_xiaohongshu_collage(
            photos, "冒烟测试"
        )
        assert result["success"], result.get("error")
        _assert_valid_png(result["base64_data"], "xhs default")
        assert result["width"] > 0 and result["height"] > 0


class TestImageDecorator:
    def test_decorate_returns_an_image(self, photos):
        from core.image_decorator import get_image_decorator

        result = get_image_decorator().decorate_image(
            image_path=photos[0], decorations={"filter": "vintage"}
        )
        assert result["status"] == "success", result

    def test_smart_decorations_returns_a_config(self):
        from core.image_decorator import get_image_decorator

        result = get_image_decorator().generate_smart_decorations(
            theme="露营", content_type="travel", mood="vibrant"
        )
        assert result["status"] == "success", result


class TestPhotoRanking:
    def test_ranking_runs_and_differentiates(self, photos):
        """Exercises the numpy/Pillow code that replaced OpenCV."""
        from core.advanced_photo_ranking import get_advanced_photo_service

        ranked = get_advanced_photo_service().rank_photos_advanced(photos, top_k=3)
        assert len(ranked) == 3
        scores = [item.get("final_score", 0) for item in ranked]
        assert all(0 <= score <= 1 for score in scores), scores
        assert scores == sorted(scores, reverse=True), "results are not ranked"


class TestSmartCover:
    @pytest.mark.parametrize("layout", ["auto", "grid_3x3", "grid_2x3", "collage_mixed", "magazine"])
    def test_every_cover_layout_runs(self, photos, layout):
        from pathlib import Path

        from core.smart_cover_generator import get_smart_cover_generator

        result = get_smart_cover_generator().generate_cover(
            images=photos, title="冒烟测试", layout=layout, theme="auto"
        )
        assert result["status"] == "success", result
        path = Path(result["data"]["cover_path"])
        try:
            assert path.is_file()
            with Image.open(path) as image:
                image.verify()
        finally:
            path.unlink(missing_ok=True)


@pytest.mark.integration
class TestAudioProcessing:
    """Audio feature analysis must produce features, not silently degrade.

    Loosening the dependency pins moved librosa 0.10 -> 0.11, where
    beat_track returns tempo as an array rather than a scalar. float() on it
    raised, the broad except swallowed it, and the whole analysis fell back to
    defaults while still reporting success.
    """

    @pytest.fixture(scope="class")
    def video(self):
        import shutil
        import subprocess
        from pathlib import Path

        from core.runtime import INPUT_DIR

        if not shutil.which("ffmpeg"):
            pytest.skip("FFmpeg is required")
        path = INPUT_DIR / "audio_smoke.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15:duration=8",
             "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=8",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
            check=True, capture_output=True,
        )
        try:
            yield str(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_analysis_returns_real_features(self, video):
        from core.audio_processing import get_audio_processing_service

        result = get_audio_processing_service().process_video_audio(video, style="治愈")
        assert result.get("success")

        analysis = result.get("audio_analysis") or {}
        assert not analysis.get("degraded"), analysis.get("degraded_reason")
        # The fallback dict has none of these.
        assert "tempo" in analysis, f"feature analysis degraded: {sorted(analysis)}"
        assert "spectral" in analysis
        assert isinstance(analysis["tempo"], float)
        assert analysis["volume"]["mean"] > 0
