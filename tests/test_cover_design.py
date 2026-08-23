"""Cover rendering: fonts, palette and layout.

Three defects motivated these:
- the hardcoded font path /System/Library/Fonts/PingFang.ttc does not exist on
  current macOS, so every cover fell back to a bitmap font with no Chinese
  glyphs and the title came out as a few unreadable pixels
- the renderer only had five fixed themes, so a cover's colours had nothing to
  do with the photos in it
- the grid was always 3x3, leaving a band of empty background under any cover
  built from fewer than nine images
"""

import pytest
from PIL import Image, ImageFont

from core import fonts
from core.color_palette import (
    NEUTRAL_PALETTE,
    complement,
    hex_to_rgb,
    palette_from_images,
    rgb_to_hex,
    shade,
)
from core.runtime import PHOTO_UPLOAD_DIR
from core.smart_cover_generator import get_smart_cover_generator


def _hue(hex_color: str) -> float:
    import colorsys

    r, g, b = hex_to_rgb(hex_color)
    return colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[0]


@pytest.fixture
def photos(tmp_path):
    """Write real image files; the palette code reads pixels."""

    def _make(colors, prefix="p"):
        paths = []
        for index, color in enumerate(colors):
            path = tmp_path / f"{prefix}_{index}.jpg"
            Image.new("RGB", (400, 400), color).save(path)
            paths.append(str(path))
        return paths

    return _make


class TestFontResolution:
    def test_a_cjk_capable_font_is_found(self):
        path = fonts.find_cjk_font()
        if path is None:
            pytest.skip("no CJK font installed on this machine")
        assert fonts._can_render_cjk(path)

    def test_latin_only_fonts_are_rejected(self):
        """A file-exists check passes these; rendering Chinese gives tofu."""
        latin_only = "/System/Library/Fonts/Helvetica.ttc"
        try:
            ImageFont.truetype(latin_only, 20)
        except Exception:
            pytest.skip("Helvetica not present")
        assert not fonts._can_render_cjk(latin_only)

    def test_load_font_always_returns_something(self, monkeypatch):
        monkeypatch.setattr(fonts, "CJK_FONT_CANDIDATES", [])
        monkeypatch.setattr(fonts, "LATIN_FONT_CANDIDATES", [])
        fonts.reset_cache()
        try:
            assert fonts.load_font(24) is not None
        finally:
            fonts.reset_cache()

    def test_env_override_is_honoured(self, monkeypatch):
        real = fonts.find_cjk_font()
        if real is None:
            pytest.skip("no CJK font installed on this machine")
        fonts.reset_cache()
        monkeypatch.setenv("AIVIDEO_CJK_FONT", real)
        try:
            assert fonts.find_cjk_font() == real
        finally:
            fonts.reset_cache()

    def test_unusable_override_falls_back(self, monkeypatch):
        fonts.reset_cache()
        monkeypatch.setenv("AIVIDEO_CJK_FONT", "/nonexistent/font.ttf")
        try:
            assert fonts.find_cjk_font() != "/nonexistent/font.ttf"
        finally:
            fonts.reset_cache()

    def test_rendered_chinese_is_not_blank(self):
        if fonts.find_cjk_font() is None:
            pytest.skip("no CJK font installed on this machine")
        font = fonts.load_font(48)
        canvas = Image.new("L", (200, 80), 0)
        from PIL import ImageDraw

        ImageDraw.Draw(canvas).text((10, 10), "周末露营", font=font, fill=255)
        # The old default bitmap font produced a handful of pixels at most.
        assert sum(1 for pixel in canvas.convert("L").tobytes() if pixel > 40) > 200


class TestPalette:
    def test_hex_round_trip(self):
        assert hex_to_rgb(rgb_to_hex((18, 110, 32))) == (18, 110, 32)

    def test_palette_follows_the_photos(self, photos):
        green = palette_from_images(photos([(34, 110, 45)] * 3, "g"))
        orange = palette_from_images(photos([(230, 120, 50)] * 3, "o"))
        assert green["primary"] != orange["primary"]
        # green hue sits near 1/3 of the wheel, orange near 1/12
        assert 0.2 < _hue(green["primary"]) < 0.5
        assert _hue(orange["primary"]) < 0.15

    def test_secondary_shares_the_primary_hue(self, photos):
        """The background is a primary->secondary gradient.

        Two opposing hues there read as a mistake, which is why every built-in
        theme pairs primary with a darker sibling.
        """
        palette = palette_from_images(photos([(34, 110, 45)] * 3, "s"))
        assert abs(_hue(palette["primary"]) - _hue(palette["secondary"])) < 0.05

    def test_accent_contrasts_with_primary(self, photos):
        palette = palette_from_images(photos([(34, 110, 45)] * 3, "a"))
        gap = abs(_hue(palette["primary"]) - _hue(palette["accent"]))
        assert 0.3 < min(gap, 1 - gap) <= 0.5

    def test_no_readable_images_gives_the_neutral_palette(self):
        assert palette_from_images(["/nope/a.jpg", "/nope/b.jpg"]) == NEUTRAL_PALETTE

    def test_empty_input_is_safe(self):
        assert palette_from_images([]) == NEUTRAL_PALETTE

    def test_shade_is_darker_at_the_same_hue(self):
        base = (34, 160, 60)
        darker = shade(base)
        assert sum(darker) < sum(base)

    def test_complement_is_opposite(self):
        gap = abs(_hue(rgb_to_hex(complement((230, 120, 50)))) - _hue("#E67832"))
        assert 0.4 < min(gap, 1 - gap) <= 0.5


class TestGridSizing:
    @pytest.mark.parametrize(
        "count,expected",
        [(1, (1, 1)), (2, (1, 2)), (3, (1, 3)), (4, (2, 2)),
         (5, (2, 3)), (6, (2, 3)), (9, (3, 3)), (12, (3, 3))],
    )
    def test_grid_matches_image_count(self, count, expected):
        assert get_smart_cover_generator()._grid_for(count) == expected

    def test_every_grid_has_room_for_its_images(self):
        for count in range(1, 10):
            rows, cols = get_smart_cover_generator()._grid_for(count)
            assert rows * cols >= count, f"{count} images do not fit {rows}x{cols}"

    def test_no_grid_wastes_more_than_two_cells(self):
        """A 3x3 holding four images is what left the empty band."""
        for count in range(1, 10):
            rows, cols = get_smart_cover_generator()._grid_for(count)
            assert rows * cols - count <= 2, f"{count} images in {rows}x{cols} wastes too much"


class TestPaletteResolution:
    def test_named_theme_is_used_verbatim(self):
        generator = get_smart_cover_generator()
        colors, theme = generator._resolve_palette("blue_gradient", [], None)
        assert theme == "blue_gradient"
        assert colors == generator.color_themes["blue_gradient"]

    def test_unknown_theme_falls_back_to_a_preset(self):
        generator = get_smart_cover_generator()
        colors, _ = generator._resolve_palette("no_such_theme", [], None)
        assert colors == generator.color_themes["pink_gradient"]

    def test_auto_samples_the_images(self, photos):
        colors, theme = get_smart_cover_generator()._resolve_palette(
            "auto", photos([(34, 110, 45)] * 3, "auto"), None
        )
        assert theme == "custom"
        assert 0.2 < _hue(colors["primary"]) < 0.5

    def test_explicit_palette_wins(self, photos):
        colors, theme = get_smart_cover_generator()._resolve_palette(
            "auto", photos([(34, 110, 45)] * 3, "x"), {"palette": {"primary": "#123456"}}
        )
        assert theme == "custom"
        assert colors["primary"] == "#123456"
        # unspecified keys still need values for the renderer
        assert colors["background"]


@pytest.mark.integration
class TestRendering:
    def _cleanup(self, result):
        from pathlib import Path

        path = Path((result.get("data") or {}).get("cover_path", ""))
        if path.is_file():
            path.unlink()

    def test_auto_theme_renders_and_reports_its_palette(self, photos):
        paths = photos([(34, 110, 45), (60, 140, 70), (25, 90, 38), (80, 160, 90)], "r")
        result = get_smart_cover_generator().generate_cover(
            images=paths, title="周末露营好去处", subtitle="8个宝藏营地", theme="auto")
        try:
            assert result["status"] == "success"
            palette = result["data"]["palette"]
            assert 0.2 < _hue(palette["primary"]) < 0.5
            from pathlib import Path

            assert Path(result["data"]["cover_path"]).is_file()
        finally:
            self._cleanup(result)

    def test_no_usable_images_is_an_error_not_a_crash(self):
        result = get_smart_cover_generator().generate_cover(
            images=["/nope/a.jpg"], title="标题")
        assert result["status"] == "error"


@pytest.mark.integration
def test_smart_cover_designer_writes_a_real_file(tmp_path):
    """datetime was never imported, so the save step raised NameError, the
    broad except swallowed it and cover_path came back as an empty string."""
    from pathlib import Path

    from core.smart_cover_design import get_smart_cover_designer

    source = PHOTO_UPLOAD_DIR / "designer_probe.jpg"
    Image.new("RGB", (800, 600), (34, 139, 200)).save(source)
    cover = None
    try:
        result = get_smart_cover_designer().generate_smart_cover(
            [], [{"path": str(source), "final_score": 0.9}], "测试标题", "治愈")
        cover = Path(result.get("cover_path") or "")
        assert result.get("cover_path"), "cover_path is empty; the save step failed silently"
        assert cover.is_file()
        assert cover.stat().st_size > 1024
    finally:
        source.unlink(missing_ok=True)
        if cover and cover.is_file():
            cover.unlink()


class TestFontDirectoryScan:
    """The container fallback.

    Known font paths differ between distributions and move between releases,
    so when none of them match the resolver looks at what is actually
    installed. Without this, getting fonts-noto-cjk into the image would only
    help if Debian kept the file exactly where the candidate list expects.
    """

    def test_scan_finds_a_font_when_known_paths_all_miss(self, monkeypatch):
        monkeypatch.setattr(fonts, "CJK_FONT_CANDIDATES", ["/nope/none.ttc"])
        fonts.reset_cache()
        try:
            found = fonts.find_cjk_font()
            if found is None:
                pytest.skip("no CJK font installed on this machine")
            assert fonts._can_render_cjk(found)
        finally:
            fonts.reset_cache()

    def test_scan_returns_none_when_there_is_nothing_to_find(self, monkeypatch):
        monkeypatch.setattr(fonts, "CJK_FONT_CANDIDATES", [])
        monkeypatch.setattr(fonts, "FONT_DIRS", ["/nonexistent/fonts"])
        fonts.reset_cache()
        try:
            assert fonts.find_cjk_font() is None
        finally:
            fonts.reset_cache()

    def test_scan_tolerates_unreadable_directories(self, monkeypatch, tmp_path):
        monkeypatch.setattr(fonts, "CJK_FONT_CANDIDATES", [])
        monkeypatch.setattr(fonts, "FONT_DIRS", [str(tmp_path), "/proc/1/root"])
        fonts.reset_cache()
        try:
            assert fonts.find_cjk_font() is None  # must not raise
        finally:
            fonts.reset_cache()


class TestCoverPipeline:
    """Selection feeding rendering.

    The designer scored candidates and the renderer composited images, but the
    ranking never reached the renderer: whatever the caller passed got used,
    in the order given.
    """

    @pytest.fixture
    def mixed_photos(self, tmp_path):
        """Three sharp, detailed images and three flat blurred ones."""
        import numpy as np
        from PIL import ImageFilter

        rng = np.random.default_rng(5)
        sharp, blurred = [], []
        for index in range(3):
            array = (
                np.tile(np.linspace(0, 255, 600), (600, 1))[:, :, None].repeat(3, 2)
                + rng.normal(0, 30, (600, 600, 3))
            ).clip(0, 255).astype(np.uint8)
            array[150:400, 100:500] = (210, 90, 50)
            path = tmp_path / f"sharp_{index}.jpg"
            Image.fromarray(array).save(path)
            sharp.append(str(path))

            flat = Image.fromarray(np.full((600, 600, 3), (120, 120, 125), np.uint8))
            path = tmp_path / f"blur_{index}.jpg"
            flat.filter(ImageFilter.GaussianBlur(12)).save(path)
            blurred.append(str(path))
        return sharp, blurred

    def test_ranking_is_exposed(self, mixed_photos):
        from core.smart_cover_design import get_smart_cover_designer

        sharp, blurred = mixed_photos
        ranked = get_smart_cover_designer().rank_cover_candidates(
            [], [{"path": p, "final_score": 0.5} for p in sharp + blurred]
        )
        assert len(ranked) == 6
        scores = [item["cover_score"] for item in ranked]
        assert scores == sorted(scores, reverse=True), "candidates are not ranked"

    def test_the_better_material_is_the_material_rendered(self, mixed_photos):
        from pathlib import Path

        from core.cover_pipeline import build_cover

        sharp, blurred = mixed_photos
        result = build_cover(
            photos=[{"path": p, "final_score": 0.5} for p in sharp + blurred],
            title="自动选材", image_count=3,
        )
        cover = Path((result.get("data") or {}).get("cover_path", ""))
        try:
            assert result["status"] == "success", result
            used = {item["path"] for item in result["selection"] if item["used"]}
            assert used == set(sharp), f"picked the blurred images: {used}"
            assert cover.is_file()
        finally:
            cover.unlink(missing_ok=True)

    def test_the_selection_is_reported(self, mixed_photos):
        """Which frames won, and what they scored, should be inspectable."""
        from pathlib import Path

        from core.cover_pipeline import build_cover

        sharp, _ = mixed_photos
        result = build_cover(photos=[{"path": p} for p in sharp], title="t", image_count=2)
        cover = Path((result.get("data") or {}).get("cover_path", ""))
        try:
            selection = result["selection"]
            assert len(selection) == 3
            assert sum(1 for item in selection if item["used"]) == 2
            assert all("cover_score" in item for item in selection)
        finally:
            cover.unlink(missing_ok=True)

    def test_no_material_degrades_visibly(self):
        from core.cover_pipeline import build_cover
        from core.degradation import is_degraded

        result = build_cover(photos=[{"path": "/nope/missing.jpg"}], title="t")
        assert result["status"] == "error"
        assert is_degraded(result)

    def test_duplicate_frames_are_collapsed(self, mixed_photos):
        """Key frames from one clip are near-identical; a cover of six copies
        of the same shot is not a cover."""
        from pathlib import Path

        from core.cover_pipeline import build_cover

        sharp, _ = mixed_photos
        duplicated = [{"path": sharp[0]} for _ in range(5)] + [{"path": sharp[1]}]
        result = build_cover(photos=duplicated, title="t", image_count=4)
        cover = Path((result.get("data") or {}).get("cover_path", ""))
        try:
            used = [item["path"] for item in result["selection"] if item["used"]]
            assert len(used) == len(set(used)), used
        finally:
            cover.unlink(missing_ok=True)
