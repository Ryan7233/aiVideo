"""Execute the content services once each.

These sat between 7% and 20% coverage -- essentially never run. The same gap
in the media generators hid a collage endpoint that raised on every request
for days, so this does the same thing for the text and export side: call each
entry point with realistic input and check the shape of what comes back.

Breadth, not depth. Getting these executed at all is the point.
"""

import pytest

from core.degradation import is_degraded

TRANSCRIPT = [
    {"start": 0.0, "end": 6.0, "text": "大家好，今天来聊聊小红书的内容运营", "timestamp": "00:00"},
    {"start": 6.0, "end": 13.0, "text": "第一个重点是发布时间，晚上七点到九点打开率最高", "timestamp": "00:06"},
    {"start": 13.0, "end": 20.0, "text": "第二个重点是封面，字不要超过十二个", "timestamp": "00:13"},
    {"start": 20.0, "end": 27.0, "text": "真的太好用了，强烈推荐给大家", "timestamp": "00:20"},
]


@pytest.fixture
def photos(tmp_path):
    from PIL import Image

    paths = []
    for index, colour in enumerate([(200, 90, 60), (240, 160, 80), (170, 60, 55)]):
        path = tmp_path / f"photo_{index}.jpg"
        Image.new("RGB", (600, 600), colour).save(path)
        paths.append(str(path))
    return paths


@pytest.fixture
def clips(tmp_path):
    return [
        {"output_path": str(tmp_path / "clip1.mp4"), "start_time": 0, "duration": 10,
         "start_hms": "00:00:00", "reason": "开场"},
        {"output_path": str(tmp_path / "clip2.mp4"), "start_time": 12, "duration": 10,
         "start_hms": "00:00:12", "reason": "重点"},
    ]


class TestSemanticHighlights:
    def test_detects_highlights_from_a_transcript(self):
        from core.semantic_highlights import get_semantic_highlight_detector

        highlights = get_semantic_highlight_detector().detect_highlights(
            TRANSCRIPT, {"city": "上海", "style": "治愈"}
        )
        assert isinstance(highlights, list)
        for item in highlights:
            assert item.get("start") is not None and item.get("end") is not None
            assert item["end"] >= item["start"]

    def test_an_empty_transcript_is_handled(self):
        from core.semantic_highlights import get_semantic_highlight_detector

        assert get_semantic_highlight_detector().detect_highlights([], {}) == []


class TestAsrSmartClipping:
    @pytest.mark.integration
    def test_selection_runs_against_a_real_video(self, tmp_path):
        import shutil
        import subprocess

        if not shutil.which("ffmpeg"):
            pytest.skip("FFmpeg is required")
        from core.runtime import INPUT_DIR
        from core.asr_smart_clipping import get_asr_smart_engine

        video = INPUT_DIR / "asr_clip_smoke.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15:duration=30",
             "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=30",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video)],
            check=True, capture_output=True,
        )
        try:
            segments = get_asr_smart_engine().select_best_segments_with_asr(
                str(video), {"segments": TRANSCRIPT, "language": "zh"}, 8, 15, 2
            )
            assert isinstance(segments, list)
            for segment in segments:
                assert segment.get("duration", 0) > 0
        finally:
            video.unlink(missing_ok=True)


class TestPersonalizedWriting:
    def test_learning_a_style_then_writing_with_it(self):
        from core.personalized_writing import get_personalized_writing_service

        service = get_personalized_writing_service()
        profile = service.learn_user_style(
            "smoke_user",
            [{"content": "今天去了一家咖啡店，拿铁真的好喝，环境也舒服", "likes": 100},
             {"content": "周末露营超治愈，推荐这个营地给大家", "likes": 200}],
        )
        assert isinstance(profile, dict)

        content = service.generate_personalized_content(
            "smoke_user", {"topic": "咖啡店探店", "city": "上海"}, "note"
        )
        assert isinstance(content, dict)
        assert content.get("title") or content.get("content") or is_degraded(content)


class TestSubtitleAndCover:
    def test_subtitles_are_generated_for_clips(self, clips):
        from core.subtitle_service import get_subtitle_generator

        result = get_subtitle_generator().generate_subtitles(clips, TRANSCRIPT, "可爱")
        assert isinstance(result, dict)
        assert "srt_files" in result or is_degraded(result)

    def test_cover_suggestions_are_produced(self, clips, photos):
        from core.subtitle_service import get_cover_service

        result = get_cover_service().suggest_cover(
            clips, [{"path": p, "final_score": 0.8} for p in photos], "咖啡店探店"
        )
        assert isinstance(result, dict)


class TestXiaohongshuPipelineParts:
    def test_photo_ranking(self, photos):
        from core.xiaohongshu_pipeline import get_photo_ranking_service

        ranked = get_photo_ranking_service().rank_photos(photos, top_k=2)
        assert len(ranked) <= 2
        for item in ranked:
            assert "path" in item

    def test_storyline_then_draft(self):
        from core.xiaohongshu_pipeline import get_draft_generator, get_storyline_generator

        storyline = get_storyline_generator().generate_storyline(
            TRANSCRIPT, "周末去了咖啡店", "上海", "治愈"
        )
        assert isinstance(storyline, dict)

        draft = get_draft_generator().generate_draft(storyline, "治愈")
        assert isinstance(draft, dict)
        assert draft.get("title") or draft.get("content") or is_degraded(draft)


class TestExportService:
    def test_exporting_a_pipeline_result(self, clips, photos):
        from core.export_service import get_export_service

        result = get_export_service().export_xiaohongshu_content(
            {
                "processing_id": "smoke_export",
                "clips": clips,
                "photos_ranked": [{"path": p, "final_score": 0.8} for p in photos],
                "draft": {"title": "标题", "content": "正文", "hashtags": ["#测试"]},
                "subtitles": {"srt_files": []},
            },
            "json",
        )
        assert isinstance(result, dict)

    def test_cleanup_accepts_a_window(self):
        from core.export_service import get_export_service

        result = get_export_service().cleanup_old_exports(days=3650)
        assert isinstance(result, dict)
