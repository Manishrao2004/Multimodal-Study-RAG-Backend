"""Multimodal ingestion: figure filtering, dedup, transcript chunking, quiz parsing."""

from __future__ import annotations

import pytest
from PIL import Image

from app.core.generation.llm_client import build_vision_message, strip_think_tags
from app.core.generation.study_tools import _extract_json_array
from app.core.ingestion.figures import _passes_geometry, dhash, hamming_distance
from app.core.ingestion.transcription import TranscriptSegment, format_timestamp, group_segments
from app.models.schemas import QuizFormat  # noqa: F401  (kept for parity with study tools)


class TestFigureFiltering:
    def test_reasonable_diagram_is_kept(self, settings):
        assert _passes_geometry(600, 400, settings) is True

    def test_icon_is_rejected_on_size(self, settings):
        assert _passes_geometry(48, 48, settings) is False

    def test_horizontal_rule_is_rejected_on_aspect_ratio(self, settings):
        # Wide enough and tall enough individually, but a 20:1 sliver.
        assert _passes_geometry(2000, 130, settings) is False

    def test_large_enough_but_thin_area_is_rejected(self, settings):
        assert _passes_geometry(130, 130, settings) is False  # area < 40_000


class TestPerceptualHash:
    def test_identical_images_hash_identically(self):
        image = Image.new("RGB", (200, 200), "white")
        for x in range(0, 200, 20):
            for y in range(0, 200, 20):
                image.putpixel((x, y), (0, 0, 0))
        assert dhash(image) == dhash(image.copy())

    def test_rescaled_image_stays_within_dedup_distance(self):
        image = Image.new("RGB", (400, 400), "white")
        for x in range(0, 400, 8):
            image.putpixel((x, x), (0, 0, 0))
        resized = image.resize((200, 200), Image.LANCZOS)
        assert hamming_distance(dhash(image), dhash(resized)) <= 8

    def test_different_images_hash_far_apart(self):
        solid = Image.new("RGB", (100, 100), "white")
        gradient = Image.new("RGB", (100, 100))
        for x in range(100):
            for y in range(100):
                gradient.putpixel((x, y), (x * 2, y * 2, 128))
        assert hamming_distance(dhash(solid), dhash(gradient)) > 4


class TestVisionMessageFormat:
    def test_ollama_uses_a_parallel_images_list(self):
        message = build_vision_message("describe", "BASE64", "ollama")
        assert message["images"] == ["BASE64"]
        assert message["content"] == "describe"

    def test_openai_format_uses_a_data_url_content_part(self):
        message = build_vision_message("describe", "BASE64", "openai_compatible")
        parts = message["content"]
        assert parts[0] == {"type": "text", "text": "describe"}
        assert parts[1]["image_url"]["url"] == "data:image/png;base64,BASE64"


class TestThinkTagStripping:
    def test_reasoning_block_is_removed(self):
        assert strip_think_tags("<think>hmm...</think>The answer is 4.") == "The answer is 4."

    def test_multiline_reasoning_is_removed(self):
        raw = "<think>\nstep 1\nstep 2\n</think>\nFinal answer [1]."
        assert strip_think_tags(raw) == "Final answer [1]."

    def test_unclosed_reasoning_block_does_not_leak(self):
        # A truncated response must not surface raw chain-of-thought.
        assert strip_think_tags("Answer here.<think>unfinished reasoning") == "Answer here."

    def test_plain_text_is_untouched(self):
        assert strip_think_tags("Just an answer [1].") == "Just an answer [1]."


class TestTranscriptChunking:
    def test_segments_are_merged_to_the_target_size(self):
        segments = [TranscriptSegment(i * 5.0, i * 5.0 + 5, "word " * 20) for i in range(10)]
        grouped = group_segments(segments, target_chars=300)
        assert len(grouped) < len(segments)

    def test_grouped_chunk_spans_first_start_to_last_end(self):
        segments = [
            TranscriptSegment(0.0, 5.0, "a" * 200),
            TranscriptSegment(5.0, 10.0, "b" * 200),
        ]
        grouped = group_segments(segments, target_chars=300)
        assert grouped[0].start == 0.0
        assert grouped[0].end == 10.0

    def test_trailing_partial_segment_is_not_dropped(self):
        segments = [TranscriptSegment(0.0, 1.0, "short text")]
        grouped = group_segments(segments, target_chars=10_000)
        assert len(grouped) == 1
        assert grouped[0].text == "short text"

    def test_no_segments_yields_no_chunks(self):
        assert group_segments([], target_chars=500) == []

    @pytest.mark.parametrize(
        "seconds,expected",
        [(0, "0:00"), (65, "1:05"), (3661, "1:01:01"), (599, "9:59")],
    )
    def test_timestamps_are_human_readable(self, seconds, expected):
        assert format_timestamp(seconds) == expected


class TestQuizJsonParsing:
    def test_bare_array_is_parsed(self):
        assert _extract_json_array('[{"question": "q"}]') == [{"question": "q"}]

    def test_fenced_json_is_unwrapped(self):
        raw = 'Here you go:\n```json\n[{"question": "q"}]\n```'
        assert _extract_json_array(raw) == [{"question": "q"}]

    def test_array_embedded_in_prose_is_recovered(self):
        raw = 'Sure! [{"question": "q"}] Hope that helps.'
        assert _extract_json_array(raw) == [{"question": "q"}]

    def test_malformed_json_yields_nothing_rather_than_raising(self):
        assert _extract_json_array("[{broken json") == []

    def test_non_array_json_is_rejected(self):
        assert _extract_json_array('{"question": "q"}') == []


class TestRetryAfterParsing:
    """Free-tier rate limits are the main reason evaluation runs lose queries,
    so the wait hint must be read from either place a provider puts it."""

    def _response(self, headers=None, text=""):
        import httpx

        return httpx.Response(429, headers=headers or {}, text=text)

    def test_standard_header_is_used(self):
        from app.core.generation.llm_client import parse_retry_after

        assert parse_retry_after(self._response(headers={"retry-after": "7"})) == 7.0

    def test_groq_message_body_is_used_when_header_is_absent(self):
        from app.core.generation.llm_client import parse_retry_after

        body = '{"error":{"message":"Rate limit reached. Please try again in 6.32s."}}'
        assert parse_retry_after(self._response(text=body)) == 6.32

    def test_header_wins_over_body(self):
        from app.core.generation.llm_client import parse_retry_after

        response = self._response(
            headers={"retry-after": "2"}, text="Please try again in 60s"
        )
        assert parse_retry_after(response) == 2.0

    def test_absent_hint_returns_none_so_backoff_applies(self):
        from app.core.generation.llm_client import parse_retry_after

        assert parse_retry_after(self._response(text="slow down")) is None

    def test_unparseable_header_falls_back_to_body(self):
        from app.core.generation.llm_client import parse_retry_after

        response = self._response(
            headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"},
            text="Please try again in 4.5s",
        )
        assert parse_retry_after(response) == 4.5


class TestCitationMarkerParsing:
    """Models are inconsistent about JSON number types; a well-formed quiz item
    must not be discarded over how its citation happens to be spelled."""

    @pytest.mark.parametrize("value,expected", [(3, 3), (3.0, 3), ("3", 3), (" 3 ", 3), ("[3]", 3)])
    def test_whole_number_shapes_all_parse(self, value, expected):
        from app.core.generation.study_tools import _parse_marker

        assert _parse_marker(value) == expected

    @pytest.mark.parametrize("value", [None, "", "abc", 2.5, True, [], {}])
    def test_non_indices_are_rejected(self, value):
        from app.core.generation.study_tools import _parse_marker

        assert _parse_marker(value) is None
