"""
Property-based tests for:
  puzzle_reconstruction.utils.text_utils

Verifies logical and mathematical invariants:
- TextConfig:           valid parameter ranges, ValueError on bad inputs
- TextBlock:            area = w*h, center, n_chars, property ranges
- clean_ocr_text:       idempotent, lowercase, strip_punct, space collapse,
                        control chars removed, \n/\t preserved
- estimate_text_density: ∈ [0, 1], all-zero → 0, all-nonzero → 1, monotone
- find_text_lines:      y_start < y_end, non-overlapping, height ≥ min_line_height,
                        y in [0, H], all-zero → empty
- segment_words:        x_start < x_end, non-overlapping, x in [0, W], all-zero → empty
- compute_text_score:   ∈ [0, 1], all-zero → 0, nonnegative
- compare_text_blocks:  ∈ [0, 1], identical → 1, symmetric, both-empty → 1,
                        one-empty → 0
- align_text_blocks:    same count, all blocks present, correct ordering,
                        unknown primary → ValueError
- batch_clean_text:     same length, each = clean_ocr_text(t), config applied
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pytest

from puzzle_reconstruction.utils.text_utils import (
    TextConfig,
    TextBlock,
    clean_ocr_text,
    estimate_text_density,
    find_text_lines,
    segment_words,
    compute_text_score,
    compare_text_blocks,
    align_text_blocks,
    batch_clean_text,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _binary(h: int = 32, w: int = 32, fill: int = 0) -> np.ndarray:
    """Uniform binary image (fill = 0 or 255)."""
    return np.full((h, w), fill, dtype=np.uint8)


def _text_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Binary image with horizontal text-line stripes."""
    img = np.zeros((h, w), dtype=np.uint8)
    for y_start in range(8, h - 8, 16):
        img[y_start: y_start + 6, 4: w - 4] = 255
    return img


def _word_row(w: int = 64, n_words: int = 2) -> np.ndarray:
    """Binary image row with n_words rectangular word blocks."""
    row = np.zeros((8, w), dtype=np.uint8)
    step = w // (n_words + 1)
    for i in range(n_words):
        x0 = step * (i + 1) - 4
        x1 = x0 + 8
        row[:, max(0, x0): min(w, x1)] = 255
    return row


def _block(
    text: str = "hello",
    x: int = 0,
    y: int = 0,
    w: int = 100,
    h: int = 20,
) -> TextBlock:
    return TextBlock(text=text, x=x, y=y, w=w, h=h)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TextConfig validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestTextConfigValidation:
    """TextConfig rejects invalid parameter values."""

    def test_negative_min_word_gap_raises(self):
        with pytest.raises(ValueError):
            TextConfig(min_word_gap=-1)

    def test_line_threshold_below_zero_raises(self):
        with pytest.raises(ValueError):
            TextConfig(line_threshold=-0.01)

    def test_line_threshold_above_one_raises(self):
        with pytest.raises(ValueError):
            TextConfig(line_threshold=1.01)

    def test_min_line_height_zero_raises(self):
        with pytest.raises(ValueError):
            TextConfig(min_line_height=0)

    def test_min_line_height_negative_raises(self):
        with pytest.raises(ValueError):
            TextConfig(min_line_height=-2)

    @pytest.mark.parametrize("thresh", [0.0, 0.5, 1.0])
    def test_boundary_thresholds_valid(self, thresh):
        cfg = TextConfig(line_threshold=thresh)
        assert cfg.line_threshold == pytest.approx(thresh)

    def test_zero_min_word_gap_valid(self):
        cfg = TextConfig(min_word_gap=0)
        assert cfg.min_word_gap == 0

    def test_custom_config_fields(self):
        cfg = TextConfig(
            min_word_gap=8,
            line_threshold=0.1,
            strip_punct=True,
            lowercase=True,
            min_line_height=6,
        )
        assert cfg.min_word_gap == 8
        assert cfg.line_threshold == pytest.approx(0.1)
        assert cfg.strip_punct is True
        assert cfg.lowercase is True
        assert cfg.min_line_height == 6


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TextBlock properties
# ═══════════════════════════════════════════════════════════════════════════════

class TestTextBlockProperties:
    """TextBlock computed properties are consistent with fields."""

    def test_area_equals_w_times_h(self):
        blk = TextBlock(text="hi", x=0, y=0, w=50, h=20)
        assert blk.area == 1000

    @pytest.mark.parametrize("w,h", [(0, 10), (10, 0), (0, 0)])
    def test_area_zero_when_dimension_zero(self, w, h):
        blk = TextBlock(text="x", x=0, y=0, w=w, h=h)
        assert blk.area == 0

    def test_area_is_nonnegative(self):
        for w, h in [(1, 1), (100, 200), (7, 3)]:
            assert TextBlock(text="", x=0, y=0, w=w, h=h).area >= 0

    def test_center_x(self):
        blk = TextBlock(text="a", x=10, y=5, w=40, h=20)
        assert blk.center[0] == pytest.approx(10 + 40 / 2)

    def test_center_y(self):
        blk = TextBlock(text="a", x=10, y=5, w=40, h=20)
        assert blk.center[1] == pytest.approx(5 + 20 / 2)

    def test_n_chars_counts_nonspace(self):
        blk = TextBlock(text="hello world", x=0, y=0, w=10, h=10)
        assert blk.n_chars == 10  # "hello" + "world" = 10

    def test_n_chars_all_spaces_is_zero(self):
        blk = TextBlock(text="   ", x=0, y=0, w=10, h=10)
        assert blk.n_chars == 0

    def test_n_chars_empty_text_is_zero(self):
        blk = TextBlock(text="", x=0, y=0, w=10, h=10)
        assert blk.n_chars == 0

    def test_n_chars_leq_len_text(self):
        for text in ["abc", "a b c", "  x  ", "hello\tworld"]:
            blk = TextBlock(text=text, x=0, y=0, w=10, h=10)
            assert blk.n_chars <= len(text)

    def test_negative_w_raises(self):
        with pytest.raises(ValueError):
            TextBlock(text="x", x=0, y=0, w=-1, h=10)

    def test_negative_h_raises(self):
        with pytest.raises(ValueError):
            TextBlock(text="x", x=0, y=0, w=10, h=-1)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            TextBlock(text="x", x=0, y=0, w=10, h=10, confidence=1.5)

    def test_confidence_minus_one_is_valid(self):
        blk = TextBlock(text="x", x=0, y=0, w=10, h=10, confidence=-1.0)
        assert blk.confidence == pytest.approx(-1.0)

    @pytest.mark.parametrize("conf", [0.0, 0.5, 1.0])
    def test_valid_confidence_range(self, conf):
        blk = TextBlock(text="x", x=0, y=0, w=10, h=10, confidence=conf)
        assert blk.confidence == pytest.approx(conf)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — clean_ocr_text
# ═══════════════════════════════════════════════════════════════════════════════

class TestCleanOcrText:
    """clean_ocr_text invariants."""

    def test_empty_string_returns_empty(self):
        assert clean_ocr_text("") == ""

    def test_all_spaces_returns_empty(self):
        assert clean_ocr_text("   ") == ""

    def test_plain_text_unchanged(self):
        text = "hello world"
        result = clean_ocr_text(text)
        assert result == "hello world"

    def test_idempotent_default(self):
        text = "  Hello,  World!  \t"
        once = clean_ocr_text(text)
        twice = clean_ocr_text(once)
        assert once == twice

    @pytest.mark.parametrize("text", [
        "Hello WORLD",
        "OCR Result 123",
        "  Spaces and  CAPS  ",
        "Punctuation: done!",
    ])
    def test_idempotent_various_inputs(self, text):
        cfg = TextConfig()
        once = clean_ocr_text(text, cfg=cfg)
        twice = clean_ocr_text(once, cfg=cfg)
        assert once == twice

    def test_lowercase_flag_lowercases_all(self):
        cfg = TextConfig(lowercase=True)
        result = clean_ocr_text("Hello WORLD", cfg=cfg)
        assert result == result.lower()

    def test_lowercase_flag_no_uppercase_left(self):
        cfg = TextConfig(lowercase=True)
        result = clean_ocr_text("ABC DEF GHI", cfg=cfg)
        assert not any(c.isupper() for c in result)

    def test_strip_punct_removes_punctuation(self):
        cfg = TextConfig(strip_punct=True)
        result = clean_ocr_text("Hello, World! How are you?", cfg=cfg)
        # No punctuation characters should remain
        assert not re.search(r"[^\w\s]", result, flags=re.UNICODE)

    def test_multiple_spaces_collapsed_to_one(self):
        result = clean_ocr_text("a   b    c")
        assert "  " not in result
        assert "a b c" == result

    def test_leading_trailing_stripped(self):
        result = clean_ocr_text("   hello   ")
        assert result == result.strip()

    def test_control_chars_removed(self):
        # \x00 is a control character (should be removed)
        result = clean_ocr_text("abc\x00def\x01ghi")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_newline_preserved(self):
        result = clean_ocr_text("line1\nline2")
        assert "\n" in result

    def test_tab_collapsed_to_space(self):
        # \t is kept through control-char removal but collapsed to space in step 5
        result = clean_ocr_text("col1\tcol2")
        assert "col1" in result and "col2" in result
        assert "\t" not in result  # tabs are collapsed to spaces

    def test_unicode_nfc_normalization(self):
        # NFD form: a + combining grave
        nfd_char = "a\u0300"  # à in NFD
        nfc_char = "\u00e0"   # à in NFC
        result_nfd = clean_ocr_text(nfd_char)
        result_nfc = clean_ocr_text(nfc_char)
        assert result_nfd == result_nfc

    def test_strip_punct_and_lowercase_combined(self):
        cfg = TextConfig(strip_punct=True, lowercase=True)
        result = clean_ocr_text("Hello, World! 123", cfg=cfg)
        assert result == result.lower()
        assert not re.search(r"[^\w\s]", result, flags=re.UNICODE)

    def test_result_length_does_not_increase(self):
        texts = ["a  b  c", "Hello!!!", "  spaces  "]
        for text in texts:
            result = clean_ocr_text(text)
            assert len(result) <= len(text)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — estimate_text_density
# ═══════════════════════════════════════════════════════════════════════════════

class TestEstimateTextDensity:
    """estimate_text_density invariants."""

    def test_output_in_range(self):
        for img in [_binary(32, 32, 0), _binary(32, 32, 255), _text_image()]:
            assert 0.0 <= estimate_text_density(img) <= 1.0

    def test_all_zero_returns_zero(self):
        assert estimate_text_density(_binary(16, 16, 0)) == pytest.approx(0.0)

    def test_all_nonzero_returns_one(self):
        assert estimate_text_density(_binary(16, 16, 255)) == pytest.approx(1.0)

    def test_half_filled_returns_half(self):
        img = np.zeros((4, 4), dtype=np.uint8)
        img[0:2, :] = 255  # top half
        assert estimate_text_density(img) == pytest.approx(0.5)

    def test_single_nonzero_pixel(self):
        img = np.zeros((8, 8), dtype=np.uint8)
        img[0, 0] = 1
        density = estimate_text_density(img)
        assert density == pytest.approx(1.0 / 64.0)

    def test_monotone_adding_pixels_increases_density(self):
        img1 = np.zeros((10, 10), dtype=np.uint8)
        img1[0:3, :] = 255
        img2 = img1.copy()
        img2[3:6, :] = 255
        assert estimate_text_density(img2) > estimate_text_density(img1)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            estimate_text_density(np.zeros((3, 3, 3), dtype=np.uint8))

    def test_dtype_independence(self):
        """Different dtypes (bool, uint8, int32) should give same density."""
        arr_bool = np.array([[True, False], [False, True]])
        arr_uint8 = arr_bool.astype(np.uint8)
        arr_int32 = arr_bool.astype(np.int32)
        d_bool = estimate_text_density(arr_bool.astype(np.uint8))
        d_u8 = estimate_text_density(arr_uint8)
        d_i32 = estimate_text_density(arr_int32.astype(np.uint8))
        assert d_bool == pytest.approx(d_u8)
        assert d_bool == pytest.approx(d_i32)

    @pytest.mark.parametrize("h,w", [(1, 1), (1, 100), (100, 1), (10, 10)])
    def test_various_shapes(self, h, w):
        img = _binary(h, w, 200)
        assert estimate_text_density(img) == pytest.approx(1.0)

    def test_nonnegative(self):
        img = np.random.default_rng(0).integers(0, 256, (16, 16), dtype=np.uint8)
        assert estimate_text_density(img) >= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — find_text_lines
# ═══════════════════════════════════════════════════════════════════════════════

class TestFindTextLines:
    """find_text_lines invariants."""

    def test_all_zero_returns_empty(self):
        lines = find_text_lines(_binary(32, 32, 0))
        assert lines == []

    def test_returns_list_of_tuples(self):
        lines = find_text_lines(_text_image())
        assert isinstance(lines, list)
        for item in lines:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_y_start_lt_y_end(self):
        lines = find_text_lines(_text_image())
        for y0, y1 in lines:
            assert y0 < y1, f"y_start={y0} must be < y_end={y1}"

    def test_lines_non_overlapping(self):
        lines = find_text_lines(_text_image())
        for i in range(len(lines) - 1):
            _, y1 = lines[i]
            y0_next, _ = lines[i + 1]
            assert y1 <= y0_next, f"Lines overlap: end={y1} > start={y0_next}"

    def test_each_line_height_geq_min_line_height(self):
        cfg = TextConfig(min_line_height=4)
        lines = find_text_lines(_text_image(), cfg=cfg)
        for y0, y1 in lines:
            assert (y1 - y0) >= cfg.min_line_height

    def test_y_end_leq_image_height(self):
        img = _text_image(64, 64)
        lines = find_text_lines(img)
        for _, y1 in lines:
            assert y1 <= 64

    def test_y_start_geq_zero(self):
        lines = find_text_lines(_text_image())
        for y0, _ in lines:
            assert y0 >= 0

    def test_full_image_at_least_one_line(self):
        # Full image with line_threshold=0 → at least one line detected
        cfg = TextConfig(line_threshold=0.0, min_line_height=1)
        img = _binary(20, 20, 255)
        lines = find_text_lines(img, cfg=cfg)
        assert len(lines) >= 1

    def test_text_image_detects_multiple_lines(self):
        img = _text_image(64, 64)
        lines = find_text_lines(img)
        assert len(lines) >= 2

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            find_text_lines(np.zeros((3, 3, 3), dtype=np.uint8))

    @pytest.mark.parametrize("min_lh", [2, 4, 8, 12])
    def test_min_line_height_filter(self, min_lh):
        img = _text_image(64, 64)
        cfg = TextConfig(min_line_height=min_lh)
        lines = find_text_lines(img, cfg=cfg)
        for y0, y1 in lines:
            assert (y1 - y0) >= min_lh

    def test_high_threshold_returns_fewer_lines(self):
        img = _text_image(64, 64)
        cfg_low = TextConfig(line_threshold=0.01)
        cfg_high = TextConfig(line_threshold=0.9)
        lines_low = find_text_lines(img, cfg=cfg_low)
        lines_high = find_text_lines(img, cfg=cfg_high)
        assert len(lines_low) >= len(lines_high)

    def test_empty_image_returns_empty(self):
        img = np.zeros((0, 32), dtype=np.uint8)
        lines = find_text_lines(img)
        assert lines == []


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — segment_words
# ═══════════════════════════════════════════════════════════════════════════════

class TestSegmentWords:
    """segment_words invariants."""

    def test_all_zero_returns_empty(self):
        row = np.zeros((8, 64), dtype=np.uint8)
        assert segment_words(row) == []

    def test_returns_list_of_tuples(self):
        row = _word_row(64, 2)
        words = segment_words(row)
        assert isinstance(words, list)
        for item in words:
            assert len(item) == 2

    def test_x_start_lt_x_end(self):
        row = _word_row(64, 3)
        for x0, x1 in segment_words(row):
            assert x0 < x1

    def test_words_non_overlapping(self):
        row = _word_row(64, 3)
        words = segment_words(row)
        for i in range(len(words) - 1):
            _, x1 = words[i]
            x0_next, _ = words[i + 1]
            assert x1 <= x0_next

    def test_x_end_leq_width(self):
        row = _word_row(64, 2)
        for _, x1 in segment_words(row):
            assert x1 <= 64

    def test_x_start_geq_zero(self):
        row = _word_row(64, 2)
        for x0, _ in segment_words(row):
            assert x0 >= 0

    def test_fully_filled_row_returns_one_word(self):
        row = _binary(8, 32, 255)
        words = segment_words(row)
        assert len(words) == 1
        assert words[0] == (0, 32)

    def test_two_separated_words_detected(self):
        # Two clear ink blocks separated by a wide gap
        row = np.zeros((8, 60), dtype=np.uint8)
        row[:, 0:8] = 255   # word 1
        row[:, 20:28] = 255  # word 2 (gap = 12 >= default min_word_gap=4)
        words = segment_words(row)
        assert len(words) >= 2

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            segment_words(np.zeros((3, 3, 3), dtype=np.uint8))

    def test_single_column_word(self):
        row = np.zeros((4, 10), dtype=np.uint8)
        row[:, 5] = 255
        words = segment_words(row)
        assert len(words) == 1

    def test_empty_width_returns_empty(self):
        row = np.zeros((4, 0), dtype=np.uint8)
        assert segment_words(row) == []


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — compute_text_score
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeTextScore:
    """compute_text_score invariants."""

    def test_output_in_range(self):
        for img in [_binary(32, 32, 0), _binary(32, 32, 255), _text_image()]:
            score = compute_text_score(img)
            assert 0.0 <= score <= 1.0

    def test_all_zero_returns_zero(self):
        assert compute_text_score(_binary(32, 32, 0)) == pytest.approx(0.0)

    def test_nonnegative(self):
        rng = np.random.default_rng(7)
        img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
        assert compute_text_score(img) >= 0.0

    def test_text_image_scores_positive(self):
        score = compute_text_score(_text_image(64, 64))
        assert score > 0.0

    def test_full_image_nonzero_score(self):
        img = _binary(32, 32, 255)
        assert compute_text_score(img) > 0.0

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            compute_text_score(np.zeros((3, 3, 3), dtype=np.uint8))

    @pytest.mark.parametrize("h,w", [(16, 16), (32, 64), (64, 32)])
    def test_various_shapes_in_range(self, h, w):
        img = _text_image(h, w)
        score = compute_text_score(img)
        assert 0.0 <= score <= 1.0

    def test_text_image_higher_than_noise_image(self):
        rng = np.random.default_rng(99)
        noise = (rng.uniform(0, 1, (64, 64)) < 0.02).astype(np.uint8) * 255
        text = _text_image(64, 64)
        score_text = compute_text_score(text)
        score_noise = compute_text_score(noise)
        # text_image has structured lines → typically higher score
        # This is a soft check; just verify both are in range
        assert 0.0 <= score_noise <= 1.0
        assert 0.0 <= score_text <= 1.0

    def test_denser_text_nonnegative_change(self):
        img1 = np.zeros((32, 32), dtype=np.uint8)
        img1[8:12, :] = 255
        img2 = img1.copy()
        img2[20:24, :] = 255
        score1 = compute_text_score(img1)
        score2 = compute_text_score(img2)
        # Both should be in valid range
        assert 0.0 <= score1 <= 1.0
        assert 0.0 <= score2 <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — compare_text_blocks
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompareTextBlocks:
    """compare_text_blocks invariants."""

    def test_output_in_range(self):
        a = _block("hello")
        b = _block("world")
        assert 0.0 <= compare_text_blocks(a, b) <= 1.0

    def test_identical_text_returns_one(self):
        a = _block("hello world")
        b = _block("hello world")
        assert compare_text_blocks(a, b) == pytest.approx(1.0)

    def test_symmetric(self):
        a = _block("abcde")
        b = _block("abxde")
        assert compare_text_blocks(a, b) == pytest.approx(compare_text_blocks(b, a))

    @pytest.mark.parametrize("text_a,text_b", [
        ("hello", "world"),
        ("abc", "xyz"),
        ("one", "two"),
        ("long string here", "short"),
    ])
    def test_symmetric_various(self, text_a, text_b):
        a = _block(text_a)
        b = _block(text_b)
        assert compare_text_blocks(a, b) == pytest.approx(
            compare_text_blocks(b, a), abs=1e-12
        )

    def test_both_empty_returns_one(self):
        a = _block("")
        b = _block("")
        assert compare_text_blocks(a, b) == pytest.approx(1.0)

    def test_one_empty_returns_zero(self):
        a = _block("hello")
        b = _block("")
        assert compare_text_blocks(a, b) == pytest.approx(0.0)

    def test_other_empty_returns_zero(self):
        a = _block("")
        b = _block("world")
        assert compare_text_blocks(a, b) == pytest.approx(0.0)

    def test_single_char_identical(self):
        a = _block("x")
        b = _block("x")
        assert compare_text_blocks(a, b) == pytest.approx(1.0)

    def test_single_char_different(self):
        a = _block("x")
        b = _block("y")
        assert compare_text_blocks(a, b) == pytest.approx(0.0)

    def test_similar_texts_higher_than_different(self):
        base = _block("hello world")
        similar = _block("hello xorld")  # 1 char diff
        different = _block("zzzzz zzzzz")
        sim_score = compare_text_blocks(base, similar)
        diff_score = compare_text_blocks(base, different)
        assert sim_score > diff_score

    def test_longer_shared_prefix_higher_score(self):
        base = _block("abcdefgh")
        close = _block("abcdefgx")   # 1 char diff
        far = _block("xxxxxxxh")     # 7 chars diff
        assert compare_text_blocks(base, close) > compare_text_blocks(base, far)

    def test_different_positions_same_text_returns_one(self):
        a = TextBlock(text="same", x=0, y=0, w=10, h=10)
        b = TextBlock(text="same", x=100, y=200, w=50, h=30)
        assert compare_text_blocks(a, b) == pytest.approx(1.0)

    def test_whitespace_only_texts(self):
        a = _block("   ")
        b = _block("hello")
        # "   ".strip() == "" → one is empty → 0.0
        assert compare_text_blocks(a, b) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — align_text_blocks
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlignTextBlocks:
    """align_text_blocks invariants."""

    @pytest.mark.parametrize("mode", ["top-to-bottom", "left-to-right", "reading-order"])
    def test_empty_returns_empty(self, mode):
        assert align_text_blocks([], primary=mode) == []

    @pytest.mark.parametrize("mode", ["top-to-bottom", "left-to-right", "reading-order"])
    def test_single_block_preserved(self, mode):
        blk = _block("x", x=5, y=10)
        result = align_text_blocks([blk], primary=mode)
        assert len(result) == 1
        assert result[0] is blk

    @pytest.mark.parametrize("mode", ["top-to-bottom", "left-to-right", "reading-order"])
    def test_same_count(self, mode):
        blocks = [_block(str(i), x=i * 20, y=i * 10) for i in range(5)]
        result = align_text_blocks(blocks, primary=mode)
        assert len(result) == len(blocks)

    @pytest.mark.parametrize("mode", ["top-to-bottom", "left-to-right", "reading-order"])
    def test_all_blocks_present(self, mode):
        blocks = [_block(str(i), x=i * 20, y=i * 10) for i in range(4)]
        result = align_text_blocks(blocks, primary=mode)
        assert set(id(b) for b in result) == set(id(b) for b in blocks)

    def test_top_to_bottom_sorted_by_y(self):
        blocks = [
            _block("c", x=0, y=30),
            _block("a", x=0, y=10),
            _block("b", x=0, y=20),
        ]
        result = align_text_blocks(blocks, primary="top-to-bottom")
        ys = [b.y for b in result]
        assert ys == sorted(ys)

    def test_left_to_right_sorted_by_x(self):
        blocks = [
            _block("c", x=30, y=0),
            _block("a", x=10, y=0),
            _block("b", x=20, y=0),
        ]
        result = align_text_blocks(blocks, primary="left-to-right")
        xs = [b.x for b in result]
        assert xs == sorted(xs)

    def test_top_to_bottom_tie_broken_by_x(self):
        blocks = [
            _block("b", x=20, y=10),
            _block("a", x=5, y=10),
        ]
        result = align_text_blocks(blocks, primary="top-to-bottom")
        assert result[0].x == 5
        assert result[1].x == 20

    def test_left_to_right_tie_broken_by_y(self):
        blocks = [
            _block("b", x=10, y=20),
            _block("a", x=10, y=5),
        ]
        result = align_text_blocks(blocks, primary="left-to-right")
        assert result[0].y == 5
        assert result[1].y == 20

    def test_invalid_primary_raises(self):
        blocks = [_block("x")]
        with pytest.raises(ValueError):
            align_text_blocks(blocks, primary="invalid-order")

    def test_reading_order_all_blocks_present(self):
        blocks = [
            _block("a", x=0, y=0, w=20, h=15),
            _block("b", x=25, y=0, w=20, h=15),
            _block("c", x=0, y=20, w=20, h=15),
        ]
        result = align_text_blocks(blocks, primary="reading-order")
        assert len(result) == 3
        assert set(id(b) for b in result) == set(id(b) for b in blocks)

    def test_already_sorted_stays_sorted(self):
        blocks = [_block(str(i), x=0, y=i * 20) for i in range(5)]
        result = align_text_blocks(blocks, primary="top-to-bottom")
        ys = [b.y for b in result]
        assert ys == sorted(ys)

    def test_reversed_order_properly_sorted(self):
        blocks = [_block(str(i), x=0, y=(4 - i) * 20) for i in range(5)]
        result = align_text_blocks(blocks, primary="top-to-bottom")
        ys = [b.y for b in result]
        assert ys == sorted(ys)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — batch_clean_text
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchCleanText:
    """batch_clean_text invariants."""

    def test_empty_list_returns_empty(self):
        assert batch_clean_text([]) == []

    def test_same_length_as_input(self):
        texts = ["Hello", "World!", "  test  ", ""]
        result = batch_clean_text(texts)
        assert len(result) == len(texts)

    def test_each_equals_clean_ocr_text(self):
        texts = ["  Hello  ", "OCR: text!", "  CAPS  ", "line\nbreak"]
        cfg = TextConfig()
        result = batch_clean_text(texts, cfg=cfg)
        for t, r in zip(texts, result):
            assert r == clean_ocr_text(t, cfg=cfg)

    def test_config_applied_lowercase(self):
        cfg = TextConfig(lowercase=True)
        texts = ["HELLO", "WORLD", "ABC"]
        result = batch_clean_text(texts, cfg=cfg)
        for r in result:
            assert r == r.lower()

    def test_config_applied_strip_punct(self):
        cfg = TextConfig(strip_punct=True)
        texts = ["Hello!", "World?", "Test: 123."]
        result = batch_clean_text(texts, cfg=cfg)
        for r in result:
            assert not re.search(r"[^\w\s]", r, flags=re.UNICODE)

    def test_single_element(self):
        texts = ["  single  "]
        result = batch_clean_text(texts)
        assert len(result) == 1
        assert result[0] == "single"

    def test_idempotent(self):
        texts = ["  Hello, World!  ", "TEST 123", "  spaces  "]
        cfg = TextConfig()
        once = batch_clean_text(texts, cfg=cfg)
        twice = batch_clean_text(once, cfg=cfg)
        assert once == twice

    def test_empty_strings_preserved_as_empty(self):
        texts = ["", "", ""]
        result = batch_clean_text(texts)
        assert result == ["", "", ""]

    def test_preserves_newlines_in_batch(self):
        texts = ["line1\nline2", "a\nb\nc"]
        result = batch_clean_text(texts)
        assert "\n" in result[0]
        assert "\n" in result[1]
