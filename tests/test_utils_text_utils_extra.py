"""Extra tests for puzzle_reconstruction/utils/text_utils.py."""
from __future__ import annotations

import pytest
import numpy as np

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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _blank(h=32, w=64) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _full(h=32, w=64) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def _block(text="hello", x=0, y=0, w=50, h=10, conf=0.9) -> TextBlock:
    return TextBlock(text=text, x=x, y=y, w=w, h=h, confidence=conf)


def _text_image() -> np.ndarray:
    """Simple binary image simulating text lines."""
    img = np.zeros((64, 128), dtype=np.uint8)
    img[10:16, 10:110] = 255
    img[30:36, 10:110] = 255
    return img


# ─── TextConfig ───────────────────────────────────────────────────────────────

class TestTextConfigExtra:
    def test_default_min_word_gap(self):
        assert TextConfig().min_word_gap == 4

    def test_default_line_threshold(self):
        assert TextConfig().line_threshold == pytest.approx(0.05)

    def test_default_strip_punct(self):
        assert TextConfig().strip_punct is False

    def test_default_lowercase(self):
        assert TextConfig().lowercase is False

    def test_default_min_line_height(self):
        assert TextConfig().min_line_height == 4

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

    def test_custom_values(self):
        cfg = TextConfig(min_word_gap=8, lowercase=True)
        assert cfg.min_word_gap == 8
        assert cfg.lowercase is True


# ─── TextBlock ────────────────────────────────────────────────────────────────

class TestTextBlockExtra:
    def test_stores_text(self):
        assert _block(text="foo").text == "foo"

    def test_stores_position(self):
        b = _block(x=10, y=20)
        assert b.x == 10 and b.y == 20

    def test_negative_w_raises(self):
        with pytest.raises(ValueError):
            TextBlock(text="a", x=0, y=0, w=-1, h=10)

    def test_negative_h_raises(self):
        with pytest.raises(ValueError):
            TextBlock(text="a", x=0, y=0, w=10, h=-1)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            TextBlock(text="a", x=0, y=0, w=10, h=5, confidence=1.5)

    def test_minus_one_confidence_ok(self):
        b = TextBlock(text="a", x=0, y=0, w=10, h=5, confidence=-1.0)
        assert b.confidence == pytest.approx(-1.0)

    def test_area_property(self):
        b = _block(w=10, h=5)
        assert b.area == 50

    def test_center_property(self):
        b = TextBlock(text="a", x=0, y=0, w=10, h=4)
        cx, cy = b.center
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(2.0)

    def test_n_chars_property(self):
        b = _block(text="hello world")
        assert b.n_chars == 10  # non-whitespace chars

    def test_n_chars_empty_text(self):
        b = _block(text="")
        assert b.n_chars == 0

    def test_source_id_default_none(self):
        b = _block()
        assert b.source_id is None

    def test_zero_w_h_ok(self):
        b = TextBlock(text="", x=0, y=0, w=0, h=0)
        assert b.area == 0


# ─── clean_ocr_text ───────────────────────────────────────────────────────────

class TestCleanOcrTextExtra:
    def test_returns_str(self):
        assert isinstance(clean_ocr_text("hello"), str)

    def test_strips_control_chars(self):
        out = clean_ocr_text("he\x01llo")
        assert "\x01" not in out

    def test_collapses_spaces(self):
        out = clean_ocr_text("hello   world")
        assert "  " not in out

    def test_lowercase(self):
        cfg = TextConfig(lowercase=True)
        assert clean_ocr_text("HELLO", cfg=cfg) == "hello"

    def test_strip_punct(self):
        cfg = TextConfig(strip_punct=True)
        out = clean_ocr_text("hello!", cfg=cfg)
        assert "!" not in out

    def test_empty_string(self):
        assert clean_ocr_text("") == ""

    def test_none_cfg(self):
        assert isinstance(clean_ocr_text("test", cfg=None), str)

    def test_preserves_words(self):
        out = clean_ocr_text("hello world")
        assert "hello" in out and "world" in out


# ─── estimate_text_density ────────────────────────────────────────────────────

class TestEstimateTextDensityExtra:
    def test_returns_float(self):
        assert isinstance(estimate_text_density(_blank()), float)

    def test_blank_is_zero(self):
        assert estimate_text_density(_blank()) == pytest.approx(0.0)

    def test_full_is_one(self):
        assert estimate_text_density(_full()) == pytest.approx(1.0)

    def test_in_range(self):
        d = estimate_text_density(_text_image())
        assert 0.0 <= d <= 1.0

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            estimate_text_density(np.zeros((2, 4, 4)))

    def test_half_filled(self):
        img = np.zeros((4, 4), dtype=np.uint8)
        img[:2] = 255
        d = estimate_text_density(img)
        assert d == pytest.approx(0.5)


# ─── find_text_lines ──────────────────────────────────────────────────────────

class TestFindTextLinesExtra:
    def test_returns_list(self):
        assert isinstance(find_text_lines(_blank()), list)

    def test_blank_no_lines(self):
        assert find_text_lines(_blank()) == []

    def test_elements_are_tuples(self):
        lines = find_text_lines(_text_image())
        for item in lines:
            assert isinstance(item, tuple) and len(item) == 2

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            find_text_lines(np.zeros((2, 4, 4)))

    def test_none_cfg(self):
        result = find_text_lines(_blank(), cfg=None)
        assert isinstance(result, list)

    def test_text_image_finds_lines(self):
        lines = find_text_lines(_text_image())
        assert len(lines) >= 1


# ─── segment_words ────────────────────────────────────────────────────────────

class TestSegmentWordsExtra:
    def test_returns_list(self):
        assert isinstance(segment_words(_blank()), list)

    def test_blank_no_words(self):
        assert segment_words(_blank()) == []

    def test_elements_are_tuples(self):
        line = np.zeros((10, 64), dtype=np.uint8)
        line[:, 5:15] = 255
        line[:, 30:45] = 255
        for item in segment_words(line):
            assert isinstance(item, tuple) and len(item) == 2

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            segment_words(np.zeros((2, 4, 4)))

    def test_none_cfg(self):
        result = segment_words(_blank(), cfg=None)
        assert isinstance(result, list)

    def test_two_word_blocks(self):
        line = np.zeros((8, 80), dtype=np.uint8)
        line[:, 5:20] = 255
        line[:, 50:70] = 255
        segs = segment_words(line)
        assert len(segs) >= 2


# ─── compute_text_score ───────────────────────────────────────────────────────

class TestComputeTextScoreExtra:
    def test_returns_float(self):
        assert isinstance(compute_text_score(_blank()), float)

    def test_in_range(self):
        score = compute_text_score(_text_image())
        assert 0.0 <= score <= 1.0

    def test_blank_zero_or_low(self):
        score = compute_text_score(_blank())
        assert score <= 0.1

    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            compute_text_score(np.zeros((2, 4, 4)))

    def test_none_cfg(self):
        score = compute_text_score(_blank(), cfg=None)
        assert isinstance(score, float)


# ─── compare_text_blocks ──────────────────────────────────────────────────────

class TestCompareTextBlocksExtra:
    def test_returns_float(self):
        b = _block("hello")
        assert isinstance(compare_text_blocks(b, b), float)

    def test_identical_returns_one(self):
        b = _block("hello")
        assert compare_text_blocks(b, b) == pytest.approx(1.0)

    def test_empty_both_returns_one(self):
        a = _block("")
        b = _block("")
        result = compare_text_blocks(a, b)
        assert result == pytest.approx(1.0)

    def test_empty_vs_nonempty_returns_zero(self):
        a = _block("")
        b = _block("hello")
        assert compare_text_blocks(a, b) == pytest.approx(0.0)

    def test_in_range(self):
        a = _block("hello")
        b = _block("world")
        sim = compare_text_blocks(a, b)
        assert 0.0 <= sim <= 1.0

    def test_symmetric(self):
        a = _block("hello")
        b = _block("world")
        assert compare_text_blocks(a, b) == pytest.approx(compare_text_blocks(b, a))


# ─── align_text_blocks ────────────────────────────────────────────────────────

class TestAlignTextBlocksExtra:
    def test_returns_list(self):
        blocks = [_block(x=5, y=2), _block(x=1, y=8)]
        assert isinstance(align_text_blocks(blocks), list)

    def test_length_preserved(self):
        blocks = [_block(x=i, y=10 - i) for i in range(5)]
        assert len(align_text_blocks(blocks)) == 5

    def test_top_to_bottom_order(self):
        blocks = [_block(x=0, y=20), _block(x=0, y=5), _block(x=0, y=10)]
        result = align_text_blocks(blocks, primary="top-to-bottom")
        ys = [b.y for b in result]
        assert ys == sorted(ys)

    def test_left_to_right_order(self):
        blocks = [_block(x=30, y=0), _block(x=5, y=0), _block(x=15, y=0)]
        result = align_text_blocks(blocks, primary="left-to-right")
        xs = [b.x for b in result]
        assert xs == sorted(xs)

    def test_invalid_mode_raises(self):
        blocks = [_block()]
        with pytest.raises(ValueError):
            align_text_blocks(blocks, primary="diagonal")

    def test_empty_list_returns_empty(self):
        assert align_text_blocks([]) == []


# ─── batch_clean_text ─────────────────────────────────────────────────────────

class TestBatchCleanTextExtra:
    def test_returns_list(self):
        assert isinstance(batch_clean_text(["hello", "world"]), list)

    def test_length_matches(self):
        result = batch_clean_text(["a", "b", "c"])
        assert len(result) == 3

    def test_each_element_str(self):
        for s in batch_clean_text(["hello", "WORLD"]):
            assert isinstance(s, str)

    def test_lowercase_applied(self):
        cfg = TextConfig(lowercase=True)
        result = batch_clean_text(["HELLO", "WORLD"], cfg=cfg)
        assert result[0] == "hello"
        assert result[1] == "world"

    def test_empty_input_returns_empty(self):
        assert batch_clean_text([]) == []

    def test_none_cfg(self):
        result = batch_clean_text(["test"], cfg=None)
        assert len(result) == 1

    def test_strips_control_chars(self):
        result = batch_clean_text(["he\x01llo"])
        assert "\x01" not in result[0]
