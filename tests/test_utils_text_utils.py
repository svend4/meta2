"""Тесты для puzzle_reconstruction/utils/text_utils.py."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_binary(h=32, w=32, fill=0):
    return np.full((h, w), fill, dtype=np.uint8)


def make_text_binary():
    """Бинарное изображение с 'текстовыми строками' (горизонтальные полосы)."""
    img = np.zeros((64, 64), dtype=np.uint8)
    # Строка 1: строки 8-12
    img[8:12, 10:54] = 255
    # Строка 2: строки 24-28
    img[24:28, 10:54] = 255
    # Строка 3: строки 40-44
    img[40:44, 10:54] = 255
    return img


def make_block(text="hello", x=0, y=0, w=100, h=20, confidence=-1.0):
    return TextBlock(text=text, x=x, y=y, w=w, h=h, confidence=confidence)


# ─── TextConfig ───────────────────────────────────────────────────────────────

class TestTextConfig:
    def test_defaults(self):
        cfg = TextConfig()
        assert cfg.min_word_gap == 4
        assert cfg.line_threshold == pytest.approx(0.05)
        assert cfg.strip_punct is False
        assert cfg.lowercase is False
        assert cfg.min_line_height == 4

    def test_negative_min_word_gap_raises(self):
        with pytest.raises(ValueError, match="min_word_gap"):
            TextConfig(min_word_gap=-1)

    def test_line_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError, match="line_threshold"):
            TextConfig(line_threshold=1.5)

    def test_min_line_height_zero_raises(self):
        with pytest.raises(ValueError, match="min_line_height"):
            TextConfig(min_line_height=0)

    def test_custom_values_stored(self):
        cfg = TextConfig(min_word_gap=8, line_threshold=0.1, strip_punct=True,
                         lowercase=True, min_line_height=6)
        assert cfg.min_word_gap == 8
        assert cfg.line_threshold == pytest.approx(0.1)
        assert cfg.strip_punct is True
        assert cfg.lowercase is True
        assert cfg.min_line_height == 6

    def test_zero_min_word_gap_valid(self):
        cfg = TextConfig(min_word_gap=0)
        assert cfg.min_word_gap == 0


# ─── TextBlock ────────────────────────────────────────────────────────────────

class TestTextBlock:
    def test_creation(self):
        b = TextBlock(text="hello", x=10, y=20, w=100, h=30)
        assert b.text == "hello"
        assert b.x == 10
        assert b.y == 20
        assert b.w == 100
        assert b.h == 30
        assert b.confidence == pytest.approx(-1.0)
        assert b.source_id is None

    def test_negative_w_raises(self):
        with pytest.raises(ValueError, match="w"):
            TextBlock(text="x", x=0, y=0, w=-1, h=10)

    def test_negative_h_raises(self):
        with pytest.raises(ValueError, match="h"):
            TextBlock(text="x", x=0, y=0, w=10, h=-1)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            TextBlock(text="x", x=0, y=0, w=10, h=10, confidence=1.5)

    def test_confidence_minus_one_valid(self):
        b = TextBlock(text="x", x=0, y=0, w=10, h=10, confidence=-1.0)
        assert b.confidence == pytest.approx(-1.0)

    def test_area_property(self):
        b = make_block(w=50, h=20)
        assert b.area == 1000

    def test_center_property(self):
        b = make_block(x=10, y=20, w=40, h=20)
        cx, cy = b.center
        assert cx == pytest.approx(30.0)
        assert cy == pytest.approx(30.0)

    def test_n_chars_property(self):
        b = make_block(text="hello world")
        assert b.n_chars == 10  # 'hello' + 'world' = 10 non-space chars

    def test_n_chars_empty(self):
        b = make_block(text="   ")
        assert b.n_chars == 0

    def test_source_id_stored(self):
        b = TextBlock(text="x", x=0, y=0, w=10, h=10, source_id=42)
        assert b.source_id == 42


# ─── clean_ocr_text ───────────────────────────────────────────────────────────

class TestCleanOcrText:
    def test_returns_string(self):
        assert isinstance(clean_ocr_text("hello"), str)

    def test_strips_whitespace(self):
        result = clean_ocr_text("  hello  ")
        assert result == "hello"

    def test_collapses_spaces(self):
        result = clean_ocr_text("hello   world")
        assert result == "hello world"

    def test_none_cfg_uses_defaults(self):
        result = clean_ocr_text("test", cfg=None)
        assert isinstance(result, str)

    def test_lowercase_flag(self):
        cfg = TextConfig(lowercase=True)
        result = clean_ocr_text("Hello World", cfg=cfg)
        assert result == "hello world"

    def test_lowercase_false_preserves_case(self):
        cfg = TextConfig(lowercase=False)
        result = clean_ocr_text("Hello World", cfg=cfg)
        assert result == "Hello World"

    def test_strip_punct_removes_punctuation(self):
        cfg = TextConfig(strip_punct=True)
        result = clean_ocr_text("hello, world!", cfg=cfg)
        assert "," not in result
        assert "!" not in result

    def test_strip_punct_false_keeps_punctuation(self):
        cfg = TextConfig(strip_punct=False)
        result = clean_ocr_text("hello, world!", cfg=cfg)
        assert "," in result

    def test_empty_string(self):
        result = clean_ocr_text("")
        assert result == ""

    def test_unicode_nfc_normalization(self):
        # Decomposed 'é' (e + combining accent) → NFC composed 'é'
        decomposed = "e\u0301"  # e + combining acute accent
        result = clean_ocr_text(decomposed)
        assert "\u00e9" in result  # é (precomposed)

    def test_preserves_newlines(self):
        result = clean_ocr_text("line1\nline2")
        assert "\n" in result


# ─── estimate_text_density ────────────────────────────────────────────────────

class TestEstimateTextDensity:
    def test_returns_float(self):
        img = make_binary()
        result = estimate_text_density(img)
        assert isinstance(result, float)

    def test_all_zeros_returns_zero(self):
        img = make_binary(fill=0)
        assert estimate_text_density(img) == pytest.approx(0.0)

    def test_all_ones_returns_one(self):
        img = make_binary(fill=255)
        assert estimate_text_density(img) == pytest.approx(1.0)

    def test_half_filled_returns_half(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        img[:5] = 255
        result = estimate_text_density(img)
        assert result == pytest.approx(0.5)

    def test_3d_array_raises(self):
        with pytest.raises(ValueError):
            estimate_text_density(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_in_0_1(self):
        img = make_text_binary()
        result = estimate_text_density(img)
        assert 0.0 <= result <= 1.0


# ─── find_text_lines ──────────────────────────────────────────────────────────

class TestFindTextLines:
    def test_returns_list(self):
        img = make_text_binary()
        result = find_text_lines(img)
        assert isinstance(result, list)

    def test_blank_image_returns_empty(self):
        img = make_binary(fill=0)
        result = find_text_lines(img)
        assert result == []

    def test_finds_lines(self):
        img = make_text_binary()
        result = find_text_lines(img)
        assert len(result) >= 1

    def test_tuples_format(self):
        img = make_text_binary()
        result = find_text_lines(img)
        for item in result:
            y0, y1 = item
            assert y0 < y1

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            find_text_lines(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_none_cfg_uses_defaults(self):
        img = make_text_binary()
        result = find_text_lines(img, cfg=None)
        assert isinstance(result, list)

    def test_high_threshold_fewer_lines(self):
        img = make_text_binary()
        cfg_low = TextConfig(line_threshold=0.01)
        cfg_high = TextConfig(line_threshold=0.99)
        low = find_text_lines(img, cfg=cfg_low)
        high = find_text_lines(img, cfg=cfg_high)
        assert len(high) <= len(low)


# ─── segment_words ────────────────────────────────────────────────────────────

class TestSegmentWords:
    def test_returns_list(self):
        img = make_binary(h=16, w=64)
        result = segment_words(img)
        assert isinstance(result, list)

    def test_blank_line_returns_empty(self):
        img = make_binary(h=16, w=64, fill=0)
        result = segment_words(img)
        assert result == []

    def test_single_word_block(self):
        img = np.zeros((16, 64), dtype=np.uint8)
        img[4:12, 10:54] = 255  # one continuous word block
        result = segment_words(img)
        assert len(result) >= 1

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            segment_words(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_tuples_format(self):
        img = np.zeros((16, 64), dtype=np.uint8)
        img[4:12, 5:20] = 255
        img[4:12, 35:55] = 255
        result = segment_words(img)
        for x0, x1 in result:
            assert x0 < x1

    def test_none_cfg_uses_defaults(self):
        img = np.zeros((16, 64), dtype=np.uint8)
        img[4:12, 5:20] = 255
        result = segment_words(img, cfg=None)
        assert isinstance(result, list)


# ─── compute_text_score ───────────────────────────────────────────────────────

class TestComputeTextScore:
    def test_returns_float(self):
        img = make_text_binary()
        result = compute_text_score(img)
        assert isinstance(result, float)

    def test_in_0_1(self):
        img = make_text_binary()
        result = compute_text_score(img)
        assert 0.0 <= result <= 1.0

    def test_blank_image_returns_zero(self):
        img = make_binary(fill=0)
        result = compute_text_score(img)
        assert result == pytest.approx(0.0)

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            compute_text_score(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_text_image_higher_than_blank(self):
        text_img = make_text_binary()
        blank = make_binary(fill=0)
        assert compute_text_score(text_img) > compute_text_score(blank)

    def test_none_cfg_uses_defaults(self):
        img = make_text_binary()
        result = compute_text_score(img, cfg=None)
        assert isinstance(result, float)


# ─── compare_text_blocks ──────────────────────────────────────────────────────

class TestCompareTextBlocks:
    def test_returns_float(self):
        a = make_block("hello")
        b = make_block("hello")
        assert isinstance(compare_text_blocks(a, b), float)

    def test_identical_texts_returns_1(self):
        a = make_block("hello world")
        b = make_block("hello world")
        assert compare_text_blocks(a, b) == pytest.approx(1.0)

    def test_completely_different_returns_low(self):
        a = make_block("aaaa")
        b = make_block("bbbb")
        result = compare_text_blocks(a, b)
        assert 0.0 <= result <= 1.0

    def test_empty_both_returns_1(self):
        a = make_block("")
        b = make_block("")
        assert compare_text_blocks(a, b) == pytest.approx(1.0)

    def test_empty_vs_nonempty_returns_0(self):
        a = make_block("hello")
        b = make_block("")
        assert compare_text_blocks(a, b) == pytest.approx(0.0)

    def test_in_0_1(self):
        a = make_block("abc")
        b = make_block("axc")
        result = compare_text_blocks(a, b)
        assert 0.0 <= result <= 1.0

    def test_partial_match(self):
        a = make_block("hello")
        b = make_block("hell")
        result = compare_text_blocks(a, b)
        assert result > 0.5

    def test_symmetry(self):
        a = make_block("hello")
        b = make_block("world")
        assert compare_text_blocks(a, b) == pytest.approx(compare_text_blocks(b, a))


# ─── align_text_blocks ────────────────────────────────────────────────────────

class TestAlignTextBlocks:
    def _blocks(self):
        return [
            make_block("c", x=50, y=0),
            make_block("a", x=0, y=50),
            make_block("b", x=0, y=0),
        ]

    def test_top_to_bottom_sorts_by_y_then_x(self):
        blocks = self._blocks()
        result = align_text_blocks(blocks, primary="top-to-bottom")
        ys = [b.y for b in result]
        # Not strictly monotone (ties by x), but y[0] <= y[1] <= y[2]
        assert ys[0] <= ys[1] <= ys[2]

    def test_left_to_right_sorts_by_x(self):
        blocks = self._blocks()
        result = align_text_blocks(blocks, primary="left-to-right")
        xs = [b.x for b in result]
        assert xs[0] <= xs[1] <= xs[2]

    def test_reading_order_returns_list(self):
        blocks = self._blocks()
        result = align_text_blocks(blocks, primary="reading-order")
        assert isinstance(result, list)
        assert len(result) == 3

    def test_unknown_primary_raises(self):
        blocks = self._blocks()
        with pytest.raises(ValueError):
            align_text_blocks(blocks, primary="zigzag")

    def test_empty_input_returns_empty(self):
        result = align_text_blocks([], primary="top-to-bottom")
        assert result == []

    def test_preserves_all_blocks(self):
        blocks = self._blocks()
        result = align_text_blocks(blocks, primary="top-to-bottom")
        assert len(result) == 3

    def test_reading_order_empty_returns_empty(self):
        result = align_text_blocks([], primary="reading-order")
        assert result == []

    def test_reading_order_single(self):
        result = align_text_blocks([make_block("x", x=0, y=0)], primary="reading-order")
        assert len(result) == 1


# ─── batch_clean_text ─────────────────────────────────────────────────────────

class TestBatchCleanText:
    def test_returns_list(self):
        result = batch_clean_text(["hello", "world"])
        assert isinstance(result, list)

    def test_length_preserved(self):
        result = batch_clean_text(["a", "b", "c"])
        assert len(result) == 3

    def test_empty_input(self):
        result = batch_clean_text([])
        assert result == []

    def test_all_strings(self):
        result = batch_clean_text(["Hello", "WORLD"])
        for r in result:
            assert isinstance(r, str)

    def test_lowercase_applied(self):
        cfg = TextConfig(lowercase=True)
        result = batch_clean_text(["Hello", "WORLD"], cfg=cfg)
        assert result == ["hello", "world"]

    def test_none_cfg_uses_defaults(self):
        result = batch_clean_text(["test"], cfg=None)
        assert result == ["test"]
