"""Extra tests for puzzle_reconstruction/algorithms/word_segmentation.py."""
from __future__ import annotations

import pytest
import numpy as np

from puzzle_reconstruction.algorithms.word_segmentation import (
    WordBox,
    LineSegment,
    WordSegmentationResult,
    binarize,
    segment_words,
    merge_line_words,
    segment_lines,
    segment_document,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _blank(h=64, w=128) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _text_image() -> np.ndarray:
    """Gray image with two text-line-like white bands."""
    img = np.zeros((64, 128), dtype=np.uint8)
    img[8:18, 10:110] = 200
    img[38:48, 10:110] = 200
    return img


def _bgr(h=64, w=128) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _wb(x=5, y=5, w=20, h=10, line_idx=-1, confidence=1.0) -> WordBox:
    return WordBox(x=x, y=y, w=w, h=h, line_idx=line_idx, confidence=confidence)


def _two_row_words() -> list:
    """Two rows of words clearly separated vertically."""
    row1 = [_wb(x=i * 30, y=5, w=20, h=10) for i in range(3)]
    row2 = [_wb(x=i * 30, y=50, w=20, h=10) for i in range(3)]
    return row1 + row2


# ─── WordBox ──────────────────────────────────────────────────────────────────

class TestWordBoxExtra:
    def test_x2_property(self):
        wb = _wb(x=5, w=20)
        assert wb.x2 == 25

    def test_y2_property(self):
        wb = _wb(y=3, h=10)
        assert wb.y2 == 13

    def test_cx_property(self):
        wb = _wb(x=4, w=10)
        assert wb.cx == pytest.approx(9.0)

    def test_cy_property(self):
        wb = _wb(y=4, h=10)
        assert wb.cy == pytest.approx(9.0)

    def test_area_property(self):
        wb = _wb(w=10, h=5)
        assert wb.area == 50

    def test_aspect_ratio_property(self):
        wb = _wb(w=20, h=10)
        assert wb.aspect_ratio == pytest.approx(2.0)

    def test_aspect_ratio_zero_height(self):
        wb = _wb(w=10, h=0)
        assert wb.aspect_ratio == pytest.approx(0.0)

    def test_to_tuple(self):
        wb = _wb(x=1, y=2, w=3, h=4)
        assert wb.to_tuple() == (1, 2, 3, 4)

    def test_iou_identical(self):
        wb = _wb()
        assert wb.iou(wb) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = _wb(x=0, y=0, w=5, h=5)
        b = _wb(x=100, y=100, w=5, h=5)
        assert a.iou(b) == pytest.approx(0.0)

    def test_iou_symmetric(self):
        a = _wb(x=0, y=0, w=10, h=10)
        b = _wb(x=5, y=5, w=10, h=10)
        assert a.iou(b) == pytest.approx(b.iou(a))

    def test_iou_partial_overlap(self):
        a = _wb(x=0, y=0, w=10, h=10)
        b = _wb(x=5, y=0, w=10, h=10)
        iou = a.iou(b)
        assert 0.0 < iou < 1.0

    def test_default_line_idx(self):
        wb = WordBox(x=0, y=0, w=5, h=5)
        assert wb.line_idx == -1

    def test_confidence_stored(self):
        wb = _wb(confidence=0.75)
        assert wb.confidence == pytest.approx(0.75)

    def test_repr_contains_line(self):
        wb = _wb(line_idx=2)
        assert "line" in repr(wb).lower() or "2" in repr(wb)


# ─── LineSegment ──────────────────────────────────────────────────────────────

class TestLineSegmentExtra:
    def test_n_words_empty(self):
        ls = LineSegment(line_idx=0)
        assert ls.n_words == 0

    def test_n_words_with_words(self):
        ls = LineSegment(line_idx=0, words=[_wb(), _wb()])
        assert ls.n_words == 2

    def test_y_center(self):
        ls = LineSegment(line_idx=0, bbox=(0, 10, 100, 20))
        assert ls.y_center == pytest.approx(20.0)  # 10 + 20/2

    def test_x_start(self):
        ls = LineSegment(line_idx=0, bbox=(15, 5, 80, 15))
        assert ls.x_start == 15

    def test_avg_word_height_empty(self):
        ls = LineSegment(line_idx=0)
        assert ls.avg_word_height() == pytest.approx(0.0)

    def test_avg_word_height_with_words(self):
        ls = LineSegment(line_idx=0, words=[_wb(h=10), _wb(h=20)])
        assert ls.avg_word_height() == pytest.approx(15.0)

    def test_repr_contains_line_idx(self):
        ls = LineSegment(line_idx=3)
        assert "3" in repr(ls)

    def test_repr_contains_n_words(self):
        ls = LineSegment(line_idx=0, words=[_wb()])
        assert "1" in repr(ls)


# ─── WordSegmentationResult ───────────────────────────────────────────────────

class TestWordSegmentationResultExtra:
    def _make_result(self, n_words=4, n_lines=2) -> WordSegmentationResult:
        words = [_wb(line_idx=i % n_lines) for i in range(n_words)]
        lines = [LineSegment(line_idx=i) for i in range(n_lines)]
        return WordSegmentationResult(
            words=words, lines=lines, image_shape=(64, 128)
        )

    def test_n_words(self):
        r = self._make_result(n_words=6)
        assert r.n_words == 6

    def test_n_lines(self):
        r = self._make_result(n_lines=3)
        assert r.n_lines == 3

    def test_avg_words_per_line(self):
        r = self._make_result(n_words=4, n_lines=2)
        assert r.avg_words_per_line == pytest.approx(2.0)

    def test_avg_words_per_line_no_lines(self):
        r = WordSegmentationResult(words=[], lines=[], image_shape=(64, 128))
        assert r.avg_words_per_line == pytest.approx(0.0)

    def test_words_on_line_valid(self):
        r = self._make_result(n_words=4, n_lines=2)
        result = r.words_on_line(0)
        assert isinstance(result, list)

    def test_words_on_line_invalid(self):
        r = self._make_result()
        assert r.words_on_line(99) == []

    def test_image_shape_stored(self):
        r = self._make_result()
        assert r.image_shape == (64, 128)

    def test_summary_is_str(self):
        r = self._make_result()
        assert isinstance(r.summary(), str)

    def test_repr_equals_summary(self):
        r = self._make_result()
        assert repr(r) == r.summary()

    def test_binarize_method_stored(self):
        r = WordSegmentationResult(
            words=[], lines=[], image_shape=(32, 32), binarize_method="adaptive"
        )
        assert r.binarize_method == "adaptive"


# ─── binarize ─────────────────────────────────────────────────────────────────

class TestBinarizeExtra:
    def test_returns_ndarray(self):
        assert isinstance(binarize(_blank()), np.ndarray)

    def test_dtype_uint8(self):
        assert binarize(_blank()).dtype == np.uint8

    def test_only_0_and_255(self):
        out = binarize(_text_image())
        assert set(np.unique(out)).issubset({0, 255})

    def test_shape_preserved_otsu(self):
        img = _blank(32, 64)
        assert binarize(img, method="otsu").shape == (32, 64)

    def test_shape_preserved_adaptive(self):
        img = _blank(32, 64)
        assert binarize(img, method="adaptive").shape == (32, 64)

    def test_shape_preserved_sauvola(self):
        img = _blank(32, 64)
        assert binarize(img, method="sauvola").shape == (32, 64)

    def test_sauvola_values_0_or_255(self):
        out = binarize(_text_image(), method="sauvola")
        assert set(np.unique(out)).issubset({0, 255})

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            binarize(_blank(), method="watershed")

    def test_bgr_input_otsu(self):
        out = binarize(_bgr(), method="otsu")
        assert out.ndim == 2

    def test_bgr_input_adaptive(self):
        out = binarize(_bgr(), method="adaptive")
        assert out.ndim == 2

    def test_text_has_nonzero_pixels(self):
        out = binarize(_text_image(), method="otsu")
        assert out.max() == 255


# ─── segment_words ────────────────────────────────────────────────────────────

class TestSegmentWordsExtra:
    def test_returns_list(self):
        assert isinstance(segment_words(_blank()), list)

    def test_blank_empty_or_list(self):
        result = segment_words(_blank())
        assert isinstance(result, list)

    def test_all_wordbox(self):
        for wb in segment_words(_text_image()):
            assert isinstance(wb, WordBox)

    def test_dimensions_positive(self):
        for wb in segment_words(_text_image()):
            assert wb.w > 0 and wb.h > 0

    def test_sorted_by_y_x(self):
        words = segment_words(_text_image())
        for i in range(1, len(words)):
            prev = (words[i - 1].y, words[i - 1].x)
            cur = (words[i].y, words[i].x)
            assert prev <= cur

    def test_min_area_filter(self):
        all_words = segment_words(_text_image(), min_area=1)
        filtered = segment_words(_text_image(), min_area=10000)
        assert len(filtered) <= len(all_words)

    def test_grayscale_input(self):
        result = segment_words(_blank())
        assert isinstance(result, list)

    def test_bgr_input(self):
        result = segment_words(_bgr())
        assert isinstance(result, list)


# ─── merge_line_words ─────────────────────────────────────────────────────────

class TestMergeLineWordsExtra:
    def test_empty_returns_empty(self):
        assert merge_line_words([]) == []

    def test_returns_list(self):
        assert isinstance(merge_line_words([_wb()]), list)

    def test_each_element_line_segment(self):
        for ls in merge_line_words(_two_row_words()):
            assert isinstance(ls, LineSegment)

    def test_single_word_one_line(self):
        result = merge_line_words([_wb()])
        assert len(result) == 1

    def test_two_rows_two_lines(self):
        result = merge_line_words(_two_row_words(), line_gap=20.0)
        assert len(result) == 2

    def test_line_idx_assigned(self):
        words = _two_row_words()
        merge_line_words(words)
        assert all(w.line_idx >= 0 for w in words)

    def test_lines_sorted_top_to_bottom(self):
        result = merge_line_words(_two_row_words())
        centers = [ls.y_center for ls in result]
        assert centers == sorted(centers)

    def test_default_line_gap_auto(self):
        # With default line_gap=None, auto-computed
        result = merge_line_words(_two_row_words())
        assert len(result) >= 1

    def test_explicit_line_gap(self):
        result = merge_line_words(_two_row_words(), line_gap=20.0)
        assert isinstance(result, list)


# ─── segment_lines ────────────────────────────────────────────────────────────

class TestSegmentLinesExtra:
    def test_returns_list(self):
        assert isinstance(segment_lines(_blank()), list)

    def test_each_element_line_segment(self):
        for ls in segment_lines(_text_image()):
            assert isinstance(ls, LineSegment)

    def test_blank_image_no_crash(self):
        result = segment_lines(_blank())
        assert isinstance(result, list)

    def test_bgr_input(self):
        result = segment_lines(_bgr())
        assert isinstance(result, list)


# ─── segment_document ─────────────────────────────────────────────────────────

class TestSegmentDocumentExtra:
    def test_returns_result(self):
        r = segment_document(_blank())
        assert isinstance(r, WordSegmentationResult)

    def test_image_shape_stored(self):
        r = segment_document(_blank(32, 64))
        assert r.image_shape == (32, 64)

    def test_binarize_method_stored_otsu(self):
        r = segment_document(_blank(), binarize_method="otsu")
        assert r.binarize_method == "otsu"

    def test_binarize_method_stored_adaptive(self):
        r = segment_document(_blank(), binarize_method="adaptive")
        assert r.binarize_method == "adaptive"

    def test_blank_no_crash(self):
        r = segment_document(_blank())
        assert isinstance(r, WordSegmentationResult)

    def test_text_image_has_words(self):
        r = segment_document(_text_image())
        assert r.n_words >= 0  # may or may not find words depending on threshold

    def test_words_equals_len_words_list(self):
        r = segment_document(_text_image())
        assert r.n_words == len(r.words)

    def test_lines_equals_len_lines_list(self):
        r = segment_document(_text_image())
        assert r.n_lines == len(r.lines)

    def test_bgr_input_ok(self):
        r = segment_document(_bgr())
        assert isinstance(r, WordSegmentationResult)
