"""Extra tests for puzzle_reconstruction.algorithms.word_segmentation."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.word_segmentation import (
    LineSegment,
    WordBox,
    WordSegmentationResult,
    binarize,
    merge_line_words,
    segment_document,
    segment_lines,
    segment_words,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _blank(h=64, w=64):
    return np.zeros((h, w), dtype=np.uint8)


def _text(h=100, w=200):
    img = np.ones((h, w), dtype=np.uint8) * 255
    img[20:30, 10:50] = 0
    img[20:30, 60:100] = 0
    img[60:70, 10:50] = 0
    img[60:70, 60:100] = 0
    return img


def _word(x=0, y=0, w=30, h=15):
    return WordBox(x=x, y=y, w=w, h=h)


# ─── TestWordBoxExtra ─────────────────────────────────────────────────────────

class TestWordBoxExtra:
    def test_x2_equals_x_plus_w(self):
        wb = WordBox(x=5, y=10, w=20, h=10)
        assert wb.x2 == 25

    def test_y2_equals_y_plus_h(self):
        wb = WordBox(x=5, y=10, w=20, h=10)
        assert wb.y2 == 20

    def test_cx_center(self):
        wb = WordBox(x=0, y=0, w=20, h=10)
        assert wb.cx == pytest.approx(10.0)

    def test_cy_center(self):
        wb = WordBox(x=0, y=0, w=20, h=10)
        assert wb.cy == pytest.approx(5.0)

    def test_area_width_times_height(self):
        wb = WordBox(x=0, y=0, w=8, h=4)
        assert wb.area == 32

    def test_aspect_ratio_wide(self):
        wb = WordBox(x=0, y=0, w=40, h=10)
        assert wb.aspect_ratio == pytest.approx(4.0)

    def test_aspect_ratio_zero_height(self):
        wb = WordBox(x=0, y=0, w=10, h=0)
        assert wb.aspect_ratio == pytest.approx(0.0)

    def test_to_tuple(self):
        wb = WordBox(x=1, y=2, w=3, h=4)
        assert wb.to_tuple() == (1, 2, 3, 4)

    def test_iou_identical(self):
        wb = WordBox(x=0, y=0, w=10, h=10)
        assert wb.iou(wb) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = WordBox(x=0, y=0, w=10, h=10)
        b = WordBox(x=20, y=0, w=10, h=10)
        assert a.iou(b) == pytest.approx(0.0)

    def test_iou_partial_overlap_in_range(self):
        a = WordBox(x=0, y=0, w=10, h=10)
        b = WordBox(x=5, y=0, w=10, h=10)
        assert 0.0 < a.iou(b) < 1.0

    def test_default_line_idx(self):
        assert WordBox(x=0, y=0, w=5, h=5).line_idx == -1

    def test_default_confidence_1(self):
        assert WordBox(x=0, y=0, w=5, h=5).confidence == pytest.approx(1.0)

    def test_repr_has_coords(self):
        wb = WordBox(x=3, y=7, w=20, h=10)
        assert "x=3" in repr(wb) and "y=7" in repr(wb)


# ─── TestLineSegmentExtra ─────────────────────────────────────────────────────

class TestLineSegmentExtra:
    def test_default_words_empty(self):
        ls = LineSegment(line_idx=0)
        assert ls.words == []

    def test_default_bbox_zeros(self):
        ls = LineSegment(line_idx=0)
        assert ls.bbox == (0, 0, 0, 0)

    def test_n_words_count(self):
        ls = LineSegment(line_idx=0, words=[_word(), _word(), _word()])
        assert ls.n_words == 3

    def test_y_center(self):
        ls = LineSegment(line_idx=0, bbox=(0, 20, 100, 10))
        assert ls.y_center == pytest.approx(25.0)

    def test_x_start(self):
        ls = LineSegment(line_idx=0, bbox=(15, 0, 100, 10))
        assert ls.x_start == 15

    def test_avg_word_height_empty(self):
        ls = LineSegment(line_idx=0)
        assert ls.avg_word_height() == pytest.approx(0.0)

    def test_avg_word_height_two(self):
        words = [WordBox(x=0, y=0, w=20, h=10), WordBox(x=30, y=0, w=20, h=20)]
        ls = LineSegment(line_idx=0, words=words)
        assert ls.avg_word_height() == pytest.approx(15.0)

    def test_repr_contains_n_words(self):
        ls = LineSegment(line_idx=0, words=[_word()])
        assert "n_words=1" in repr(ls)


# ─── TestWordSegmentationResultExtra ─────────────────────────────────────────

class TestWordSegmentationResultExtra:
    def _result(self, n_words=4, n_lines=2):
        words = [_word(x=i * 10) for i in range(n_words)]
        lines = [LineSegment(line_idx=i, words=[words[i]]) for i in range(n_lines)]
        return WordSegmentationResult(words=words, lines=lines, image_shape=(100, 200))

    def test_n_words(self):
        assert self._result(n_words=5).n_words == 5

    def test_n_lines(self):
        assert self._result(n_lines=3).n_lines == 3

    def test_avg_words_per_line(self):
        r = self._result(n_words=4, n_lines=2)
        assert r.avg_words_per_line == pytest.approx(2.0)

    def test_avg_words_no_lines(self):
        r = WordSegmentationResult(words=[], lines=[], image_shape=(50, 50))
        assert r.avg_words_per_line == pytest.approx(0.0)

    def test_words_on_valid_line(self):
        words = [_word(x=0), _word(x=10)]
        line = LineSegment(line_idx=0, words=words)
        r = WordSegmentationResult(words=words, lines=[line], image_shape=(100, 200))
        assert len(r.words_on_line(0)) == 2

    def test_words_on_invalid_line_empty(self):
        r = WordSegmentationResult(words=[], lines=[], image_shape=(50, 50))
        assert r.words_on_line(99) == []

    def test_default_binarize_method(self):
        r = WordSegmentationResult(words=[], lines=[], image_shape=(50, 50))
        assert r.binarize_method == "otsu"


# ─── TestBinarizeExtra ────────────────────────────────────────────────────────

class TestBinarizeExtra:
    def test_otsu_shape(self):
        bw = binarize(_text(), method="otsu")
        assert bw.shape == _text().shape

    def test_otsu_dtype_uint8(self):
        assert binarize(_text(), method="otsu").dtype == np.uint8

    def test_adaptive_shape(self):
        bw = binarize(_text(), method="adaptive", block=21)
        assert bw.shape == _text().shape

    def test_sauvola_shape(self):
        bw = binarize(_text(h=20, w=40), method="sauvola", block=7)
        assert bw.shape == (20, 40)

    def test_binary_values(self):
        bw = binarize(_text(), method="otsu")
        assert set(np.unique(bw)).issubset({0, 255})

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            binarize(_blank(), method="magic")

    def test_bgr_input_2d_output(self):
        img = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
        assert binarize(img, method="otsu").ndim == 2

    def test_even_block_adaptive_ok(self):
        bw = binarize(_text(), method="adaptive", block=20)
        assert bw.shape == _text().shape


# ─── TestSegmentWordsExtra ────────────────────────────────────────────────────

class TestSegmentWordsExtra:
    def test_blank_returns_list(self):
        assert isinstance(segment_words(_blank()), list)

    def test_text_finds_words(self):
        words = segment_words(_text())
        assert len(words) >= 1

    def test_returns_word_boxes(self):
        for w in segment_words(_text()):
            assert isinstance(w, WordBox)

    def test_sorted_by_y_then_x(self):
        words = segment_words(_text())
        yx = [(w.y, w.x) for w in words]
        assert yx == sorted(yx)

    def test_min_area_filter(self):
        words = segment_words(_text(), min_area=50)
        for w in words:
            assert w.area >= 50


# ─── TestMergeLineWordsExtra ──────────────────────────────────────────────────

class TestMergeLineWordsExtra:
    def test_empty_returns_empty(self):
        assert merge_line_words([]) == []

    def test_single_word_one_line(self):
        lines = merge_line_words([WordBox(x=0, y=5, w=20, h=10)])
        assert len(lines) == 1

    def test_same_row_grouped(self):
        words = [WordBox(x=0, y=10, w=20, h=10), WordBox(x=30, y=12, w=20, h=10)]
        lines = merge_line_words(words, line_gap=20.0)
        assert len(lines) == 1

    def test_different_rows_separated(self):
        words = [WordBox(x=0, y=5, w=20, h=10), WordBox(x=0, y=60, w=20, h=10)]
        lines = merge_line_words(words, line_gap=5.0)
        assert len(lines) == 2

    def test_line_idx_assigned(self):
        words = [WordBox(x=0, y=5, w=20, h=10), WordBox(x=30, y=5, w=20, h=10)]
        merge_line_words(words)
        for w in words:
            assert w.line_idx >= 0

    def test_returns_line_segments(self):
        lines = merge_line_words([WordBox(x=0, y=5, w=20, h=10)])
        assert all(isinstance(ln, LineSegment) for ln in lines)


# ─── TestSegmentDocumentExtra ─────────────────────────────────────────────────

class TestSegmentDocumentExtra:
    def test_returns_segmentation_result(self):
        assert isinstance(segment_document(_text()), WordSegmentationResult)

    def test_image_shape_stored(self):
        r = segment_document(_text(h=80, w=160))
        assert r.image_shape == (80, 160)

    def test_binarize_method_stored(self):
        r = segment_document(_text(), binarize_method="adaptive")
        assert r.binarize_method == "adaptive"

    def test_words_lines_consistent(self):
        r = segment_document(_text())
        total = sum(ln.n_words for ln in r.lines)
        assert total == r.n_words

    def test_blank_image_no_error(self):
        r = segment_document(_blank())
        assert isinstance(r, WordSegmentationResult)
        assert r.n_words >= 0

    def test_segment_lines_returns_list(self):
        assert isinstance(segment_lines(_text()), list)
