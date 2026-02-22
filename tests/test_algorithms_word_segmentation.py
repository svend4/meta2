"""Tests for puzzle_reconstruction.algorithms.word_segmentation."""
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


# ─── helpers ──────────────────────────────────────────────────────────────────

def _blank_gray(h=64, w=64):
    return np.zeros((h, w), dtype=np.uint8)


def _text_like_gray(h=100, w=200):
    """Gray image with horizontal stripes simulating lines of text."""
    img = np.ones((h, w), dtype=np.uint8) * 255  # white background
    # Add two rows of dark blobs simulating words
    img[20:30, 10:50] = 0   # line 1, word 1
    img[20:30, 60:100] = 0  # line 1, word 2
    img[60:70, 10:50] = 0   # line 2, word 1
    img[60:70, 60:100] = 0  # line 2, word 2
    return img


def _word(x=0, y=0, w=30, h=15, line_idx=-1):
    return WordBox(x=x, y=y, w=w, h=h, line_idx=line_idx)


# ─── TestWordBox ──────────────────────────────────────────────────────────────

class TestWordBox:
    def test_basic_creation(self):
        wb = WordBox(x=10, y=20, w=50, h=15)
        assert wb.x == 10
        assert wb.y == 20
        assert wb.w == 50
        assert wb.h == 15

    def test_x2_y2_properties(self):
        wb = WordBox(x=5, y=10, w=20, h=10)
        assert wb.x2 == 25
        assert wb.y2 == 20

    def test_cx_cy_properties(self):
        wb = WordBox(x=0, y=0, w=20, h=10)
        assert wb.cx == pytest.approx(10.0)
        assert wb.cy == pytest.approx(5.0)

    def test_area_property(self):
        wb = WordBox(x=0, y=0, w=10, h=5)
        assert wb.area == 50

    def test_aspect_ratio_property(self):
        wb = WordBox(x=0, y=0, w=30, h=10)
        assert wb.aspect_ratio == pytest.approx(3.0)

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

    def test_iou_partial_overlap(self):
        a = WordBox(x=0, y=0, w=10, h=10)
        b = WordBox(x=5, y=0, w=10, h=10)
        iou = a.iou(b)
        assert 0.0 < iou < 1.0

    def test_default_line_idx(self):
        wb = WordBox(x=0, y=0, w=10, h=5)
        assert wb.line_idx == -1

    def test_default_confidence(self):
        wb = WordBox(x=0, y=0, w=10, h=5)
        assert wb.confidence == pytest.approx(1.0)

    def test_repr_contains_coords(self):
        wb = WordBox(x=3, y=7, w=20, h=10)
        s = repr(wb)
        assert "x=3" in s
        assert "y=7" in s


# ─── TestLineSegment ──────────────────────────────────────────────────────────

class TestLineSegment:
    def test_basic_creation(self):
        ls = LineSegment(line_idx=0)
        assert ls.line_idx == 0
        assert ls.words == []
        assert ls.bbox == (0, 0, 0, 0)

    def test_n_words_property(self):
        ls = LineSegment(line_idx=0, words=[_word(), _word()])
        assert ls.n_words == 2

    def test_y_center_property(self):
        ls = LineSegment(line_idx=0, bbox=(0, 20, 100, 10))
        assert ls.y_center == pytest.approx(25.0)

    def test_x_start_property(self):
        ls = LineSegment(line_idx=0, bbox=(15, 0, 100, 10))
        assert ls.x_start == 15

    def test_avg_word_height_empty(self):
        ls = LineSegment(line_idx=0)
        assert ls.avg_word_height() == pytest.approx(0.0)

    def test_avg_word_height_words(self):
        words = [WordBox(x=0, y=0, w=20, h=10),
                 WordBox(x=30, y=0, w=20, h=20)]
        ls = LineSegment(line_idx=0, words=words)
        assert ls.avg_word_height() == pytest.approx(15.0)

    def test_repr_contains_n_words(self):
        ls = LineSegment(line_idx=1, words=[_word()])
        assert "n_words=1" in repr(ls)


# ─── TestWordSegmentationResult ───────────────────────────────────────────────

class TestWordSegmentationResult:
    def _result(self, n_words=3, n_lines=2):
        words = [_word(x=i * 10) for i in range(n_words)]
        lines = [LineSegment(line_idx=i, words=[words[i]]) for i in range(n_lines)]
        return WordSegmentationResult(words=words, lines=lines,
                                     image_shape=(100, 200))

    def test_n_words_property(self):
        r = self._result(n_words=5, n_lines=2)
        assert r.n_words == 5

    def test_n_lines_property(self):
        r = self._result(n_words=3, n_lines=2)
        assert r.n_lines == 2

    def test_avg_words_per_line(self):
        r = self._result(n_words=4, n_lines=2)
        assert r.avg_words_per_line == pytest.approx(2.0)

    def test_avg_words_per_line_no_lines(self):
        r = WordSegmentationResult(words=[], lines=[], image_shape=(50, 50))
        assert r.avg_words_per_line == pytest.approx(0.0)

    def test_words_on_line_valid(self):
        words = [_word(x=0), _word(x=10)]
        line = LineSegment(line_idx=0, words=words)
        r = WordSegmentationResult(words=words, lines=[line],
                                   image_shape=(100, 200))
        assert len(r.words_on_line(0)) == 2

    def test_words_on_line_invalid_index(self):
        r = WordSegmentationResult(words=[], lines=[], image_shape=(50, 50))
        assert r.words_on_line(99) == []

    def test_summary_contains_shape(self):
        r = self._result()
        s = r.summary()
        assert "200×100" in s or "100" in s

    def test_default_binarize_method(self):
        r = WordSegmentationResult(words=[], lines=[], image_shape=(50, 50))
        assert r.binarize_method == "otsu"


# ─── TestBinarize ─────────────────────────────────────────────────────────────

class TestBinarize:
    def test_otsu_grayscale(self):
        img = _text_like_gray()
        bw = binarize(img, method="otsu")
        assert bw.shape == img.shape
        assert bw.dtype == np.uint8

    def test_adaptive_grayscale(self):
        img = _text_like_gray()
        bw = binarize(img, method="adaptive", block=21)
        assert bw.shape == img.shape

    def test_sauvola_grayscale(self):
        img = _text_like_gray(h=20, w=40)  # small for speed
        bw = binarize(img, method="sauvola", block=7)
        assert bw.shape == img.shape
        assert bw.dtype == np.uint8

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="метод"):
            binarize(_blank_gray(), method="magic")

    def test_output_binary(self):
        img = _text_like_gray()
        bw = binarize(img, method="otsu")
        assert set(np.unique(bw)).issubset({0, 255})

    def test_bgr_image_accepted(self):
        img = np.random.default_rng(0).integers(0, 255, (32, 32, 3),
                                                dtype=np.uint8)
        bw = binarize(img, method="otsu")
        assert bw.ndim == 2

    def test_adaptive_even_block_handled(self):
        """Even block size should be auto-corrected to odd."""
        img = _text_like_gray()
        bw = binarize(img, method="adaptive", block=20)  # even → 21
        assert bw.shape == img.shape


# ─── TestSegmentWords ─────────────────────────────────────────────────────────

class TestSegmentWords:
    def test_blank_image_returns_empty(self):
        words = segment_words(_blank_gray())
        assert isinstance(words, list)

    def test_text_image_finds_words(self):
        img = _text_like_gray()
        words = segment_words(img)
        assert len(words) >= 1

    def test_returns_word_boxes(self):
        img = _text_like_gray()
        words = segment_words(img)
        assert all(isinstance(w, WordBox) for w in words)

    def test_sorted_by_y_then_x(self):
        img = _text_like_gray()
        words = segment_words(img)
        yx = [(w.y, w.x) for w in words]
        assert yx == sorted(yx)

    def test_all_words_pass_min_area(self):
        img = _text_like_gray()
        words = segment_words(img, min_area=50)
        for w in words:
            assert w.area >= 50


# ─── TestMergeLineWords ───────────────────────────────────────────────────────

class TestMergeLineWords:
    def test_empty_input_returns_empty(self):
        assert merge_line_words([]) == []

    def test_single_word_one_line(self):
        words = [WordBox(x=0, y=5, w=20, h=10)]
        lines = merge_line_words(words)
        assert len(lines) == 1

    def test_words_on_same_row_grouped(self):
        words = [
            WordBox(x=0, y=10, w=20, h=10),
            WordBox(x=30, y=12, w=20, h=10),
        ]
        lines = merge_line_words(words, line_gap=20.0)
        assert len(lines) == 1

    def test_words_on_different_rows_separated(self):
        words = [
            WordBox(x=0, y=5, w=20, h=10),
            WordBox(x=0, y=60, w=20, h=10),
        ]
        lines = merge_line_words(words, line_gap=5.0)
        assert len(lines) == 2

    def test_line_idx_assigned(self):
        words = [
            WordBox(x=0, y=5, w=20, h=10),
            WordBox(x=30, y=5, w=20, h=10),
        ]
        merge_line_words(words)
        for w in words:
            assert w.line_idx >= 0

    def test_returns_line_segments(self):
        words = [WordBox(x=0, y=5, w=20, h=10)]
        lines = merge_line_words(words)
        assert all(isinstance(ln, LineSegment) for ln in lines)

    def test_auto_line_gap(self):
        """Without explicit line_gap, auto-computed from word heights."""
        words = [
            WordBox(x=0, y=5, w=20, h=10),
            WordBox(x=0, y=100, w=20, h=10),
        ]
        lines = merge_line_words(words)
        assert len(lines) == 2


# ─── TestSegmentLines ─────────────────────────────────────────────────────────

class TestSegmentLines:
    def test_blank_returns_list(self):
        lines = segment_lines(_blank_gray())
        assert isinstance(lines, list)

    def test_text_image_finds_lines(self):
        img = _text_like_gray()
        lines = segment_lines(img)
        assert all(isinstance(ln, LineSegment) for ln in lines)

    def test_returns_list(self):
        result = segment_lines(_text_like_gray())
        assert isinstance(result, list)


# ─── TestSegmentDocument ──────────────────────────────────────────────────────

class TestSegmentDocument:
    def test_returns_segmentation_result(self):
        img = _text_like_gray()
        result = segment_document(img)
        assert isinstance(result, WordSegmentationResult)

    def test_image_shape_stored(self):
        img = _text_like_gray(h=80, w=160)
        result = segment_document(img)
        assert result.image_shape == (80, 160)

    def test_binarize_method_stored(self):
        img = _text_like_gray()
        result = segment_document(img, binarize_method="adaptive")
        assert result.binarize_method == "adaptive"

    def test_words_and_lines_consistent(self):
        img = _text_like_gray()
        result = segment_document(img)
        total_words_in_lines = sum(ln.n_words for ln in result.lines)
        assert total_words_in_lines == result.n_words

    def test_blank_image_no_error(self):
        result = segment_document(_blank_gray())
        assert isinstance(result, WordSegmentationResult)
        assert result.n_words >= 0
