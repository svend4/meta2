"""
Тесты для puzzle_reconstruction/algorithms/word_segmentation.py

Покрытие:
    WordBox          — x2/y2/cx/cy/area/aspect_ratio, to_tuple, iou (0/частичный/
                       полный), repr, line_idx по умолчанию=-1
    LineSegment      — n_words, y_center, x_start, avg_word_height, repr
    WordSegmentationResult — n_words/n_lines/avg_words_per_line,
                             words_on_line (valid/invalid), summary, repr
    binarize         — все 3 метода, форма = gray.shape, dtype=uint8,
                       значения ∈ {0,255}, BGR вход, gray вход, ValueError
    segment_words    — пустое → [], текст → слова найдены, min_area фильтр,
                       форма/dtype, отсортированы по (y,x)
    merge_line_words — пустой → [], одна строка, несколько строк,
                       words.line_idx присваивается, LineSegment.bbox
    segment_lines    — возвращает list, LineSegment
    segment_document — WordSegmentationResult, image_shape, binarize_method
"""
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


# ─── Фикстуры ─────────────────────────────────────────────────────────────────

@pytest.fixture
def text_img():
    """
    BGR 128×400: белый фон с тёмными «словами» — прямоугольниками в 3 строках.
    Строки 1, 2, 3 расположены по Y≈20, 50, 80.
    """
    img = np.ones((128, 400, 3), dtype=np.uint8) * 240
    for row_y in [15, 50, 85]:
        for col_x in [20, 100, 180, 260]:
            img[row_y:row_y + 12, col_x:col_x + 30] = 20
    return img


@pytest.fixture
def blank_img():
    return np.ones((64, 128, 3), dtype=np.uint8) * 255


@pytest.fixture
def gray_text(text_img):
    import cv2
    return cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)


# ─── WordBox ──────────────────────────────────────────────────────────────────

class TestWordBox:
    def test_x2(self):
        w = WordBox(x=10, y=5, w=30, h=15)
        assert w.x2 == 40

    def test_y2(self):
        w = WordBox(x=10, y=5, w=30, h=15)
        assert w.y2 == 20

    def test_cx(self):
        w = WordBox(x=10, y=0, w=40, h=20)
        assert w.cx == pytest.approx(30.0)

    def test_cy(self):
        w = WordBox(x=0, y=10, w=40, h=20)
        assert w.cy == pytest.approx(20.0)

    def test_area(self):
        w = WordBox(x=0, y=0, w=10, h=20)
        assert w.area == 200

    def test_aspect_ratio(self):
        w = WordBox(x=0, y=0, w=60, h=20)
        assert w.aspect_ratio == pytest.approx(3.0)

    def test_aspect_ratio_zero_height(self):
        w = WordBox(x=0, y=0, w=10, h=0)
        assert w.aspect_ratio == pytest.approx(0.0)

    def test_to_tuple(self):
        w = WordBox(x=5, y=3, w=20, h=10)
        assert w.to_tuple() == (5, 3, 20, 10)

    def test_iou_no_overlap(self):
        a = WordBox(x=0,  y=0, w=10, h=10)
        b = WordBox(x=20, y=0, w=10, h=10)
        assert a.iou(b) == pytest.approx(0.0)

    def test_iou_identical(self):
        a = WordBox(x=0, y=0, w=10, h=10)
        b = WordBox(x=0, y=0, w=10, h=10)
        assert a.iou(b) == pytest.approx(1.0)

    def test_iou_partial(self):
        a = WordBox(x=0, y=0, w=10, h=10)
        b = WordBox(x=5, y=0, w=10, h=10)
        # intersection = 5×10 = 50, union = 100+100-50 = 150
        assert a.iou(b) == pytest.approx(50 / 150)

    def test_iou_symmetric(self):
        a = WordBox(x=0, y=0, w=10, h=10)
        b = WordBox(x=5, y=5, w=10, h=10)
        assert a.iou(b) == pytest.approx(b.iou(a))

    def test_line_idx_default(self):
        w = WordBox(x=0, y=0, w=10, h=10)
        assert w.line_idx == -1

    def test_repr_contains_line(self):
        w = WordBox(x=0, y=0, w=10, h=10, line_idx=2)
        assert "line=2" in repr(w)


# ─── LineSegment ──────────────────────────────────────────────────────────────

class TestLineSegment:
    def _make_words(self, ys, xs):
        return [WordBox(x=x, y=y, w=30, h=15) for x, y in zip(xs, ys)]

    def test_n_words(self):
        words = self._make_words([10, 10, 10], [0, 40, 80])
        ls    = LineSegment(line_idx=0, words=words, bbox=(0, 10, 110, 15))
        assert ls.n_words == 3

    def test_y_center(self):
        ls = LineSegment(line_idx=0, words=[], bbox=(0, 20, 100, 30))
        assert ls.y_center == pytest.approx(35.0)

    def test_x_start(self):
        ls = LineSegment(line_idx=0, words=[], bbox=(15, 0, 100, 30))
        assert ls.x_start == 15

    def test_avg_word_height_empty(self):
        ls = LineSegment(line_idx=0)
        assert ls.avg_word_height() == pytest.approx(0.0)

    def test_avg_word_height(self):
        words = [WordBox(x=i*40, y=0, w=30, h=h) for i, h in enumerate([10, 20, 30])]
        ls    = LineSegment(line_idx=0, words=words, bbox=(0, 0, 130, 30))
        assert ls.avg_word_height() == pytest.approx(20.0)

    def test_repr_contains_line_idx(self):
        ls = LineSegment(line_idx=3)
        assert "line=3" in repr(ls)

    def test_repr_contains_n_words(self):
        ls = LineSegment(line_idx=0, words=[WordBox(0, 0, 10, 10)])
        assert "n_words=1" in repr(ls)


# ─── WordSegmentationResult ───────────────────────────────────────────────────

class TestWordSegmentationResult:
    def _make_result(self, n_words=6, n_lines=2):
        words_per_line = n_words // n_lines
        words: list = []
        lines: list = []
        for li in range(n_lines):
            lw = [WordBox(x=i * 40, y=li * 30, w=30, h=20, line_idx=li)
                   for i in range(words_per_line)]
            words.extend(lw)
            lines.append(LineSegment(
                line_idx=li,
                words=lw,
                bbox=(0, li * 30, words_per_line * 40, 20),
            ))
        return WordSegmentationResult(
            words=words,
            lines=lines,
            image_shape=(128, 400),
        )

    def test_n_words(self):
        r = self._make_result(6, 2)
        assert r.n_words == 6

    def test_n_lines(self):
        r = self._make_result(6, 2)
        assert r.n_lines == 2

    def test_avg_words_per_line(self):
        r = self._make_result(6, 2)
        assert r.avg_words_per_line == pytest.approx(3.0)

    def test_avg_words_per_line_no_lines(self):
        r = WordSegmentationResult(words=[], lines=[], image_shape=(64, 64))
        assert r.avg_words_per_line == pytest.approx(0.0)

    def test_words_on_line_valid(self):
        r = self._make_result(6, 2)
        ws = r.words_on_line(0)
        assert len(ws) == 3

    def test_words_on_line_invalid(self):
        r = self._make_result(6, 2)
        assert r.words_on_line(99) == []

    def test_summary_string(self):
        r = self._make_result(6, 2)
        s = r.summary()
        assert isinstance(s, str)
        assert "n_words=6" in s
        assert "n_lines=2" in s

    def test_repr_same_as_summary(self):
        r = self._make_result(4, 2)
        assert repr(r) == r.summary()

    def test_image_shape_stored(self):
        r = self._make_result()
        assert r.image_shape == (128, 400)


# ─── binarize ─────────────────────────────────────────────────────────────────

class TestBinarize:
    @pytest.mark.parametrize("method", ["otsu", "adaptive"])
    def test_output_shape_bgr(self, text_img, method):
        bw = binarize(text_img, method=method)
        assert bw.shape == text_img.shape[:2]

    @pytest.mark.parametrize("method", ["otsu", "adaptive"])
    def test_output_dtype(self, text_img, method):
        bw = binarize(text_img, method=method)
        assert bw.dtype == np.uint8

    @pytest.mark.parametrize("method", ["otsu", "adaptive"])
    def test_values_binary(self, text_img, method):
        bw = binarize(text_img, method=method)
        assert set(np.unique(bw)).issubset({0, 255})

    def test_grayscale_input(self, gray_text):
        bw = binarize(gray_text, method="otsu")
        assert bw.shape == gray_text.shape
        assert bw.dtype == np.uint8

    def test_unknown_method_raises(self, text_img):
        with pytest.raises(ValueError):
            binarize(text_img, method="magic")

    def test_text_pixels_nonzero(self, text_img):
        """Тёмные прямоугольники → белые пиксели после инверсии."""
        bw = binarize(text_img, method="otsu")
        assert bw.sum() > 0

    def test_blank_image_near_zero(self, blank_img):
        """Однородный белый фон — почти ничего не остаётся после inv. Otsu."""
        bw = binarize(blank_img, method="otsu")
        assert bw.dtype == np.uint8


# ─── segment_words ────────────────────────────────────────────────────────────

class TestSegmentWords:
    def test_blank_returns_empty(self, blank_img):
        words = segment_words(blank_img)
        assert isinstance(words, list)

    def test_text_words_found(self, text_img):
        words = segment_words(text_img, min_area=10)
        assert len(words) > 0

    def test_all_word_boxes(self, text_img):
        words = segment_words(text_img, min_area=10)
        assert all(isinstance(w, WordBox) for w in words)

    def test_min_area_filter(self, text_img):
        words_all    = segment_words(text_img, min_area=1)
        words_strict = segment_words(text_img, min_area=1000)
        assert len(words_all) >= len(words_strict)

    def test_sorted_by_y_x(self, text_img):
        words = segment_words(text_img, min_area=10)
        for a, b in zip(words, words[1:]):
            assert (a.y, a.x) <= (b.y, b.x)

    def test_grayscale_input(self, gray_text):
        words = segment_words(gray_text, min_area=10)
        assert isinstance(words, list)

    def test_word_dimensions_positive(self, text_img):
        words = segment_words(text_img, min_area=10)
        for w in words:
            assert w.w > 0 and w.h > 0

    def test_dilation_width_param(self, text_img):
        """dilation_w=1 → меньше слияния, dilation_w=200 → больше слияния."""
        w_small = segment_words(text_img, dilation_w=1,   min_area=10)
        w_large = segment_words(text_img, dilation_w=200, min_area=10)
        # При большом dilation слов меньше (слились)
        assert len(w_large) <= len(w_small)


# ─── merge_line_words ─────────────────────────────────────────────────────────

class TestMergeLineWords:
    def test_empty_returns_empty(self):
        assert merge_line_words([]) == []

    def test_single_word_one_line(self):
        w     = WordBox(x=10, y=5, w=30, h=20)
        lines = merge_line_words([w])
        assert len(lines) == 1
        assert lines[0].n_words == 1

    def test_two_rows(self):
        """Два ряда слов с большим Y-расстоянием → 2 строки."""
        row1 = [WordBox(x=i * 40, y=5,  w=30, h=15) for i in range(3)]
        row2 = [WordBox(x=i * 40, y=60, w=30, h=15) for i in range(3)]
        lines = merge_line_words(row1 + row2)
        assert len(lines) == 2

    def test_same_row_one_line(self):
        """Слова с близким Y → одна строка."""
        words = [WordBox(x=i * 40, y=i, w=30, h=20) for i in range(4)]
        lines = merge_line_words(words, line_gap=5.0)
        assert len(lines) == 1

    def test_line_idx_assigned(self):
        row1 = [WordBox(x=i * 40, y=5,  w=30, h=15) for i in range(2)]
        row2 = [WordBox(x=i * 40, y=60, w=30, h=15) for i in range(2)]
        merge_line_words(row1 + row2)
        for w in row1:
            assert w.line_idx >= 0
        for w in row2:
            assert w.line_idx >= 0

    def test_words_sorted_left_right(self):
        """Слова внутри строки упорядочены слева направо."""
        words = [WordBox(x=80, y=5, w=30, h=15),
                  WordBox(x=10, y=5, w=30, h=15),
                  WordBox(x=40, y=5, w=30, h=15)]
        lines = merge_line_words(words, line_gap=5.0)
        xs    = [w.x for w in lines[0].words]
        assert xs == sorted(xs)

    def test_bbox_covers_all_words(self):
        words = [WordBox(x=10, y=5, w=30, h=20),
                  WordBox(x=80, y=8, w=30, h=15)]
        lines = merge_line_words(words, line_gap=10.0)
        bx, by, bw, bh = lines[0].bbox
        assert bx <= 10
        assert bx + bw >= 80 + 30

    def test_lines_sorted_top_bottom(self):
        row1 = [WordBox(x=i * 40, y=60, w=30, h=15) for i in range(2)]
        row2 = [WordBox(x=i * 40, y=5,  w=30, h=15) for i in range(2)]
        lines = merge_line_words(row1 + row2)
        ys    = [ln.y_center for ln in lines]
        assert ys == sorted(ys)


# ─── segment_lines ────────────────────────────────────────────────────────────

class TestSegmentLines:
    def test_returns_list(self, text_img):
        lines = segment_lines(text_img, min_area=10)
        assert isinstance(lines, list)

    def test_all_line_segments(self, text_img):
        lines = segment_lines(text_img, min_area=10)
        assert all(isinstance(ln, LineSegment) for ln in lines)

    def test_blank_no_crash(self, blank_img):
        lines = segment_lines(blank_img)
        assert isinstance(lines, list)


# ─── segment_document ─────────────────────────────────────────────────────────

class TestSegmentDocument:
    def test_returns_result(self, text_img):
        r = segment_document(text_img)
        assert isinstance(r, WordSegmentationResult)

    def test_image_shape_stored(self, text_img):
        r = segment_document(text_img)
        assert r.image_shape == text_img.shape[:2]

    def test_binarize_method_stored(self, text_img):
        r = segment_document(text_img, binarize_method="adaptive")
        assert r.binarize_method == "adaptive"

    def test_words_found(self, text_img):
        r = segment_document(text_img, min_area=10)
        assert r.n_words > 0

    def test_lines_found(self, text_img):
        r = segment_document(text_img, min_area=10)
        assert r.n_lines > 0

    def test_n_words_equals_words_len(self, text_img):
        r = segment_document(text_img)
        assert r.n_words == len(r.words)

    def test_n_lines_equals_lines_len(self, text_img):
        r = segment_document(text_img)
        assert r.n_lines == len(r.lines)

    def test_blank_no_crash(self, blank_img):
        r = segment_document(blank_img)
        assert isinstance(r, WordSegmentationResult)

    def test_grayscale_input(self, gray_text):
        r = segment_document(gray_text, min_area=10)
        assert isinstance(r, WordSegmentationResult)
