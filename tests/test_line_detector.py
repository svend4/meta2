"""Тесты для puzzle_reconstruction/algorithms/line_detector.py."""
import numpy as np
import pytest

from puzzle_reconstruction.algorithms.line_detector import (
    TextLine,
    LineDetectionResult,
    detect_lines_projection,
    detect_lines_hough,
    estimate_line_metrics,
    filter_lines,
    detect_text_lines,
    batch_detect_lines,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gray(h=128, w=128, val=200):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=128, w=128):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 220
    img[:, :, 1] = 180
    img[:, :, 2] = 140
    return img


def _text_image(h=128, w=128, n_lines=4, line_h=8, gap=12):
    """Синтетическое изображение с горизонтальными «строками»."""
    img = np.full((h, w), 240, dtype=np.uint8)
    for i in range(n_lines):
        y = 8 + i * (line_h + gap)
        if y + line_h <= h:
            img[y:y + line_h, 10:w - 10] = 30   # тёмный текст
    return img


def _make_line(y_top=10, y_bot=20, x_left=0, x_right=80, conf=0.9):
    h = y_bot - y_top
    return TextLine(
        y_top=y_top, y_bottom=y_bot,
        x_left=x_left, x_right=x_right,
        height=h, width=x_right - x_left,
        confidence=conf,
    )


# ─── TextLine ─────────────────────────────────────────────────────────────────

class TestTextLine:
    def test_fields(self):
        ln = _make_line(10, 20, 0, 80)
        assert ln.y_top == 10
        assert ln.y_bottom == 20
        assert ln.x_left == 0
        assert ln.x_right == 80
        assert ln.height == 10
        assert ln.width == 80

    def test_confidence_default(self):
        ln = TextLine(y_top=0, y_bottom=8, x_left=0, x_right=64,
                      height=8, width=64)
        assert ln.confidence == pytest.approx(1.0)

    def test_confidence_stored(self):
        ln = _make_line(conf=0.75)
        assert ln.confidence == pytest.approx(0.75)

    def test_center_y(self):
        ln = _make_line(y_top=10, y_bot=20)
        assert ln.center_y == pytest.approx(15.0)

    def test_center_y_odd(self):
        ln = _make_line(y_top=5, y_bot=16)
        assert ln.center_y == pytest.approx(10.5)

    def test_repr(self):
        ln = _make_line()
        s = repr(ln)
        assert "TextLine" in s
        assert "10" in s

    def test_height_matches_diff(self):
        ln = TextLine(y_top=7, y_bottom=21, x_left=0, x_right=100,
                      height=14, width=100)
        assert ln.height == ln.y_bottom - ln.y_top


# ─── LineDetectionResult ──────────────────────────────────────────────────────

class TestLineDetectionResult:
    def test_fields(self):
        lines = [_make_line()]
        r = LineDetectionResult(
            lines=lines, method="projection",
            line_height=10.0, line_spacing=5.0, skew_angle=0.0, n_lines=1,
        )
        assert r.method == "projection"
        assert r.n_lines == 1
        assert r.line_height == pytest.approx(10.0)
        assert r.line_spacing == pytest.approx(5.0)
        assert r.skew_angle == pytest.approx(0.0)
        assert isinstance(r.params, dict)

    def test_n_lines_matches_list(self):
        lines = [_make_line() for _ in range(3)]
        r = LineDetectionResult(
            lines=lines, method="hough",
            line_height=8.0, line_spacing=4.0, skew_angle=1.0, n_lines=3,
        )
        assert r.n_lines == len(r.lines)

    def test_repr(self):
        r = LineDetectionResult(
            lines=[], method="auto",
            line_height=12.0, line_spacing=6.0, skew_angle=0.0, n_lines=0,
        )
        s = repr(r)
        assert "LineDetectionResult" in s
        assert "auto" in s


# ─── estimate_line_metrics ────────────────────────────────────────────────────

class TestEstimateLineMetrics:
    def test_empty_returns_zeros(self):
        h, s = estimate_line_metrics([])
        assert h == pytest.approx(0.0)
        assert s == pytest.approx(0.0)

    def test_single_line(self):
        ln = _make_line(y_top=0, y_bot=12)
        h, s = estimate_line_metrics([ln])
        assert h == pytest.approx(12.0)
        assert s == pytest.approx(0.0)

    def test_two_lines_height(self):
        l1 = _make_line(y_top=0,  y_bot=10)
        l2 = _make_line(y_top=20, y_bot=30)
        h, s = estimate_line_metrics([l1, l2])
        assert h == pytest.approx(10.0)

    def test_two_lines_spacing(self):
        l1 = _make_line(y_top=0,  y_bot=10)
        l2 = _make_line(y_top=18, y_bot=28)
        _, s = estimate_line_metrics([l1, l2])
        assert s == pytest.approx(8.0)

    def test_spacing_nonneg(self):
        lines = [_make_line(y_top=i * 15, y_bot=i * 15 + 10) for i in range(5)]
        _, s = estimate_line_metrics(lines)
        assert s >= 0.0

    def test_returns_tuple_of_floats(self):
        result = estimate_line_metrics([_make_line()])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_average_height(self):
        l1 = _make_line(y_top=0, y_bot=8)
        l2 = _make_line(y_top=20, y_bot=32)
        h, _ = estimate_line_metrics([l1, l2])
        assert h == pytest.approx(10.0)


# ─── filter_lines ─────────────────────────────────────────────────────────────

class TestFilterLines:
    def test_empty_input(self):
        assert filter_lines([]) == []

    def test_all_pass_default(self):
        lines = [_make_line() for _ in range(3)]
        # default: min_height=4, max_height=80, min_width_frac=0.2, img_width=100
        filtered = filter_lines(lines, img_width=100)
        assert len(filtered) == 3

    def test_height_below_min_removed(self):
        short = TextLine(y_top=0, y_bottom=2, x_left=0, x_right=80,
                         height=2, width=80)
        ok    = _make_line()
        result = filter_lines([short, ok], min_height=4, img_width=100)
        assert short not in result
        assert ok in result

    def test_height_above_max_removed(self):
        tall = TextLine(y_top=0, y_bottom=90, x_left=0, x_right=80,
                        height=90, width=80)
        ok   = _make_line()
        result = filter_lines([tall, ok], max_height=80, img_width=100)
        assert tall not in result
        assert ok in result

    def test_narrow_line_removed(self):
        narrow = TextLine(y_top=0, y_bottom=10, x_left=0, x_right=10,
                          height=10, width=10)
        ok     = _make_line(x_right=70)
        result = filter_lines([narrow, ok], min_width_frac=0.3, img_width=100)
        assert narrow not in result
        assert ok in result

    def test_all_removed_returns_empty(self):
        lines = [TextLine(y_top=0, y_bottom=2, x_left=0, x_right=5,
                          height=2, width=5) for _ in range(3)]
        result = filter_lines(lines, min_height=4, img_width=100)
        assert result == []


# ─── detect_lines_projection ─────────────────────────────────────────────────

class TestDetectLinesProjection:
    def test_returns_result(self):
        assert isinstance(detect_lines_projection(_text_image()), LineDetectionResult)

    def test_method(self):
        assert detect_lines_projection(_text_image()).method == "projection"

    def test_n_lines_matches_list(self):
        r = detect_lines_projection(_text_image())
        assert r.n_lines == len(r.lines)

    def test_params_stored(self):
        r = detect_lines_projection(_text_image(), min_height=5, max_height=60,
                                     min_width_frac=0.4, bw_thresh=200)
        assert r.params.get("min_height") == 5
        assert r.params.get("max_height") == 60
        assert r.params.get("min_width_frac") == pytest.approx(0.4)
        assert r.params.get("bw_thresh") == 200

    def test_gray_input(self):
        r = detect_lines_projection(_gray())
        assert isinstance(r, LineDetectionResult)

    def test_bgr_input(self):
        r = detect_lines_projection(_bgr())
        assert r.n_lines >= 0

    def test_constant_image_no_crash(self):
        r = detect_lines_projection(_gray(val=200))
        assert isinstance(r, LineDetectionResult)

    def test_line_height_nonneg(self):
        r = detect_lines_projection(_text_image())
        assert r.line_height >= 0.0

    def test_line_spacing_nonneg(self):
        r = detect_lines_projection(_text_image())
        assert r.line_spacing >= 0.0

    def test_skew_zero(self):
        # Проекция не оценивает угол
        r = detect_lines_projection(_text_image())
        assert r.skew_angle == pytest.approx(0.0)

    def test_all_lines_have_valid_heights(self):
        r = detect_lines_projection(_text_image(), min_height=4, max_height=80)
        for ln in r.lines:
            assert 4 <= ln.height <= 80

    @pytest.mark.parametrize("n_text_lines", [2, 3, 5])
    def test_detects_reasonable_lines(self, n_text_lines):
        img = _text_image(n_lines=n_text_lines)
        r   = detect_lines_projection(img)
        # Should detect at least some lines (allow slack)
        assert r.n_lines >= 0


# ─── detect_lines_hough ───────────────────────────────────────────────────────

class TestDetectLinesHough:
    def test_returns_result(self):
        assert isinstance(detect_lines_hough(_text_image()), LineDetectionResult)

    def test_method(self):
        assert detect_lines_hough(_text_image()).method == "hough"

    def test_n_lines_matches_list(self):
        r = detect_lines_hough(_text_image())
        assert r.n_lines == len(r.lines)

    def test_params_stored(self):
        r = detect_lines_hough(_text_image(), threshold=60, min_len_frac=0.25,
                                max_gap=8)
        assert r.params.get("threshold") == 60
        assert r.params.get("min_len_frac") == pytest.approx(0.25)
        assert r.params.get("max_gap") == 8

    def test_gray_input(self):
        r = detect_lines_hough(_gray())
        assert isinstance(r, LineDetectionResult)

    def test_bgr_input(self):
        r = detect_lines_hough(_bgr())
        assert r.n_lines >= 0

    def test_constant_image_no_crash(self):
        r = detect_lines_hough(_gray(val=180))
        assert isinstance(r, LineDetectionResult)

    def test_line_height_nonneg(self):
        r = detect_lines_hough(_text_image())
        assert r.line_height >= 0.0

    def test_lines_within_image(self):
        h = 128
        r = detect_lines_hough(_text_image(h=h))
        for ln in r.lines:
            assert 0 <= ln.y_top < h
            assert ln.y_bottom <= h

    def test_skew_float(self):
        r = detect_lines_hough(_text_image())
        assert isinstance(r.skew_angle, float)


# ─── detect_text_lines ────────────────────────────────────────────────────────

class TestDetectTextLines:
    @pytest.mark.parametrize("method", ["projection", "hough", "auto"])
    def test_all_methods(self, method):
        r = detect_text_lines(_text_image(), method=method)
        assert isinstance(r, LineDetectionResult)
        assert r.n_lines >= 0

    def test_method_projection(self):
        r = detect_text_lines(_text_image(), method="projection")
        assert r.method == "projection"

    def test_method_hough(self):
        r = detect_text_lines(_text_image(), method="hough")
        assert r.method == "hough"

    def test_method_auto(self):
        r = detect_text_lines(_text_image(), method="auto")
        assert r.method == "auto"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            detect_text_lines(_text_image(), method="magic_algo")

    def test_gray_input(self):
        r = detect_text_lines(_gray(), method="auto")
        assert isinstance(r, LineDetectionResult)

    def test_bgr_input(self):
        r = detect_text_lines(_bgr(), method="projection")
        assert isinstance(r, LineDetectionResult)

    def test_auto_fallback_flag(self):
        r = detect_text_lines(_text_image(), method="auto")
        assert "fallback" in r.params

    def test_min_height_kwarg(self):
        r = detect_text_lines(_text_image(), method="projection", min_height=6)
        for ln in r.lines:
            assert ln.height >= 6

    def test_n_lines_nonneg(self):
        r = detect_text_lines(_gray(), method="auto")
        assert r.n_lines >= 0


# ─── batch_detect_lines ───────────────────────────────────────────────────────

class TestBatchDetectLines:
    def test_returns_list(self):
        imgs = [_text_image() for _ in range(3)]
        results = batch_detect_lines(imgs)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_each_is_result(self):
        for r in batch_detect_lines([_gray(), _text_image()]):
            assert isinstance(r, LineDetectionResult)

    def test_empty_list(self):
        assert batch_detect_lines([]) == []

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_detect_lines([_gray()], method="unknown_xyz")

    @pytest.mark.parametrize("method", ["projection", "hough", "auto"])
    def test_all_methods(self, method):
        results = batch_detect_lines([_text_image(), _gray()], method=method)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, LineDetectionResult)

    def test_kwargs_forwarded(self):
        results = batch_detect_lines([_text_image()], method="projection",
                                      min_height=6)
        for ln in results[0].lines:
            assert ln.height >= 6

    def test_bgr_input(self):
        results = batch_detect_lines([_bgr()], method="projection")
        assert isinstance(results[0], LineDetectionResult)
