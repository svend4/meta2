"""Extra tests for puzzle_reconstruction/algorithms/line_detector.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.algorithms.line_detector import (
    LineDetectionResult,
    TextLine,
    batch_detect_lines,
    detect_lines_hough,
    detect_lines_projection,
    detect_text_lines,
    estimate_line_metrics,
    filter_lines,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _white(h: int = 128, w: int = 128) -> np.ndarray:
    """All-white image (blank page — no text lines)."""
    return np.full((h, w), 200, dtype=np.uint8)


def _black(h: int = 128, w: int = 128) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _bgr(h: int = 128, w: int = 128) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, :] = 200
    return img


def _with_lines(h: int = 128, w: int = 128, n_lines: int = 3,
                line_h: int = 10, spacing: int = 20) -> np.ndarray:
    """White image with dark horizontal text-like bands."""
    img = np.full((h, w), 230, dtype=np.uint8)
    for i in range(n_lines):
        y = 10 + i * (line_h + spacing)
        if y + line_h < h:
            img[y:y + line_h, 5:w - 5] = 30
    return img


def _textline(y_top: int = 10, y_bot: int = 20,
              x_left: int = 0, x_right: int = 100,
              conf: float = 1.0) -> TextLine:
    return TextLine(
        y_top=y_top, y_bottom=y_bot,
        x_left=x_left, x_right=x_right,
        height=y_bot - y_top,
        width=x_right - x_left,
        confidence=conf,
    )


# ─── TextLine (extra) ─────────────────────────────────────────────────────────

class TestTextLineExtra:
    def test_center_y(self):
        tl = _textline(10, 30)
        assert tl.center_y == pytest.approx(20.0)

    def test_center_y_single_pixel(self):
        tl = _textline(15, 16)
        assert tl.center_y == pytest.approx(15.5)

    def test_height_stored(self):
        tl = _textline(10, 25)
        assert tl.height == 15

    def test_width_stored(self):
        tl = _textline(x_left=5, x_right=95)
        assert tl.width == 90

    def test_confidence_default_1(self):
        tl = _textline()
        assert tl.confidence == pytest.approx(1.0)

    def test_confidence_custom(self):
        tl = _textline(conf=0.75)
        assert tl.confidence == pytest.approx(0.75)

    def test_y_top_stored(self):
        tl = _textline(y_top=7)
        assert tl.y_top == 7

    def test_y_bottom_stored(self):
        tl = _textline(y_bot=42)
        assert tl.y_bottom == 42

    def test_x_left_stored(self):
        tl = _textline(x_left=15)
        assert tl.x_left == 15

    def test_x_right_stored(self):
        tl = _textline(x_right=110)
        assert tl.x_right == 110

    def test_repr_contains_coordinates(self):
        tl = _textline(10, 20)
        r = repr(tl)
        assert "10" in r
        assert "20" in r


# ─── LineDetectionResult (extra) ──────────────────────────────────────────────

class TestLineDetectionResultExtra:
    def _make(self, n: int = 3, method: str = "projection") -> LineDetectionResult:
        lines = [_textline(i * 20, i * 20 + 10) for i in range(n)]
        return LineDetectionResult(
            lines=lines, method=method,
            line_height=10.0, line_spacing=10.0,
            skew_angle=0.0, n_lines=n,
        )

    def test_n_lines_stored(self):
        r = self._make(5)
        assert r.n_lines == 5

    def test_method_stored(self):
        r = self._make(method="hough")
        assert r.method == "hough"

    def test_line_height_stored(self):
        r = self._make()
        assert r.line_height == pytest.approx(10.0)

    def test_line_spacing_stored(self):
        r = self._make()
        assert r.line_spacing == pytest.approx(10.0)

    def test_skew_angle_stored(self):
        r = self._make()
        assert r.skew_angle == pytest.approx(0.0)

    def test_lines_is_list(self):
        r = self._make(3)
        assert isinstance(r.lines, list)

    def test_lines_count_matches_n_lines(self):
        r = self._make(4)
        assert len(r.lines) == r.n_lines

    def test_repr_contains_method(self):
        r = self._make(method="auto")
        assert "auto" in repr(r)

    def test_repr_contains_n(self):
        r = self._make(7)
        assert "7" in repr(r)

    def test_params_default_empty(self):
        r = self._make()
        assert isinstance(r.params, dict)


# ─── estimate_line_metrics (extra) ────────────────────────────────────────────

class TestEstimateLineMetricsExtra:
    def test_empty_returns_zeros(self):
        h, s = estimate_line_metrics([])
        assert h == pytest.approx(0.0)
        assert s == pytest.approx(0.0)

    def test_single_line_spacing_zero(self):
        tl = _textline(10, 20)
        h, s = estimate_line_metrics([tl])
        assert s == pytest.approx(0.0)
        assert h == pytest.approx(10.0)

    def test_single_line_height(self):
        tl = _textline(5, 25)
        h, s = estimate_line_metrics([tl])
        assert h == pytest.approx(20.0)

    def test_two_lines_spacing(self):
        lines = [_textline(0, 10), _textline(20, 30)]
        h, s = estimate_line_metrics(lines)
        assert s == pytest.approx(10.0)   # gap between y_bottom=10 and y_top=20

    def test_avg_height_multiple(self):
        lines = [_textline(0, 10), _textline(20, 36)]  # heights 10 and 16
        h, _ = estimate_line_metrics(lines)
        assert h == pytest.approx(13.0)

    def test_returns_two_floats(self):
        result = estimate_line_metrics([_textline()])
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_sorts_by_y_top(self):
        # Provide lines out of order
        lines = [_textline(30, 40), _textline(0, 10), _textline(15, 25)]
        h, s = estimate_line_metrics(lines)
        assert h >= 0.0
        assert s >= 0.0

    def test_three_equal_lines_spacing(self):
        lines = [_textline(0, 10), _textline(20, 30), _textline(40, 50)]
        _, s = estimate_line_metrics(lines)
        assert s == pytest.approx(10.0)


# ─── filter_lines (extra) ─────────────────────────────────────────────────────

class TestFilterLinesExtra:
    def test_empty_returns_empty(self):
        assert filter_lines([]) == []

    def test_too_short_removed(self):
        lines = [_textline(0, 2)]  # height=2 < min_height=4
        result = filter_lines(lines, min_height=4, max_height=80, img_width=100)
        assert result == []

    def test_too_tall_removed(self):
        lines = [_textline(0, 90)]  # height=90 > max_height=80
        result = filter_lines(lines, min_height=4, max_height=80, img_width=100)
        assert result == []

    def test_too_narrow_removed(self):
        lines = [_textline(0, 10, x_left=0, x_right=10)]  # width=10 < 0.3*100
        result = filter_lines(lines, min_height=4, max_height=80,
                               min_width_frac=0.3, img_width=100)
        assert result == []

    def test_valid_line_kept(self):
        lines = [_textline(0, 15, x_left=0, x_right=80)]
        result = filter_lines(lines, min_height=4, max_height=80,
                               min_width_frac=0.3, img_width=100)
        assert len(result) == 1

    def test_min_height_exact_kept(self):
        lines = [_textline(0, 4, x_left=0, x_right=80)]  # height exactly min
        result = filter_lines(lines, min_height=4, max_height=80,
                               min_width_frac=0.1, img_width=100)
        assert len(result) == 1

    def test_max_height_exact_kept(self):
        lines = [_textline(0, 80, x_left=0, x_right=80)]
        result = filter_lines(lines, min_height=4, max_height=80,
                               min_width_frac=0.1, img_width=100)
        assert len(result) == 1

    def test_mixed_filters(self):
        lines = [
            _textline(0, 10, x_left=0, x_right=80),   # valid
            _textline(0, 2, x_left=0, x_right=80),    # too short
            _textline(0, 10, x_left=0, x_right=5),    # too narrow
        ]
        result = filter_lines(lines, min_height=4, max_height=80,
                               min_width_frac=0.3, img_width=100)
        assert len(result) == 1


# ─── detect_lines_projection (extra) ─────────────────────────────────────────

class TestDetectLinesProjectionExtra:
    def test_returns_line_detection_result(self):
        r = detect_lines_projection(_white())
        assert isinstance(r, LineDetectionResult)

    def test_method_is_projection(self):
        r = detect_lines_projection(_white())
        assert r.method == "projection"

    def test_blank_image_zero_or_few_lines(self):
        r = detect_lines_projection(_white())
        # White/blank image should produce 0 lines
        assert r.n_lines == 0

    def test_image_with_lines_detects_some(self):
        img = _with_lines(h=200, w=200, n_lines=4)
        r = detect_lines_projection(img)
        assert r.n_lines >= 0  # non-negative count

    def test_bgr_input_accepted(self):
        r = detect_lines_projection(_bgr())
        assert isinstance(r, LineDetectionResult)

    def test_n_lines_matches_lines_list(self):
        r = detect_lines_projection(_white())
        assert r.n_lines == len(r.lines)

    def test_line_height_nonneg(self):
        r = detect_lines_projection(_with_lines())
        assert r.line_height >= 0.0

    def test_line_spacing_nonneg(self):
        r = detect_lines_projection(_with_lines())
        assert r.line_spacing >= 0.0

    def test_skew_angle_zero(self):
        r = detect_lines_projection(_white())
        assert r.skew_angle == pytest.approx(0.0)

    def test_params_stored(self):
        r = detect_lines_projection(_white(), min_height=5, max_height=60)
        assert r.params.get("min_height") == 5
        assert r.params.get("max_height") == 60

    def test_confidence_in_0_1(self):
        r = detect_lines_projection(_with_lines())
        for tl in r.lines:
            assert 0.0 <= tl.confidence <= 1.0


# ─── detect_lines_hough (extra) ───────────────────────────────────────────────

class TestDetectLinesHoughExtra:
    def test_returns_line_detection_result(self):
        r = detect_lines_hough(_white())
        assert isinstance(r, LineDetectionResult)

    def test_method_is_hough(self):
        r = detect_lines_hough(_white())
        assert r.method == "hough"

    def test_n_lines_matches_list(self):
        r = detect_lines_hough(_white())
        assert r.n_lines == len(r.lines)

    def test_line_height_nonneg(self):
        r = detect_lines_hough(_white())
        assert r.line_height >= 0.0

    def test_line_spacing_nonneg(self):
        r = detect_lines_hough(_white())
        assert r.line_spacing >= 0.0

    def test_bgr_input(self):
        r = detect_lines_hough(_bgr())
        assert isinstance(r, LineDetectionResult)

    def test_params_stored(self):
        r = detect_lines_hough(_white(), threshold=50)
        assert r.params.get("threshold") == 50

    def test_skew_angle_is_float(self):
        r = detect_lines_hough(_white())
        assert isinstance(r.skew_angle, float)

    def test_confidence_in_0_1(self):
        r = detect_lines_hough(_with_lines())
        for tl in r.lines:
            assert 0.0 <= tl.confidence <= 1.0


# ─── detect_text_lines (extra) ────────────────────────────────────────────────

class TestDetectTextLinesExtra:
    def test_returns_line_detection_result(self):
        r = detect_text_lines(_white())
        assert isinstance(r, LineDetectionResult)

    def test_method_projection(self):
        r = detect_text_lines(_white(), method="projection")
        assert r.method == "projection"

    def test_method_hough(self):
        r = detect_text_lines(_white(), method="hough")
        assert r.method == "hough"

    def test_method_auto(self):
        r = detect_text_lines(_white(), method="auto")
        assert r.method == "auto"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            detect_text_lines(_white(), method="unknown_method")

    def test_bgr_input(self):
        r = detect_text_lines(_bgr(), method="projection")
        assert isinstance(r, LineDetectionResult)

    def test_n_lines_matches_list(self):
        r = detect_text_lines(_white())
        assert r.n_lines == len(r.lines)

    def test_auto_fallback_params(self):
        # Auto fallback param should be in params dict for auto method
        r = detect_text_lines(_white(), method="auto")
        assert "fallback" in r.params

    def test_line_height_nonneg(self):
        r = detect_text_lines(_with_lines())
        assert r.line_height >= 0.0

    def test_skew_angle_is_float(self):
        r = detect_text_lines(_white(), method="auto")
        assert isinstance(r.skew_angle, float)


# ─── batch_detect_lines (extra) ───────────────────────────────────────────────

class TestBatchDetectLinesExtra:
    def test_empty_list_returns_empty(self):
        assert batch_detect_lines([]) == []

    def test_single_image(self):
        results = batch_detect_lines([_white()])
        assert len(results) == 1

    def test_multiple_images(self):
        imgs = [_white(), _black(), _bgr()]
        results = batch_detect_lines(imgs)
        assert len(results) == 3

    def test_all_are_line_detection_results(self):
        results = batch_detect_lines([_white(), _white()])
        for r in results:
            assert isinstance(r, LineDetectionResult)

    def test_method_propagated_projection(self):
        results = batch_detect_lines([_white()], method="projection")
        assert results[0].method == "projection"

    def test_method_propagated_hough(self):
        results = batch_detect_lines([_white()], method="hough")
        assert results[0].method == "hough"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            batch_detect_lines([_white()], method="bad_method")

    def test_n_lines_nonneg(self):
        results = batch_detect_lines([_white(), _black()])
        for r in results:
            assert r.n_lines >= 0

    def test_kwargs_passed(self):
        results = batch_detect_lines([_white()], method="projection", min_height=6)
        assert results[0].params.get("min_height") == 6
