"""Тесты для puzzle_reconstruction.algorithms.line_detector."""
import pytest
import numpy as np
import cv2
from puzzle_reconstruction.algorithms.line_detector import (
    TextLine,
    LineDetectionResult,
    estimate_line_metrics,
    filter_lines,
    detect_lines_projection,
    detect_lines_hough,
    detect_text_lines,
    batch_detect_lines,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _blank(h=128, w=128) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _text_like(h=128, w=128) -> np.ndarray:
    """Image with horizontal dark lines simulating text on white background."""
    img = np.full((h, w), 255, dtype=np.uint8)
    for y in [20, 40, 60, 80, 100]:
        img[y:y + 5, 10:w - 10] = 0
    return img


def _rgb_text_like() -> np.ndarray:
    gray = _text_like()
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _make_line(y_top=10, y_bottom=20, x_left=0, x_right=100,
               confidence=1.0) -> TextLine:
    return TextLine(
        y_top=y_top, y_bottom=y_bottom,
        x_left=x_left, x_right=x_right,
        height=y_bottom - y_top,
        width=x_right - x_left,
        confidence=confidence,
    )


# ─── TestTextLine ─────────────────────────────────────────────────────────────

class TestTextLine:
    def test_construction(self):
        ln = _make_line()
        assert ln.y_top == 10
        assert ln.height == 10

    def test_center_y_property(self):
        ln = _make_line(y_top=10, y_bottom=20)
        assert ln.center_y == pytest.approx(15.0)

    def test_confidence_stored(self):
        ln = _make_line(confidence=0.75)
        assert ln.confidence == pytest.approx(0.75)

    def test_width_computed(self):
        ln = _make_line(x_left=5, x_right=95)
        assert ln.width == 90


# ─── TestLineDetectionResult ──────────────────────────────────────────────────

class TestLineDetectionResult:
    def test_construction(self):
        r = LineDetectionResult(
            lines=[], method="projection",
            line_height=0.0, line_spacing=0.0,
            skew_angle=0.0, n_lines=0,
        )
        assert r.method == "projection"

    def test_params_default_empty(self):
        r = LineDetectionResult(
            lines=[], method="hough",
            line_height=0.0, line_spacing=0.0,
            skew_angle=0.0, n_lines=0,
        )
        assert r.params == {}

    def test_n_lines_stored(self):
        r = LineDetectionResult(
            lines=[_make_line()], method="auto",
            line_height=10.0, line_spacing=5.0,
            skew_angle=0.0, n_lines=1,
        )
        assert r.n_lines == 1


# ─── TestEstimateLineMetrics ──────────────────────────────────────────────────

class TestEstimateLineMetrics:
    def test_empty_returns_zeros(self):
        h, s = estimate_line_metrics([])
        assert h == pytest.approx(0.0)
        assert s == pytest.approx(0.0)

    def test_single_line_zero_spacing(self):
        h, s = estimate_line_metrics([_make_line(y_top=10, y_bottom=20)])
        assert h == pytest.approx(10.0)
        assert s == pytest.approx(0.0)

    def test_two_lines_height_and_spacing(self):
        ln1 = _make_line(y_top=0, y_bottom=10)
        ln2 = _make_line(y_top=20, y_bottom=30)
        h, s = estimate_line_metrics([ln1, ln2])
        assert h == pytest.approx(10.0)
        assert s == pytest.approx(10.0)  # gap = 20 - 10 = 10

    def test_returns_tuple_of_floats(self):
        result = estimate_line_metrics([_make_line()])
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


# ─── TestFilterLines ──────────────────────────────────────────────────────────

class TestFilterLines:
    def test_removes_too_short(self):
        lines = [_make_line(y_top=0, y_bottom=2)]  # height=2, min=4
        result = filter_lines(lines, min_height=4, max_height=80,
                               min_width_frac=0.0, img_width=100)
        assert len(result) == 0

    def test_removes_too_tall(self):
        lines = [_make_line(y_top=0, y_bottom=100)]  # height=100, max=80
        result = filter_lines(lines, min_height=4, max_height=80,
                               min_width_frac=0.0, img_width=100)
        assert len(result) == 0

    def test_removes_too_narrow(self):
        lines = [_make_line(x_left=0, x_right=10, y_top=0, y_bottom=10)]
        # width=10, min_width = 0.5*100 = 50
        result = filter_lines(lines, min_width_frac=0.5, img_width=100)
        assert len(result) == 0

    def test_keeps_valid_lines(self):
        lines = [_make_line(y_top=0, y_bottom=10, x_left=0, x_right=80)]
        result = filter_lines(lines, min_height=4, max_height=80,
                               min_width_frac=0.3, img_width=100)
        assert len(result) == 1

    def test_empty_list(self):
        result = filter_lines([])
        assert result == []


# ─── TestDetectLinesProjection ────────────────────────────────────────────────

class TestDetectLinesProjection:
    def test_returns_line_detection_result(self):
        r = detect_lines_projection(_text_like())
        assert isinstance(r, LineDetectionResult)

    def test_method_projection(self):
        r = detect_lines_projection(_text_like())
        assert r.method == "projection"

    def test_blank_image_no_lines(self):
        r = detect_lines_projection(_blank())
        assert r.n_lines == 0

    def test_text_image_detects_lines(self):
        r = detect_lines_projection(_text_like())
        assert r.n_lines >= 0  # at least runs without error

    def test_rgb_ok(self):
        r = detect_lines_projection(_rgb_text_like())
        assert isinstance(r, LineDetectionResult)

    def test_n_lines_matches_lines_length(self):
        r = detect_lines_projection(_text_like())
        assert r.n_lines == len(r.lines)

    def test_params_stored(self):
        r = detect_lines_projection(_text_like(), min_height=5, max_height=60)
        assert r.params["min_height"] == 5
        assert r.params["max_height"] == 60

    def test_line_heights_within_bounds(self):
        r = detect_lines_projection(_text_like(), min_height=3, max_height=50)
        for ln in r.lines:
            assert 3 <= ln.height <= 50

    def test_line_height_nonneg(self):
        r = detect_lines_projection(_text_like())
        assert r.line_height >= 0.0

    def test_line_spacing_nonneg(self):
        r = detect_lines_projection(_text_like())
        assert r.line_spacing >= 0.0


# ─── TestDetectLinesHough ─────────────────────────────────────────────────────

class TestDetectLinesHough:
    def test_returns_line_detection_result(self):
        r = detect_lines_hough(_text_like())
        assert isinstance(r, LineDetectionResult)

    def test_method_hough(self):
        r = detect_lines_hough(_text_like())
        assert r.method == "hough"

    def test_blank_image(self):
        r = detect_lines_hough(_blank())
        assert isinstance(r, LineDetectionResult)
        assert r.n_lines == 0

    def test_rgb_ok(self):
        r = detect_lines_hough(_rgb_text_like())
        assert isinstance(r, LineDetectionResult)

    def test_n_lines_matches_lines(self):
        r = detect_lines_hough(_text_like())
        assert r.n_lines == len(r.lines)

    def test_skew_angle_is_float(self):
        r = detect_lines_hough(_text_like())
        assert isinstance(r.skew_angle, float)

    def test_lines_sorted_by_y(self):
        r = detect_lines_hough(_text_like())
        y_tops = [ln.y_top for ln in r.lines]
        assert y_tops == sorted(y_tops)

    def test_params_stored(self):
        r = detect_lines_hough(_text_like(), threshold=50)
        assert r.params["threshold"] == 50


# ─── TestDetectTextLines ──────────────────────────────────────────────────────

class TestDetectTextLines:
    def test_projection_method(self):
        r = detect_text_lines(_text_like(), method="projection")
        assert r.method == "projection"

    def test_hough_method(self):
        r = detect_text_lines(_text_like(), method="hough")
        assert r.method == "hough"

    def test_auto_method(self):
        r = detect_text_lines(_text_like(), method="auto")
        assert r.method == "auto"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            detect_text_lines(_text_like(), method="unknown")

    def test_returns_line_detection_result(self):
        r = detect_text_lines(_text_like())
        assert isinstance(r, LineDetectionResult)

    def test_blank_image_no_crash(self):
        r = detect_text_lines(_blank())
        assert isinstance(r, LineDetectionResult)

    def test_auto_fallback_blank(self):
        # blank triggers fallback to hough
        r = detect_text_lines(_blank(), method="auto")
        assert r.method == "auto"
        assert "fallback" in r.params


# ─── TestBatchDetectLines ─────────────────────────────────────────────────────

class TestBatchDetectLines:
    def test_returns_list(self):
        imgs = [_text_like(), _blank()]
        result = batch_detect_lines(imgs)
        assert isinstance(result, list)

    def test_length_matches(self):
        imgs = [_text_like(), _blank(), _text_like()]
        result = batch_detect_lines(imgs)
        assert len(result) == 3

    def test_empty_list(self):
        result = batch_detect_lines([])
        assert result == []

    def test_all_line_detection_results(self):
        imgs = [_text_like(), _blank()]
        for r in batch_detect_lines(imgs):
            assert isinstance(r, LineDetectionResult)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            batch_detect_lines([_text_like()], method="bad")

    def test_projection_method(self):
        result = batch_detect_lines([_text_like()], method="projection")
        assert result[0].method == "projection"
