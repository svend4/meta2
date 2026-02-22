"""Extra tests for puzzle_reconstruction.algorithms.line_detector."""
import numpy as np
import pytest
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

def _blank(h=64, w=128):
    return np.zeros((h, w), dtype=np.uint8)


def _text_like(h=128, w=256):
    img = np.full((h, w), 255, dtype=np.uint8)
    for y in [20, 45, 70, 95]:
        img[y:y + 6, 15:w - 15] = 0
    return img


def _make_line(y_top=10, y_bottom=20, x_left=0, x_right=100, conf=1.0):
    return TextLine(
        y_top=y_top, y_bottom=y_bottom,
        x_left=x_left, x_right=x_right,
        height=y_bottom - y_top,
        width=x_right - x_left,
        confidence=conf,
    )


# ─── TextLine extras ──────────────────────────────────────────────────────────

class TestTextLineExtra:
    def test_center_y_arithmetic(self):
        ln = _make_line(y_top=0, y_bottom=40)
        assert ln.center_y == pytest.approx(20.0)

    def test_repr_is_string(self):
        assert isinstance(repr(_make_line()), str)

    def test_confidence_zero(self):
        ln = _make_line(conf=0.0)
        assert ln.confidence == pytest.approx(0.0)

    def test_confidence_one(self):
        ln = _make_line(conf=1.0)
        assert ln.confidence == pytest.approx(1.0)

    def test_height_stored(self):
        ln = _make_line(y_top=5, y_bottom=25)
        assert ln.height == 20

    def test_width_stored(self):
        ln = _make_line(x_left=10, x_right=90)
        assert ln.width == 80

    def test_height_zero_line(self):
        ln = _make_line(y_top=10, y_bottom=10)
        assert ln.height == 0

    def test_x_left_equals_right(self):
        ln = _make_line(x_left=50, x_right=50)
        assert ln.width == 0


# ─── LineDetectionResult extras ───────────────────────────────────────────────

class TestLineDetectionResultExtra:
    def _make(self, method="projection", n=0):
        return LineDetectionResult(
            lines=[_make_line() for _ in range(n)],
            method=method,
            line_height=10.0, line_spacing=5.0,
            skew_angle=0.0, n_lines=n,
        )

    def test_repr_is_string(self):
        assert isinstance(repr(self._make()), str)

    def test_lines_stored(self):
        r = self._make(n=3)
        assert len(r.lines) == 3

    def test_method_hough_stored(self):
        r = self._make(method="hough")
        assert r.method == "hough"

    def test_method_auto_stored(self):
        r = self._make(method="auto")
        assert r.method == "auto"

    def test_line_height_stored(self):
        r = self._make()
        assert r.line_height == pytest.approx(10.0)

    def test_line_spacing_stored(self):
        r = self._make()
        assert r.line_spacing == pytest.approx(5.0)

    def test_skew_angle_stored(self):
        r = LineDetectionResult(
            lines=[], method="hough",
            line_height=0.0, line_spacing=0.0,
            skew_angle=2.5, n_lines=0,
        )
        assert r.skew_angle == pytest.approx(2.5)


# ─── estimate_line_metrics extras ─────────────────────────────────────────────

class TestEstimateLineMetricsExtra:
    def test_three_lines_spacing(self):
        lines = [
            _make_line(y_top=0,  y_bottom=10),
            _make_line(y_top=20, y_bottom=30),
            _make_line(y_top=40, y_bottom=50),
        ]
        h, s = estimate_line_metrics(lines)
        assert h == pytest.approx(10.0)
        assert s == pytest.approx(10.0)

    def test_equal_spacing(self):
        lines = [_make_line(y_top=i * 15, y_bottom=i * 15 + 10) for i in range(4)]
        h, s = estimate_line_metrics(lines)
        assert h == pytest.approx(10.0)
        assert s == pytest.approx(5.0)

    def test_result_types_float(self):
        h, s = estimate_line_metrics([_make_line()])
        assert isinstance(h, float)
        assert isinstance(s, float)

    def test_height_nonneg(self):
        lines = [_make_line(y_top=i, y_bottom=i + 5) for i in range(3)]
        h, s = estimate_line_metrics(lines)
        assert h >= 0.0

    def test_spacing_nonneg(self):
        lines = [_make_line(y_top=i * 10, y_bottom=i * 10 + 5) for i in range(4)]
        _, s = estimate_line_metrics(lines)
        assert s >= 0.0


# ─── filter_lines extras ──────────────────────────────────────────────────────

class TestFilterLinesExtra:
    def test_keeps_all_valid(self):
        lines = [_make_line(y_top=0, y_bottom=12, x_left=0, x_right=80)]
        result = filter_lines(lines, min_height=5, max_height=50,
                               min_width_frac=0.5, img_width=100)
        assert len(result) == 1

    def test_returns_list(self):
        assert isinstance(filter_lines([_make_line()]), list)

    def test_all_removed_by_height(self):
        lines = [_make_line(y_top=0, y_bottom=3)]  # height=3 < min=5
        result = filter_lines(lines, min_height=5, max_height=100,
                               min_width_frac=0.0, img_width=100)
        assert len(result) == 0

    def test_width_boundary_kept(self):
        # width=50, min_width_frac=0.5, img_width=100 → min_width=50 → keep
        lines = [_make_line(y_top=0, y_bottom=10, x_left=0, x_right=50)]
        result = filter_lines(lines, min_height=4, max_height=80,
                               min_width_frac=0.5, img_width=100)
        assert len(result) == 1

    def test_multiple_lines_filtered(self):
        lines = [
            _make_line(y_top=0, y_bottom=10, x_left=0, x_right=80),  # valid
            _make_line(y_top=0, y_bottom=2, x_left=0, x_right=80),   # too short
            _make_line(y_top=0, y_bottom=10, x_left=0, x_right=10),  # too narrow
        ]
        result = filter_lines(lines, min_height=4, max_height=80,
                               min_width_frac=0.5, img_width=100)
        assert len(result) == 1


# ─── detect_lines_projection extras ──────────────────────────────────────────

class TestDetectLinesProjectionExtra:
    def test_large_image(self):
        img = np.full((256, 512), 255, dtype=np.uint8)
        for y in [30, 80, 130, 180, 230]:
            img[y:y + 5, 20:490] = 0
        r = detect_lines_projection(img)
        assert isinstance(r, LineDetectionResult)

    def test_non_square_image(self):
        img = np.full((64, 200), 255, dtype=np.uint8)
        r = detect_lines_projection(img)
        assert isinstance(r, LineDetectionResult)

    def test_small_image(self):
        img = np.zeros((16, 16), dtype=np.uint8)
        r = detect_lines_projection(img)
        assert isinstance(r, LineDetectionResult)

    def test_n_lines_equals_length(self):
        r = detect_lines_projection(_text_like())
        assert r.n_lines == len(r.lines)

    def test_all_lines_have_positive_height(self):
        r = detect_lines_projection(_text_like())
        for ln in r.lines:
            assert ln.height > 0

    def test_method_is_projection(self):
        r = detect_lines_projection(_blank())
        assert r.method == "projection"

    def test_skew_angle_is_float(self):
        r = detect_lines_projection(_text_like())
        assert isinstance(r.skew_angle, float)

    def test_lines_sorted_by_y_top(self):
        r = detect_lines_projection(_text_like())
        ytops = [ln.y_top for ln in r.lines]
        assert ytops == sorted(ytops)


# ─── detect_lines_hough extras ───────────────────────────────────────────────

class TestDetectLinesHoughExtra:
    def test_large_image_runs(self):
        img = np.full((256, 512), 255, dtype=np.uint8)
        for y in [50, 100, 150, 200]:
            img[y:y + 4, 10:490] = 0
        r = detect_lines_hough(img)
        assert isinstance(r, LineDetectionResult)

    def test_small_image_runs(self):
        r = detect_lines_hough(np.zeros((16, 16), dtype=np.uint8))
        assert isinstance(r, LineDetectionResult)

    def test_n_lines_nonneg(self):
        r = detect_lines_hough(_blank())
        assert r.n_lines >= 0

    def test_line_height_nonneg(self):
        r = detect_lines_hough(_text_like())
        assert r.line_height >= 0.0

    def test_line_spacing_nonneg(self):
        r = detect_lines_hough(_text_like())
        assert r.line_spacing >= 0.0

    def test_method_hough(self):
        r = detect_lines_hough(_text_like())
        assert r.method == "hough"


# ─── detect_text_lines extras ────────────────────────────────────────────────

class TestDetectTextLinesExtra:
    def test_all_methods_return_result(self):
        for method in ("projection", "hough", "auto"):
            r = detect_text_lines(_text_like(), method=method)
            assert isinstance(r, LineDetectionResult)

    def test_projection_result_method(self):
        r = detect_text_lines(_text_like(), method="projection")
        assert r.method == "projection"

    def test_hough_result_method(self):
        r = detect_text_lines(_text_like(), method="hough")
        assert r.method == "hough"

    def test_blank_image_any_method_ok(self):
        for method in ("projection", "hough", "auto"):
            r = detect_text_lines(_blank(), method=method)
            assert isinstance(r, LineDetectionResult)

    def test_n_lines_nonneg(self):
        r = detect_text_lines(_text_like())
        assert r.n_lines >= 0

    def test_params_fallback_in_auto_blank(self):
        r = detect_text_lines(_blank(), method="auto")
        assert "fallback" in r.params

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            detect_text_lines(_blank(), method="unknown_xyz")


# ─── batch_detect_lines extras ───────────────────────────────────────────────

class TestBatchDetectLinesExtra:
    def test_all_blank_ok(self):
        results = batch_detect_lines([_blank() for _ in range(3)])
        assert len(results) == 3
        assert all(r.n_lines == 0 for r in results)

    def test_projection_method(self):
        results = batch_detect_lines([_text_like()], method="projection")
        assert results[0].method == "projection"

    def test_hough_method(self):
        results = batch_detect_lines([_text_like()], method="hough")
        assert results[0].method == "hough"

    def test_single_image(self):
        results = batch_detect_lines([_text_like()])
        assert len(results) == 1

    def test_all_results_are_line_detection_result(self):
        imgs = [_blank(), _text_like(), _blank()]
        for r in batch_detect_lines(imgs):
            assert isinstance(r, LineDetectionResult)

    def test_five_images(self):
        imgs = [_text_like() if i % 2 == 0 else _blank() for i in range(5)]
        results = batch_detect_lines(imgs)
        assert len(results) == 5
