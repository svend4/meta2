"""Extra tests for puzzle_reconstruction.preprocessing.skew_correction."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.skew_correction import (
    SkewResult,
    auto_correct_skew,
    batch_correct_skew,
    correct_skew,
    detect_skew_fft,
    detect_skew_hough,
    detect_skew_projection,
    skew_confidence,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _bgr_lines(h=128, w=256):
    img = np.ones((h, w, 3), dtype=np.uint8) * 240
    for y in range(15, h, 25):
        img[y:y + 3, 10:w - 10] = 30
    return img


def _gray_lines(h=128, w=256):
    img = np.ones((h, w), dtype=np.uint8) * 240
    for y in range(15, h, 25):
        img[y:y + 3, 10:w - 10] = 30
    return img


def _blank(h=64, w=128):
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def _blank_gray(h=64, w=128):
    return np.ones((h, w), dtype=np.uint8) * 255


# ─── TestSkewResultExtra ─────────────────────────────────────────────────────

class TestSkewResultExtra:
    def _make(self, method="hough"):
        return auto_correct_skew(_bgr_lines(), method=method)

    def test_corrected_image_ndarray(self):
        r = self._make()
        assert isinstance(r.corrected_image, np.ndarray)

    def test_angle_deg_is_float(self):
        r = self._make()
        assert isinstance(r.angle_deg, float)

    def test_confidence_in_0_1(self):
        r = self._make()
        assert 0.0 <= r.confidence <= 1.0

    def test_method_stored_hough(self):
        r = self._make("hough")
        assert r.method == "hough"

    def test_method_stored_projection(self):
        r = self._make("projection")
        assert r.method == "projection"

    def test_method_stored_fft(self):
        r = self._make("fft")
        assert r.method == "fft"

    def test_params_is_dict(self):
        r = self._make()
        assert isinstance(r.params, dict)

    def test_repr_is_str(self):
        r = self._make()
        assert isinstance(repr(r), str)

    def test_repr_contains_method(self):
        r = self._make("projection")
        assert "projection" in repr(r)

    def test_shape_preserved_bgr(self):
        img = _bgr_lines()
        r = auto_correct_skew(img, method="hough")
        assert r.corrected_image.shape == img.shape


# ─── TestDetectSkewHoughExtra ─────────────────────────────────────────────────

class TestDetectSkewHoughExtra:
    def test_returns_float(self):
        assert isinstance(detect_skew_hough(_bgr_lines()), float)

    def test_blank_returns_zero(self):
        assert detect_skew_hough(_blank()) == pytest.approx(0.0)

    def test_gray_input_ok(self):
        assert isinstance(detect_skew_hough(_gray_lines()), float)

    def test_angle_range_param_ok(self):
        a = detect_skew_hough(_bgr_lines(), angle_range=5.0)
        assert isinstance(a, float)

    def test_horizontal_lines_small_angle(self):
        a = detect_skew_hough(_bgr_lines())
        assert abs(a) < 15.0


# ─── TestDetectSkewProjectionExtra ───────────────────────────────────────────

class TestDetectSkewProjectionExtra:
    def test_returns_float(self):
        assert isinstance(detect_skew_projection(_bgr_lines()), float)

    def test_in_lo_hi_range(self):
        lo, hi = -20.0, 20.0
        a = detect_skew_projection(_bgr_lines(), lo=lo, hi=hi)
        assert lo <= a <= hi

    def test_gray_input_ok(self):
        assert isinstance(detect_skew_projection(_gray_lines()), float)

    def test_n_angles_param_ok(self):
        assert isinstance(detect_skew_projection(_bgr_lines(), n_angles=20), float)

    def test_blank_no_crash(self):
        assert isinstance(detect_skew_projection(_blank()), float)

    def test_horizontal_small_angle(self):
        a = detect_skew_projection(_bgr_lines(), n_angles=60)
        assert abs(a) < 20.0


# ─── TestDetectSkewFftExtra ───────────────────────────────────────────────────

class TestDetectSkewFftExtra:
    def test_returns_float(self):
        assert isinstance(detect_skew_fft(_bgr_lines()), float)

    def test_in_clamp_range(self):
        a = detect_skew_fft(_bgr_lines())
        assert -45.0 <= a <= 45.0

    def test_gray_input_ok(self):
        assert isinstance(detect_skew_fft(_gray_lines()), float)

    def test_sigma_0_no_crash(self):
        assert isinstance(detect_skew_fft(_bgr_lines(), sigma=0.0), float)

    def test_sigma_large_no_crash(self):
        assert isinstance(detect_skew_fft(_bgr_lines(), sigma=5.0), float)

    def test_blank_no_crash(self):
        assert isinstance(detect_skew_fft(_blank()), float)

    def test_n_angles_param_ok(self):
        assert isinstance(detect_skew_fft(_bgr_lines(), n_angles=36), float)


# ─── TestCorrectSkewExtra ─────────────────────────────────────────────────────

class TestCorrectSkewExtra:
    def test_shape_preserved_bgr(self):
        img = _bgr_lines()
        assert correct_skew(img, 5.0).shape == img.shape

    def test_shape_preserved_gray(self):
        img = _gray_lines()
        assert correct_skew(img, 5.0).shape == img.shape

    def test_dtype_uint8(self):
        assert correct_skew(_bgr_lines(), 3.0).dtype == np.uint8

    def test_zero_angle_near_identity(self):
        img = _bgr_lines()
        out = correct_skew(img, 0.0)
        diff = np.abs(out.astype(int) - img.astype(int))
        assert diff.mean() < 5.0

    def test_negative_angle_ok(self):
        img = _bgr_lines()
        assert correct_skew(img, -10.0).shape == img.shape

    def test_large_angle_ok(self):
        img = _bgr_lines()
        assert correct_skew(img, 45.0).shape == img.shape


# ─── TestSkewConfidenceExtra ──────────────────────────────────────────────────

class TestSkewConfidenceExtra:
    def test_empty_zero(self):
        assert skew_confidence([]) == pytest.approx(0.0)

    def test_single_half(self):
        assert skew_confidence([5.0]) == pytest.approx(0.5)

    def test_identical_one(self):
        assert skew_confidence([3.0, 3.0, 3.0]) == pytest.approx(1.0)

    def test_large_spread_low(self):
        assert skew_confidence([-30.0, 0.0, 30.0], tol=2.0) < 0.5

    def test_small_spread_high(self):
        assert skew_confidence([1.0, 1.1, 1.2], tol=2.0) > 0.8

    def test_result_in_0_1(self):
        for angles in [[], [0.0], [5.0, 6.0], [-20.0, 20.0]]:
            c = skew_confidence(angles)
            assert 0.0 <= c <= 1.0

    def test_two_identical_one(self):
        assert skew_confidence([7.0, 7.0]) == pytest.approx(1.0)


# ─── TestAutoCorrectSkewExtra ─────────────────────────────────────────────────

class TestAutoCorrectSkewExtra:
    @pytest.mark.parametrize("method", ["hough", "projection", "fft", "auto"])
    def test_returns_skew_result(self, method):
        assert isinstance(auto_correct_skew(_bgr_lines(), method=method), SkewResult)

    @pytest.mark.parametrize("method", ["hough", "projection", "fft", "auto"])
    def test_shape_preserved(self, method):
        img = _bgr_lines()
        r = auto_correct_skew(img, method=method)
        assert r.corrected_image.shape == img.shape

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            auto_correct_skew(_bgr_lines(), method="magic")

    def test_method_stored(self):
        r = auto_correct_skew(_bgr_lines(), method="fft")
        assert r.method == "fft"

    def test_confidence_in_range(self):
        for method in ["hough", "projection", "fft", "auto"]:
            r = auto_correct_skew(_bgr_lines(), method=method)
            assert 0.0 <= r.confidence <= 1.0

    def test_auto_angles_all_in_params(self):
        r = auto_correct_skew(_bgr_lines(), method="auto")
        assert "angles_all" in r.params
        assert len(r.params["angles_all"]) == 3

    def test_grayscale_input(self):
        r = auto_correct_skew(_gray_lines(), method="hough")
        assert r.corrected_image.shape == _gray_lines().shape

    def test_blank_no_crash(self):
        for method in ["hough", "projection", "fft", "auto"]:
            assert isinstance(auto_correct_skew(_blank(), method=method), SkewResult)


# ─── TestBatchCorrectSkewExtra ────────────────────────────────────────────────

class TestBatchCorrectSkewExtra:
    def test_empty_list(self):
        assert batch_correct_skew([]) == []

    def test_single_image(self):
        results = batch_correct_skew([_bgr_lines()])
        assert len(results) == 1

    def test_two_images_length(self):
        results = batch_correct_skew([_bgr_lines(), _blank()])
        assert len(results) == 2

    def test_three_images_length(self):
        imgs = [_bgr_lines(), _blank(), _gray_lines()]
        assert len(batch_correct_skew(imgs)) == 3

    def test_all_skew_results(self):
        results = batch_correct_skew([_bgr_lines(), _blank()])
        assert all(isinstance(r, SkewResult) for r in results)

    def test_method_propagated(self):
        results = batch_correct_skew([_bgr_lines()], method="fft")
        assert results[0].method == "fft"
