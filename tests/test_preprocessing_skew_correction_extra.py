"""Extra tests for puzzle_reconstruction/preprocessing/skew_correction.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.skew_correction import (
    SkewResult,
    detect_skew_hough,
    detect_skew_projection,
    detect_skew_fft,
    correct_skew,
    skew_confidence,
    auto_correct_skew,
    batch_correct_skew,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _gray(h=80, w=200, val=240):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(h=80, w=200, val=240):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _text_like(h=100, w=200):
    """Image with horizontal dark lines simulating text."""
    img = np.full((h, w), 240, dtype=np.uint8)
    for y in range(10, h - 10, 12):
        img[y:y + 2, 20:w - 20] = 30
    return img


# ─── SkewResult ──────────────────────────────────────────────────────────────

class TestSkewResultExtra:
    def test_fields(self):
        r = SkewResult(corrected_image=_gray(), angle_deg=2.5,
                       confidence=0.8, method="hough")
        assert r.angle_deg == pytest.approx(2.5)
        assert r.method == "hough"
        assert r.confidence == pytest.approx(0.8)

    def test_repr(self):
        r = SkewResult(corrected_image=_gray(), angle_deg=3.14,
                       confidence=0.9, method="projection")
        s = repr(r)
        assert "3.14" in s
        assert "projection" in s

    def test_default_params(self):
        r = SkewResult(corrected_image=_gray(), angle_deg=0.0,
                       confidence=0.0, method="fft")
        assert r.params == {}

    def test_custom_params(self):
        r = SkewResult(corrected_image=_gray(), angle_deg=1.0,
                       confidence=0.5, method="auto",
                       params={"n_angles": 180})
        assert r.params["n_angles"] == 180


# ─── detect_skew_hough ───────────────────────────────────────────────────────

class TestDetectSkewHoughExtra:
    def test_returns_float(self):
        angle = detect_skew_hough(_text_like())
        assert isinstance(angle, float)

    def test_blank_image_zero(self):
        angle = detect_skew_hough(_gray(val=255))
        assert angle == pytest.approx(0.0)

    def test_bgr_input(self):
        angle = detect_skew_hough(_bgr())
        assert isinstance(angle, float)

    def test_angle_within_range(self):
        angle = detect_skew_hough(_text_like(), angle_range=10.0)
        assert -10.0 <= angle <= 10.0


# ─── detect_skew_projection ─────────────────────────────────────────────────

class TestDetectSkewProjectionExtra:
    def test_returns_float(self):
        angle = detect_skew_projection(_text_like())
        assert isinstance(angle, float)

    def test_angle_in_default_range(self):
        angle = detect_skew_projection(_text_like())
        assert -45.0 <= angle <= 45.0

    def test_custom_range(self):
        angle = detect_skew_projection(_text_like(), lo=-10.0, hi=10.0)
        assert -10.0 <= angle <= 10.0

    def test_bgr_input(self):
        angle = detect_skew_projection(_bgr())
        assert isinstance(angle, float)


# ─── detect_skew_fft ────────────────────────────────────────────────────────

class TestDetectSkewFftExtra:
    def test_returns_float(self):
        angle = detect_skew_fft(_text_like())
        assert isinstance(angle, float)

    def test_angle_bounded(self):
        angle = detect_skew_fft(_text_like())
        assert -45.0 <= angle <= 45.0

    def test_bgr_input(self):
        angle = detect_skew_fft(_bgr())
        assert isinstance(angle, float)

    def test_zero_sigma(self):
        angle = detect_skew_fft(_text_like(), sigma=0.0)
        assert isinstance(angle, float)


# ─── correct_skew ───────────────────────────────────────────────────────────

class TestCorrectSkewExtra:
    def test_shape_preserved(self):
        img = _bgr()
        out = correct_skew(img, 5.0)
        assert out.shape == img.shape

    def test_zero_angle(self):
        img = _gray()
        out = correct_skew(img, 0.0)
        assert out.shape == img.shape

    def test_dtype_preserved(self):
        img = _gray()
        out = correct_skew(img, 3.0)
        assert out.dtype == np.uint8

    def test_grayscale(self):
        img = _gray()
        out = correct_skew(img, 2.0)
        assert out.ndim == 2

    def test_bgr(self):
        img = _bgr()
        out = correct_skew(img, 2.0)
        assert out.ndim == 3


# ─── skew_confidence ────────────────────────────────────────────────────────

class TestSkewConfidenceExtra:
    def test_empty_returns_zero(self):
        assert skew_confidence([]) == pytest.approx(0.0)

    def test_single_angle(self):
        assert skew_confidence([5.0]) == pytest.approx(0.5)

    def test_identical_angles(self):
        conf = skew_confidence([2.0, 2.0, 2.0])
        assert conf == pytest.approx(1.0)

    def test_close_angles_high(self):
        conf = skew_confidence([1.0, 1.5, 1.2], tol=2.0)
        assert conf > 0.9

    def test_spread_angles_low(self):
        conf = skew_confidence([0.0, 10.0], tol=2.0)
        assert conf < 0.1

    def test_range_zero_to_one(self):
        conf = skew_confidence([1.0, 3.0], tol=2.0)
        assert 0.0 <= conf <= 1.0


# ─── auto_correct_skew ──────────────────────────────────────────────────────

class TestAutoCorrectSkewExtra:
    def test_hough_method(self):
        r = auto_correct_skew(_text_like(), method="hough")
        assert isinstance(r, SkewResult)
        assert r.method == "hough"

    def test_projection_method(self):
        r = auto_correct_skew(_text_like(), method="projection")
        assert r.method == "projection"

    def test_fft_method(self):
        r = auto_correct_skew(_text_like(), method="fft")
        assert r.method == "fft"

    def test_auto_method(self):
        r = auto_correct_skew(_text_like(), method="auto")
        assert r.method == "auto"
        assert "angles_all" in r.params

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            auto_correct_skew(_text_like(), method="bad")

    def test_shape_preserved(self):
        img = _text_like()
        r = auto_correct_skew(img)
        assert r.corrected_image.shape == img.shape

    def test_bgr_input(self):
        r = auto_correct_skew(_bgr(), method="hough")
        assert r.corrected_image.shape == _bgr().shape


# ─── batch_correct_skew ─────────────────────────────────────────────────────

class TestBatchCorrectSkewExtra:
    def test_empty(self):
        assert batch_correct_skew([]) == []

    def test_length(self):
        results = batch_correct_skew([_text_like(), _text_like()])
        assert len(results) == 2

    def test_result_type(self):
        results = batch_correct_skew([_text_like()])
        assert isinstance(results[0], SkewResult)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_correct_skew([_text_like()], method="bad")
