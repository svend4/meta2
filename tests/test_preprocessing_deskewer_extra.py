"""Extra tests for puzzle_reconstruction/preprocessing/deskewer.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.deskewer import (
    DeskewResult,
    estimate_skew_projection,
    estimate_skew_hough,
    deskew_image,
    auto_deskew,
    batch_deskew,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _bgr(h=50, w=80, val=200):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _gray(h=50, w=80, val=200):
    return np.full((h, w), val, dtype=np.uint8)


def _text_like(h=100, w=200):
    """Create a simple image with horizontal dark lines (text-like)."""
    img = np.full((h, w), 240, dtype=np.uint8)
    for y in range(10, h - 10, 12):
        img[y:y + 2, 20:w - 20] = 30
    return img


# ─── DeskewResult ─────────────────────────────────────────────────────────────

class TestDeskewResultExtra:
    def test_fields(self):
        r = DeskewResult(corrected=_gray(), angle=1.5,
                         method="projection", confidence=0.9)
        assert r.angle == pytest.approx(1.5)
        assert r.method == "projection"
        assert r.confidence == pytest.approx(0.9)

    def test_repr(self):
        r = DeskewResult(corrected=_gray(), angle=2.0,
                         method="hough", confidence=0.8)
        s = repr(r)
        assert "2.00" in s
        assert "hough" in s

    def test_default_confidence(self):
        r = DeskewResult(corrected=_gray(), angle=0.0, method="projection")
        assert r.confidence == pytest.approx(0.0)

    def test_default_params(self):
        r = DeskewResult(corrected=_gray(), angle=0.0, method="projection")
        assert r.params == {}


# ─── estimate_skew_projection ─────────────────────────────────────────────────

class TestEstimateSkewProjectionExtra:
    def test_returns_tuple(self):
        angle, conf = estimate_skew_projection(_text_like())
        assert isinstance(angle, float)
        assert isinstance(conf, float)

    def test_confidence_range(self):
        _, conf = estimate_skew_projection(_text_like())
        assert 0.0 <= conf <= 1.0

    def test_angle_in_range(self):
        angle, _ = estimate_skew_projection(_text_like(), angle_range=(-10, 10))
        assert -10.0 <= angle <= 10.0

    def test_gray_input(self):
        angle, conf = estimate_skew_projection(_gray())
        assert isinstance(angle, float)

    def test_bgr_input(self):
        angle, conf = estimate_skew_projection(_bgr())
        assert isinstance(angle, float)


# ─── estimate_skew_hough ──────────────────────────────────────────────────────

class TestEstimateSkewHoughExtra:
    def test_returns_tuple(self):
        angle, conf = estimate_skew_hough(_text_like())
        assert isinstance(angle, float)
        assert isinstance(conf, float)

    def test_confidence_range(self):
        _, conf = estimate_skew_hough(_text_like())
        assert 0.0 <= conf <= 1.0

    def test_blank_image(self):
        # Blank white image: no lines found → (0.0, 0.0)
        angle, conf = estimate_skew_hough(_gray(val=255))
        assert angle == pytest.approx(0.0)
        assert conf == pytest.approx(0.0)

    def test_bgr_input(self):
        angle, conf = estimate_skew_hough(_bgr())
        assert isinstance(angle, float)


# ─── deskew_image ─────────────────────────────────────────────────────────────

class TestDeskewImageExtra:
    def test_shape_preserved(self):
        img = _bgr()
        out = deskew_image(img, angle=5.0)
        assert out.shape == img.shape

    def test_zero_angle_identity(self):
        img = _gray()
        out = deskew_image(img, angle=0.0)
        # With zero angle, output is essentially the same
        assert out.shape == img.shape

    def test_dtype_preserved(self):
        img = _gray()
        out = deskew_image(img, angle=3.0)
        assert out.dtype == np.uint8

    def test_grayscale(self):
        img = _gray()
        out = deskew_image(img, angle=2.0)
        assert out.ndim == 2

    def test_bgr(self):
        img = _bgr()
        out = deskew_image(img, angle=2.0)
        assert out.ndim == 3


# ─── auto_deskew ──────────────────────────────────────────────────────────────

class TestAutoDeskewExtra:
    def test_projection_method(self):
        r = auto_deskew(_text_like(), method="projection")
        assert isinstance(r, DeskewResult)
        assert r.method == "projection"

    def test_hough_method(self):
        r = auto_deskew(_text_like(), method="hough")
        assert r.method == "hough"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            auto_deskew(_text_like(), method="unknown")

    def test_shape_preserved(self):
        img = _text_like()
        r = auto_deskew(img)
        assert r.corrected.shape == img.shape

    def test_params_populated(self):
        r = auto_deskew(_text_like(), method="projection")
        assert "n_angles" in r.params

    def test_bgr_input(self):
        r = auto_deskew(_bgr(), method="projection")
        assert r.corrected.shape == _bgr().shape


# ─── batch_deskew ─────────────────────────────────────────────────────────────

class TestBatchDeskewExtra:
    def test_empty(self):
        assert batch_deskew([]) == []

    def test_length(self):
        imgs = [_text_like(), _text_like()]
        results = batch_deskew(imgs)
        assert len(results) == 2

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_deskew([_text_like()], method="bad")

    def test_result_types(self):
        results = batch_deskew([_text_like()])
        assert isinstance(results[0], DeskewResult)
