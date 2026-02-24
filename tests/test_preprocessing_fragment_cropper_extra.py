"""Extra tests for puzzle_reconstruction/preprocessing/fragment_cropper.py."""
from __future__ import annotations

import numpy as np
import pytest

from puzzle_reconstruction.preprocessing.fragment_cropper import (
    CropResult,
    find_content_bbox,
    pad_image,
    crop_to_content,
    auto_crop,
    batch_crop,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _white(h=50, w=50):
    return np.full((h, w), 255, dtype=np.uint8)


def _bgr(h=50, w=50, val=128):
    return np.full((h, w, 3), val, dtype=np.uint8)


def _gray(h=50, w=50, val=128):
    return np.full((h, w), val, dtype=np.uint8)


def _with_dark_rect(h=60, w=60, top=10, left=10, rh=20, rw=20):
    """White image with a dark rectangle inside."""
    img = np.full((h, w), 255, dtype=np.uint8)
    img[top:top + rh, left:left + rw] = 30
    return img


# ─── CropResult ───────────────────────────────────────────────────────────────

class TestCropResultExtra:
    def test_fields(self):
        r = CropResult(cropped=_gray(20, 20), bbox=(5, 5, 20, 20),
                        padding=4, original_shape=(50, 50), method="content")
        assert r.bbox == (5, 5, 20, 20)
        assert r.method == "content"

    def test_repr(self):
        r = CropResult(cropped=_gray(20, 20), bbox=(5, 5, 20, 20),
                        padding=4, original_shape=(50, 50), method="auto")
        s = repr(r)
        assert "auto" in s

    def test_default_params(self):
        r = CropResult(cropped=_gray(), bbox=(0, 0, 50, 50),
                        padding=0, original_shape=(50, 50), method="content")
        assert r.params == {}


# ─── find_content_bbox ────────────────────────────────────────────────────────

class TestFindContentBboxExtra:
    def test_no_content(self):
        # All white → returns full frame
        x, y, w, h = find_content_bbox(_white())
        assert (x, y) == (0, 0)
        assert w == 50 and h == 50

    def test_with_content(self):
        img = _with_dark_rect(60, 60, 10, 10, 20, 20)
        x, y, w, h = find_content_bbox(img)
        assert x <= 10 and y <= 10
        assert x + w >= 30 and y + h >= 30

    def test_bgr_input(self):
        img = _bgr()
        x, y, w, h = find_content_bbox(img)
        assert w > 0 and h > 0

    def test_margin(self):
        img = _with_dark_rect(60, 60, 20, 20, 10, 10)
        x1, y1, w1, h1 = find_content_bbox(img, margin=0)
        x2, y2, w2, h2 = find_content_bbox(img, margin=5)
        assert w2 >= w1 and h2 >= h1


# ─── pad_image ────────────────────────────────────────────────────────────────

class TestPadImageExtra:
    def test_zero_padding(self):
        img = _gray()
        out = pad_image(img, padding=0)
        assert np.array_equal(out, img)

    def test_positive_padding(self):
        img = _gray(20, 20)
        out = pad_image(img, padding=5)
        assert out.shape == (30, 30)

    def test_fill_value(self):
        img = _gray(10, 10, val=0)
        out = pad_image(img, padding=2, fill=255)
        # Top border should be white
        assert np.all(out[:2, :] == 255)

    def test_bgr(self):
        img = _bgr(20, 20)
        out = pad_image(img, padding=3)
        assert out.shape == (26, 26, 3)

    def test_negative_padding_treated_as_zero(self):
        img = _gray()
        out = pad_image(img, padding=-5)
        assert np.array_equal(out, img)


# ─── crop_to_content ──────────────────────────────────────────────────────────

class TestCropToContentExtra:
    def test_returns_result(self):
        img = _with_dark_rect()
        r = crop_to_content(img)
        assert isinstance(r, CropResult)
        assert r.method == "content"

    def test_shape_smaller(self):
        img = _with_dark_rect(100, 100, 30, 30, 20, 20)
        r = crop_to_content(img, padding=2)
        ch, cw = r.cropped.shape[:2]
        assert ch < 100 and cw < 100

    def test_min_size(self):
        img = _with_dark_rect(100, 100, 45, 45, 2, 2)
        r = crop_to_content(img, min_size=10, padding=0)
        assert r.cropped.shape[0] >= 10
        assert r.cropped.shape[1] >= 10

    def test_original_shape(self):
        img = _gray(40, 60)
        r = crop_to_content(img)
        assert r.original_shape == (40, 60)

    def test_bgr(self):
        img = _bgr(50, 50, val=100)
        r = crop_to_content(img)
        assert r.cropped.ndim == 3


# ─── auto_crop ────────────────────────────────────────────────────────────────

class TestAutoCropExtra:
    def test_returns_result(self):
        img = _with_dark_rect()
        r = auto_crop(img)
        assert isinstance(r, CropResult)
        assert r.method == "auto"

    def test_all_white_image(self):
        img = _white(50, 50)
        r = auto_crop(img)
        assert isinstance(r, CropResult)

    def test_bgr(self):
        img = _bgr()
        r = auto_crop(img)
        assert r.cropped.ndim == 3

    def test_min_size(self):
        img = _white(50, 50)
        r = auto_crop(img, min_size=20)
        assert r.cropped.shape[0] >= 20
        assert r.cropped.shape[1] >= 20


# ─── batch_crop ───────────────────────────────────────────────────────────────

class TestBatchCropExtra:
    def test_empty(self):
        assert batch_crop([]) == []

    def test_length(self):
        results = batch_crop([_with_dark_rect(), _with_dark_rect()])
        assert len(results) == 2

    def test_content_method(self):
        results = batch_crop([_with_dark_rect()], method="content")
        assert results[0].method == "content"

    def test_auto_method(self):
        results = batch_crop([_with_dark_rect()], method="auto")
        assert results[0].method == "auto"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            batch_crop([_with_dark_rect()], method="bad")
